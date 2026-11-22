"""Train the decoder of model.py on shots drawn on the fly from the detector error model.

Run from challenge-1-ml-decoder-kit/. A first run, and the same run continued later:

  python -m neural.train --out runs/x1 --seed 1
  python -m neural.train --out runs/x1 --resume runs/x1/checkpoint.pt

The recipe of arXiv:2604.08358 at a small scale:
- a curriculum in three stages: `--steps-start` steps at the low noise `--p-start`, then
  `--steps-anneal` steps that climb to `--p-train` in `--anneal-levels` equal jumps, then the
  rest of the run at `--p-train`;
- a moving average of the weights (the reports and the saved decoder use it), mixed precision
  on a GPU, gradient clipping, a warm-up and a cosine decay of the learning rate to one tenth.
The paper's optimizers (Muon and Lion) are replaced by AdamW.

checkpoint.pt is rewritten at every report, and every `--checkpoint-minutes` in between. A copy
checkpoint_step<N>.pt is kept every `--snapshot-every` steps: the decoders along the climb,
for a later comparison at each noise. To stop a run cleanly, create an empty file named STOP
in `--out`: the run prints a report, writes the checkpoint, removes the file and ends.

Every schedule counts steps, never seconds, and the shots of step s depend only on `--seed`
and s. A resumed run therefore continues exactly where the first one stopped.

Each report prints, for p = 0.001 to 0.005, on fresh shots from the error model (not Stim
shots; the same shots in every report):
  shots wrong   the share of shots with at least one of the 12 logical bits predicted wrong
  bits right    the share of single bits predicted right
and next to each the value of always answering 0. The lines that start with "speed" depend
on the machine; every other line is the same when the same command is run twice on it.
"""
import argparse
import math
import os
import shutil
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.nn import functional as F

from neural.model import CascadeDecoder, count_parameters
from neural.sampler import DemSampler

DEFAULTS = {
    "code": "gross", "basis": "X", "rounds": None, "detectors": "all",
    "width": 128, "depth": 8, "bottleneck": 4, "residual": "branch",
    "batch": 128, "steps": 400000, "lr": 1e-3, "warmup_steps": 1000, "lr_decay_steps": None,
    "weight_decay": 3e-3, "clip": 1.0, "ema": 0.9998,
    "p_start": 0.001, "p_train": 0.0055, "steps_start": 4000, "steps_anneal": 24000, "anneal_levels": 12,
    "report_every": 2000, "report_shots": 2000, "report_p": [0.001, 0.002, 0.003, 0.004, 0.005],
    "snapshot_every": 10000, "seed": 0, "amp": "auto",
}
TINY = {"width": 32, "depth": 3, "batch": 32}   # for tests and smoke runs on a CPU
MODEL_KEYS = ["code", "basis", "rounds", "detectors", "width", "depth", "bottleneck", "residual"]
# Settings that may change when a run is resumed. Everything else is fixed by the checkpoint.
RUN_CONTROL = ["steps", "lr_decay_steps", "report_every", "report_shots", "snapshot_every"]
TRAIN_STREAM, REPORT_STREAM = 1, 2


@dataclass
class Run:
    """Everything a training run owns. `step` counts finished optimizer steps."""
    config: dict
    device: torch.device
    model: CascadeDecoder
    averaged_model: CascadeDecoder                 # filled from `average` when it is asked for: see ema_model
    optimizer: torch.optim.Optimizer
    scaler: torch.amp.GradScaler
    sampler: DemSampler
    generator: torch.Generator
    step: int = 0
    history: list = field(default_factory=list)    # one entry per report
    loss_sum: float = 0.0                          # training loss since the last report
    loss_steps: int = 0
    average: torch.Tensor = None                   # the moving average of all weights, as one long vector
    weights: list = None                           # the model's tensors that the average follows

    @property
    def ema_model(self):
        """The decoder with the averaged weights.

        The average is kept as one vector, so that a training step updates it with one
        operation instead of one per tensor. Here it is written into a model. The counters of
        the normalisation layers (not floating point) are copied from the trained model.
        """
        with torch.no_grad():
            start = 0
            for tensor in float_tensors(self.averaged_model):
                tensor.copy_(self.average[start:start + tensor.numel()].view_as(tensor))
                start += tensor.numel()
            current = self.model.state_dict()
            for name, tensor in self.averaged_model.state_dict().items():
                if not tensor.dtype.is_floating_point:
                    tensor.copy_(current[name])
        return self.averaged_model

    def start_average(self, model):
        """Begin the moving average at the weights of `model`."""
        self.weights = float_tensors(self.model)
        self.average = flat(float_tensors(model)).clone()

    @property
    def amp_dtype(self):
        """The mixed-precision type, or None for plain 32-bit floats."""
        choice = self.config["amp"]
        if choice == "auto":
            choice = "fp16" if self.device.type == "cuda" else "off"
        return {"fp16": torch.float16, "bf16": torch.bfloat16, "off": None}[choice]


def float_tensors(model):
    """The weights and the normalisation statistics of a model, in a fixed order."""
    return [tensor for tensor in model.state_dict().values() if tensor.dtype.is_floating_point]


def flat(tensors):
    """One long vector of all the numbers in a list of tensors."""
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


# ---------- schedules: plain functions of the step ----------

def noise_levels(config):
    """The noise strengths of the curriculum: p_start, then anneal_levels equal jumps up to p_train."""
    n, low, high = config["anneal_levels"], config["p_start"], config["p_train"]
    return [round(low + (high - low) * k / n, 7) for k in range(n + 1)]


def noise_at(step, config):
    """The noise strength of the shots of one step."""
    levels, into_anneal = noise_levels(config), step - config["steps_start"]
    if into_anneal < 0:
        return levels[0]
    if into_anneal >= config["steps_anneal"]:
        return levels[-1]
    return levels[1 + into_anneal * config["anneal_levels"] // config["steps_anneal"]]


def learning_rate_at(step, config):
    """Linear warm-up, then a cosine from the peak down to a tenth of it, then constant."""
    warm_up = min(1.0, (step + 1) / max(1, config["warmup_steps"]))
    progress = min(1.0, step / max(1, config["lr_decay_steps"]))
    return config["lr"] * warm_up * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def stream_seed(seed, stream, index):
    """A well-mixed 63-bit seed for one batch of shots (the splitmix64 finalizer)."""
    x = (seed * 0x9E3779B97F4A7C15 + stream * 0xBF58476D1CE4E5B9 + index * 0x94D049BB133111EB) % 2 ** 64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) % 2 ** 64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) % 2 ** 64
    return (x ^ (x >> 31)) >> 1


# ---------- building, saving and loading a run ----------

def new_run(config, device):
    """A run at step 0. The seed fixes the initial weights and, with the step, every batch of shots."""
    device = torch.device(device)
    torch.manual_seed(config["seed"])
    model = CascadeDecoder(**{key: config[key] for key in MODEL_KEYS}).to(device)
    config = dict(config, rounds=model.config["rounds"])
    matrices = [p for p in model.parameters() if p.ndim > 1]
    others = [p for p in model.parameters() if p.ndim <= 1]
    # On a GPU the fused AdamW updates a tensor in one kernel instead of about ten.
    optimizer = torch.optim.AdamW([{"params": matrices, "weight_decay": config["weight_decay"]},
                                   {"params": others, "weight_decay": 0.0}], lr=config["lr"],
                                  fused=device.type == "cuda")
    levels = sorted(set(noise_levels(config)) | set(config["report_p"]))
    sampler = DemSampler(config["code"], config["basis"], config["rounds"], config["detectors"], levels, device)
    run = Run(config, device, model, deepcopy(model), optimizer, None, sampler, torch.Generator(device=device))
    run.scaler = torch.amp.GradScaler(device.type, enabled=run.amp_dtype == torch.float16)
    run.start_average(model)
    return run


def save_checkpoint(run, path):
    """Write everything a resumed run needs. The file replaces the old one only when it is complete."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"config": run.config, "step": run.step, "history": run.history,
             "loss_sum": run.loss_sum, "loss_steps": run.loss_steps,
             "model": run.model.state_dict(), "ema_model": run.ema_model.state_dict(),
             "optimizer": run.optimizer.state_dict(), "scaler": run.scaler.state_dict(),
             "commit": git_commit(), "torch": str(torch.__version__)}
    partial = path.with_name(path.name + ".partial")
    torch.save(state, partial)
    os.replace(partial, path)


def load_checkpoint(path, device, changes=None):
    """Rebuild a run from a checkpoint. `changes` may replace the settings listed in RUN_CONTROL."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    run = new_run(DEFAULTS | state["config"] | (changes or {}), device)   # DEFAULTS fills settings newer than the file
    run.model.load_state_dict(state["model"])
    run.averaged_model.load_state_dict(state["ema_model"])
    run.start_average(run.averaged_model)
    for group in state["optimizer"]["param_groups"]:      # the file may come from a CPU, or from code before "fused"
        group["fused"] = run.optimizer.defaults["fused"]
    run.optimizer.load_state_dict(state["optimizer"])
    run.scaler.load_state_dict(state["scaler"])
    run.step, run.history = state["step"], state["history"]
    run.loss_sum, run.loss_steps = state.get("loss_sum", 0.0), state.get("loss_steps", 0)
    return run


def load_decoder(path, device="cpu"):
    """The trained decoder of a checkpoint (the moving-average weights), ready to predict, and the checkpoint's contents."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = CascadeDecoder(**{key: state["config"][key] for key in MODEL_KEYS})
    model.load_state_dict(state["ema_model"])
    return model.to(device).eval(), state


def git_commit():
    """The commit of the code, or "unknown" outside a git checkout."""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ---------- one step, one report ----------

def train_step(run):
    """Draw one batch at the step's noise strength and take one optimizer step."""
    config = run.config
    run.generator.manual_seed(stream_seed(config["seed"], TRAIN_STREAM, run.step))
    detectors, observables = run.sampler.sample(config["batch"], noise_at(run.step, config), run.generator)
    for group in run.optimizer.param_groups:
        group["lr"] = learning_rate_at(run.step, config)
    run.model.train()
    with torch.autocast(run.device.type, dtype=run.amp_dtype, enabled=run.amp_dtype is not None):
        logits = run.model(detectors)            # the labels never enter the model
    loss = F.binary_cross_entropy_with_logits(logits.float(), observables.float())
    if not torch.isfinite(loss):
        raise FloatingPointError(f"the loss is {loss.item()} at step {run.step}")
    run.optimizer.zero_grad(set_to_none=True)
    run.scaler.scale(loss).backward()
    run.scaler.unscale_(run.optimizer)
    torch.nn.utils.clip_grad_norm_(run.model.parameters(), config["clip"])
    run.scaler.step(run.optimizer)
    run.scaler.update()
    run.step += 1
    update_average(run)
    run.loss_sum += loss.item()
    run.loss_steps += 1


def update_average(run):
    """Move the averaged weights a little towards the current ones.

    The decay starts small and rises to config["ema"], so that the average of a short run is
    not stuck at the initial weights.
    """
    decay = min(run.config["ema"], run.step / (run.step + 9))
    with torch.no_grad():
        run.average.lerp_(flat(run.weights), 1 - decay)


def predict(model, detectors, amp_dtype=None):
    """0/1 predictions [shots, observables] of a decoder in evaluation mode."""
    with torch.no_grad(), torch.autocast(detectors.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        return (model(detectors) > 0).to(torch.uint8)


def score(predictions, observables):
    """Counts over a batch: shots with a wrong bit, right bits, and the same for the always-0 answer."""
    wrong = predictions != observables
    return {"shots": len(observables), "bits": observables.numel(),
            "shots_wrong": int(wrong.any(dim=1).sum()), "bits_right": int((~wrong).sum()),
            "shots_wrong_zero": int(observables.any(dim=1).sum()), "bits_right_zero": int((observables == 0).sum())}


def evaluate(run, p, shots, chunk=500):
    """Score the averaged weights on `shots` fresh error-model shots at noise strength p. The shots are the same in every call."""
    model, totals = run.ema_model.eval(), {}
    for index, start in enumerate(range(0, shots, chunk)):
        seed = stream_seed(run.config["seed"], REPORT_STREAM, int(round(p * 1e7)) * 100000 + index)
        detectors, observables = run.sampler.sample(min(chunk, shots - start), p, run.generator.manual_seed(seed))
        for key, value in score(predict(model, detectors, run.amp_dtype), observables).items():
            totals[key] = totals.get(key, 0) + value
    return {"p": p, "shots": totals["shots"],
            "shots_wrong": totals["shots_wrong"] / totals["shots"], "bits_right": totals["bits_right"] / totals["bits"],
            "shots_wrong_zero": totals["shots_wrong_zero"] / totals["shots"],
            "bits_right_zero": totals["bits_right_zero"] / totals["bits"]}


def report(run, say):
    """Print one report and remember it in the run's history."""
    config = run.config
    loss = run.loss_sum / run.loss_steps if run.loss_steps else None
    run.loss_sum, run.loss_steps = 0.0, 0
    last = max(run.step - 1, 0)
    rows = [evaluate(run, p, config["report_shots"]) for p in config["report_p"]]
    loss_text = "no training yet" if loss is None else f"mean training loss {loss:.4f}"
    say(f"step {run.step} of {config['steps']} | training noise p={noise_at(last, config):g} | "
        f"{loss_text} | learning rate {learning_rate_at(last, config):.2e}")
    say(f"  averaged weights on {config['report_shots']} error-model shots per p (not Stim shots); "
        f"95% interval of a share: +-{1.96 * math.sqrt(0.25 / config['report_shots']):.3f} or less")
    for row in rows:
        say(f"  p={row['p']:g}  shots wrong {row['shots_wrong']:.4f} (always-0 {row['shots_wrong_zero']:.4f})  "
            f"bits right {row['bits_right']:.4f} (always-0 {row['bits_right_zero']:.4f})")
    run.history.append({"step": run.step, "loss": loss, "rows": rows})
    return rows


# ---------- the command ----------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add = parser.add_argument
    add("--out", required=True, help="folder for checkpoint.pt and train.log")
    add("--resume", help="a checkpoint to continue; the run's settings come from it")
    add("--device", default="auto", help="auto, cpu or cuda")
    add("--max-hours", type=float, help="stop after this time, with a report and a checkpoint")
    add("--checkpoint-minutes", type=float, default=10.0,
        help="also write the checkpoint when this many minutes have passed since the last one (default 10)")
    add("--threads", type=int, help="CPU threads for torch (default: torch's own choice)")
    add("--tiny", action="store_true", help=f"a small model for CPU tests: {TINY}")
    add("--code", help="gross, bb72, bb90 or bb108 (default gross)")
    add("--basis", choices=["X", "Z"], help="memory basis; the labels are its logical operators (default X)")
    add("--rounds", type=int, help="noisy cycles (default: the code's distance)")
    add("--detectors", choices=["all", "basis"], help="all detectors, or only those on checks of the memory basis (default all)")
    add("--width", type=int, help="numbers per check and round (default 128)")
    add("--depth", type=int, help="residual blocks (default 8)")
    add("--bottleneck", type=int, help="the blocks work at width / bottleneck (default 4)")
    add("--residual", choices=["branch", "skip"], help="which side of the residual sum is divided by sqrt(2 depth) (default branch)")
    add("--batch", type=int, help="shots per step (default 128)")
    add("--steps", type=int, help="optimizer steps in all (default 400000)")
    add("--lr", type=float, help="peak learning rate (default 1e-3)")
    add("--warmup-steps", type=int, help="steps of linear warm-up (default 1000)")
    add("--lr-decay-steps", type=int, help="length of the cosine decay (default: --steps)")
    add("--weight-decay", type=float, help="on weight matrices only (default 3e-3)")
    add("--clip", type=float, help="largest gradient norm (default 1.0)")
    add("--ema", type=float, help="decay of the moving average of the weights (default 0.9998)")
    add("--p-start", type=float, help="noise of the first stage (default 0.001)")
    add("--p-train", type=float, help="noise of the last stage (default 0.0055)")
    add("--steps-start", type=int, help="steps of the first stage (default 4000)")
    add("--steps-anneal", type=int, help="steps of the climb from --p-start to --p-train (default 24000)")
    add("--anneal-levels", type=int, help="jumps of the climb (default 12)")
    add("--report-every", type=int, help="steps between reports and checkpoints (default 2000)")
    add("--report-shots", type=int, help="error-model shots per noise strength in a report (default 2000)")
    add("--report-p", type=float, nargs="+", help="noise strengths of the reports (default 0.001 0.002 0.003 0.004 0.005)")
    add("--snapshot-every", type=int, help="steps between kept copies checkpoint_step<N>.pt (default 10000)")
    add("--seed", type=int, help="fixes the initial weights and every shot (default 0)")
    add("--amp", choices=["auto", "fp16", "bf16", "off"], help="mixed precision; auto is fp16 on cuda and off on a CPU")
    return parser.parse_args(argv)


def settings(args):
    """(config, checkpoint changes): the full settings of a new run, and what a resumed run may change."""
    given = {key: getattr(args, key) for key in DEFAULTS if getattr(args, key, None) is not None}
    if args.resume:
        fixed = sorted(set(given) - set(RUN_CONTROL))
        if fixed:
            raise SystemExit(f"a resumed run keeps its settings; remove --{' --'.join(k.replace('_', '-') for k in fixed)}")
        return None, given
    config = DEFAULTS | (TINY if args.tiny else {}) | given
    if config["lr_decay_steps"] is None:
        config["lr_decay_steps"] = config["steps"]
    return config, None


def main(argv=None):
    args = parse_args(argv)
    config, changes = settings(args)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    if args.threads:
        torch.set_num_threads(args.threads)

    def say(line):
        print(line, flush=True)
        with open(out / "train.log", "a", encoding="utf-8") as log:
            log.write(line + "\n")

    started = time.perf_counter()
    run = load_checkpoint(args.resume, device, changes) if args.resume else new_run(config, device)
    config = run.config
    say(f"{'resumed at step ' + str(run.step) if args.resume else 'new run'} | seed {config['seed']} | "
        f"{config['code']} code, {config['basis']} memory, {config['rounds']} cycles, detectors={config['detectors']} "
        f"({run.sampler.num_detectors} detectors, {run.sampler.num_mechanisms} error mechanisms)")
    say(f"model: width {config['width']}, depth {config['depth']}, bottleneck {config['bottleneck']}, "
        f"residual {config['residual']}, {count_parameters(run.model)} parameters | batch {config['batch']}")
    say(f"curriculum: {config['steps_start']} steps at p={config['p_start']:g}, {config['steps_anneal']} steps climbing "
        f"in {config['anneal_levels']} jumps, then p={config['p_train']:g} | peak learning rate {config['lr']:g}")
    say(f"commit {git_commit()} | torch {torch.__version__}")
    amp = {torch.float16: "fp16", torch.bfloat16: "bf16", None: "off"}[run.amp_dtype]
    say(f"speed: device {device}, mixed precision {amp}, start-up {time.perf_counter() - started:.0f} s")
    if run.step == 0:
        report(run, say)

    stop_file, checkpoint, saved = out / "STOP", out / "checkpoint.pt", time.perf_counter()
    while run.step < config["steps"]:
        tick, first, stop = time.perf_counter(), run.step, None
        while run.step < config["steps"] and (run.step == first or run.step % config["report_every"]) and not stop:
            train_step(run)
            now = time.perf_counter()
            if args.max_hours is not None and now - started > args.max_hours * 3600:
                stop = "--max-hours"
            elif stop_file.exists():
                stop = f"the file {stop_file}"
            elif now - saved > args.checkpoint_minutes * 60 and run.step % config["report_every"]:
                save_checkpoint(run, checkpoint)      # by the clock, without a report
                saved = time.perf_counter()
        trained = time.perf_counter() - tick
        report(run, say)
        save_checkpoint(run, checkpoint)
        saved = time.perf_counter()
        if run.step // config["snapshot_every"] > first // config["snapshot_every"]:
            shutil.copyfile(checkpoint, out / f"checkpoint_step{run.step}.pt")
        memory = (f", peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB in tensors and "
                  f"{torch.cuda.max_memory_reserved() / 2 ** 30:.2f} GB held by PyTorch") if run.device.type == "cuda" else ""
        say(f"speed: {(run.step - first) * config['batch'] / trained:.0f} shots/s in training, "
            f"{trained / (run.step - first):.3f} s per step, report {time.perf_counter() - tick - trained:.1f} s{memory}")
        if stop:
            stop_file.unlink(missing_ok=True)
            say(f"speed: stopped by {stop} at step {run.step}; continue with --resume {checkpoint}")
            break
    say(f"checkpoint: {out / 'checkpoint.pt'} at step {run.step}")


if __name__ == "__main__":
    main()
