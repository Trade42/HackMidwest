"""Train the predecoder head on a trained X network, with the truly fired fault mechanisms as labels.

Run from challenge-1-ml-decoder-kit/ (any machine; every path is an option):

  python -m predecoder.train --trunk <checkpoint of neural.train> --out <folder> --seed 1

Every step draws a fresh batch from the kit's FULL detector error model (neural/sampler.py,
the network's own training distribution) at one of the training noise strengths, in turn. The
network (the "trunk") reads the 1,728 detector bits; the head reads the trunk's vectors, the
936 basis detector bits and the kit's priors, and predicts for each of the 8,784 columns of
the kit's basis problem whether it fired. The label of a column is the parity of its fired
mechanisms (mapping.py, which is checked on every report batch). The labels reach the loss
only.

The loss is the plain binary cross-entropy, averaged over columns and shots: a proper score,
so the head's probabilities are calibrated. The head starts at exactly the kit's priors (its
last layer is zero), so the first report shows the kit's own log-loss. `--pos-weight w`
weights the fired columns w times; the head then learns odds w times too high, and every
probability this folder reports or uses is corrected by log(w) first.

The trunk is frozen (evaluation mode, no gradient) unless `--finetune-from-step` is given; from
that step on its weights train too, at `--trunk-lr`, with its normalisation statistics kept.

Each report prints, on held-out error-model shots (fixed seeds, the same shots in every
report; not Stim shots), per noise strength:
  log-loss       the mean binary cross-entropy per column, of the head and of the kit's priors
  calibration    for columns the head puts in each band of probability: their number, the mean
                 probability and the share that really fired
  fired columns  how many really fired per shot, and how many of them the head puts above 0.5;
                 how many that did NOT fire it puts above 0.5

The checkpoint holds the head, its settings, the trunk's file name and sha256 (and the trunk's
weights only when they were fine-tuned), every report, and the git commit. To stop a run,
create an empty file named STOP in --out.
"""
import argparse
import hashlib
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from circuit import dem_matrices
from make_circuit import memory_experiment
from neural.sampler import DemSampler
from neural.train import load_decoder, stream_seed
from predecoder.head import build_head, prior_logit
from predecoder.mapping import column_map

TRAIN_STREAM, REPORT_STREAM = 11, 12
BANDS = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 0.5, 0.7, 0.9, 0.97, 1.0 + 1e-9]
HEAD_SIZES = ("project", "hidden", "shape_width")


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class Problem:
    """The sampler, the column map and the kit's prior log-odds, at the training and report noise strengths."""

    def __init__(self, trunk_config, levels, device):
        code, basis, rounds = trunk_config["code"], trunk_config["basis"], trunk_config["rounds"]
        # The trunk may predict the kit's observables or lighter ones (logicals=light, used for Z memory):
        # the head never reads that readout, only the trunk's vectors. The sampler here always labels with
        # the kit's observables (DemSampler's default), which is what the column map checks against.
        self.levels = sorted(set(levels))
        self.sampler = DemSampler(code, basis, rounds, "all", self.levels, device)
        self.map = column_map(code, basis, rounds, self.levels[0], self.sampler)
        for p in self.levels[1:]:
            column_map(code, basis, rounds, p, self.sampler)          # the same columns at every p (raises otherwise)
        self.basis_rows = torch.from_numpy(self.map.basis_rows).to(device)
        self.prior_logit = {}
        for p in self.levels:
            _, _, priors = dem_matrices(memory_experiment(code, basis, rounds, p), basis)
            self.prior_logit[p] = prior_logit(priors).to(device)
        self.trunk_reads_all = trunk_config["detectors"] == "all"

    def batch(self, shots, p, generator):
        """(trunk input, basis bits, column labels, detectors, observables) of fresh shots."""
        detectors, observables, (shot, mechanism) = self.sampler.sample(shots, p, generator, return_faults=True)
        labels = self.map.columns_fired(shot, mechanism, shots)
        basis_bits = detectors[:, self.basis_rows]
        trunk_input = detectors if self.trunk_reads_all else basis_bits
        return trunk_input, basis_bits, labels, detectors, observables


def head_logits(trunk, head, trunk_input, basis_bits, logit_prior, amp_dtype, trunk_grad=False):
    """The head's logits for a batch (raw: still to be corrected by log(pos_weight))."""
    with torch.set_grad_enabled(trunk_grad and torch.is_grad_enabled()), \
            torch.autocast(trunk_input.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        features = trunk.features(trunk_input)
    return head(features, basis_bits, logit_prior)


def learning_rate_at(step, config, peak):
    warm = min(1.0, (step + 1) / max(1, config["warmup_steps"]))
    progress = min(1.0, step / max(1, config["steps"]))
    return peak * warm * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def evaluate(trunk, head, problem, config, amp_dtype, generator, chunk=50):
    """The report numbers on held-out error-model shots, per noise strength."""
    rows = []
    shift = math.log(config["pos_weight"])
    head.eval()
    with torch.no_grad():
        for p in config["report_p"]:
            counts = np.zeros(len(BANDS) - 1)
            sums_q, sums_fired = np.zeros(len(BANDS) - 1), np.zeros(len(BANDS) - 1)
            loss_head = loss_kit = 0.0
            cells = fired = found = false_alarms = 0
            shots = config["report_shots"]
            for index, start in enumerate(range(0, shots, chunk)):
                size = min(chunk, shots - start)
                generator.manual_seed(stream_seed(config["seed"], REPORT_STREAM, int(round(p * 1e7)) * 100000 + index))
                trunk_input, basis_bits, labels, detectors, observables = problem.batch(size, p, generator)
                if index == 0:
                    problem.map.check(labels.cpu().numpy(), detectors.cpu().numpy(), observables.cpu().numpy())
                logits = head_logits(trunk, head, trunk_input, basis_bits, problem.prior_logit[p], amp_dtype) - shift
                target = labels.float()
                kit = problem.prior_logit[p][None].expand_as(logits)
                loss_head += float(F.binary_cross_entropy_with_logits(logits, target, reduction="sum"))
                loss_kit += float(F.binary_cross_entropy_with_logits(kit, target, reduction="sum"))
                cells += target.numel()
                q = torch.sigmoid(logits).double().cpu().numpy().ravel()
                y = labels.cpu().numpy().ravel()
                band = np.clip(np.searchsorted(BANDS, q, side="right") - 1, 0, len(BANDS) - 2)
                counts += np.bincount(band, minlength=len(counts))
                sums_q += np.bincount(band, weights=q, minlength=len(counts))
                sums_fired += np.bincount(band, weights=y, minlength=len(counts))
                fired += int(y.sum())
                found += int(((q > 0.5) & (y == 1)).sum())
                false_alarms += int(((q > 0.5) & (y == 0)).sum())
            ece = float(np.abs(sums_q - sums_fired).sum() / cells)
            rows.append({"p": p, "shots": shots, "log_loss_head": loss_head / cells, "log_loss_kit": loss_kit / cells,
                         "bands": [{"low": BANDS[b], "high": min(BANDS[b + 1], 1.0), "columns": int(counts[b]),
                                    "mean_q": float(sums_q[b] / counts[b]) if counts[b] else None,
                                    "fired": float(sums_fired[b] / counts[b]) if counts[b] else None}
                                   for b in range(len(counts))],
                         "fired_per_shot": fired / shots, "found_per_shot": found / shots,
                         "false_above_half_per_shot": false_alarms / shots, "calibration_error": ece})
    head.train()
    return rows


def report(state, say):
    config = state["config"]
    rows = evaluate(state["trunk"], state["head"], state["problem"], config, state["amp"], state["report_generator"])
    loss = state["loss_sum"] / state["loss_steps"] if state["loss_steps"] else None
    state["loss_sum"], state["loss_steps"] = 0.0, 0
    say(f"step {state['step']} of {config['steps']} | " + ("no training yet" if loss is None else f"mean training loss {loss:.6f}")
        + f" | head learning rate {learning_rate_at(max(state['step'] - 1, 0), config, config['lr']):.2e}")
    say(f"  held-out error-model shots (not Stim shots), {config['report_shots']} per p; log-loss = mean cross-entropy per column")
    for row in rows:
        gain = 1 - row["log_loss_head"] / row["log_loss_kit"]
        say(f"  p={row['p']:g}  log-loss head {row['log_loss_head']:.6f}  kit priors {row['log_loss_kit']:.6f}  "
            f"({gain:+.1%} of the kit's)  | fired per shot {row['fired_per_shot']:.1f}, of them above 0.5 "
            f"{row['found_per_shot']:.1f}; not fired but above 0.5 {row['false_above_half_per_shot']:.2f} | "
            f"calibration error {row['calibration_error']:.2e}")
        bands = [f"[{b['low']:g},{b['high']:g}) n={b['columns']} q={b['mean_q']:.4f} fired={b['fired']:.4f}"
                 for b in row["bands"] if b["columns"]]
        say("    calibration: " + " | ".join(bands))
    state["history"].append({"step": state["step"], "loss": loss, "rows": rows})


def save(state, path):
    path = Path(path)
    head, config = state["head"], state["config"]
    payload = {"config": config, "head_config": head.config, "head": head.state_dict(), "step": state["step"],
               "optimizer": state["optimizer"].state_dict(), "history": state["history"],
               "trunk_file": config["trunk"], "trunk_sha256": state["trunk_sha256"], "trunk_step": state["trunk_step"],
               "trunk_commit": state["trunk_commit"], "trunk_state": state["trunk"].state_dict() if state["finetuned"] else None,
               "commit": git_commit(), "torch": str(torch.__version__)}
    partial = path.with_name(path.name + ".partial")
    torch.save(payload, partial)
    os.replace(partial, path)


def load_head(path, device="cpu", trunk_path=None):
    """(trunk, head, checkpoint) ready to predict. The trunk file must have the sha256 the head was trained on."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    trunk_path = trunk_path or saved["trunk_file"]
    if sha256_of(trunk_path) != saved["trunk_sha256"]:
        raise SystemExit(f"{trunk_path} is not the trunk this head was trained on (sha256 {saved['trunk_sha256']})")
    trunk, trunk_saved = load_decoder(trunk_path, device)
    if saved.get("trunk_state") is not None:
        trunk.load_state_dict(saved["trunk_state"])
    problem_H, basis_rows = basis_problem(trunk_saved["config"], saved["config"]["p_train"][0])
    head = build_head(problem_H, basis_rows, trunk_saved["config"],
                      **{key: saved["head_config"][key] for key in HEAD_SIZES})
    head.load_state_dict(saved["head"])
    return trunk.eval(), head.to(device).eval(), saved


def basis_problem(trunk_config, p):
    """(H, detector ids of its rows) of the kit's basis problem for the trunk's circuit."""
    from circuit import detector_ids
    circuit = memory_experiment(trunk_config["code"], trunk_config["basis"], trunk_config["rounds"], p)
    H, _, _ = dem_matrices(circuit, trunk_config["basis"])
    return H, detector_ids(circuit, trunk_config["basis"])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add = parser.add_argument
    add("--trunk", required=True, help="a checkpoint of neural.train (the trained network)")
    add("--out", required=True, help="folder for head.pt and train.log")
    add("--resume", action="store_true", help="continue from <out>/head.pt")
    add("--seed", type=int, default=1)
    add("--steps", type=int, default=30000)
    add("--batch", type=int, default=64)
    add("--lr", type=float, default=1e-3)
    add("--warmup-steps", type=int, default=500)
    add("--weight-decay", type=float, default=1e-4)
    add("--clip", type=float, default=1.0)
    add("--p-train", type=float, nargs="+", default=[0.004],
        help="training noise strengths, taken in turn, one per step (default 0.004)")
    add("--report-p", type=float, nargs="+", default=None, help="default: the training strengths")
    add("--report-every", type=int, default=2000)
    add("--report-shots", type=int, default=500)
    add("--snapshot-every", type=int, default=10000)
    add("--pos-weight", type=float, default=1.0, help="weight of the fired columns in the loss (default 1: plain cross-entropy)")
    add("--project", type=int, default=32)
    add("--hidden", type=int, default=64)
    add("--shape-width", type=int, default=16)
    add("--finetune-from-step", type=int, default=None, help="train the trunk too from this step on")
    add("--trunk-lr", type=float, default=5e-5)
    add("--device", default="auto")
    add("--amp", choices=["auto", "fp16", "bf16", "off"], default="auto", help="for the trunk; the head runs in float32")
    add("--threads", type=int)
    add("--max-hours", type=float)
    add("--checkpoint-minutes", type=float, default=10.0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("high")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "head.pt"

    def say(line):
        print(line, flush=True)
        with open(out / "train.log", "a", encoding="utf-8") as log:
            log.write(line + "\n")

    started = time.perf_counter()
    previous = torch.load(checkpoint, map_location="cpu", weights_only=True) if args.resume else None
    config = previous["config"] if previous else {
        "trunk": str(args.trunk), "seed": args.seed, "steps": args.steps, "batch": args.batch, "lr": args.lr,
        "warmup_steps": args.warmup_steps, "weight_decay": args.weight_decay, "clip": args.clip,
        "p_train": list(args.p_train), "report_p": list(args.report_p or args.p_train), "report_every": args.report_every,
        "report_shots": args.report_shots, "snapshot_every": args.snapshot_every, "pos_weight": args.pos_weight,
        "project": args.project, "hidden": args.hidden, "shape_width": args.shape_width,
        "finetune_from_step": args.finetune_from_step, "trunk_lr": args.trunk_lr, "amp": args.amp}
    if previous:
        config["steps"] = args.steps                 # the one setting a resumed run may change: the total length
    trunk_sha = sha256_of(args.trunk)
    if previous and previous["trunk_sha256"] != trunk_sha:
        raise SystemExit("--trunk is not the trunk of the run being resumed")
    trunk, trunk_saved = load_decoder(args.trunk, device)
    trunk.eval()
    if previous and previous.get("trunk_state") is not None:
        trunk.load_state_dict(previous["trunk_state"])
    for parameter in trunk.parameters():
        parameter.requires_grad_(False)
    problem = Problem(trunk_saved["config"], config["p_train"] + config["report_p"], device)
    torch.manual_seed(config["seed"])
    head = build_head(problem.map.basis_H, problem.map.basis_rows, trunk_saved["config"],
                      **{key: config[key] for key in HEAD_SIZES}).to(device)
    head_parameters = list(head.parameters())
    groups = [{"params": [q for q in head_parameters if q.ndim > 1], "weight_decay": config["weight_decay"], "name": "head"},
              {"params": [q for q in head_parameters if q.ndim <= 1], "weight_decay": 0.0, "name": "head"}]
    if config["finetune_from_step"] is not None:
        groups.append({"params": list(trunk.parameters()), "weight_decay": 0.0, "name": "trunk"})
    optimizer = torch.optim.AdamW(groups, lr=config["lr"])
    amp_choice = config["amp"] if config["amp"] != "auto" else ("fp16" if device.type == "cuda" else "off")
    amp = {"fp16": torch.float16, "bf16": torch.bfloat16, "off": None}[amp_choice]
    state = {"config": config, "trunk": trunk, "head": head, "problem": problem, "optimizer": optimizer, "amp": amp,
             "step": 0, "history": [], "loss_sum": 0.0, "loss_steps": 0, "finetuned": False,
             "trunk_sha256": trunk_sha, "trunk_step": trunk_saved["step"], "trunk_commit": trunk_saved["commit"],
             "report_generator": torch.Generator(device=device)}
    if previous:
        head.load_state_dict(previous["head"])
        optimizer.load_state_dict(previous["optimizer"])
        state.update(step=previous["step"], history=previous["history"],
                     finetuned=previous.get("trunk_state") is not None)
    generator = torch.Generator(device=device)

    say(f"{'resumed at step ' + str(state['step']) if previous else 'new run'} | seed {config['seed']} | trunk {args.trunk} "
        f"(sha256 {trunk_sha[:16]}..., step {trunk_saved['step']}, {trunk_saved['config']['basis']} memory, "
        f"width {trunk_saved['config']['width']}, depth {trunk_saved['config']['depth']}, reads "
        f"{trunk_saved['config']['detectors']} detectors)")
    say(f"problem: {problem.sampler.num_mechanisms} mechanisms of the full model onto {problem.map.num_columns} columns of the "
        f"kit's basis problem ({problem.map.basis_H.shape[0]} detectors); merged priors equal the kit's to "
        f"{problem.map.prior_error:.1e}")
    say(f"head: {sum(q.numel() for q in head.parameters())} parameters, {head.config['shapes']} column shapes | "
        f"batch {config['batch']}, training p {config['p_train']}, pos-weight {config['pos_weight']:g}, peak lr {config['lr']:g}"
        + (f", trunk trains from step {config['finetune_from_step']} at lr {config['trunk_lr']:g}"
           if config["finetune_from_step"] is not None else ", trunk frozen"))
    say(f"commit {git_commit()} | torch {torch.__version__} | device {device}, trunk mixed precision {amp_choice} | "
        f"start-up {time.perf_counter() - started:.0f} s")
    if state["step"] == 0:
        report(state, say)
        save(state, checkpoint)

    stop_file, saved_at = out / "STOP", time.perf_counter()
    weight = torch.tensor(config["pos_weight"], device=device)
    while state["step"] < config["steps"]:
        tick, first, stop = time.perf_counter(), state["step"], None
        while state["step"] < config["steps"] and (state["step"] == first or state["step"] % config["report_every"]) and not stop:
            step = state["step"]
            tuning_trunk = config["finetune_from_step"] is not None and step >= config["finetune_from_step"]
            if tuning_trunk and not state["finetuned"]:
                for parameter in trunk.parameters():
                    parameter.requires_grad_(True)
                state["finetuned"] = True
                say(f"step {step}: the trunk trains from now on (normalisation statistics kept)")
            for group in optimizer.param_groups:
                group["lr"] = (learning_rate_at(step, config, config["lr"]) if group["name"] == "head" else
                               config["trunk_lr"] * min(1.0, (step - (config["finetune_from_step"] or 0) + 1) / 500))
            p = config["p_train"][step % len(config["p_train"])]
            generator.manual_seed(stream_seed(config["seed"], TRAIN_STREAM, step))
            trunk_input, basis_bits, labels, _, _ = problem.batch(config["batch"], p, generator)
            logits = head_logits(trunk, head, trunk_input, basis_bits, problem.prior_logit[p], amp, trunk_grad=tuning_trunk)
            loss = F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"the loss is {loss.item()} at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([q for group in optimizer.param_groups for q in group["params"]], config["clip"])
            optimizer.step()
            state["step"] += 1
            state["loss_sum"] += loss.item()
            state["loss_steps"] += 1
            now = time.perf_counter()
            if args.max_hours is not None and now - started > args.max_hours * 3600:
                stop = "--max-hours"
            elif stop_file.exists():
                stop = f"the file {stop_file}"
            elif now - saved_at > args.checkpoint_minutes * 60:
                save(state, checkpoint)
                saved_at = time.perf_counter()
        trained = time.perf_counter() - tick
        report(state, say)
        save(state, checkpoint)
        saved_at = time.perf_counter()
        if state["step"] // config["snapshot_every"] > first // config["snapshot_every"]:
            shutil.copyfile(checkpoint, out / f"head_step{state['step']}.pt")
        memory = (f", peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB"
                  if device.type == "cuda" else "")
        say(f"speed: {(state['step'] - first) * config['batch'] / trained:.0f} shots/s, {trained / (state['step'] - first):.3f} s "
            f"per step, report {time.perf_counter() - tick - trained:.1f} s{memory}")
        if stop:
            stop_file.unlink(missing_ok=True)
            say(f"stopped by {stop} at step {state['step']}; continue with --resume")
            break
    say(f"checkpoint: {checkpoint} at step {state['step']} (sha256 {sha256_of(checkpoint)})")


if __name__ == "__main__":
    main()
