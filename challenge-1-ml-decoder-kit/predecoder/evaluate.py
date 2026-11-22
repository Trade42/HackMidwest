"""Run the network and the head on a shot file; save every shot's logits, and time one shot at a time.

Run from challenge-1-ml-decoder-kit/ (any machine; every path is an option):

  python -m predecoder.evaluate --head <head.pt> --trunk <the trunk checkpoint> \\
      --data data/p0.004/validation_X.npz --out <folder> [--device cuda] [--time-shots 200]

The network reads each shot's 1,728 detector bits and nothing else; the head reads the network's
vectors, the 936 basis bits and the kit's priors at the file's p. The truth in the file is not
read here at all.

It writes, in --out:
  logits.npy   float16 [shots, 8784]: the head's logit of every column of the kit's H, per shot,
               in file order (the decoders read these through predecoder.priors)
  meta.json    the files and their sha256, the settings, and the time per shot

The time per shot is measured one shot at a time (batch one, as a decoder would run), over the
first --time-shots shots, after one warm-up shot: the network plus the head, on --device. The
logits themselves are computed in batches; batch and batch-one results agree to float rounding.

--precision fp32 runs the network in float32 on a GPU too: no float16 autocast, and no TF32 in
matrix products or convolutions. The default ("auto") keeps float16 autocast on a GPU, as before;
on a CPU both are float32. (On the laptop, float16 moved 33 of 5,000 relay-BP outcomes at p = 0.005.)

--base-priors <vector.npy> gives the head another base prior (its input, and the vector it corrects) in place of
the kit's priors at the file's p. A head trained with --noise-factor (predecoder/noise.py) needs it, and the vector
must be the one it was trained with (checked by sha256 of its float64 values). A kit head takes it too (the arm
"kit head on the matched base" of neutral_atom_noise/tcbp_retrain.py); meta.json then says the head was not
trained with this base. The decoder must mix with the same vector (predecoder.priors.shot_priors' kit_priors).

Files named test_* or go_* are refused without --final.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from evaluation.data import load_dataset
from predecoder.decode import kit_problem
from predecoder.files import HELD_OUT, read_logits, vector_sha256  # noqa: F401 (read_logits: kept importable from here)
from predecoder.head import prior_logit
from predecoder.train import load_head, sha256_of



class Predecoder:
    """The network and the head, loaded once: shot(s) -> logits of the kit's columns."""

    def __init__(self, head_path, trunk_path=None, device="cpu", threads=None, precision="auto"):
        if threads:
            torch.set_num_threads(int(threads))
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if precision not in ("auto", "fp32"):
            raise ValueError("precision is auto or fp32")
        self.float32 = precision == "fp32"
        if self.float32 and self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
        self.trunk, self.head, self.saved = load_head(head_path, self.device, trunk_path)
        self.logit_by_p = {}
        self.shift = math.log(self.saved["config"]["pos_weight"])     # a weighted loss learns odds pos_weight times too high
        self.record = {"head": str(head_path), "head_sha256": sha256_of(head_path), "head_step": self.saved["step"],
                       "trunk": str(trunk_path or self.saved["trunk_file"]), "trunk_sha256": self.saved["trunk_sha256"],
                       "trunk_step": self.saved["trunk_step"], "head_commit": self.saved["commit"],
                       "device": str(self.device), "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
                       "precision": "float32" if self.float32 or self.device.type != "cuda" else "float16 autocast",
                       "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None}

    def prior_logit(self, p, priors):
        key = (p, vector_sha256(priors))            # the same p may come with another base vector (--base-priors)
        if key not in self.logit_by_p:
            self.logit_by_p[key] = prior_logit(priors).to(self.device)
        return self.logit_by_p[key]

    def __call__(self, detectors, basis_bits, p, priors):
        """detectors [shots, 1728] and basis bits [shots, 936] (uint8 numpy) -> float32 logits [shots, columns]."""
        with torch.no_grad():
            full = torch.from_numpy(np.ascontiguousarray(detectors, dtype=np.uint8)).to(self.device)
            basis = torch.from_numpy(np.ascontiguousarray(basis_bits, dtype=np.uint8)).to(self.device)
            trunk_input = full if self.trunk.num_inputs == full.shape[1] else basis
            amp = torch.float16 if self.device.type == "cuda" and not self.float32 else None
            with torch.autocast(self.device.type, dtype=amp, enabled=amp is not None):
                features = self.trunk.features(trunk_input)
            logits = self.head(features, basis, self.prior_logit(p, priors)) - self.shift
        return logits.float().cpu().numpy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", required=True, help="head.pt of predecoder.train")
    parser.add_argument("--trunk", help="the trunk checkpoint (default: the path stored in head.pt); its sha256 is checked")
    parser.add_argument("--data", required=True, help="a shot file of make_dataset.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--precision", choices=("auto", "fp32"), default="auto",
                        help="fp32: no float16 autocast and no TF32 on a GPU (default auto: float16 autocast on a GPU)")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--time-shots", type=int, default=200)
    parser.add_argument("--limit", type=int, help="only the first this many shots (smoke runs)")
    parser.add_argument("--final", action="store_true", help="allow a held-out file (test_*, go_*)")
    parser.add_argument("--base-priors", help="a .npy prior vector on the kit's columns: the head's base in place of the kit's "
                                              "priors at the file's p (required for a head trained with --noise-factor)")
    args = parser.parse_args(argv)
    if Path(args.data).name.startswith(HELD_OUT) and not args.final:
        raise SystemExit(f"{args.data} is a held-out file: it is run once, from a frozen commit, with --final")

    start = time.time()
    data = load_dataset(args.data)
    problem = kit_problem(data.basis, data.p)
    model = Predecoder(args.head, args.trunk, args.device, args.threads, args.precision)
    if model.head.num_columns != problem.H.shape[1]:
        raise SystemExit("the head does not fit this problem")
    base, trained_on = problem.priors, model.saved["config"].get("base_sha256")
    base_record = {"what": f"the kit's priors at the file's p = {data.p:g}", "sha256": vector_sha256(base),
                   "sum": float(base.sum()), "trained_with_this_base": trained_on is None}
    if args.base_priors:
        base = np.load(args.base_priors).astype(np.float64)
        if base.shape != problem.priors.shape or not ((0 < base) & (base < 1)).all():
            raise SystemExit(f"{args.base_priors}: shape {base.shape} or values outside (0, 1)")
        if trained_on and vector_sha256(base) not in trained_on.values():
            raise SystemExit(f"{args.base_priors} is not the base this head was trained with ({trained_on})")
        base_record = {"what": "--base-priors", "path": str(args.base_priors), "sha256": vector_sha256(base),
                       "file_sha256": sha256_of(args.base_priors), "sum": float(base.sum()),
                       "trained_with_this_base": bool(trained_on and vector_sha256(base) in trained_on.values())}
    elif trained_on:
        raise SystemExit(f"this head was trained on another noise model's base priors ({trained_on}): give that vector "
                         "with --base-priors")
    shots = len(data.detectors) if args.limit is None else min(args.limit, len(data.detectors))
    detectors, basis = data.detectors[:shots], data.basis_input[:shots]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logits = np.lib.format.open_memmap(out / "logits.npy", mode="w+", dtype=np.float16, shape=(shots, problem.H.shape[1]))
    for first in range(0, shots, args.batch):
        last = min(first + args.batch, shots)
        logits[first:last] = model(detectors[first:last], basis[first:last], data.p, base)
    logits.flush()
    batch_seconds = time.time() - start

    timed = min(args.time_shots, shots)
    seconds = []
    if timed:
        model(detectors[:1], basis[:1], data.p, base)                      # warm-up
        for k in range(timed):
            tick = time.perf_counter()
            one = model(detectors[k:k + 1], basis[k:k + 1], data.p, base)
            if model.device.type == "cuda":
                torch.cuda.synchronize()
            seconds.append(time.perf_counter() - tick)
        gap = float(np.abs(one[0].astype(np.float16).astype(np.float32) - logits[timed - 1].astype(np.float32)).max())
    else:
        gap = None
    probability_sum = sum(float((1 / (1 + np.exp(-np.asarray(logits[i:i + 1000], np.float32)))).sum())
                          for i in range(0, shots, 1000))
    meta = {"data": data.identity(), "shots": shots, "columns": int(problem.H.shape[1]), "model": model.record,
            "seconds_batched_total": batch_seconds,
            "per_shot_seconds": {"what": "network + head, one shot at a time, on the device named", "shots": timed,
                                 "mean": float(np.mean(seconds)) if seconds else None,
                                 "median": float(np.median(seconds)) if seconds else None,
                                 "p99": float(np.percentile(seconds, 99)) if seconds else None},
            "batch_one_against_batched_largest_logit_gap": gap,
            "mean_probability_sum_per_shot": probability_sum / shots, "kit_prior_sum": float(problem.priors.sum()),
            "base_priors": base_record}
    with open(out / "meta.json", "w") as file:
        json.dump(meta, file, indent=1)
    print(f"{data.path.name}: {shots} shots, logits of {problem.H.shape[1]} columns in {out / 'logits.npy'} "
          f"({batch_seconds:.0f} s in batches of {args.batch} on {model.device})")
    if seconds:
        print(f"network + head, one shot at a time on {model.device} ({torch.get_num_threads()} threads), {timed} shots: "
              f"mean {meta['per_shot_seconds']['mean'] * 1e3:.1f} ms, median {meta['per_shot_seconds']['median'] * 1e3:.1f} ms, "
              f"p99 {meta['per_shot_seconds']['p99'] * 1e3:.1f} ms; batch one against batched: largest logit gap {gap:.2e}")
    print(f"the head's probabilities sum to {meta['mean_probability_sum_per_shot']:.1f} per shot (the kit's priors: "
          f"{meta['kit_prior_sum']:.1f}; the head's base, {base_record['what']}: {base_record['sum']:.1f})")


if __name__ == "__main__":
    main()
