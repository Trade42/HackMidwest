"""Sal's rule: relay-BP with the head's priors where it converges; where it does not, TCBP-JOBS's answer.

Run from challenge-1-ml-decoder-kit/:

  python -m predecoder.fallback --data data/p0.005/validation_X.npz --ours <compare folder>/per_shot.npz \\
      --tcbp <validation_X.tcbp.npz>

The rule has no knob: "converged" is relay-BP's own flag (its correction reproduces the
syndrome), recorded per shot by compare.py. It prints the failures of H (relay-BP with the
head's priors), T (TCBP-JOBS) and the rule, the shots that fall back, and the rule paired
against T and against H with the exact two-sided test. Files named test_* or go_* need --final.
"""
import argparse
import json

import numpy as np

from evaluation.data import load_dataset
from predecoder.decode import paired, wilson
from predecoder.files import HELD_OUT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--ours", required=True, help="per_shot.npz of predecoder.compare with the head arm")
    parser.add_argument("--tcbp", required=True, help="TCBP-JOBS's per-shot file of the same shot file")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--out", help="a JSON file for the numbers")
    args = parser.parse_args(argv)
    data = load_dataset(args.data)
    if data.path.name.startswith(HELD_OUT) and not args.final:
        raise SystemExit("a held-out file: give --final")
    truth = data.observables
    with np.load(args.ours) as file:
        ours = {key: file[key] for key in ("truth", "predicted_head", "converged_head")}
    with np.load(args.tcbp, allow_pickle=False) as file:
        tcbp = {key: file[key] for key in ("truth", "predicted_T", "data_sha256")}
    checks = {"ours: truth equals the file's": bool(np.array_equal(ours["truth"], truth)),
              "TCBP-JOBS: data_sha256 equals the file's": str(tcbp["data_sha256"]) == data.sha256,
              "TCBP-JOBS: truth equals the file's": bool(np.array_equal(tcbp["truth"], truth))}
    for name, ok in checks.items():
        print(f"check: {name}: {'yes' if ok else 'NO'}")
    if not all(checks.values()):
        raise SystemExit("a file does not belong to these shots")
    converged = ours["converged_head"] > 0
    rule = np.where(converged[:, None], ours["predicted_head"], tcbp["predicted_T"])
    failed = {"H": (ours["predicted_head"] != truth).any(axis=1), "T": (tcbp["predicted_T"] != truth).any(axis=1),
              "rule": (rule != truth).any(axis=1)}
    shots = len(truth)
    print(f"{data.path.name} (sha256 {data.sha256[:12]}...), {shots} shots; the rule: H where relay-BP with the head's "
          f"priors converged, else T. It falls back to T on {int((~converged).sum())} shots.")
    summary = {"data": data.identity(), "checks": checks, "fallback_shots": int((~converged).sum()), "failures": {},
               "pairs": {}, "rule": "H where relay-BP with the head's priors converged, else TCBP-JOBS"}
    for name, values in failed.items():
        low, high = wilson(int(values.sum()), shots)
        summary["failures"][name] = {"failures": int(values.sum()), "wilson95": [low, high]}
        print(f"  {name:5s} {int(values.sum()):5d} failures, {values.mean():.4f} [{low:.4f}, {high:.4f}]")
    print(f"  H fails {int((failed['H'] & ~converged).sum())} of its unconverged shots and {int((failed['H'] & converged).sum())} "
          f"converged ones; T fails {int((failed['T'] & ~converged).sum())} of the shots that fall back")
    for other in ("T", "H"):
        both, only_rule, only_other, p_value = paired(failed["rule"], failed[other])
        summary["pairs"][f"rule against {other}"] = {"both": both, "only_rule": only_rule, f"only_{other}": only_other,
                                                     "p": p_value}
        print(f"  the rule against {other}: both fail {both}, only the rule {only_rule}, only {other} {only_other}, "
              f"exact two-sided p = {p_value:.3g}")
    if args.out:
        with open(args.out, "w") as file:
            json.dump(summary, file, indent=1)


if __name__ == "__main__":
    main()
