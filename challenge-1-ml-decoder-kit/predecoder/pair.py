"""The kit's score: pairs of an X shot and a Z shot. Shot k of X is paired with shot k of Z; the pair fails when either fails.

Run from challenge-1-ml-decoder-kit/ (after the X and Z test runs of compare.py):

  python -m predecoder.pair --final --data-x data/p0.005/test_X.npz --data-z data/p0.005/test_Z.npz \\
      --ours-x <X compare folder>/per_shot.npz --ours-z <Z compare folder>/per_shot.npz \\
      --tcbp-x <test_X.tcbp.npz> --tcbp-z <test_Z.tcbp.npz> \\
      --evaluator-shots <01_relay-bp_shots.npz> --evaluator-summary <its summary.json>

Decoders, each per basis, then paired:
  H     relay-BP with the head's priors (compare.py's head arm)
  rule  the frozen fallback rule: H where that relay-BP converged, else TCBP-JOBS (logs/fallback_rule_frozen.md)
  T     TCBP-JOBS
  R     the team evaluator's relay-bp row (built once, timed), read from its own pair file
For each: failed pairs, a 95% Wilson interval, the rate per cycle (decoders.per_cycle_rate, 12 cycles),
and H and the rule paired against T and against R with the exact two-sided test. Every file is
checked against its shot file (sha256 and truth, or its own stored failures).
"""
import argparse
import json

import numpy as np

from decoders import per_cycle_rate
from evaluation.data import load_dataset
from predecoder.decode import paired, wilson
from predecoder.files import HELD_OUT


def basis_failures(data, ours_path, tcbp_path, evaluator, summary):
    """{decoder: failed [shots]} for one basis, after the checks."""
    truth = data.observables
    with np.load(ours_path) as file:
        ours = {key: file[key] for key in ("truth", "predicted_head", "converged_head")}
    with np.load(tcbp_path, allow_pickle=False) as file:
        tcbp = {key: file[key] for key in ("truth", "predicted_T", "data_sha256")}
    b = data.basis
    checks = {f"{b}: ours' truth equals the file's": bool(np.array_equal(ours["truth"], truth)),
              f"{b}: TCBP-JOBS data_sha256 equals the file's": str(tcbp["data_sha256"]) == data.sha256,
              f"{b}: TCBP-JOBS truth equals the file's": bool(np.array_equal(tcbp["truth"], truth)),
              f"{b}: the evaluator's summary names this file": summary["datasets"]["test"][b]["sha256"] == data.sha256,
              f"{b}: the evaluator's failed_{b} equals its predictions against the truth": bool(np.array_equal(
                  evaluator[f"failed_{b}"], (evaluator[f"predictions_{b}"] != truth).any(axis=1)))}
    converged = ours["converged_head"] > 0
    rule = np.where(converged[:, None], ours["predicted_head"], tcbp["predicted_T"])
    failed = {"H": (ours["predicted_head"] != truth).any(axis=1), "rule": (rule != truth).any(axis=1),
              "T": (tcbp["predicted_T"] != truth).any(axis=1), "R": evaluator[f"failed_{b}"].astype(bool)}
    return failed, checks, int((~converged).sum())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("data-x", "data-z", "ours-x", "ours-z", "tcbp-x", "tcbp-z", "evaluator-shots", "evaluator-summary"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    data_x, data_z = load_dataset(args.data_x), load_dataset(args.data_z)
    if (data_x.path.name.startswith(HELD_OUT) or data_z.path.name.startswith(HELD_OUT)) and not args.final:
        raise SystemExit("held-out files: give --final")
    if (data_x.basis, data_z.basis) != ("X", "Z") or data_x.p != data_z.p or len(data_x.detectors) != len(data_z.detectors):
        raise SystemExit("give an X file and a Z file of the same p and size")
    with np.load(args.evaluator_shots, allow_pickle=False) as file:
        evaluator = {key: file[key] for key in file.files}
    with open(args.evaluator_summary) as file:
        summary = json.load(file)
    failed_x, checks_x, fallback_x = basis_failures(data_x, args.ours_x, args.tcbp_x, evaluator, summary)
    failed_z, checks_z, fallback_z = basis_failures(data_z, args.ours_z, args.tcbp_z, evaluator, summary)
    checks = checks_x | checks_z
    checks["the evaluator's failed_pair equals failed_X or failed_Z"] = bool(np.array_equal(
        evaluator["failed_pair"], evaluator["failed_X"] | evaluator["failed_Z"]))
    for name, ok in checks.items():
        print(f"check: {name}: {'yes' if ok else 'NO'}")
    if not all(checks.values()):
        raise SystemExit("a file does not belong to these shots")
    pairs = len(data_x.detectors)
    print(f"p = {data_x.p:g}: {pairs} pairs (X {data_x.path.name} {data_x.sha256[:12]}..., Z {data_z.path.name} "
          f"{data_z.sha256[:12]}...). The rule falls back to T on {fallback_x} X shots and {fallback_z} Z shots.")
    result = {"p": data_x.p, "pairs": pairs, "checks": checks, "fallback_shots": {"X": fallback_x, "Z": fallback_z},
              "decoders": {}, "paired": {}}
    failed_pair = {}
    for name in ("H", "rule", "T", "R"):
        values = failed_x[name] | failed_z[name]
        failed_pair[name] = values
        low, high = wilson(int(values.sum()), pairs)
        rate, error = per_cycle_rate(values, args.cycles)
        result["decoders"][name] = {"failures_X": int(failed_x[name].sum()), "failures_Z": int(failed_z[name].sum()),
                                    "failed_pairs": int(values.sum()), "wilson95": [low, high],
                                    "per_cycle": rate, "per_cycle_sigma": error}
        print(f"  {name:5s} failures [X {int(failed_x[name].sum()):4d}  Z {int(failed_z[name].sum()):4d}]  failed pairs "
              f"{int(values.sum()):4d}  {values.mean():.4f} [{low:.4f}, {high:.4f}]  per cycle {rate:.3e} +- {error:.1e}")
    for ours in ("H", "rule"):
        for other in ("T", "R"):
            both, only_ours, only_other, p_value = paired(failed_pair[ours], failed_pair[other])
            result["paired"][f"{ours} against {other}"] = {"both": both, f"only_{ours}": only_ours,
                                                           f"only_{other}": only_other, "p": p_value}
            print(f"  pairs, {ours} against {other}: both fail {both}, only {ours} {only_ours}, only {other} {only_other}, "
                  f"exact two-sided p = {p_value:.3g}")
    if args.out:
        with open(args.out, "w") as file:
            json.dump(result, file, indent=1)


if __name__ == "__main__":
    main()
