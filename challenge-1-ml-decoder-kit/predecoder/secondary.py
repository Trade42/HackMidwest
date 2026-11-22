"""The secondary rows of the one test run: relay-BP with the head's priors against two other decoders' files, same shots.

Run from challenge-1-ml-decoder-kit/, after the test run of compare.py, with the two files the
orchestrator named (from origin/server-logs; nothing else of those folders is read):

  python -m predecoder.secondary --final --data data/p0.005/test_X.npz --ours <test compare folder>/per_shot.npz \\
      --tcbp <test_X.tcbp.npz> --evaluator-shots <01_relay-bp_shots.npz> --evaluator-summary <its summary.json>

Secondary, chosen after the p = 0.004 gate failed, X memory only, accuracy only: it counts the
shots each decoder gets wrong and the paired counts with an exact two-sided test. It checks that
every file was made from this shot file (its sha256, and the truth or the failures it holds).
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
    parser.add_argument("--ours", required=True, help="per_shot.npz of predecoder.compare on that file")
    parser.add_argument("--tcbp", required=True, help="the TCBP-JOBS per-shot file of that file (predicted_T, truth)")
    parser.add_argument("--evaluator-shots", required=True, help="the evaluator's relay-bp per-shot file (predictions_X)")
    parser.add_argument("--evaluator-summary", required=True, help="its summary.json, which names the data file")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--out", help="a JSON file for the numbers")
    args = parser.parse_args(argv)
    data = load_dataset(args.data)
    if data.path.name.startswith(HELD_OUT) and not args.final:
        raise SystemExit("a held-out file: give --final")
    truth = data.observables
    with np.load(args.ours) as file:
        ours = {key: file[key] for key in file.files}
    with np.load(args.tcbp, allow_pickle=False) as file:
        tcbp = {key: file[key] for key in file.files}
    with np.load(args.evaluator_shots, allow_pickle=False) as file:
        evaluator = {key: file[key] for key in ("predictions_X", "failed_X")}
    with open(args.evaluator_summary) as file:
        evaluator_data = json.load(file)["datasets"]["test"]["X"]
    checks = {
        "ours: truth equals the file's observables": bool(np.array_equal(ours["truth"], truth)),
        "TCBP-JOBS: data_sha256 equals the file's": str(tcbp["data_sha256"]) == data.sha256,
        "TCBP-JOBS: truth equals the file's observables": bool(np.array_equal(tcbp["truth"], truth)),
        "TCBP-JOBS: failed_T equals its predictions against the truth": bool(
            np.array_equal(tcbp["failed_T"], (tcbp["predicted_T"] != truth).any(axis=1))),
        "evaluator: its summary's X data sha256 equals the file's": evaluator_data["sha256"] == data.sha256,
        "evaluator: failed_X equals its X predictions against the file's observables": bool(
            np.array_equal(evaluator["failed_X"], (evaluator["predictions_X"] != truth).any(axis=1))),
    }
    for name, ok in checks.items():
        print(f"check: {name}: {'yes' if ok else 'NO'}")
    if not all(checks.values()):
        raise SystemExit("a file does not belong to these shots")
    failed = {"H (relay-BP, the head's priors)": (ours["predicted_head"] != truth).any(axis=1),
              "fresh (relay-BP, the kit's priors, per shot)": (ours["predicted_fresh"] != truth).any(axis=1)
              if "predicted_fresh" in ours else ours["failed_fresh"] > 0,
              "T (TCBP-JOBS)": (tcbp["predicted_T"] != truth).any(axis=1),
              "R (the evaluator's relay-bp row, X)": (evaluator["predictions_X"] != truth).any(axis=1)}
    shots = len(truth)
    print(f"{data.path.name} (sha256 {data.sha256}), {shots} shots, X memory. Secondary, chosen after the p = 0.004 gate "
          "failed, X only, accuracy only.")
    summary = {"data": data.identity(), "checks": checks, "failures": {}, "pairs": {},
               "label": "secondary, chosen after the p = 0.004 gate failed, X only, accuracy only"}
    for name, values in failed.items():
        low, high = wilson(int(values.sum()), shots)
        summary["failures"][name] = {"failures": int(values.sum()), "wilson95": [low, high]}
        print(f"  {name:46s} {int(values.sum()):5d} failures, {values.mean():.4f} [{low:.4f}, {high:.4f}]")
    head = failed["H (relay-BP, the head's priors)"]
    for name, values in list(failed.items())[1:]:
        both, only_h, only_other, p_value = paired(head, values)
        summary["pairs"][f"H against {name}"] = {"both": both, "only_H": only_h, "only_other": only_other, "p": p_value}
        print(f"  H against {name}: both fail {both}, only H {only_h}, only the other {only_other}, "
              f"exact two-sided p = {p_value:.3g}")
    if args.out:
        with open(args.out, "w") as file:
            json.dump(summary, file, indent=1)


if __name__ == "__main__":
    main()
