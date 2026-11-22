"""The plumbing check: does this folder's pipeline, given the TRUE loss round, reproduce loss-gates' ORACLE?

usage (from challenge-1-ml-decoder-kit/):
    python -m loss_tracker.plumbing --mine 'MY_PARTS/*.npz' --theirs 'LOSS_GATES_PARTS/*.npz' --out loss_tracker/results/plumbing

Compares the condition `plumb-oracle-round` of the first glob with `oracle-round` of the second, shot by shot:
failed, converged, the number of relay-BP iterations and the mask of wrong observables.
"""
import argparse
import json

import numpy as np

from loss.summarize import load_parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine", required=True)
    ap.add_argument("--theirs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    mine, theirs = load_parts([a.mine])["plumb-oracle-round"], load_parts([a.theirs])["oracle-round"]
    n = min(len(mine["failed"]), len(theirs["failed"]))
    out = {"shots_compared": n, "my_failures": int(mine["failed"][:n].sum()), "their_failures": int(theirs["failed"][:n].sum())}
    for field in ("failed", "converged", "iterations", "wrong_mask"):
        out[f"shots_with_equal_{field}"] = int((mine[field][:n] == theirs[field][:n]).sum())
    out["passed"] = all(out[f"shots_with_equal_{f}"] == n for f in ("failed", "converged", "iterations", "wrong_mask"))
    with open(a.out + ".json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
