"""Merge step-9 chunk results of one file into results_step9/selector_<stem>.npz (R = relay-BP, P1 = PriorTilt seed 1 +
chunk), one row per shot in file order, in the same layout as the step-8b selector files (without P2).

  python -I -B merge_step9.py <stem> [<stem> ...]
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = {"R": "relay", "P1": "priortilt"}
FIELDS = ("predicted", "converged", "iterations", "n_faults", "weight", "seconds", "failed")

for stem in sys.argv[1:]:
    merged, truth, sha = {}, None, set()
    for name, label in RUNS.items():
        parts = [np.load(f) for f in sorted(glob.glob(os.path.join(HERE, "results_step9", f"{stem}.{label}.c*.npz")))]
        shot = np.concatenate([q["shot"] for q in parts])
        order = np.argsort(shot)
        if not np.array_equal(shot[order], np.arange(len(shot))):
            sys.exit(f"{stem} {label}: incomplete ({len(shot)} shots)")
        for k in FIELDS:
            merged[f"{k}_{name}"] = np.concatenate([q[k] for q in parts])[order]
        t = np.concatenate([q["truth"] for q in parts])[order]
        if truth is not None and not np.array_equal(truth, t):
            sys.exit(f"{stem}: truth differs between runs")
        truth = t
        sha |= {str(q["data_sha256"]) for q in parts}
        merged[f"settings_{name}"] = sorted({str(q["desc_settings"]) for q in parts})[0]
    if len(sha) != 1:
        sys.exit(f"{stem}: more than one data file")
    out = os.path.join(HERE, "results_step9", f"selector_{stem}.npz")
    np.savez_compressed(out, truth=truth.astype(np.uint8), data_sha256=sha.pop(), **merged)
    print(f"{out}: {len(truth)} shots, failures R {int(merged['failed_R'].sum())}, P1 {int(merged['failed_P1'].sum())}")
