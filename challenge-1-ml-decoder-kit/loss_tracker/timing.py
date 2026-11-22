"""Step 4: the tracker's cost per shot, in milliseconds.

usage (from challenge-1-ml-decoder-kit/, one process pinned to one CPU, e.g. `taskset -c 0`):
    python -m loss_tracker.timing --data DATA.npz --tables loss_tracker/tables/X_p0.004_s1e-3.npz --shots 300 \
        --out loss_tracker/results/timing

The timer wraps everything the tracker does for a shot: the features (detector patterns, herald marks) and the
posterior of every flagged data atom, in numpy, in one process. Two ways of calling it:
  one shot at a time (what a decoder that handles shots one by one would pay), and
  batches of 100 shots (the time per shot when shots are processed together).
The relay-BP decode that follows is not included; loss/RESULTS.md (L4) has it.
"""
import argparse
import json
import platform
import time

import numpy as np

from loss_tracker.track import read_inputs
from loss_tracker.tracker import Tables, Tracker, features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--shots", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    detectors, heralds, meta = read_inputs(a.data)
    tracker = Tracker(Tables.load(a.tables))
    tracker.tables.fire_table()
    rows = []
    for name, kw in (("explain, 2 sweeps (the tracker of step 3)", dict(explain=True, sweeps=2)),
                     ("coupled, 2 sweeps", dict(sweeps=2)), ("coupled, no sweep", dict(sweeps=0)),
                     ("independent rounds, coupled, 2 sweeps", dict(sweeps=2, order=0))):
        tracker.loglik(features(tracker.geo, detectors[:2], heralds[:2]), **kw)          # warm up
        single = []
        for k in range(a.shots):
            t0 = time.perf_counter()
            feat = features(tracker.geo, detectors[k:k + 1], heralds[k:k + 1])
            _, _, ll = tracker.loglik(feat, **kw)
            tracker.posterior(ll)
            single.append(1e3 * (time.perf_counter() - t0))
        t0 = time.perf_counter()
        for lo in range(0, a.shots, 100):
            feat = features(tracker.geo, detectors[lo:lo + 100], heralds[lo:lo + 100])
            _, _, ll = tracker.loglik(feat, **kw)
            tracker.posterior(ll)
        batch = 1e3 * (time.perf_counter() - t0) / a.shots
        rows.append({"version": name, "one_shot_mean_ms": float(np.mean(single)), "one_shot_median_ms": float(np.median(single)),
                     "one_shot_p99_ms": float(np.percentile(single, 99)), "batch_of_100_ms_per_shot": float(batch)})
    flagged = float(heralds[:a.shots, tracker.geo.data_rec].sum(axis=1).mean())
    out = {"dataset": {**meta, "file": a.data.split("/")[-1]}, "shots": a.shots, "flagged_data_atoms_per_shot": flagged,
           "machine": platform.processor() or platform.machine(), "node": "this PC (AMD Ryzen 7 7730U), one process", "rows": rows}
    md = [f"{a.shots} shots of {a.data.split('/')[-1]} ({flagged:.1f} flagged data atoms per shot), one process, numpy.", "",
          "| tracker version | one shot at a time: mean ms | median ms | 99th percentile ms | batches of 100: ms per shot |", "|---|---|---|---|---|"]
    md += [f"| {r['version']} | {r['one_shot_mean_ms']:.2f} | {r['one_shot_median_ms']:.2f} | {r['one_shot_p99_ms']:.2f} | {r['batch_of_100_ms_per_shot']:.2f} |" for r in rows]
    with open(a.out + ".json", "w") as f:
        json.dump(out, f, indent=1)
    with open(a.out + ".md", "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
