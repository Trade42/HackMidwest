"""Sample a loss dataset with Clifft and save it (with the true loss times from the probes).

usage (from challenge-1-ml-decoder-kit/):
    python -m loss.sample --basis X --p 0.004 --s 1e-3 --shots 20000 --seed 11 --out DATA.npz

The shots come in chunks of 2,000; chunk k uses seed + k. The file holds detectors (all 1728), observables,
heralds (one per measurement record, in the original record order), data_slot and check_k (see
lossmodel.loss_times) and meta. It prints the sha256 of the file and the loss statistics.
"""
import argparse
import json
import time

import numpy as np

from loss.lossmodel import make_dataset, save_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", default="X")
    ap.add_argument("--p", type=float, default=0.004)
    ap.add_argument("--s", type=float, required=True)
    ap.add_argument("--shots", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.perf_counter()
    data = make_dataset(a.basis, a.p, a.s, a.shots, a.seed, threads=a.threads)
    seconds = time.perf_counter() - t0
    sha = save_dataset(data, a.out)
    lost = (data["data_slot"] > 0).sum(axis=1)
    stats = {"file": a.out, "sha256": sha, **json.loads(str(data["meta"])), "sample_seconds": round(seconds, 1),
             "threads": a.threads, "shots_per_second": round(a.shots / seconds, 1),
             "lost_data_atoms_per_shot": round(float(lost.mean()), 3),
             "shots_with_no_lost_data_atom": int((lost == 0).sum()),
             "check_heralds_per_shot": round(float(data["heralds"][:, :1728].sum(axis=1).mean()), 3),
             "detector_weight_per_shot": round(float(data["detectors"].sum(axis=1).mean()), 2)}
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
