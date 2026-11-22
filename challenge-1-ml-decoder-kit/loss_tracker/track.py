"""Run the tracker on a dataset and save, for every flagged data atom, log P(detector history | loss slot).

usage (from challenge-1-ml-decoder-kit/):
    python -m loss_tracker.track --data DATA.npz --tables loss_tracker/tables/X_p0.004_s1e-3.npz --out POST.npz

This script is the tracker's only door to a dataset, and it opens two arrays of it: `detectors` and `heralds`.
It never reads `data_slot`, `check_k` or `observables`. Everything downstream (accuracy.py, decode.py) takes the
tracker's output from the file written here.

The file holds: shot (entries,), atom (entries,) = index 0..143 of the flagged data atom, loglik (entries, 72)
over the loss slots, log_prior (72,), and meta (the tables, the mode, the dataset's name and sha256).
"""
import argparse
import hashlib
import json
import time

import numpy as np

from loss_tracker.tracker import Tables, Tracker, features


def read_inputs(path):
    """(detectors, heralds, meta) of a dataset file. Nothing else is read from it."""
    with np.load(path) as f:
        return f["detectors"], f["heralds"], json.loads(str(f["meta"]))


def track(tables_path, detectors, heralds, mode="coupled", sweeps=2, order=1, batch=500):
    tracker = Tracker(Tables.load(tables_path))
    out = [[], [], []]
    for a in range(0, len(detectors), batch):
        feat = features(tracker.geo, detectors[a:a + batch], heralds[a:a + batch])
        shot, atom, ll = tracker.loglik(feat, mode=mode, sweeps=sweeps, order=order)
        for lst, x in zip(out, (shot + a, atom, ll.astype(np.float32)), strict=True):
            lst.append(x)
    return tuple(np.concatenate(x) for x in out) + (tracker.log_prior,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--mode", default="coupled", choices=("naive", "drop", "coupled"))
    ap.add_argument("--sweeps", type=int, default=2)
    ap.add_argument("--order", type=int, default=1, choices=(0, 1))
    ap.add_argument("--shots", type=int, help="the first SHOTS shots (default: all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    detectors, heralds, meta = read_inputs(a.data)
    n = a.shots or len(detectors)
    t0 = time.perf_counter()
    shot, atom, ll, log_prior = track(a.tables, detectors[:n], heralds[:n], a.mode, a.sweeps, a.order)
    seconds = time.perf_counter() - t0
    info = {"data": a.data.split("/")[-1], "data_sha256": hashlib.sha256(open(a.data, "rb").read()).hexdigest(),
            "data_seed": meta["seed"], "basis": meta["basis"], "p": meta["p"], "s": meta["s"], "shots": n,
            "tables": a.tables, "tables_sha256": hashlib.sha256(open(a.tables, "rb").read()).hexdigest(),
            "mode": a.mode, "sweeps": a.sweeps, "order": a.order, "flagged_data_atoms": int(len(shot))}
    np.savez_compressed(a.out, shot=shot, atom=atom, loglik=ll, log_prior=log_prior, meta=np.array(json.dumps(info)))
    print(json.dumps({**info, "out": a.out, "seconds": round(seconds, 1),
                      "ms_per_shot_one_process": round(1e3 * seconds / n, 2)}), flush=True)


if __name__ == "__main__":
    main()
