"""Fit the tracker's emission tables on TRAINING shots. This is the one place where the tracker meets the truth.

usage (from challenge-1-ml-decoder-kit/):
    python -m loss_tracker.fit --data TRAIN.npz --out loss_tracker/tables/X_p0.004_s1e-3.npz

For every data atom, check type and detector round of every training shot, the pattern of the atom's 3 detector
bits is counted under the class the status probes give (present, lost this cycle after CX k, lost last cycle
after CX k, two cycles ago after CX k, lost earlier: tracker.py). Only CLEAN triples are counted: no bit of the triple is unreliable by a
herald (tracker.features) and no other lost data atom has left one of the 3 checks yet (from the probes).
A second table counts the pattern together with the pattern of the round before, when both rounds are clean.
The file holds these counts; tracker.Tables turns them into probabilities.
"""
import argparse
import hashlib
import json

import numpy as np

from loss.lossmodel import load_dataset
from loss_tracker.tracker import CLS, N_CLASS, ROUNDS, SLOTS, TYPES, Geometry, Tables, features, others_dirt


def true_cdf(slots):
    """cdf[e, n] = 1 if the atom's true slot <= n, for n = 0..72."""
    return (slots[:, None] <= np.arange(SLOTS + 1)[None, :]).astype(np.float64)


def clean_and_class(geo, feat, data_slot):
    """Per check type: clean (shots, 144, 13) bool, and the class (shots, 144, 13) of every atom in every round.

    data_slot is the truth from the probes: 0 = never lost, else the loss slot.
    """
    shots, atoms = data_slot.shape
    shot_idx, atom_idx = np.nonzero(data_slot)
    cdf = true_cdf(data_slot[shot_idx, atom_idx])
    cls = np.zeros((shots, atoms, ROUNDS), np.int64)
    cls[shot_idx, atom_idx] = CLS[:, data_slot[shot_idx, atom_idx] - 1].T
    all_s, all_a = (x.ravel() for x in np.indices((shots, atoms)))
    clean = {}
    for T in TYPES:
        dirt = others_dirt(geo, T, shot_idx, atom_idx, cdf, shots, all_s, all_a).reshape(shots, atoms, 3, ROUNDS) > 0.5
        clean[T] = ~(dirt | feat["herald_dirt"][T]).any(axis=2)
        clean[T][:, :, np.setdiff1d(np.arange(ROUNDS), geo.rounds[T])] = False
    return clean, cls


def count_tables(geo, detectors, heralds, data_slot, chunk=500):
    """(counts1, counts2) of tracker.Tables, from training shots and their true loss slots."""
    atoms = len(geo.side)
    counts1 = np.zeros((2, 2, ROUNDS, N_CLASS, 8), np.int64)
    counts2 = np.zeros((2, 2, ROUNDS, N_CLASS, 8, 8), np.int64)
    for a in range(0, len(detectors), chunk):
        feat = features(geo, detectors[a:a + chunk], heralds[a:a + chunk])
        slot = data_slot[a:a + chunk].astype(np.int64)
        assert ((slot > 0) == feat["flagged"]).all(), "the probes and the final heralds disagree"
        clean, cls = clean_and_class(geo, feat, slot)
        key = (np.broadcast_to(geo.side[None, :, None], cls.shape) * ROUNDS
               + np.broadcast_to(np.arange(ROUNDS)[None, None, :], cls.shape)) * N_CLASS + cls
        for t, T in enumerate(TYPES):
            y, ok = feat["pattern"][T].astype(np.int64), clean[T]
            counts1[:, t] += np.bincount(key[ok] * 8 + y[ok], minlength=counts1[:, t].size).reshape(counts1[:, t].shape)
            ok2 = ok[:, :, 1:] & ok[:, :, :-1]
            idx = (key[:, :, 1:][ok2] * 8 + y[:, :, :-1][ok2]) * 8 + y[:, :, 1:][ok2]
            counts2[:, t] += np.bincount(idx, minlength=counts2[:, t].size).reshape(counts2[:, t].shape)
    return counts1, counts2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="the TRAINING dataset (with probes)")
    ap.add_argument("--shots", type=int, help="use the first SHOTS shots (default: all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = load_dataset(a.data)
    meta = data["meta"]
    n = a.shots or meta["shots"]
    geo = Geometry(meta["basis"], meta["p"])
    counts1, counts2 = count_tables(geo, data["detectors"][:n], data["heralds"][:n], data["data_slot"][:n])
    sha = hashlib.sha256(open(a.data, "rb").read()).hexdigest()
    tables = Tables(counts1, counts2, meta["basis"], meta["p"], meta["s"],
                    {"train_file": a.data.split("/")[-1], "train_sha256": sha, "train_seed": meta["seed"], "train_shots": n,
                     "lost_data_atoms": int((data["data_slot"][:n] > 0).sum())})
    tables.save(a.out)
    lost_tables = [tables.pooled1[side, t, r, c].sum() for side in range(2) for t, T in enumerate(TYPES)
                   for r in geo.rounds[T] for c in range(1, N_CLASS) if CLS[r].tolist().count(c)]
    print(json.dumps({"out": a.out, **tables.meta, "clean_triple_rounds_counted": int(counts1.sum()),
                      "of_them_in_lost_classes": int(counts1[:, :, :, 1:].sum()),
                      "clean_pairs_of_rounds_counted": int(counts2.sum()),
                      "smallest_lost_table_counts": int(min(lost_tables)),
                      "median_lost_table_counts": int(np.median(lost_tables))}), flush=True)


if __name__ == "__main__":
    main()
