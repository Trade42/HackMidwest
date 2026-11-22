"""L0: with no loss, Clifft's shots must match Stim's and the kit's relay-BP failure rate.

usage (from challenge-1-ml-decoder-kit/):
    python -m loss.l0_check --data DATA_s0.npz --parts 'PARTS/*.npz' --out loss/results/L0_X_p0.004

1. Detector and observable statistics of the Clifft dataset (s = 0) against Stim shots of the same kit circuit
   (circuit.sample, seed 1000, the same shot count): mean detector weight, the largest per-detector z-score,
   and the per-observable flip rate.
2. Relay-BP failures on the Clifft shots (from the decode parts) against the kit's reference at p = 0.004
   (README.md: relay-BP failures X = 45 of 10,000; the relay-BP authors: 53 of 10,000).
"""
import argparse
import json

import numpy as np

from circuit import sample
from loss.lossmodel import kit_circuit, load_dataset
from loss.summarize import load_parts, wilson

KIT = {"kit README, relay-BP X": (45, 10000), "relay-BP authors, X": (53, 10000)}


def two_proportion_z(k1, n1, k2, n2):
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return (k1 / n1 - k2 / n2) / se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--parts", required=True)
    ap.add_argument("--stim-seed", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = load_dataset(a.data)
    meta = data["meta"]
    assert meta["s"] == 0
    n = meta["shots"]
    det_c, obs_c = data["detectors"].astype(np.int64), data["observables"].astype(np.int64)
    det_s, obs_s = sample(kit_circuit(meta["basis"], meta["p"]), n, a.stim_seed)
    det_s, obs_s = det_s.astype(np.int64), obs_s.astype(np.int64)
    rc, rs = det_c.mean(0), det_s.mean(0)
    pooled = (rc + rs) / 2
    z = (rc - rs) / np.sqrt(np.maximum(pooled * (1 - pooled) * 2 / n, 1e-12))
    wc, ws = det_c.sum(1), det_s.sum(1)
    out = {"dataset": meta, "stim_seed": a.stim_seed, "shots": n,
           "mean_detector_weight": {"clifft": float(wc.mean()), "stim": float(ws.mean()),
                                    "z": float((wc.mean() - ws.mean()) / np.sqrt(wc.var() / n + ws.var() / n))},
           "max_abs_detector_z": float(np.abs(z).max()), "detectors": int(len(z)),
           "detectors_with_abs_z_over_4": int((np.abs(z) > 4).sum()),
           "observable_flip_rate": {"clifft": obs_c.mean(0).round(4).tolist(), "stim": obs_s.mean(0).round(4).tolist()}}
    per = load_parts([a.parts])["blind"]
    k, m = int(per["failed"].sum()), len(per["failed"])
    lo, hi = wilson(k, m)
    out["relay_bp_on_clifft"] = {"failures": k, "shots": m, "fail_fraction": k / m, "wilson95": [lo, hi],
                                 "unconverged": int((~per["converged"].astype(bool)).sum()),
                                 "mean_iterations": float(per["iterations"].mean())}
    out["references"] = {}
    for name, (kr, nr) in KIT.items():
        out["references"][name] = {"failures": kr, "shots": nr, "wilson95": list(wilson(kr, nr)),
                                   "z_vs_clifft": float(two_proportion_z(k, m, kr, nr))}
    with open(a.out + ".json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
