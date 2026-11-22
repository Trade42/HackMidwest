"""Tune relay-BP's schedule on real kit shots (X memory): first-leg length, leg length, stopping rule.
usage: python relay_knobs.py KIT_DIR P N_SHOTS
Every configuration decodes the same Stim shots. Iterations do not depend on the machine.
"""
import sys
import time
import numpy as np

kit, p, n = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
sys.path.insert(0, kit)
from circuit import GROSS_CIRCUITS, dem_matrices, detector_ids, load_circuit, sample
from decoders import relay_decoder

circ = load_circuit(GROSS_CIRCUITS["X"], p)
H, L, pri = dem_matrices(circ, "X")
det, obs = sample(circ, n, 1)
det = det[:, detector_ids(circ, "X")]
configs = [("kit baseline: 80 / 60 / stop at 5", dict()),
           ("80 / 60 / stop at 1", dict(stop_nconv=1)),
           ("80 / 60 / stop at 2", dict(stop_nconv=2)),
           ("40 / 60 / stop at 1", dict(stop_nconv=1, pre_iter=40)),
           ("20 / 60 / stop at 1", dict(stop_nconv=1, pre_iter=20)),
           ("40 / 30 / stop at 1", dict(stop_nconv=1, pre_iter=40, set_max_iter=30)),
           ("20 / 30 / stop at 1", dict(stop_nconv=1, pre_iter=20, set_max_iter=30)),
           ("20 / 30 / stop at 2", dict(stop_nconv=2, pre_iter=20, set_max_iter=30))]
print(f"p={p}, {n} Stim shots, X memory. Columns: first leg / later legs / stopping rule", flush=True)
for name, changes in configs:
    dec = relay_decoder(H, pri, seed=1, **changes)
    t0 = time.perf_counter()
    its, fails = [], 0
    for k in range(n):
        r = dec.decode_detailed(det[k])
        its.append(r.iterations)
        fails += (not r.success) or bool((((L @ np.asarray(r.decoding, np.uint8)) % 2) != obs[k]).any())
    its = np.array(its)
    print(f"  {name:34s} mean {its.mean():6.0f} | median {np.median(its):5.0f} | p99 {np.percentile(its, 99):6.0f} iterations"
          f" | failures {fails}/{n} | {(time.perf_counter() - t0) / n * 1e3:5.1f} ms per shot here", flush=True)
