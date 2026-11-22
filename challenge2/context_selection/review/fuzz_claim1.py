"""Fuzz: random circuits over the whole documented gate set, random measurement maps, k = 1..6."""
import math, time
import numpy as np
from qiskit import QuantumCircuit
import iceberg
from vlib import *

rng = np.random.default_rng(31337)
ONEQ_Z = ["rz", "p", "s", "sdg", "t", "tdg", "z"]
def rand_circ(k, allow_rxx):
    nc = k + int(rng.integers(0, 3))
    qc = QuantumCircuit(k, nc)
    if rng.random() < 0.8:
        qc.h(range(k))
        if rng.random() < 0.3:
            qc.barrier()
    for _ in range(int(rng.integers(0, 30))):
        r = rng.random()
        if r < 0.25 and k >= 2:
            i, j = (int(x) for x in rng.choice(k, 2, replace=False))
            qc.rzz(rng.uniform(-3, 3), i, j)
        elif r < 0.32 and k >= 2:
            i, j = (int(x) for x in rng.choice(k, 2, replace=False))
            qc.cz(i, j)
        elif r < 0.37 and k >= 2 and allow_rxx:
            i, j = (int(x) for x in rng.choice(k, 2, replace=False))
            qc.rxx(rng.uniform(-3, 3), i, j)
        elif r < 0.6:
            qc.rx(rng.uniform(-3, 3), int(rng.integers(k)))
        elif r < 0.85:
            g = str(rng.choice(ONEQ_Z)); i = int(rng.integers(k))
            if g in ("rz", "p"):
                getattr(qc, g)(rng.uniform(-3, 3), i)
            else:
                getattr(qc, g)(i)
        elif r < 0.93:
            getattr(qc, str(rng.choice(["x", "y"])))(int(rng.integers(k)))
        elif r < 0.97:
            qc.barrier()
        else:
            qc.id(int(rng.integers(k)))
    perm = rng.permutation(nc)[:k]
    which = [i for i in range(k) if rng.random() < 0.9] or [0]
    for i in which:
        qc.measure(i, int(perm[i]))
    return qc

t0 = time.time(); n = 0; worst = 0.0; fails = []
for trial in range(400):
    k = int(rng.integers(1, 7))
    for style in ("direct", "bus"):
        bare = rand_circ(k, allow_rxx=(style == "direct"))
        pb = exact_dist(bare)
        if style == "direct":
            opts = [dict(), dict(checks="z"), dict(checks="x"), dict(barriers=False, init_flag=bool(rng.integers(2)), sx_flag=bool(rng.integers(2)))]
            encs = [iceberg.encode_direct(bare, **o) for o in opts]
        else:
            K = k + k % 2
            opts = [dict(), dict(edge_mode="bus"), dict(placement=[int(x) for x in rng.permutation(K)], edge_mode=str(rng.choice(["direct", "bus"])), precopy=bool(rng.integers(2)), init_flag=bool(rng.integers(2)), barriers=bool(rng.integers(2)))]
            encs = [iceberg.encode_bus(bare, **o) for o in opts]
        for o, enc in zip(opts, encs):
            lay = iceberg.layout_of(enc)
            pe = exact_dist(enc)
            lg, kept = my_postselect_dist(pe, lay)
            d = max(np.abs(lg - pb).max(), abs(kept - 1))
            # package path on proportional counts
            cnt = {format(i, f"0{enc.num_clbits}b"): int(round(x * 10**12)) for i, x in enumerate(pe) if round(x * 10**12)}
            lc, st = iceberg.postselect(cnt, enc)
            pk = np.zeros(len(pb))
            for key, c in lc.items():
                pk[int(key, 2)] += c
            d2 = np.abs(pk / max(pk.sum(), 1) - pb).max() if st.kept else 1.0
            n += 1; worst = max(worst, d)
            if d > 1e-9 or d2 > 1e-8 or st.kept != st.shots:
                fails.append((trial, k, style, o, d, d2))
print(f"{n} encoded circuits from 800 random bare circuits, worst deviation {worst:.2e}, failures {len(fails)} ({time.time()-t0:.0f}s)")
for f in fails[:10]:
    print("  ", f)
