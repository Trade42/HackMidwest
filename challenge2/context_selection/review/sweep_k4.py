"""k=4: every placement x both edge modes x several graphs, p=1 (and a subset at p=2)."""
import itertools, sys, time
from collections import Counter
import numpy as np
import iceberg, qsel
from qsel import Angles
from vlib import *
from faultlib import *
from explore_bus import footprint

rng = np.random.default_rng(7)
t0 = time.time()
tot = Counter()
part_viol = []
maxw = Counter()
latent = Counter()
meas_bad = 0
def analyse(tag, enc, p):
    global meas_bad
    an = Analysis(enc)
    faults = an.enumerate()
    assert all(f.pred_ok for f in faults)
    assert set(round(f.kept, 6) for f in faults) <= {0.0, 1.0}
    style = an.lay.style
    tot[(style, p, "faults")] += len(faults)
    tot[(style, p, "circuits")] += 1
    for f in faults:
        if f.kind == "meas" and f.status != "discarded":
            meas_bad += 1
        if f.status == "harmful":
            tot[(style, p, "harmful")] += 1
            if f.block in ("init", "syndrome", "measure"):
                part_viol.append((tag, describe(enc, f)))
            w, clean = footprint(an, f)
            maxw[(style, p, "clean" if clean else "bus-residual", w)] += 1
        elif f.status == "harmless" and f.kind == "gate" and (f.anc_x or f.lx < 0):
            latent[(style, p, f.block)] += 1

graphs = {"path": [(0, 1), (1, 2), (2, 3)], "K4": list(itertools.combinations(range(4), 2)), "star": [(0, 1), (0, 2), (0, 3)], "none": []}
from qiskit import QuantumCircuit
def bare_circ(edges, p):
    qc = QuantumCircuit(4, 4)
    qc.h(range(4))
    for _ in range(p):
        for i, j in edges:
            qc.rzz(rng.uniform(0.3, 1.3) * rng.choice([-1, 1]), i, j)
        for i in range(4):
            qc.rz(rng.uniform(0.3, 1.3), i)
        for i in range(4):
            qc.rx(rng.uniform(0.3, 1.3), i)
    qc.measure(range(4), range(4))
    return qc

for gname, edges in graphs.items():
    for p in (1, 2):
        bare = bare_circ(edges, p)
        analyse(f"direct {gname} p{p}", iceberg.encode_direct(bare), p)
        perms = list(itertools.permutations(range(4))) if p == 1 else [(0, 1, 2, 3), (2, 0, 3, 1), (3, 2, 1, 0), (1, 3, 0, 2)]
        for perm in perms:
            for mode in ("direct", "bus"):
                analyse(f"bus {gname} p{p} placement={perm} mode={mode}", iceberg.encode_bus(bare, placement=perm, edge_mode=mode), p)
    print(f"{gname}: elapsed {time.time()-t0:.0f}s", flush=True)
print("totals:", dict(tot))
print("readout flips not discarded:", meas_bad)
print("undetected harmful in init/syndrome/measure:", len(part_viol))
for v in part_viol[:5]:
    print("   ", v)
print("undetected harmful by (style, p, kind, logical footprint):")
for key in sorted(maxw):
    print("   ", key, maxw[key])
print("latent (kept, harmless here, but residual is not a clean logical operator):", dict(latent))
