"""(b) exhaustive single-fault enumeration, new default encode_bus: k=4 at p=2,3 (exact simulation),
k=6 at p=2 (tracker; two circuits also exact)."""
import itertools, sys, time
from collections import Counter
import numpy as np
from qiskit import QuantumCircuit
import iceberg
from faultlib import Analysis
from tracker_large import TrackerOnly
from rerun_lib import all_faults, report

rng = np.random.default_rng(2027)
def bare_circ(k, edges, p):
    qc = QuantumCircuit(k, k)
    qc.h(range(k))
    for _ in range(p):
        for i, j in edges:
            qc.rzz(rng.uniform(0.3, 1.3) * rng.choice([-1, 1]), i, j)
        for i in range(k):
            qc.rz(rng.uniform(0.3, 1.3), i)
        for i in range(k):
            qc.rx(rng.uniform(0.3, 1.3), i)
    qc.measure(range(k), range(k))
    return qc

t0 = time.time()
tot = Counter(); offenders = []; blocks = Counter(); maxw = Counter()
def go(tag, enc, exact, key):
    an = Analysis(enc) if exact else TrackerOnly(enc)
    faults = all_faults(an)
    n, off = report(tag, an, faults)
    tot[key + ("faults",)] += n; tot[key + ("circuits",)] += 1
    for f in faults:
        if f.kept > 0.5 and f.status == "harmful":
            blocks[key + (f.block,)] += 1
            maxw[key] = max(maxw[key], f.footprint)
    offenders.extend((tag, f) for f in off)

G4 = {"K4": list(itertools.combinations(range(4), 2)), "path": [(0, 1), (1, 2), (2, 3)], "star": [(0, 1), (0, 2), (0, 3)],
      "cycle": [(0, 1), (1, 2), (2, 3), (0, 3)], "rand": [(0, 2), (1, 2), (1, 3), (2, 3)]}
print("== k=4, p=2 (exact)")
for g, edges in G4.items():
    bare = bare_circ(4, edges, 2)
    for mode in ("direct", "bus"):
        for pl in (None, (2, 0, 3, 1), (3, 2, 1, 0)):
            go(f"k4 p2 {g} mode={mode} placement={pl}", iceberg.encode_bus(bare, edge_mode=mode, placement=pl), True, (4, 2))
print(f"   elapsed {time.time()-t0:.0f}s")
print("== k=4, p=3 (exact)")
for g in ("K4", "path", "rand"):
    bare = bare_circ(4, G4[g], 3)
    for mode in ("direct", "bus"):
        for pl in (None, (1, 3, 0, 2)):
            go(f"k4 p3 {g} mode={mode} placement={pl}", iceberg.encode_bus(bare, edge_mode=mode, placement=pl), True, (4, 3))
print(f"   elapsed {time.time()-t0:.0f}s")
print("== k=6, p=2 (tracker; first two also exact)")
G6 = {"K6": list(itertools.combinations(range(6), 2)), "ring": [(i, (i + 1) % 6) for i in range(6)],
      "rand": [(0, 1), (0, 2), (0, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5)], "bip": [(i, j) for i in range(3) for j in range(3, 6)]}
first = True
for g, edges in G6.items():
    bare = bare_circ(6, [tuple(sorted(e)) for e in edges], 2)
    for mode in ("direct", "bus"):
        for pl in (None, (4, 0, 5, 1, 3, 2)):
            enc = iceberg.encode_bus(bare, edge_mode=mode, placement=pl)
            go(f"k6 p2 {g} mode={mode} placement={pl}", enc, False, (6, 2))
            if g == "K6" and pl is None:
                go(f"   (exact) k6 p2 {g} mode={mode}", enc, True, (6, 2, "exact"))
print(f"   elapsed {time.time()-t0:.0f}s")
print("totals:", {k: v for k, v in tot.items()})
print("undetected harmful per block:", {k: v for k, v in sorted(blocks.items(), key=str)})
print("max logical weight of an undetected harmful fault:", dict(maxw))
print("offenders (kept with bus flipped / non-logical residual / >2 logical / harmful in init-syndrome-measure):", len(offenders))
