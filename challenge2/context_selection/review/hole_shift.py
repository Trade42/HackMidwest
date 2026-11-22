"""Look for a bus-encoder single fault whose effect is an exact XOR shift of >2 logical bits."""
import itertools, math
import numpy as np
from qiskit import QuantumCircuit
import iceberg
from vlib import *
from faultlib import *

def bare_circ(k, edges1, edges2, seed=0):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(k, k)
    qc.h(range(k))
    for i, j in edges1:
        qc.rzz(rng.uniform(0.3, 1.2), i, j)
    for i in range(k):
        qc.rz(rng.uniform(0.3, 1.2), i)
    for i in range(k):
        qc.rx(rng.uniform(0.3, 1.2), i)
    for i, j in edges2:
        qc.rzz(math.pi / 2, i, j)
    for i in range(k):
        qc.rz(rng.uniform(0.3, 1.2), i)
    for i in range(k):
        qc.rx(math.pi / 2, i)
    qc.measure(range(k), range(k))
    return qc

K4 = list(itertools.combinations(range(4), 2))
cands = {
    "K4/K4": (K4, K4),
    "cycle+diag": ([(0, 1), (1, 2), (2, 3), (0, 3), (0, 2)], [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2)]),
    "K4 then two disjoint": (K4, [(0, 1), (2, 3)]),
    "K4 then (0,2),(1,3)": (K4, [(0, 2), (1, 3)]),
    "K4 then (0,3),(1,2)": (K4, [(0, 3), (1, 2)]),
    "star": ([(0, 1), (0, 2), (0, 3)], [(0, 1), (0, 2), (0, 3)]),
    "path": ([(0, 1), (1, 2), (2, 3)], [(0, 1), (1, 2), (2, 3)]),
}
for name, (e1, e2) in cands.items():
    bare = bare_circ(4, e1, e2)
    for mode in ("direct", "bus"):
        enc = iceberg.encode_bus(bare, edge_mode=mode)
        an = Analysis(enc)
        faults = an.enumerate()
        assert all(f.pred_ok for f in faults)
        harmful = [f for f in faults if f.status == "harmful"]
        big = [f for f in harmful if f.pure_shift >= 0 and popc(f.pure_shift) > 2]
        print(f"{name:24s} edge_mode={mode:6s} placement={enc.metadata['placement']} harmful={len(harmful)} "
              f"pure-shift weights={sorted(set(popc(f.pure_shift) for f in harmful if f.pure_shift>=0))} #shift>2: {len(big)}")
        for f in big[:2]:
            print("     ", describe(enc, f))
