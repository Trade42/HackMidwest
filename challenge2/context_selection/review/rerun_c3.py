"""More hunting at K >= 4 (no-H circuits, ragged circuits, random placements) and a positive control.
Run as a script.  Tracker only, except the control (exact)."""
import itertools, time
from collections import Counter
import numpy as np
from qiskit import QuantumCircuit
import iceberg
from faultlib import Analysis, describe
from tracker_large import TrackerOnly
from rerun_lib import all_faults

rng = np.random.default_rng(777)
def qaoa_like(k, edges, p, h=True):
    qc = QuantumCircuit(k, k)
    if h: qc.h(range(k))
    for _ in range(p):
        for i, j in edges: qc.rzz(rng.uniform(0.3, 1.3), i, j)
        for i in range(k): qc.rz(rng.uniform(0.3, 1.3), i)
        for i in range(k): qc.rx(rng.uniform(0.3, 1.3), i)
    qc.measure(range(k), range(k))
    return qc
def ragged(k, layers, h=True):
    qc = QuantumCircuit(k, k)
    if h: qc.h(range(k))
    for _ in range(layers):
        if rng.random() < 0.8:
            for i, j in itertools.combinations(range(k), 2):
                if rng.random() < 0.4: qc.rzz(rng.uniform(0.3, 1.3), i, j)
        if rng.random() < 0.8:
            for i in range(k):
                if rng.random() < 0.6: qc.rz(rng.uniform(0.3, 1.3), i)
        for i in ([i for i in range(k) if rng.random() < 0.6] or [int(rng.integers(k))]):
            qc.rx(rng.uniform(0.3, 1.3), i)
    qc.measure(range(k), range(k))
    return qc
def offenders(enc, exact=False):
    an = Analysis(enc) if exact else TrackerOnly(enc)
    fs = all_faults(an)
    off = [f for f in fs if f.kept > 0.5 and ((f.block not in ("syndrome", "measure") and (f.blk_anc_x or not f.blk_clean))
           or (f.status == "harmful" and (f.footprint > 2 or f.block in ("init", "syndrome", "measure"))))]
    return an, fs, off

if __name__ == "__main__":
    t0 = time.time(); tot = Counter()
    for k in (3, 4, 5, 6, 7, 8, 9, 10, 12):
        K = k + k % 2
        full = list(itertools.combinations(range(k), 2))
        jobs = []
        for p in (1, 2, 3):
            if k >= 9 and p == 3: continue
            edges = full if k <= 6 else [e for e in full if rng.random() < 0.35]
            jobs.append((f"noH p={p}", qaoa_like(k, edges, p, h=False)))
        for trial in range(6 if k <= 6 else 2):
            jobs.append((f"ragged h={trial % 2 == 0}", ragged(k, int(rng.integers(1, 4)), h=(trial % 2 == 0))))
        for tag, bare in jobs:
            for mode in ("direct", "bus"):
                enc = iceberg.encode_bus(bare, edge_mode=mode, placement=[int(x) for x in rng.permutation(K)])
                an, fs, off = offenders(enc)
                tot["circuits"] += 1; tot["faults"] += len(fs); tot["offenders"] += len(off)
                tot["maxw"] = max(tot["maxw"], max((f.footprint for f in fs if f.kept > 0.5 and f.status == "harmful"), default=0))
                for f in off[:2]:
                    print("   OFFENDER", k, tag, mode, describe(enc, f).split(": kept")[0])
    print(f"K>=4 extra hunt: {tot['circuits']} circuits, {tot['faults']} faults, offenders {tot['offenders']}, max logical weight {tot['maxw']} ({time.time()-t0:.0f}s)")
    bare = qaoa_like(4, list(itertools.combinations(range(4), 2)), 2)
    for pc in (True, 1, False):
        an, fs, off = offenders(iceberg.encode_bus(bare, precopy=pc), exact=True)
        harm = [f for f in off if f.status == "harmful"]
        print(f"control k=4 K4 p=2 precopy={pc!r}: faults {len(fs)}, offenders {len(off)} (harmful {len(harm)}), blocks {dict(Counter(f.block for f in harm))}")
