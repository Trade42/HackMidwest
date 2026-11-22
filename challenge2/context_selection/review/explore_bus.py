"""Wider sweep: both encoders over k=2..6, p=1,2; classify undetected harmful faults."""
import sys, time
from collections import Counter
import numpy as np
import iceberg, qsel
from qsel import Angles
from vlib import *
from faultlib import *

def footprint(an, f):
    """Logical qubits touched: residual logical Pauli (if clean) + supports of sign-flipped
    rotations that the residual does not explain + readout shift."""
    K = an.lay.k_code
    rot = {j: (lx, lz) for j, th, s, lx, lz in an.rot}
    clean = (f.anc_x == 0 and f.lx >= 0)
    sup = 0
    if clean:
        sup |= f.lx | f.lz
        return popc(sup), True
    for j in f.flipped:
        lx, lz = rot[j]
        sup |= lx | lz
    # shift is in bare-clbit numbering; map back to logical qubits
    for i, cl in enumerate(an.lay.logical_clbits):
        if cl >= 0 and (f.shift >> cl) & 1:
            sup |= 1 << i
    return popc(sup), False

def run(tag, enc, parts=("init", "syndrome", "measure")):
    an = Analysis(enc)
    faults = an.enumerate()
    assert all(f.pred_ok for f in faults), "tracker mismatch"
    assert set(round(f.kept, 6) for f in faults) <= {0.0, 1.0}
    harmful = [f for f in faults if f.status == "harmful"]
    in_parts = [f for f in harmful if f.block in parts]
    meas_ok = all(f.status == "discarded" for f in faults if f.kind == "meas")
    cleanw, dirty = Counter(), Counter()
    worst = None
    for f in harmful:
        w, clean = footprint(an, f)
        (cleanw if clean else dirty)[w] += 1
        if not clean and (worst is None or w > worst[0]):
            worst = (w, f)
    nb = sum(1 for f in harmful if an.lay.bottom in f.qubits or an.lay.top in f.qubits)
    print(f"{tag:42s} n={enc.num_qubits:2d} faults={len(faults):5d} harmful={len(harmful):3d} in init/synd/meas={len(in_parts)} "
          f"meas_flips_discarded={meas_ok} clean-logical weights={dict(sorted(cleanw.items()))} bus-residual footprints={dict(sorted(dirty.items()))}")
    if worst:
        print("      worst bus-residual:", describe(enc, worst[1]), "footprint", worst[0])
    for f in in_parts[:3]:
        print("      PART VIOLATION:", describe(enc, f))
    return an, faults

if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    t0 = time.time()
    total = 0
    for k, ef, seed, p in [(2, 1.0, 0, 1), (2, 1.0, 0, 2), (3, 0.7, 1, 1), (3, 1.0, 2, 2), (4, 0.5, 3, 1), (4, 1.0, 4, 2), (4, 0.5, 5, 2), (4, 0.34, 6, 3),
                           (5, 0.5, 7, 1), (5, 0.6, 8, 2), (6, 0.4, 9, 1), (6, 0.5, 10, 2)]:
        q = qsel.random_qubo(k, edge_fraction=ef, seed=seed)
        ang = Angles(tuple(rng.uniform(0.3, 1.2, p) * rng.choice([-1, 1], p)), tuple(rng.uniform(0.3, 1.2, p)))
        bare = qsel.qaoa_circuit(q, ang)
        print(f"--- k={k} p={p} edges={q.edges}")
        for name, enc in (("direct", iceberg.encode_direct(bare)), ("bus[direct]", iceberg.encode_bus(bare)), ("bus[bus]", iceberg.encode_bus(bare, edge_mode="bus"))):
            an, faults = run(name, enc)
            total += len(faults)
        print(f"   elapsed {time.time()-t0:.0f}s")
    print("total faults enumerated:", total)
