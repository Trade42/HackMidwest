"""Look at the K=2 offenders in detail, with exact simulation."""
import itertools
import numpy as np
from qiskit import QuantumCircuit
import iceberg
from faultlib import Analysis, describe, qname
from rerun_lib import all_faults

def show(title, bare, **kw):
    enc = iceberg.encode_bus(bare, **kw)
    an = Analysis(enc)
    faults = all_faults(an)
    assert all(f.pred_ok for f in faults if f.kind != "meas")
    kept = [f for f in faults if f.kept > 0.5]
    harmful = [f for f in kept if f.status == "harmful"]
    bus = [f for f in kept if f.blk_anc_x and f.block not in ("syndrome", "measure")]
    part = [f for f in harmful if f.block in ("init", "syndrome", "measure")]
    print(f"== {title}: n={enc.num_qubits} faults={len(faults)} kept={len(kept)} harmful={len(harmful)} "
          f"per block={ {b: sum(1 for f in harmful if f.block == b) for b in dict.fromkeys(f.block for f in harmful)} } "
          f"kept with bus flipped at block end={len(bus)} (harmful among them {sum(1 for f in bus if f.status == 'harmful')}) harmful in init/syndrome/measure={len(part)}")
    for f in (part + [f for f in bus if f.status == "harmful"])[:8]:
        print("     ", f.kind, describe(enc, f), "bus_x", [qname(enc, q) for q in range(an.n) if f.blk_anc_x >> q & 1])
    return enc, an, faults

def qaoa(k, p, edges, h=True, seed=0):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(k, k)
    if h: qc.h(range(k))
    for _ in range(p):
        for i, j in edges: qc.rzz(rng.uniform(0.4, 1.2), i, j)
        for i in range(k): qc.rz(rng.uniform(0.4, 1.2), i)
        for i in range(k): qc.rx(rng.uniform(0.4, 1.2), i)
    qc.measure(range(k), range(k))
    return qc

show("k=2 p=1", qaoa(2, 1, [(0, 1)]))
show("k=2 p=2", qaoa(2, 2, [(0, 1)]))
show("k=2 p=3", qaoa(2, 3, [(0, 1)]))
show("k=1 p=2", qaoa(1, 2, []))
show("k=2 p=2 no leading H", qaoa(2, 2, [(0, 1)], h=False))
show("k=1 p=2 no leading H", qaoa(1, 2, [], h=False))
# k=1, rz then rx then rz then rx (a field after a mixer)
qc = QuantumCircuit(1, 1); qc.h(0); qc.rz(0.7, 0); qc.rx(0.9, 0); qc.rz(0.5, 0); qc.rx(0.6, 0); qc.measure(0, 0)
show("k=1 two layers of rz/rx", qc)
show("k=3 p=2 (K=4, for contrast)", qaoa(3, 2, [(0, 1), (1, 2), (0, 2)]))
