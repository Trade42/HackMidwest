"""Claim 4: transpile robustness of the bus encoding."""
import time
import numpy as np
from qiskit import transpile
from qiskit.transpiler import CouplingMap
import iceberg, qsel
from qsel import Angles
from vlib import *

def twoq_budget(enc):
    ops = enc.count_ops()
    return ops.get("cx", 0) + 2 * ops.get("rzz", 0) + 2 * ops.get("rxx", 0)

def twoq_count(circ):
    return sum(1 for inst in circ.data if inst.operation.num_qubits == 2 and inst.operation.name != "barrier")

rng = np.random.default_rng(5)
t0 = time.time()
print("k=4 on from_grid(3,3): simulate the transpiled circuit")
for ef, seed, p, mode in [(0.5, 1, 1, "direct"), (0.5, 1, 1, "bus"), (1.0, 2, 1, "direct"), (1.0, 2, 2, "direct"), (0.67, 3, 2, "bus"), (0.0, 4, 1, "direct"), (0.34, 5, 2, "direct")]:
    q = qsel.random_qubo(4, edge_fraction=ef, seed=seed)
    ang = qsel.tune_angles(q, p=p).angles
    bare = qsel.qaoa_circuit(q, ang)
    enc = iceberg.encode_bus(bare, edge_mode=mode)
    C = iceberg.layout_of(enc).k_code // 2 + 1
    pb = exact_dist(bare)
    for ts in (0, 1, 2):
        tq = transpile(enc, basis_gates=["cz", "rz", "sx", "x"], coupling_map=CouplingMap.from_grid(3, C), optimization_level=3, seed_transpiler=ts)
        names = set(tq.count_ops()) - {"barrier", "measure"}
        meta_ok = isinstance(tq.metadata, dict) and "iceberg" in tq.metadata
        lay = iceberg.layout_of(enc)
        # every 2q gate on a coupling edge
        edges = set(map(tuple, CouplingMap.from_grid(3, C).get_edges()))
        on_edges = all(tuple(tq.find_bit(x).index for x in inst.qubits) in edges for inst in tq.data if inst.operation.num_qubits == 2 and inst.operation.name != "barrier")
        # measurement map: clbit j must be written exactly once
        _, meas = parse(tq)
        clbits = sorted(c for _, _, c in meas)
        pe = exact_dist(tq)
        logical, kept = my_postselect_dist(pe, lay)
        d = np.abs(logical - pb).max()
        # also against the untranspiled encoded distribution, bit for bit
        d_enc = np.abs(pe - exact_dist(enc)).max()
        # package postselect with the *transpiled* circuit as layout source
        cnt = {format(i, f"0{tq.num_clbits}b"): int(round(x * 10**12)) for i, x in enumerate(pe) if round(x * 10**12)}
        lc, st = iceberg.postselect(cnt, tq if meta_ok else enc)
        print(f"  ef={ef} p={p} mode={mode:6s} seed_transpiler={ts}: gates={sorted(names)} 2q={twoq_count(tq)} budget={twoq_budget(enc)} "
              f"depth={tq.depth()} on_edges={on_edges} metadata_kept={meta_ok} clbits_ok={clbits == list(range(tq.num_clbits))} "
              f"max|logical-bare|={d:.1e} max|enc-transpiled|={d_enc:.1e} kept={kept:.9f} pkg_kept={st.kept_share:.6f}")
print(f"elapsed {time.time()-t0:.0f}s")

print("k=10 on from_grid(4,6): gate count only")
for ef, seed, p, mode in [(0.2, 11, 1, "direct"), (0.3, 12, 1, "direct"), (0.3, 12, 2, "direct"), (0.5, 13, 1, "bus"), (1.0, 14, 1, "direct")]:
    q = qsel.random_qubo(10, edge_fraction=ef, seed=seed)
    bare = qsel.qaoa_circuit(q, Angles(tuple(rng.uniform(0.2, 1, p)), tuple(rng.uniform(0.2, 1, p))))
    enc = iceberg.encode_bus(bare, edge_mode=mode)
    for ts in (0, 1):
        t1 = time.time()
        tq = transpile(enc, basis_gates=["cz", "rz", "sx", "x"], coupling_map=CouplingMap.from_grid(4, 6), optimization_level=3, seed_transpiler=ts)
        print(f"  ef={ef} E={q.num_edges} p={p} mode={mode:6s} seed={ts}: qubits={enc.num_qubits} 2q={twoq_count(tq)} budget={twoq_budget(enc)} "
              f"ok={twoq_count(tq) <= twoq_budget(enc)} swaps_in_layout={'swap' in tq.count_ops()} ({time.time()-t1:.1f}s)")
    # for contrast: the direct encoding on the same device
    encd = iceberg.encode_direct(bare)
    tqd = transpile(encd, basis_gates=["cz", "rz", "sx", "x"], coupling_map=CouplingMap.from_grid(4, 6), optimization_level=3, seed_transpiler=0)
    print(f"     (encode_direct for contrast: 2q={twoq_count(tqd)} budget={twoq_budget(encd)})")
print(f"elapsed {time.time()-t0:.0f}s")
