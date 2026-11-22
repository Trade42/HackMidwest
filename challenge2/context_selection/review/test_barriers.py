"""What does a compiler do to the checks when barriers=False?  And metadata 'iceberg_terms' sanity."""
import numpy as np
from qiskit import transpile
from qiskit.transpiler import CouplingMap
import iceberg, qsel
from qsel import Angles
from vlib import *
from faultlib import *

q = qsel.random_qubo(4, edge_fraction=0.5, seed=1)
bare = qsel.qaoa_circuit(q, Angles((0.6,), (0.5,)))
print("== direct encoder, all-to-all, optimization_level=3")
for barriers in (True, False):
    enc = iceberg.encode_direct(bare, barriers=barriers)
    lay = iceberg.layout_of(enc)
    for basis in (["cx", "rz", "sx", "x"], ["cz", "rz", "sx", "x"]):
        tq = transpile(enc, basis_gates=basis, optimization_level=3, seed_transpiler=0)
        twoq = [tuple(sorted(tq.find_bit(x).index for x in inst.qubits)) for inst in tq.data if inst.operation.num_qubits == 2 and inst.operation.name != "barrier"]
        f, a = lay.flag, lay.sx_ancilla
        lg, kept = my_postselect_dist(exact_dist(tq), lay)
        print(f"  barriers={barriers} basis={basis[0]}: 2q total {len(twoq)} (input budget {enc.count_ops().get('cx',0) + 2*enc.count_ops().get('rzz',0) + 2*enc.count_ops().get('rxx',0)}); "
              f"gates on (a,f): {twoq.count(tuple(sorted((a, f))))} (encoder emits 2); gates touching f: {sum(1 for p in twoq if f in p)} (encoder emits 4); "
              f"noiseless equivalence max diff {np.abs(lg - exact_dist(bare)).max():.1e}")
print("== bus encoder on from_grid(3,3)")
for barriers in (True, False):
    enc = iceberg.encode_bus(bare, barriers=barriers)
    tq = transpile(enc, basis_gates=["cz", "rz", "sx", "x"], coupling_map=CouplingMap.from_grid(3, 3), optimization_level=3, seed_transpiler=0)
    n2 = sum(1 for inst in tq.data if inst.operation.num_qubits == 2 and inst.operation.name != "barrier")
    lg, kept = my_postselect_dist(exact_dist(tq), iceberg.layout_of(enc))
    print(f"  barriers={barriers}: 2q {n2} (budget {enc.count_ops().get('cx',0) + 2*enc.count_ops().get('rzz',0)}), equivalence max diff {np.abs(lg - exact_dist(bare)).max():.1e}")

print("== metadata['iceberg_terms'] vs generators derived independently (stim)")
bad = 0; n = 0
rng = np.random.default_rng(1)
for k, ef, seed, p in [(4, 1.0, 1, 2), (5, 0.6, 2, 2), (6, 0.5, 3, 1), (3, 1.0, 4, 3)]:
    qq = qsel.random_qubo(k, edge_fraction=ef, seed=seed)
    b = qsel.qaoa_circuit(qq, Angles(tuple(rng.uniform(0.2, 1, p)), tuple(rng.uniform(0.2, 1, p))))
    for enc in (iceberg.encode_direct(b), iceberg.encode_bus(b), iceberg.encode_bus(b, edge_mode="bus")):
        an = Analysis(enc)
        by_data = {an.ops[j][0]: (lx, lz, th, s) for j, th, s, lx, lz in an.rot}
        terms = enc.metadata["iceberg_terms"]
        if len(terms) != len(by_data):
            bad += 1
        for kind, logical, idx in terms:
            n += 1
            mask = sum(1 << i for i in logical)
            lx, lz, th, s = by_data.get(idx, (None, None, None, None))
            want = (mask, 0) if kind in ("x", "xx") else (0, mask)
            if (lx, lz) != want or s != 1:
                bad += 1
print(f"  {n} term records checked, mismatches: {bad}")
