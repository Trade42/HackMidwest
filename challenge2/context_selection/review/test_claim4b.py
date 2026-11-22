import time
import numpy as np
from qiskit import transpile
from qiskit.transpiler import CouplingMap
import iceberg, qsel
from qsel import Angles
rng = np.random.default_rng(6)
def budget(enc):
    o = enc.count_ops(); return o.get("cx", 0) + 2 * o.get("rzz", 0)
def n2(c):
    return sum(1 for i in c.data if i.operation.num_qubits == 2 and i.operation.name != "barrier")
bad = 0; n = 0
t0 = time.time()
for k in (2, 3, 4, 5, 6, 8, 10, 12):
    for ef in (0.2, 0.6, 1.0):
        for p in (1, 2):
            q = qsel.random_qubo(k, edge_fraction=ef, seed=10 * k + p)
            bare = qsel.qaoa_circuit(q, Angles(tuple(rng.uniform(0.2, 1, p)), tuple(rng.uniform(0.2, 1, p))))
            for mode in ("direct", "bus"):
                enc = iceberg.encode_bus(bare, edge_mode=mode)
                C = iceberg.layout_of(enc).k_code // 2 + 1
                grids = [(3, C)] + ([(4, 6)] if k == 10 else []) + ([(C, 3)] if k in (4, 8) else [])
                for g in grids:
                    for lvl in (1, 3):
                        tq = transpile(enc, basis_gates=["cz", "rz", "sx", "x"], coupling_map=CouplingMap.from_grid(*g), optimization_level=lvl, seed_transpiler=k)
                        n += 1
                        if n2(tq) > budget(enc):
                            bad += 1
                            print(f"  EXCEEDS: k={k} ef={ef} p={p} mode={mode} grid={g} level={lvl}: {n2(tq)} > {budget(enc)}")
print(f"{n} transpilations, {bad} exceed the no-SWAP budget ({time.time()-t0:.0f}s)")
