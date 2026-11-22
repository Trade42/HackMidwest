import numpy as np
from qiskit import transpile
from qiskit.transpiler import CouplingMap
import iceberg, qsel
from vlib import *

def segments(circ):
    segs, cur = [], []
    for inst in circ.data:
        if inst.operation.name == "barrier":
            segs.append(cur); cur = []
        else:
            cur.append(inst)
    segs.append(cur)
    return segs

q = qsel.random_qubo(4, edge_fraction=0.5, seed=1)
bare = qsel.qaoa_circuit(q, qsel.tune_angles(q, p=1).angles)
for mode in ("direct", "bus"):
    enc = iceberg.encode_bus(bare, edge_mode=mode)
    tq = transpile(enc, basis_gates=["cz", "rz", "sx", "x"], coupling_map=CouplingMap.from_grid(3, 3), optimization_level=3, seed_transpiler=0)
    lab = block_labels(enc)
    se, st = segments(enc), segments(tq)
    print(mode, len(se), len(st))
    pos = 0
    for a, b in zip(se, st):
        bud = sum({"cx": 1, "rzz": 2}.get(i.operation.name, 0) for i in a)
        got = sum(1 for i in b if i.operation.num_qubits == 2)
        names = [(i.operation.name, [enc.find_bit(x).index for x in i.qubits]) for i in a]
        flag = "  <-- reduced" if got < bud else ""
        print(f"   segment: enc 2q budget {bud}, transpiled 2q {got}{flag}")
        if got < bud:
            print("      enc gates:", names)
            print("      transpiled:", [(i.operation.name, [tq.find_bit(x).index for x in i.qubits], [round(float(p), 3) for p in i.operation.params]) for i in b if i.operation.num_qubits == 2 or True][:60])
    print("   layout:", tq.layout.initial_index_layout() if tq.layout else None)
