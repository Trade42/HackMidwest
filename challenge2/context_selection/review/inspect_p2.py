import iceberg, qsel
from qsel import Angles
from vlib import *
from faultlib import *
q = qsel.random_qubo(4, edge_fraction=0.67, seed=3)
ang = Angles((0.731, -0.377), (0.413, 0.887))
bare = qsel.qaoa_circuit(q, ang)
enc = iceberg.encode_bus(bare)
an = Analysis(enc)
faults = an.enumerate()
lab = block_labels(enc)
for i, inst in enumerate(enc.data):
    qs = [enc.find_bit(x).index for x in inst.qubits]
    if inst.operation.name in ("barrier",): continue
    print(i, lab[i], inst.operation.name, [qname(enc, x) for x in qs], [round(float(p),3) for p in inst.operation.params])
print()
for f in faults:
    if "harmful" in f.status and f.anc_x:
        print(describe(enc, f), "anc_x=", [qname(enc, q) for q in range(enc.num_qubits) if f.anc_x >> q & 1], "flipped rot ops:", [an.ops[j][0] for j in f.flipped])
