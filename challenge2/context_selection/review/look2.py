import iceberg, qsel
from vlib import *
q = qsel.random_qubo(4, edge_fraction=0.5, seed=1)
ang = qsel.tune_angles(q, p=1).angles
bare = qsel.qaoa_circuit(q, ang)
for enc in (iceberg.encode_direct(bare), iceberg.encode_bus(bare)):
    lab = block_labels(enc)
    print(len(lab), len(enc.data), enc.metadata["iceberg_blocks"])
    print({k:v for k,v in enc.metadata.items() if k!="iceberg_terms"})
    names = [q._register.name+str(q._index) if hasattr(q,'_register') else str(i) for i,q in enumerate(enc.qubits)]
    for i, inst in enumerate(enc.data):
        qs = [enc.find_bit(x).index for x in inst.qubits]
        if inst.operation.name=="barrier":
            print(i, lab[i], "barrier"); continue
        print(i, lab[i], inst.operation.name, [names[x] for x in qs], [round(float(p),4) for p in inst.operation.params])
