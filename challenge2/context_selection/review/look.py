import iceberg, qsel
from vlib import *
q = qsel.random_qubo(4, edge_fraction=0.5, seed=1)
ang = qsel.tune_angles(q, p=1).angles
bare = qsel.qaoa_circuit(q, ang)
print(bare.draw(fold=200))
ed = iceberg.encode_direct(bare)
print(ed.draw(fold=250))
print(ed.metadata)
eb = iceberg.encode_bus(bare)
print(eb.draw(fold=250))
print(eb.metadata)
