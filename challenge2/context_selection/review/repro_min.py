import math, numpy as np, iceberg
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

bare = QuantumCircuit(4, 4); bare.h(range(4))
for (i, j), th in zip([(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)], [0.9,0.5,0.7,1.1,0.4,0.8]): bare.rzz(th, i, j)
for i, th in enumerate([0.6, 1.0, 0.35, 0.75]): bare.rz(th, i)
for i, th in enumerate([0.45, 0.95, 0.65, 0.85]): bare.rx(th, i)
bare.rzz(math.pi/2, 0, 3); bare.rzz(math.pi/2, 1, 2)                 # layer 2
for i, th in enumerate([0.5, 0.9, 0.3, 0.7]): bare.rz(th, i)
for i in range(4): bare.rx(math.pi/2, i)
bare.measure(range(4), range(4))

enc = iceberg.encode_bus(bare)                                       # all defaults
root = iceberg.layout_of(enc).sx_ancilla                             # qubit 8 = c_2
lab = [l for l, n in enc.metadata["iceberg_blocks"] for _ in range(n)]
idx = next(i for i, g in enumerate(enc.data) if lab[i] == "fields" and g.operation.name == "cx"
           and [enc.find_bit(q).index for q in g.qubits] == [root, root - 1])     # enc.data[42]
bad = enc.copy_empty_like()
for i, g in enumerate(enc.data):
    bad.append(g.operation, g.qubits, g.clbits)
    if i == idx: bad.y(root)                                         # the one fault

def kept_dist(circ):
    p = Statevector(circ.remove_final_measurements(inplace=False)).probabilities()
    counts = {format(i, "09b"): int(round(x * 1e9)) for i, x in enumerate(p) if round(x * 1e9)}
    logical, stats = iceberg.postselect(counts, enc)
    q = np.zeros(16)
    for k, c in logical.items(): q[int(k, 2)] = c
    return q / q.sum(), stats.kept_share
p = Statevector(bare.remove_final_measurements(inplace=False)).probabilities()
q, kept = kept_dist(bad)
print(idx, root, kept, 0.5 * abs(q - p).sum(), 0.5 * abs(q[np.arange(16) ^ 15] - p).sum())
