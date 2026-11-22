"""Self-contained reproduction: one Pauli fault, default encode_bus, every check passes,
all four logical bits flip.  Uses only qiskit + the public iceberg API."""
import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
import iceberg

k = 4
bare = QuantumCircuit(k, k)
bare.h(range(k))
# layer 1 (generic angles, complete graph)
for (i, j), th in zip([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], [0.9, 0.5, 0.7, 1.1, 0.4, 0.8]):
    bare.rzz(th, i, j)
for i, th in enumerate([0.6, 1.0, 0.35, 0.75]):
    bare.rz(th, i)
for i, th in enumerate([0.45, 0.95, 0.65, 0.85]):
    bare.rx(th, i)
# layer 2
bare.rzz(math.pi / 2, 0, 3)
bare.rzz(math.pi / 2, 1, 2)
for i, th in enumerate([0.5, 0.9, 0.3, 0.7]):
    bare.rz(th, i)
for i in range(k):
    bare.rx(math.pi / 2, i)
bare.measure(range(k), range(k))

enc = iceberg.encode_bus(bare)                 # defaults: init_flag=True, precopy=True
lay = iceberg.layout_of(enc)
root = lay.sx_ancilla                          # bus qubit c_{C-1}
labels = [l for l, n in enc.metadata["iceberg_blocks"] for _ in range(n)]
# first gate of the first "fields" fan-out that has the root bus qubit as control
idx = next(i for i, inst in enumerate(enc.data)
           if labels[i] == "fields" and inst.operation.name == "cx"
           and enc.find_bit(inst.qubits[0]).index == root and enc.find_bit(inst.qubits[1]).index == root - 1)
faulty = enc.copy_empty_like()
for i, inst in enumerate(enc.data):
    faulty.append(inst.operation, inst.qubits, inst.clbits)
    if i == idx:
        faulty.y(root)                         # the single fault
print("placement", enc.metadata["placement"], "| fault: Y on qubit", root, "after enc.data[%d] = cx(%d,%d), block '%s'" % (idx, root, root - 1, labels[idx]))

def exact_counts(circ, scale=10**9):
    p = Statevector(circ.remove_final_measurements(inplace=False)).probabilities()
    return {format(i, f"0{circ.num_qubits}b"): int(round(x * scale)) for i, x in enumerate(p) if round(x * scale)}

p_bare = Statevector(bare.remove_final_measurements(inplace=False)).probabilities()
for name, circ in (("ideal encoded", enc), ("faulty encoded", faulty)):
    logical, stats = iceberg.postselect(exact_counts(circ), enc)
    q = np.zeros(2 ** k)
    for key, c in logical.items():
        q[int(key, 2)] = c
    q /= q.sum()
    tv = 0.5 * np.abs(q - p_bare).sum()
    shifted = 0.5 * np.abs(q[np.arange(16) ^ 0b1111] - p_bare).sum()
    print(f"{name:15s}: kept share {stats.kept_share:.6f}  TV(kept, bare) = {tv:.6f}  TV(kept XOR 1111, bare) = {shifted:.2e}")

# same thing with sampled shots through Aer, to show postselect keeps everything
sim = AerSimulator(method="statevector", seed_simulator=1)
counts = sim.run(faulty, shots=20000).result().get_counts()
logical, stats = iceberg.postselect(counts, enc)
print("Aer 20000 shots of the faulty circuit:", stats)
bus = [lay.sx_ancilla, *lay.flags]
print("bus bits read 1 in any shot:", any(key[::-1][b] == "1" for key in counts for b in bus))
# and the bus really is flipped before the final check
pre = iceberg.encode_bus(bare, readout=False)
fp = QuantumCircuit(pre.num_qubits)
for i, inst in enumerate(pre.data):
    if inst.operation.name == "barrier":
        continue
    fp.append(inst.operation, [pre.find_bit(x).index for x in inst.qubits])
    if i == idx:
        fp.y(root)
probs = Statevector(fp).probabilities_dict()
print("bus state (c_root, c_root-1, ...) just before the S_X check:", {key[::-1][root] + key[::-1][root - 1] + key[::-1][root - 2] for key, v in probs.items() if v > 1e-12})
