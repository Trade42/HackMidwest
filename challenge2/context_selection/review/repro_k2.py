"""Remaining gap after the fix: K = 2 code (k = 1 or 2), where the lane has only two bus qubits
and the two-step witness silently degrades to the old single copy."""
import math, numpy as np, iceberg
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

def run(bare, after, expect_gate, shift):
    enc = iceberg.encode_bus(bare)                         # all defaults
    lay = iceberg.layout_of(enc)
    root = lay.sx_ancilla                                  # qubit 5 = c_1
    g = enc.data[after]
    assert (g.operation.name, [enc.find_bit(q).index for q in g.qubits]) == expect_gate
    bad = enc.copy_empty_like()
    for i, inst in enumerate(enc.data):
        bad.append(inst.operation, inst.qubits, inst.clbits)
        if i == after: bad.y(root)                         # the one fault
    def kept_dist(circ):
        p = Statevector(circ.remove_final_measurements(inplace=False)).probabilities()
        n = circ.num_qubits
        counts = {format(i, f"0{n}b"): int(round(x * 1e9)) for i, x in enumerate(p) if round(x * 1e9)}
        logical, stats = iceberg.postselect(counts, enc)
        q = np.zeros(2 ** bare.num_clbits)
        for k, c in logical.items(): q[int(k, 2)] = c
        return (q / q.sum() if q.sum() else q), stats.kept_share
    p = Statevector(bare.remove_final_measurements(inplace=False)).probabilities()
    q0, k0 = kept_dist(enc)
    q, kept = kept_dist(bad)
    idx = np.arange(len(p))
    print(f"  no fault: kept {k0:.3f}, TV {0.5*abs(q0-p).sum():.1e} | fault Y on qubit {root} after enc.data[{after}]: "
          f"kept {kept:.6f}, TV(kept, bare) {0.5*abs(q-p).sum():.4f}, TV(kept XOR {shift:0{bare.num_clbits}b}, bare) {0.5*abs(q[idx ^ shift]-p).sum():.1e}")

print("A. k=2, two QAOA layers, default encode_bus, Y on c_1 inside the first 'fields' block")
bare = QuantumCircuit(2, 2); bare.h([0, 1])
bare.rzz(0.9, 0, 1); bare.rz(0.6, 0); bare.rz(1.0, 1); bare.rx(0.45, 0); bare.rx(0.95, 1)
bare.rzz(math.pi/2, 0, 1); bare.rz(0.5, 0); bare.rz(0.9, 1); bare.rx(math.pi/2, 0); bare.rx(math.pi/2, 1)
bare.measure([0, 1], [0, 1])
enc = iceberg.encode_bus(bare)
lab = [l for l, n in enc.metadata["iceberg_blocks"] for _ in range(n)]
after = next(i for i, g in enumerate(enc.data) if lab[i] == "fields" and g.operation.name == "cx"
             and [enc.find_bit(q).index for q in g.qubits] == [5, 4])
run(bare, after, ("cx", [5, 4]), 0b11)

print("B. k=1, no leading H layer, default encode_bus, Y on c_1 inside state preparation")
bare = QuantumCircuit(1, 1); bare.rx(0.8, 0); bare.measure(0, 0)
run(bare, 2, ("cx", [2, 5]), 0b1)
