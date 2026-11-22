"""Same hole on a plain qsel QAOA circuit (k=4, complete graph, p=2), default encode_bus.
The fault is equivalent to running layer-2 pair terms with the wrong sign."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import iceberg, qsel
from qsel import Angles
from faultlib import Analysis, describe, popc

q = qsel.random_qubo(4, edge_fraction=1.0, seed=4)
ang = Angles((0.731, -0.377), (0.413, 0.887))
bare = qsel.qaoa_circuit(q, ang)
enc = iceberg.encode_bus(bare)
lay = iceberg.layout_of(enc)
root = lay.sx_ancilla
labels = [l for l, n in enc.metadata["iceberg_blocks"] for _ in range(n)]
idx = next(i for i, inst in enumerate(enc.data) if labels[i] == "fields" and inst.operation.name == "cx"
           and [enc.find_bit(x).index for x in inst.qubits] == [root, root - 1])
faulty = enc.copy_empty_like()
for i, inst in enumerate(enc.data):
    faulty.append(inst.operation, inst.qubits, inst.clbits)
    if i == idx:
        faulty.y(root)
def dist(circ):
    return Statevector(circ.remove_final_measurements(inplace=False)).probabilities()
def decode(circ):
    p = dist(circ)
    cnt = {format(i, f"0{circ.num_qubits}b"): int(round(x * 10**12)) for i, x in enumerate(p) if round(x * 10**12)}
    lc, st = iceberg.postselect(cnt, enc)
    out = np.zeros(16)
    for k, c in lc.items():
        out[int(k, 2)] = c
    return out / out.sum(), st.kept_share
p_bare = dist(bare)
p_f, kept = decode(faulty)
print(f"fault: Y on bus qubit {root} after enc.data[{idx}] (cx({root},{root-1}), block fields, layer 1); kept share {kept:.6f}; TV(kept, bare) = {0.5*np.abs(p_f-p_bare).sum():.4f}")

# which rotations does my tracker say are sign-flipped?
an = Analysis(enc)
f = next(f for f in an.enumerate(prep=False, meas=False) if f.data_index == idx and f.pauli == "YI")
terms = {t[2]: (t[0], t[1]) for t in enc.metadata["iceberg_terms"]}
flipped = [(an.ops[j][0], terms[an.ops[j][0]]) for j in f.flipped]
print("sign-flipped encoded rotations (enc.data index, kind, logical qubits):", flipped)
# rebuild the bare circuit with those layer-2 pair angles negated and compare
flip_pairs = {tuple(sorted(t[1])) for _, t in flipped if t[0] == "zz"}
mod = QuantumCircuit(4, 4)
seen_rx = 0
for inst in bare.data:
    name = inst.operation.name
    qs = [bare.find_bit(x).index for x in inst.qubits]
    if name == "rx":
        seen_rx += 1
    if name == "rzz" and seen_rx >= 4 and tuple(sorted(qs)) in flip_pairs:      # layer 2 only
        mod.rzz(-inst.operation.params[0], *qs)
    elif name == "measure":
        mod.measure(qs[0], bare.find_bit(inst.clbits[0]).index)
    elif name != "barrier":
        mod.append(inst.operation, qs)
print(f"bare circuit with layer-2 rzz on pairs {sorted(flip_pairs)} negated: max |p_mod - p_faulty_kept| = {np.abs(dist(mod) - p_f).max():.2e}")
print("logical qubits touched by the flipped terms:", sorted({i for p in flip_pairs for i in p}))
