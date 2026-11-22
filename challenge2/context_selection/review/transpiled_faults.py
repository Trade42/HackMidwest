"""Bonus for claim 4: single-fault enumeration (exact simulation only) on the *transpiled* bus circuit."""
import itertools, time
from collections import Counter
import numpy as np
from qiskit import transpile
from qiskit.transpiler import CouplingMap
import iceberg, qsel
from qsel import Angles
from vlib import *

TWO = ["".join(p) for p in itertools.product("IXYZ", repeat=2) if "".join(p) != "II"]

def enumerate_sim(circ, lay, seg_labels):
    n = circ.num_qubits
    # ops with segment index
    ops, meas, seg = [], [], 0
    for inst in circ.data:
        nm = inst.operation.name
        qs = [circ.find_bit(x).index for x in inst.qubits]
        if nm == "barrier":
            if len(qs) == n:
                seg += 1
            continue
        if nm == "measure":
            meas.append((0, qs[0], circ.find_bit(inst.clbits[0]).index)); continue
        ops.append((seg, nm, qs, gate_matrix(nm, [float(p) for p in inst.operation.params])))
    states = [zero_state(n)]
    for _, nm, qs, m in ops:
        states.append(apply(states[-1], m, qs, n))
    def dist(psi):
        return my_postselect_dist(clbit_dist(qubit_probs(psi), meas, n, circ.num_clbits), lay)
    ideal, kept = dist(states[-1])
    assert abs(kept - 1) < 1e-9
    res = Counter(); viol = []
    nf = 0
    for li, (sg, nm, qs, m) in enumerate(ops):
        for p in (["X", "Y", "Z"] if len(qs) == 1 else TWO):
            psi = states[li + 1]
            for ch, q in zip(p, qs):
                if ch != "I":
                    psi = apply(psi, PAULI[ch], [q], n)
            for _, nm2, q2, m2 in ops[li + 1:]:
                psi = apply(psi, m2, q2, n)
            lg, kept = dist(psi)
            nf += 1
            lab = seg_labels[sg]
            if kept < 1e-9:
                res[(lab, "discarded")] += 1
            else:
                harmful = tv(lg / kept, ideal) > 1e-7
                tag = ("harmful" if harmful else "harmless") + ("" if abs(kept - 1) < 1e-7 else "-partial")
                res[(lab, tag)] += 1
                if harmful and lab in ("init", "syndrome"):
                    viol.append((lab, li, nm, qs, p, round(kept, 4), round(tv(lg / kept, ideal), 4)))
    return nf, res, viol

rng = np.random.default_rng(3)
for ef, seed, p, mode in [(0.5, 1, 1, "direct"), (1.0, 2, 1, "bus"), (0.67, 3, 2, "direct")]:
    q = qsel.random_qubo(4, edge_fraction=ef, seed=seed)
    bare = qsel.qaoa_circuit(q, Angles(tuple(rng.uniform(0.3, 1.1, p)), tuple(rng.uniform(0.3, 1.1, p))))
    enc = iceberg.encode_bus(bare, edge_mode=mode)
    lay = iceberg.layout_of(enc)
    # segment labels from the encoded circuit's own labels (first non-barrier label of each full-width-barrier segment)
    lab = block_labels(enc)
    seg_labels, cur = [], None
    for inst, l in zip(enc.data, lab):
        if inst.operation.name == "barrier" and len(inst.qubits) == enc.num_qubits:
            seg_labels.append(cur); cur = None
        elif cur is None:
            cur = l
    seg_labels.append(cur)
    t0 = time.time()
    tq = transpile(enc, basis_gates=["cz", "rz", "sx", "x"], coupling_map=CouplingMap.from_grid(3, 3), optimization_level=3, seed_transpiler=0)
    nf, res, viol = enumerate_sim(tq, lay, seg_labels)
    print(f"k=4 ef={ef} p={p} mode={mode}: transpiled gates={sum(tq.count_ops().values())} faults={nf} segments={seg_labels} ({time.time()-t0:.0f}s)")
    for l in dict.fromkeys(seg_labels):
        print("    ", l, {k[1]: v for k, v in res.items() if k[0] == l})
    print("     undetected harmful in init/syndrome:", len(viol), viol[:6])
