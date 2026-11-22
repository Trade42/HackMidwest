"""Claim 2: stabilizers +1 and ancillas |0> just before the final check block."""
import itertools, math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli
import iceberg, qsel
from qsel import Angles
from vlib import *

def strip_to_before_syndrome(enc):
    """Strip syndrome+measure instructions myself, using the block labels."""
    lab = block_labels(enc)
    qc = QuantumCircuit(enc.num_qubits)
    for inst, l in zip(enc.data, lab):
        if l in ("syndrome", "measure") or inst.operation.name == "barrier":
            continue
        qc.append(inst.operation, [enc.find_bit(q).index for q in inst.qubits])
    return qc

def check(tag, bare, enc_nr, enc_full, expect_sx=True, expect_sz=True):
    lay = iceberg.layout_of(enc_nr)
    assert not any(i.operation.name == "measure" for i in enc_nr.data), "readout=False still measures"
    n = enc_nr.num_qubits
    res = {}
    for label, circ in (("readout=False", enc_nr.remove_final_measurements(inplace=False) if enc_nr.num_clbits else enc_nr),
                        ("stripped", strip_to_before_syndrome(enc_full))):
        qc = QuantumCircuit(n)
        for inst in circ.data:
            if inst.operation.name != "barrier":
                qc.append(inst.operation, [circ.find_bit(q).index for q in inst.qubits])
        sv = Statevector(qc)
        code = list(lay.code_qubits)
        def pauli(ch):
            s = ["I"] * n
            for j in code:
                s[n - 1 - j] = ch
            return Pauli("".join(s))
        sx = float(np.real(sv.expectation_value(pauli("X"))))
        sz = float(np.real(sv.expectation_value(pauli("Z"))))
        anc = [j for j in range(n) if j not in code]
        probs = np.asarray(sv.probabilities())
        idx = np.arange(1 << n)
        mask = sum(1 << j for j in anc)
        p_anc0 = float(probs[(idx & mask) == 0].sum())
        # isometric image of the bare state
        ops, _ = parse(bare)
        psi_b = run_ops(ops, bare.num_qubits).reshape(-1)
        k, K = lay.k, lay.k_code
        if K > k:   # padded logical qubit: |+> if the bare circuit starts with H, else |0>
            pad = np.array([1, 1]) / math.sqrt(2) if lay.init == "plus" else np.array([1.0, 0.0])
            psi_b = np.kron(pad, psi_b)
        target = np.zeros(1 << n, dtype=complex)
        for x in range(1 << K):
            par = bin(x).count("1") & 1
            i0 = x | (par << lay.top)
            allmask = (1 << (K + 2)) - 1
            i1 = i0 ^ allmask
            target[i0] += psi_b[x] / math.sqrt(2)
            target[i1] += psi_b[x] / math.sqrt(2)
        fid = abs(np.vdot(target, np.asarray(sv.data))) ** 2
        res[label] = (sx, sz, p_anc0, fid)
    return res

rows = []
cases = []
for k, ef, seed, p in [(2, 1.0, 0, 1), (3, 0.5, 2, 1), (4, 0.5, 1, 1), (4, 0.5, 4, 2), (5, 0.4, 5, 2), (6, 0.5, 8, 2), (6, 1.0, 10, 1), (4, 0.0, 14, 1)]:
    q = qsel.random_qubo(k, edge_fraction=ef, seed=seed)
    ang = qsel.tune_angles(q, p=p).angles
    cases.append((f"k={k} ef={ef} p={p}", qsel.qaoa_circuit(q, ang)))
q = qsel.random_qubo(4, edge_fraction=0.5, seed=1)
cases.append(("k=4 zero angles", qsel.qaoa_circuit(q, Angles((0.0,), (0.0,)))))

worst = {"sx": 1.0, "sz": 1.0, "anc": 1.0, "fid": 1.0}
bad = []
for tag, bare in cases:
    encs = {
        "direct[both]": (lambda ro: iceberg.encode_direct(bare, readout=ro)),
        "direct[z]": (lambda ro: iceberg.encode_direct(bare, checks="z", readout=ro)),
        "direct[x]": (lambda ro: iceberg.encode_direct(bare, checks="x", readout=ro)),
        "bus[direct]": (lambda ro: iceberg.encode_bus(bare, readout=ro)),
        "bus[bus]": (lambda ro: iceberg.encode_bus(bare, edge_mode="bus", readout=ro)),
    }
    for name, mk in encs.items():
        r = check(tag, bare, mk(False), mk(None))
        for label, (sx, sz, anc, fid) in r.items():
            exp_sz = 0.0 if name == "direct[x]" else 1.0    # product start: S_Z undefined
            ok = abs(sx - 1) < 1e-9 and abs(sz - exp_sz) < 1e-9 and abs(anc - 1) < 1e-9
            if name != "direct[x]":
                ok = ok and abs(fid - 1) < 1e-9
            rows.append((tag, name, label, sx, sz, anc, fid, ok))
            if not ok:
                bad.append(rows[-1])
print(f"{len(rows)} states checked")
for r in rows:
    if r[1] == "direct[x]" and r[0].startswith("k=4 ef=0.5 p=1"):
        print("direct[x] example:", r)
sel = [r for r in rows if r[1] != "direct[x]"]
print("non-x: min <S_X> %.12f  min <S_Z> %.12f  min P(anc=0) %.12f  min fidelity-to-encoded-bare %.12f" % (
    min(r[3] for r in sel), min(r[4] for r in sel), min(r[5] for r in sel), min(r[6] for r in sel)))
selx = [r for r in rows if r[1] == "direct[x]"]
print("checks=x: <S_X> range", min(r[3] for r in selx), max(r[3] for r in selx), " <S_Z> range", min(r[4] for r in selx), max(r[4] for r in selx),
      " fid range", min(r[6] for r in selx), max(r[6] for r in selx))
print("BAD:", len(bad))
for b in bad: print("  ", b)
