"""Claim 3: exhaustive single-fault enumeration, k=4."""
import sys, time, pickle
from collections import Counter
import numpy as np
import iceberg, qsel
from qsel import Angles
from vlib import *
from faultlib import *

def labels_sanity(enc):
    """init = everything before the first logical rotation and contains only h/cx;
    all rotations are in cost/pairs/fields/mixer; syndrome+measure are a suffix without rotations."""
    lab = block_labels(enc)
    names = [i.operation.name for i in enc.data]
    assert len(lab) == len(names)
    rot_idx = [i for i, nm in enumerate(names) if nm in ROT]
    init_idx = [i for i, l in enumerate(lab) if l == "init"]
    notes = []
    if init_idx != list(range(len(init_idx))):
        notes.append("init is not a prefix")
    if any(names[i] in ROT for i in init_idx):
        notes.append("rotation inside init")
    if not all(names[i] in ("h", "cx", "barrier") for i in init_idx):
        notes.append("unexpected gate in init")
    bad = [i for i in rot_idx if lab[i] not in ("cost", "pairs", "fields", "mixer")]
    if bad:
        notes.append(f"rotations with labels {[lab[i] for i in bad]}")
    tail = [i for i, l in enumerate(lab) if l in ("syndrome", "measure")]
    if tail and tail != list(range(tail[0], len(lab))):
        notes.append("syndrome/measure not a suffix")
    if any(names[i] in ROT for i in tail):
        notes.append("rotation in syndrome")
    if any(names[i] == "measure" and lab[i] != "measure" for i in range(len(lab))):
        notes.append("measure outside measure block")
    # which blocks contain x-type rotations vs z-type
    kinds = {}
    for i in rot_idx:
        kinds.setdefault(lab[i], set()).add(names[i])
    return notes, kinds, len(init_idx), (tail[0] if tail else None)

def init_state_check(enc):
    """State right after the init block is the encoded |+>^K: S_X, S_Z, X_t X_di all +1, ancillas |0>."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, Pauli
    lab = block_labels(enc)
    n = enc.num_qubits
    lay = iceberg.layout_of(enc)
    qc = QuantumCircuit(n)
    for inst, l in zip(enc.data, lab):
        if l == "init" and inst.operation.name != "barrier":
            qc.append(inst.operation, [enc.find_bit(q).index for q in inst.qubits])
    sv = Statevector(qc)
    def P(xq=(), zq=()):
        s = ["I"] * n
        for q in xq: s[n - 1 - q] = "X"
        for q in zq: s[n - 1 - q] = "Z"
        return Pauli("".join(s))
    vals = [sv.expectation_value(P(xq=lay.code_qubits)).real, sv.expectation_value(P(zq=lay.code_qubits)).real]
    vals += [sv.expectation_value(P(xq=(lay.top, i))).real for i in range(lay.k_code)]
    anc = [q for q in range(n) if q not in lay.code_qubits]
    vals += [sv.expectation_value(P(zq=(q,))).real for q in anc]
    return min(vals)

def run(title, enc, verbose_harmful=6, check_parts=("init", "syndrome", "measure")):
    t0 = time.time()
    notes, kinds, n_init, tail0 = labels_sanity(enc)
    an = Analysis(enc)
    faults = an.enumerate()
    c = summarize(enc, faults, title)
    print(f"   label sanity: {notes or 'ok'}; rotation kinds per block: { {k: sorted(v) for k, v in kinds.items()} }; "
          f"min stabilizer/logical-X/ancilla-Z expectation after init: {init_state_check(enc):.6f}; model_ok={an.model_ok} {an.model_notes[:3]}")
    kept_vals = sorted(set(round(f.kept, 6) for f in faults))
    print(f"   distinct kept masses: {kept_vals[:8]}{'...' if len(kept_vals) > 8 else ''}")
    bad_pred = [f for f in faults if not f.pred_ok]
    for f in bad_pred[:5]:
        print("   TRACKER MISMATCH:", describe(enc, f), "pred_kept", f.pred_kept)
    harmful = [f for f in faults if "harmful" in f.status]
    part = [f for f in harmful if f.block in check_parts]
    print(f"   undetected harmful: total {len(harmful)}; in init/syndrome/measure: {len(part)}")
    for f in part[:verbose_harmful]:
        print("      ", describe(enc, f))
    meas = [f for f in faults if f.kind == "meas"]
    print(f"   readout flips: {len(meas)} ; all discarded: {all(f.status == 'discarded' for f in meas)}")
    # logical weight of undetected harmful faults
    wc = Counter(f.lweight for f in harmful)
    ancx = [f for f in harmful if f.anc_x]
    print(f"   logical weight (skeleton residual) of undetected harmful faults: {dict(sorted(wc.items()))}; with X left on ancilla: {len(ancx)}")
    ps = Counter(bin(f.pure_shift).count('1') for f in harmful if f.pure_shift >= 0)
    print(f"   harmful faults whose kept dist is an exact XOR shift of the ideal: weights {dict(sorted(ps.items()))} (of {len(harmful)})")
    lay = iceberg.layout_of(enc)
    touch_b = Counter()
    for f in harmful:
        if lay.bottom in f.qubits:
            touch_b[f.lweight] += 1
    print(f"   ... of which on gates touching b, by weight: {dict(sorted(touch_b.items()))}")
    mw = max((f.lweight for f in harmful), default=0)
    ex = [f for f in harmful if f.lweight == mw][:3]
    for f in ex:
        print("      max-weight example:", describe(enc, f))
    print(f"   time {time.time()-t0:.1f}s")
    return an, faults

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "p1"
    q = qsel.random_qubo(4, edge_fraction=0.5, seed=1)
    if which == "p1":
        ang = qsel.tune_angles(q, p=1).angles
    elif which == "p1r":
        q = qsel.random_qubo(4, edge_fraction=1.0, seed=5)
        ang = Angles((0.731,), (0.413,))
    elif which == "p2":
        q = qsel.random_qubo(4, edge_fraction=0.67, seed=3)
        ang = Angles((0.731, -0.377), (0.413, 0.887))
    bare = qsel.qaoa_circuit(q, ang)
    print(f"instance {which}: k=4 edges={q.edges} angles={ang}")
    res = {}
    res["direct"] = run("encode_direct default", iceberg.encode_direct(bare))
    res["bus"] = run("encode_bus default", iceberg.encode_bus(bare))
    res["bus_busmode"] = run("encode_bus edge_mode=bus", iceberg.encode_bus(bare, edge_mode="bus"))
    res["direct_noinit"] = run("encode_direct init_flag=False", iceberg.encode_direct(bare, init_flag=False))
    res["direct_nosx"] = run("encode_direct sx_flag=False", iceberg.encode_direct(bare, sx_flag=False))
    res["bus_noprecopy"] = run("encode_bus precopy=False", iceberg.encode_bus(bare, precopy=False))
    res["bus_noinit"] = run("encode_bus init_flag=False", iceberg.encode_bus(bare, init_flag=False))
