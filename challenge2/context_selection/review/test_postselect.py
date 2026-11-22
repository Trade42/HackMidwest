"""postselect(): layout inference, key formats, Aer counts."""
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
import iceberg, qsel
from qsel import Angles
from vlib import *

sim = AerSimulator(method="statevector", seed_simulator=11)
def aer_counts(circ, shots=40000):
    return sim.run(circ, shots=shots).result().get_counts()

def dist_from_counts(counts, width):
    p = np.zeros(1 << width)
    for k, c in counts.items():
        p[int(k.replace(" ", ""), 2)] += c
    return p / p.sum()

print("== Aer-sampled counts through iceberg.postselect vs exact bare distribution (40000 shots; TV noise floor ~ sqrt(2^k/shots)/2)")
for k, ef, seed, p in [(4, 0.5, 1, 1), (5, 0.4, 5, 2), (6, 0.5, 8, 1), (3, 0.7, 9, 2)]:
    q = qsel.random_qubo(k, edge_fraction=ef, seed=seed)
    bare = qsel.qaoa_circuit(q, qsel.tune_angles(q, p=p).angles)
    pb = exact_dist(bare)
    bare_aer = dist_from_counts(aer_counts(bare), k)
    for name, enc in (("direct", iceberg.encode_direct(bare)), ("bus", iceberg.encode_bus(bare)), ("direct[x]", iceberg.encode_direct(bare, checks="x")), ("direct[z]", iceberg.encode_direct(bare, checks="z"))):
        cnt = aer_counts(enc)
        lc, st = iceberg.postselect(cnt, enc)
        mine, shots, kept = my_postselect_counts(cnt, iceberg.layout_of(enc))
        print(f"  k={k} p={p} {name:9s}: kept {st.kept}/{st.shots}  TV(pkg, bare exact)={tv(dist_from_counts(lc, k), pb):.4f}  (bare Aer sample TV={tv(bare_aer, pb):.4f})  pkg==mine: {lc == mine and kept == st.kept}  key width {len(next(iter(lc)))}")

print("== literal keep rule (parity AND ancilla AND flags) applied to checks='x'")
q = qsel.random_qubo(4, edge_fraction=0.5, seed=1)
bare = qsel.qaoa_circuit(q, qsel.tune_angles(q, p=1).angles)
enc = iceberg.encode_direct(bare, checks="x")
lg, kept = my_postselect_dist(exact_dist(enc), iceberg.layout_of(enc), force_parity=True)
print(f"  kept with parity enforced: {kept:.6f}; decoded/kept vs bare max diff {np.abs(lg / kept - exact_dist(bare)).max():.2e}; (package rule, no parity) kept 1.0")

print("== postselect(counts) with the layout omitted")
for k in (2, 3, 4, 5, 6):
    q = qsel.random_qubo(k, edge_fraction=0.5, seed=k)
    bare = qsel.qaoa_circuit(q, Angles((0.6,), (0.5,)))
    pb = exact_dist(bare)
    for name, enc in (("encode() default", iceberg.encode(bare)), ("encode_direct", iceberg.encode_direct(bare)), ("direct[z]", iceberg.encode_direct(bare, checks="z"))):
        cnt = sample_counts(exact_dist(enc), 50000, seed=3)
        ref, st_ref = iceberg.postselect(cnt, enc)
        try:
            lc, st = iceberg.postselect(cnt)
            w = len(next(iter(lc))) if lc else 0
            if w == k:
                msg = f"kept {st.kept_share:.3f}, TV vs bare {tv(dist_from_counts(lc, k), pb):.3f}"
            else:
                msg = f"kept {st.kept_share:.3f}, WRONG KEY WIDTH {w} (bare has {k})"
            print(f"  k={k} {name:17s} width={enc.num_clbits:2d}: no error -> {msg}   [with layout: kept {st_ref.kept_share:.3f}]")
        except Exception as e:
            print(f"  k={k} {name:17s} width={enc.num_clbits:2d}: {type(e).__name__}: {str(e)[:90]}")

print("== key formats")
enc = iceberg.encode_direct(bare)
cnt = sample_counts(exact_dist(enc), 2000, seed=1)
for label, conv in (("hex keys", lambda k: hex(int(k, 2))), ("int keys", lambda k: int(k, 2)), ("spaces", lambda k: k[:3] + " " + k[3:]), ("too short (code bits only)", lambda k: k[-8:])):
    try:
        c2 = {}
        for k_, v in cnt.items():
            kk = conv(k_)
            c2[kk] = c2.get(kk, 0) + v
        lc, st = iceberg.postselect(c2, enc)
        print(f"  {label}: no error, kept {st.kept}/{st.shots}")
    except Exception as e:
        print(f"  {label}: {type(e).__name__}: {str(e)[:100]}")
# bare circuit with two classical registers: key format of bare counts vs postselect output
qa = QuantumRegister(4, "q"); ca, cb = ClassicalRegister(2, "ca"), ClassicalRegister(2, "cb")
b2 = QuantumCircuit(qa, ca, cb); b2.h(qa); b2.rzz(0.4, 0, 1); b2.rx(0.5, 2); b2.measure(qa[:2], ca); b2.measure(qa[2:], cb)
e2 = iceberg.encode_bus(b2)
print("  bare key example:", repr(next(iter(aer_counts(b2, 10)))), " postselect key example:", repr(next(iter(iceberg.postselect(aer_counts(e2, 10), e2)[0]))))
print("  empty counts:", iceberg.postselect({}, e2))
