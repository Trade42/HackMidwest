"""Claim 1: noiseless logical equivalence, both encoders, all option combos."""
import itertools, math, sys, time
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import iceberg, qsel
from qsel import Angles
from vlib import *

TOL = 1e-9
rows = []
fails = []

def check(tag, bare, enc_name, enc):
    lay = iceberg.layout_of(enc)
    pb = exact_dist(bare)
    pe = exact_dist(enc)
    logical, kept = my_postselect_dist(pe, lay)
    d = float(np.abs(logical - pb).max())
    # package postselect on exact-proportional integer counts
    scale = 10**12
    width = enc.num_clbits
    counts = {format(i, f"0{width}b"): int(round(p * scale)) for i, p in enumerate(pe) if round(p * scale) > 0}
    lc, st = iceberg.postselect(counts, enc)
    tot = sum(counts.values())
    pk = np.zeros(1 << bare.num_clbits)
    for key, c in lc.items():
        pk[int(key.replace(" ", ""), 2)] += c / tot
    d_pkg = float(np.abs(pk - pb).max())
    kept_pkg = st.kept / st.shots
    # sampled counts: package vs my own counts post-selection
    sc = sample_counts(pe, 20000, seed=7)
    lc2, st2 = iceberg.postselect(sc, enc)
    mine, shots, mykept = my_postselect_counts(sc, lay)
    same = (lc2 == mine) and st2.kept == mykept and st2.shots == shots
    ok = d < TOL and abs(kept - 1) < TOL and d_pkg < 1e-8 and abs(kept_pkg - 1) < 1e-9 and same
    rows.append((tag, enc_name, enc.num_qubits, d, kept, d_pkg, kept_pkg, same, ok))
    if not ok:
        fails.append(rows[-1])

def variants(bare):
    out = []
    for checks in ("both", "z", "x"):
        out.append((f"direct[{checks}]", iceberg.encode_direct(bare, checks=checks)))
    out.append(("direct[noflags]", iceberg.encode_direct(bare, init_flag=False, sx_flag=False)))
    out.append(("direct[init_flag=F]", iceberg.encode_direct(bare, init_flag=False)))
    out.append(("direct[sx_flag=F]", iceberg.encode_direct(bare, sx_flag=False)))
    out.append(("direct[nobarriers]", iceberg.encode_direct(bare, barriers=False)))
    for em in ("direct", "bus"):
        out.append((f"bus[{em}]", iceberg.encode_bus(bare, edge_mode=em)))
    out.append(("bus[precopy=F]", iceberg.encode_bus(bare, precopy=False)))
    out.append(("bus[init_flag=F]", iceberg.encode_bus(bare, init_flag=False)))
    out.append(("bus[nobarriers]", iceberg.encode_bus(bare, barriers=False)))
    out.append(("encode()default", iceberg.encode(bare)))
    return out

cases = []
t0 = time.time()
# random instances, p=1 and p=2
for k, ef, seed, p in [(2, 1.0, 0, 1), (2, 0.0, 1, 1), (3, 0.5, 2, 1), (4, 0.3, 3, 1), (4, 0.5, 1, 1), (4, 0.5, 4, 2),
                       (5, 0.4, 5, 1), (5, 0.4, 6, 2), (6, 0.3, 7, 1), (6, 0.5, 8, 2), (3, 0.7, 9, 2),
                       (6, 1.0, 10, 1), (4, 1.0, 11, 2), (6, 0.0, 12, 1), (5, 1.0, 13, 1), (4, 0.0, 14, 2)]:
    q = qsel.random_qubo(k, edge_fraction=ef, seed=seed)
    ang = qsel.tune_angles(q, p=p).angles
    cases.append((f"k={k} ef={ef} seed={seed} p={p} E={q.num_edges}", qsel.qaoa_circuit(q, ang)))
# random (untuned) angles so nothing is accidentally special
rng = np.random.default_rng(123)
for k, ef, seed, p in [(4, 0.6, 21, 2), (6, 0.6, 22, 2), (5, 0.8, 23, 3), (2, 1.0, 24, 3)]:
    q = qsel.random_qubo(k, edge_fraction=ef, seed=seed)
    ang = Angles(tuple(rng.uniform(-2, 2, p)), tuple(rng.uniform(-2, 2, p)))
    cases.append((f"k={k} ef={ef} seed={seed} p={p} randangles E={q.num_edges}", qsel.qaoa_circuit(q, ang)))
# zero angles
q = qsel.random_qubo(4, edge_fraction=0.5, seed=1)
cases.append(("k=4 zero angles", qsel.qaoa_circuit(q, Angles((0.0,), (0.0,)))))
cases.append(("k=4 gamma=0 only", qsel.qaoa_circuit(q, Angles((0.0,), (0.7,)))))
cases.append(("k=4 beta=0 only", qsel.qaoa_circuit(q, Angles((0.9,), (0.0,)))))
# scrambled measurement map
def scrambled(k, perm, seed, extra_clbits=0):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(QuantumRegister(k, "q"), ClassicalRegister(k + extra_clbits, "c"))
    qc.h(range(k))
    for i, j in itertools.combinations(range(k), 2):
        if rng.random() < 0.6:
            qc.rzz(rng.uniform(-2, 2), i, j)
    for i in range(k):
        qc.rz(rng.uniform(-2, 2), i)
    for i in range(k):
        qc.rx(rng.uniform(-2, 2), i)
    for i in range(k):
        qc.measure(i, perm[i])
    return qc
cases.append(("k=4 scrambled meas [2,0,3,1]", scrambled(4, [2, 0, 3, 1], 31)))
cases.append(("k=5 scrambled meas [4,2,0,1,3]", scrambled(5, [4, 2, 0, 1, 3], 32)))
cases.append(("k=3 scrambled into 5 clbits [4,0,2]", scrambled(3, [4, 0, 2], 33, extra_clbits=2)))
cases.append(("k=6 scrambled reversed", scrambled(6, [5, 4, 3, 2, 1, 0], 34)))
# hand-built: interleaved rz/rzz/rx with different rx angles per qubit and rz before rzz
def handmade(k, seed, layers=2):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(k, k)
    qc.h(range(k))
    for _ in range(layers):
        for _ in range(2 * k):
            if rng.random() < 0.5:
                qc.rz(rng.uniform(-3, 3), int(rng.integers(k)))
            else:
                i, j = rng.choice(k, 2, replace=False)
                qc.rzz(rng.uniform(-3, 3), int(i), int(j))
        for i in rng.permutation(k):
            if rng.random() < 0.8:
                qc.rx(rng.uniform(-3, 3), int(i))
    qc.measure(range(k), range(k))
    return qc
if True:
    cases.append(("k=4 handmade interleaved (repeated edges, partial rx)", handmade(4, 41)))
    cases.append(("k=6 handmade interleaved", handmade(6, 42)))
    cases.append(("k=5 handmade interleaved 3 layers", handmade(5, 43, 3)))
    cases.append(("k=2 handmade", handmade(2, 44)))

# simulator cross-check against qiskit Statevector on a subset
xmax = 0.0
for tag, bare in cases[:6] + cases[-6:]:
    xmax = max(xmax, np.abs(exact_dist(bare) - exact_dist_qiskit(bare)).max())
    for name, enc in variants(bare)[:9:4]:
        xmax = max(xmax, np.abs(exact_dist(enc) - exact_dist_qiskit(enc)).max())
print(f"own simulator vs qiskit Statevector, max abs prob diff: {xmax:.2e}")

for tag, bare in cases:
    for name, enc in variants(bare):
        check(tag, bare, name, enc)

print(f"{len(cases)} bare circuits x {len(rows)//len(cases)} encodings = {len(rows)} checks in {time.time()-t0:.1f}s")
print("max |logical-bare| (mine):", max(r[3] for r in rows), " min kept:", min(r[4] for r in rows))
print("max |logical-bare| (package postselect on proportional counts):", max(r[5] for r in rows), " min kept:", min(r[6] for r in rows))
print("sampled-counts package == mine for all:", all(r[7] for r in rows))
print("max qubits simulated:", max(r[2] for r in rows))
print("FAILS:", len(fails))
for f in fails:
    print("  ", f)
