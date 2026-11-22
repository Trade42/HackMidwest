"""(c) hunt: any single fault (gate faults, X/Y/Z before the first gate, X/Y/Z idle at every block
boundary, readout flips) that passes all checks while leaving a bus qubit flipped at the end of its
block, leaving a non-logical residual, or acting on more than 2 logical qubits.  Tracker only."""
import itertools, sys, time
from collections import Counter, defaultdict
import numpy as np
from qiskit import QuantumCircuit
import iceberg
from faultlib import describe, qname
from tracker_large import TrackerOnly
from rerun_lib import all_faults

rng = np.random.default_rng(424242)
def qaoa_like(k, edges, p, h=True):
    qc = QuantumCircuit(k, k)
    if h: qc.h(range(k))
    for _ in range(p):
        for i, j in edges:
            qc.rzz(rng.uniform(0.3, 1.3), i, j)
        for i in range(k):
            qc.rz(rng.uniform(0.3, 1.3), i)
        for i in range(k):
            qc.rx(rng.uniform(0.3, 1.3), i)
    qc.measure(range(k), range(k))
    return qc

def ragged(k, layers, h=True):
    """Random subsets: fields / mixers on some qubits only, pair-only or field-only blocks."""
    qc = QuantumCircuit(k, k)
    if h: qc.h(range(k))
    for _ in range(layers):
        if k >= 2 and rng.random() < 0.8:
            for i, j in itertools.combinations(range(k), 2):
                if rng.random() < 0.4:
                    qc.rzz(rng.uniform(0.3, 1.3), i, j)
        if rng.random() < 0.8:
            for i in range(k):
                if rng.random() < 0.6:
                    qc.rz(rng.uniform(0.3, 1.3), i)
        sub = [i for i in range(k) if rng.random() < 0.6] or [int(rng.integers(k))]
        for i in sub:
            qc.rx(rng.uniform(0.3, 1.3), i)
    qc.measure(range(k), range(k))
    return qc

t0 = time.time()
stats = defaultdict(Counter)
found = []
def hunt(tag, enc, key):
    an = TrackerOnly(enc)
    faults = all_faults(an)
    s = stats[key]
    s["circuits"] += 1; s["faults"] += len(faults)
    for f in faults:
        if f.kept < 0.5:
            continue
        s["kept"] += 1
        harmful = f.status == "harmful"
        s["harmful"] += harmful
        in_logic = f.block not in ("syndrome", "measure")
        bus = bool(f.blk_anc_x) and in_logic
        nonlog = (not f.blk_clean) and in_logic
        wide = harmful and f.footprint > 2
        part = harmful and f.block in ("init", "syndrome", "measure")
        s["max_w"] = max(s["max_w"], f.footprint if harmful else 0)
        if bus or nonlog or wide or part:
            s["offenders"] += 1
            s["off_bus"] += bus; s["off_wide"] += wide; s["off_part"] += part; s["off_harmful"] += harmful
            if len(found) < 4000:
                found.append((key, tag, enc, f, bus, nonlog, wide, part))

def edgesets(k):
    full = list(itertools.combinations(range(k), 2))
    sparse = [e for e in full if rng.random() < 0.4] if k > 2 else full
    return {"complete": full, "sparse": sparse}

for k in (1, 2, 3, 4, 5, 6, 8, 10):
    K = k + k % 2
    for p in (1, 2, 3):
        if k == 10 and p == 3:
            continue
        for gname, edges in edgesets(k).items():
            if k == 10 and gname == "complete" and p > 1:
                continue
            bare = qaoa_like(k, edges, p)
            for mode in ("direct", "bus"):
                for pl in ([None, [int(x) for x in rng.permutation(K)]] if k <= 6 else [None]):
                    hunt(f"qaoa k={k} p={p} {gname} mode={mode} pl={pl}", iceberg.encode_bus(bare, edge_mode=mode, placement=pl), (k, p))
    # ragged circuits and a start without the H layer
    for trial in range(3 if k <= 6 else 1):
        layers = int(rng.integers(1, 4))
        bare = ragged(k, layers, h=(trial != 2))
        mode = "direct" if trial % 2 == 0 else "bus"
        hunt(f"ragged k={k} layers={layers} h={trial != 2} mode={mode}", iceberg.encode_bus(bare, edge_mode=mode, placement=[int(x) for x in rng.permutation(K)]), (k, "ragged"))
    print(f"k={k} done, {time.time()-t0:.0f}s", flush=True)

tot = sum(s["faults"] for s in stats.values())
print(f"total faults {tot} over {sum(s['circuits'] for s in stats.values())} circuits")
print("k    p        circuits  faults   kept  harmful  maxw  offenders (bus-flipped / >2 logical / in init-syndrome-measure / harmful)")
for key in sorted(stats, key=str):
    s = stats[key]
    print(f"{key[0]:<4} {str(key[1]):8s} {s['circuits']:8d} {s['faults']:7d} {s['kept']:6d} {s['harmful']:7d} {s['max_w']:5d}  {s['offenders']} ({s['off_bus']} / {s['off_wide']} / {s['off_part']} / {s['off_harmful']})")
seen = set()
for key, tag, enc, f, bus, nonlog, wide, part in found:
    sig = (key, f.block, f.gate, f.pauli, tuple(qname(enc, q) for q in f.qubits))
    if sig in seen:
        continue
    seen.add(sig)
    if len(seen) <= 14:
        print("  ", key, tag[:48], "|", f.kind, describe(enc, f).split(": kept")[0], f"status={f.status} bus_x={[qname(enc, q) for q in range(an_n) if f.blk_anc_x >> q & 1] if (an_n := enc.num_qubits) else None} footprint={f.footprint}")
print("distinct offender signatures:", len(seen))
