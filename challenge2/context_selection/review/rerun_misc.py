import math, numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import iceberg
from vlib import *

def t(label, fn):
    try:
        r = fn()
        print(f"  {label}: no error -> {r}")
    except Exception as e:
        print(f"  {label}: {type(e).__name__}: {str(e)[:110]}")

def base(k=3, clbits=None):
    qc = QuantumCircuit(k, k if clbits is None else clbits); qc.h(range(k))
    for i in range(k - 1): qc.rzz(0.4 + 0.1 * i, i, i + 1)
    for i in range(k): qc.rz(0.3 + 0.2 * i, i)
    for i in range(k): qc.rx(0.5 + 0.15 * i, i)
    return qc
b = base(4); b.measure(range(4), range(4))
enc = iceberg.encode_bus(b)
pe = exact_dist(enc)
cnt = sample_counts(pe, 2000, seed=1)
print("== postselect input validation")
t("layout omitted", lambda: iceberg.postselect(cnt)[1])
t("keys too short (8 of 9 bits)", lambda: iceberg.postselect({k[1:]: v for k, v in cnt.items()}, enc)[1])
t("keys too long", lambda: iceberg.postselect({"0" + k: v for k, v in cnt.items()}, enc)[1])
t("hex keys", lambda: iceberg.postselect({hex(int(k, 2)): v for k, v in cnt.items()}, enc)[1])
t("int keys", lambda: iceberg.postselect({int(k, 2): v for k, v in cnt.items()}, enc)[1])
t("keys with a space", lambda: iceberg.postselect({k[:4] + " " + k[4:]: v for k, v in cnt.items()}, enc)[1].kept)
t("direct counts with bus layout (8 vs 9 bits)", lambda: iceberg.postselect(sample_counts(exact_dist(iceberg.encode_direct(b)), 100, 1), enc)[1])
t("empty counts", lambda: iceberg.postselect({}, enc))
print("== measurement maps / degenerate input")
def same_clbit():
    qc = base(3); qc.measure(0, 0); qc.measure(1, 0); qc.measure(2, 2); return qc
def twice():
    qc = base(3, clbits=4); qc.measure(0, 0); qc.measure(0, 3); qc.measure(1, 1); qc.measure(2, 2); return qc
for name, f in (("encode_direct", iceberg.encode_direct), ("encode_bus", iceberg.encode_bus)):
    t(f"{name}: two qubits into one clbit", lambda: f(same_clbit()).num_qubits)
    def one_two():
        bare = twice(); e = f(bare)
        lg, kept = my_postselect_dist(exact_dist(e), iceberg.layout_of(e))
        return f"encoded; max |decoded - bare| = {np.abs(lg - exact_dist(bare)).max():.3f}"
    t(f"{name}: one qubit into two clbits", one_two)
    t(f"{name}: zero-qubit circuit", lambda: f(QuantumCircuit(0, 0)).num_qubits)
print("== options")
t('encode(style="bus", sx_flag=False)', lambda: iceberg.encode(b, style="bus", sx_flag=False).num_qubits)
t('encode(style="direct", precopy=False)', lambda: iceberg.encode(b, style="direct", precopy=False).num_qubits)
t('encode(style="direct", edge_mode="bus")', lambda: iceberg.encode(b, style="direct", edge_mode="bus").num_qubits)
t('encode(style="direct", seed=3)', lambda: iceberg.encode(b, style="direct", seed=3).num_qubits)
t('encode(style="bus", checks="z")', lambda: iceberg.encode(b, style="bus", checks="z").num_qubits)
for pc in (True, 2, 1, False, 0, 3, -1, "yes"):
    def f(pc=pc):
        e = iceberg.encode_bus(b, precopy=pc)
        lab = [l for l, n in e.metadata["iceberg_blocks"] for _ in range(n)]
        syn = [i for i, inst in enumerate(e.data) if lab[i] == "syndrome" and inst.operation.name == "cx"]
        h = next(i for i, inst in enumerate(e.data) if lab[i] == "syndrome" and inst.operation.name == "h")
        return f"{sum(1 for i in syn if i < h)} witness CX before the H"
    t(f"encode_bus(precopy={pc!r})", f)
