"""Claim 5: adversarial inputs."""
import itertools, math, traceback
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
import iceberg, qsel
from qsel import Angles
from vlib import *

ENC = {
    "direct": lambda c, **kw: iceberg.encode_direct(c, **kw),
    "bus": lambda c, **kw: iceberg.encode_bus(c, **kw),
}

def equiv(bare, enc):
    lay = iceberg.layout_of(enc)
    pb = exact_dist(bare)
    pe = exact_dist(enc)
    lg, kept = my_postselect_dist(pe, lay)
    if len(lg) != len(pb):
        return f"WIDTH MISMATCH bare {len(pb)} vs decoded {len(lg)}; kept={kept:.6f}"
    return f"max|diff|={np.abs(lg - pb).max():.2e} kept={kept:.6f}"

def attempt(title, build, encs=("direct", "bus"), check=True, **kw):
    print(f"* {title}")
    for name in encs:
        try:
            bare = build()
            enc = ENC[name](bare, **kw)
            msg = f"encoded ok ({enc.num_qubits}q, {enc.num_clbits} clbits, measures={sum(1 for i in enc.data if i.operation.name=='measure')})"
            if check and enc.num_clbits:
                msg += " | " + equiv(bare, enc)
            print(f"    {name:6s}: {msg}")
        except Exception as e:
            print(f"    {name:6s}: {type(e).__name__}: {str(e)[:150]}")

def base(k=3, measure=True, h=True, clbits=None):
    qc = QuantumCircuit(k, k if clbits is None else clbits)
    if h:
        qc.h(range(k))
    for i in range(k - 1):
        qc.rzz(0.4 + 0.1 * i, i, i + 1)
    for i in range(k):
        qc.rz(0.3 + 0.2 * i, i)
    for i in range(k):
        qc.rx(0.5 + 0.15 * i, i)
    if measure:
        qc.measure(range(k), range(k))
    return qc

# 1 gates after measurement
def after_meas_same():
    qc = base(3); qc.rx(0.3, 0); return qc
def after_meas_other():
    qc = base(3, measure=False); qc.measure(0, 0); qc.rx(0.3, 1); qc.rzz(0.2, 1, 2); qc.measure(1, 1); qc.measure(2, 2); return qc
def after_meas_remeasure():
    qc = base(3); qc.rx(0.3, 0); qc.measure(0, 0); return qc
attempt("gate on a qubit after its measurement", after_meas_same)
attempt("gates on other qubits after measuring qubit 0 (legal deferred measurement)", after_meas_other)
attempt("measure, gate, measure again on same qubit", after_meas_remeasure)

# 2 partial measurement
def partial():
    qc = base(4, measure=False); qc.measure(1, 1); qc.measure(3, 3); return qc
def partial_small_creg():
    qc = base(4, measure=False, clbits=2); qc.measure(1, 0); qc.measure(3, 1); return qc
attempt("partial measurement (qubits 1,3 of 4 into clbits 1,3)", partial)
attempt("partial measurement into a 2-bit register", partial_small_creg)

# 3 no measurement
attempt("no measurement, readout default", lambda: base(3, measure=False))
attempt("no measurement, readout=True", lambda: base(3, measure=False), check=False, readout=True)
def no_meas_ro():
    b = base(3, measure=False)
    for name in ("direct", "bus"):
        enc = ENC[name](b, readout=True)
        bm = base(3)
        print(f"    {name:6s}: readout=True on unmeasured circuit vs measured twin: {equiv(bm, enc)}")
no_meas_ro()
attempt("no measurement, but circuit has a (unused) 5-bit classical register, readout=True", lambda: base(3, measure=False, clbits=5), check=False, readout=True)

# 4 parameterised
def param():
    g, b = Parameter("g"), Parameter("b")
    qc = QuantumCircuit(3, 3); qc.h(range(3))
    qc.rzz(g, 0, 1); qc.rzz(2 * g, 1, 2)
    for i in range(3):
        qc.rz(g * (i + 1), i)
    qc.rz(0.5 * g, 0)          # second rz on the same qubit: bus adds them up
    for i in range(3):
        qc.rx(b, i)
    qc.measure(range(3), range(3))
    return qc
print("* parameterised angles left unbound")
for name in ("direct", "bus"):
    try:
        bare = param()
        enc = ENC[name](bare)
        vals = {p: v for p, v in zip(sorted(bare.parameters, key=lambda p: p.name), (0.9, 0.37))}
        bound_bare = bare.assign_parameters(vals)
        bound_enc = enc.assign_parameters({p: vals[q] for p in enc.parameters for q in vals if q.name == p.name})
        print(f"    {name:6s}: encoded with {len(enc.parameters)} free parameters; after binding: {equiv(bound_bare, bound_enc)}; metadata kept: {'iceberg' in (bound_enc.metadata or {})}")
    except Exception as e:
        print(f"    {name:6s}: {type(e).__name__}: {str(e)[:200]}")

# 5 k = 1
attempt("k = 1", lambda: base(1))
attempt("k = 1, no gates but H and measure", lambda: (lambda qc: (qc.h(0), qc.measure(0, 0), qc)[-1])(QuantumCircuit(1, 1)))

# 6 clbit collisions
def same_clbit():
    qc = base(3, measure=False); qc.measure(0, 0); qc.measure(1, 0); qc.measure(2, 2); return qc
def twice_diff_clbits():
    qc = base(3, measure=False, clbits=4); qc.measure(0, 0); qc.measure(0, 3); qc.measure(1, 1); qc.measure(2, 2); return qc
attempt("two qubits measured into the same clbit (bare: last write wins)", same_clbit)
attempt("one qubit measured into two clbits", twice_diff_clbits)

# 7 init variants
attempt("no leading H layer (logical |0..0> start)", lambda: base(4, h=False))
attempt("no leading H layer, odd k", lambda: base(3, h=False))
attempt("no leading H, direct checks='z'", lambda: base(4, h=False), encs=("direct",), checks="z")
attempt("no leading H, direct checks='x'", lambda: base(4, h=False), encs=("direct",), checks="x")
def partial_h():
    qc = QuantumCircuit(3, 3); qc.h(0); qc.h(1); qc.rx(0.3, 2); qc.measure(range(3), range(3)); return qc
attempt("H on only some qubits", partial_h)
def h_late():
    qc = QuantumCircuit(3, 3); qc.h(0); qc.h(1); qc.rz(0.3, 0); qc.h(2); qc.rx(0.4, 1); qc.measure(range(3), range(3)); return qc
attempt("H layer interrupted by another gate", h_late)
def leading_barrier():
    qc = QuantumCircuit(3, 3); qc.barrier(); qc.h(range(3)); qc.barrier(); qc.rzz(0.5, 0, 2); qc.rx(0.3, 1); qc.id(0); qc.measure(range(3), range(3)); return qc
attempt("leading barrier, id gate", leading_barrier)
def double_h():
    qc = QuantumCircuit(2, 2); qc.h(range(2)); qc.h(0); qc.measure(range(2), range(2)); return qc
attempt("H twice on a qubit", double_h)

# 8 other gates
def cliffordish():
    qc = QuantumCircuit(4, 4); qc.h(range(4))
    qc.cz(0, 1); qc.s(0); qc.t(1); qc.sdg(2); qc.tdg(3); qc.p(0.7, 2); qc.z(1); qc.rx(0.4, 0); qc.x(2); qc.y(3)
    qc.rzz(0.6, 1, 3); qc.cz(2, 3); qc.rx(0.9, 1); qc.rx(1.1, 3); qc.y(0); qc.rz(0.2, 0); qc.rx(0.3, 2); qc.rx(0.3, 0)
    qc.measure(range(4), range(4)); return qc
attempt("cz, s, t, sdg, tdg, p, x, y, z mixed in", cliffordish)
def with_rxx():
    qc = base(4, measure=False); qc.rxx(0.8, 0, 2); qc.rxx(0.5, 1, 3); qc.measure(range(4), range(4)); return qc
attempt("rxx (documented for direct only)", with_rxx)
for g in ("cx", "ry", "h_mid", "reset", "swap", "u", "c_if"):
    def bad(g=g):
        qc = base(3, measure=False)
        if g == "cx": qc.cx(0, 1)
        elif g == "ry": qc.ry(0.2, 0)
        elif g == "h_mid": qc.h(0)
        elif g == "reset": qc.reset(0)
        elif g == "swap": qc.swap(0, 1)
        elif g == "u": qc.u(0.1, 0.2, 0.3, 0)
        elif g == "c_if":
            qc.measure(0, 0)
            with qc.if_test((qc.clbits[0], 1)):
                qc.x(1)
        qc.measure(range(3), range(3)); return qc
    attempt(f"unsupported: {g}", bad)

# 9 registers
def multi_reg():
    qa, qb = QuantumRegister(2, "qa"), QuantumRegister(2, "qb")
    ca, cb = ClassicalRegister(2, "ca"), ClassicalRegister(2, "cb")
    qc = QuantumCircuit(qa, qb, ca, cb); qc.h(qc.qubits)
    qc.rzz(0.5, qa[0], qb[1]); qc.rz(0.3, qb[0]); qc.rx(0.7, qa[1]); qc.rx(0.2, qb[1])
    qc.measure(qa, cb); qc.measure(qb, ca); return qc
attempt("two quantum + two classical registers, crossed measurement", multi_reg)

# 10 misc options
attempt("direct checks='z' with init_flag=False", lambda: base(4), encs=("direct",), checks="z", init_flag=False)
attempt("bus with explicit bad placement", lambda: base(4), encs=("bus",), placement=[0, 0, 1, 2])
attempt("bus with short placement for odd k", lambda: base(3), encs=("bus",), placement=[0, 1, 2])
attempt("bus edge_mode typo", lambda: base(4), encs=("bus",), edge_mode="Bus")
attempt("direct checks typo", lambda: base(4), encs=("direct",), checks="xz")
attempt("zero-qubit circuit", lambda: QuantumCircuit(0, 0))
attempt("huge angles / multiples of 2pi", lambda: (lambda qc: (qc.h(range(2)), qc.rz(2 * math.pi, 0), qc.rzz(-7 * math.pi, 0, 1), qc.rx(1e6, 1), qc.measure(range(2), range(2)), qc)[-1])(QuantumCircuit(2, 2)))
