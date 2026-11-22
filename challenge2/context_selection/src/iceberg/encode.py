"""Encode a QAOA-style circuit in the ``[[k+2, k, 2]]`` Iceberg code (textbook mapping).

What the encoder does, block by block (``K`` data qubits, ``t``, ``b``, ancilla ``a``, flag ``f``):

1. **Start state.** The bare circuit starts with ``H`` on every qubit, i.e. logical
   ``|+>^k``. In the code that state is ``(|+>^n + |->^n)/sqrt(2)`` on the ``n = K+2``
   code qubits. It is prepared as a GHZ state (``H`` on ``t``, then a CNOT chain
   ``t -> d_0 -> ... -> d_{K-1} -> b``) followed by ``H`` on every code qubit. A fault
   in the chain can spread to many qubits, so the flag ``f`` records the parity of the
   chain's two ends, which must be even.
2. **Logical gates.** ``rz`` on logical ``i`` becomes ``rzz`` on ``(d_i, b)``; ``rx``
   becomes ``rxx`` on ``(t, d_i)``; ``rzz``/``rxx`` between logical qubits stay on the
   two data qubits. Every one of these commutes with both stabilizers.
3. **Checks.** ``S_X`` is measured with ancilla ``a`` (``H``, a CNOT from ``a`` to every
   code qubit, ``H``). The flag ``f`` is coupled to ``a`` after the first and before the
   last of those CNOTs so that a fault on ``a`` that spreads to several code qubits is
   caught. ``S_Z`` costs nothing: it is the parity of the measured code bits.
4. **Readout.** Logical bit ``i`` is ``bit(d_i) XOR bit(b)``.

Shots are kept only if the code parity is even, ``a`` reads 0 and ``f`` reads 0; see
:func:`iceberg.postselect`. The grid-native variant lives in :mod:`iceberg.bus`.

``checks="z"`` drops the ancilla (parity check only); ``checks="x"`` drops the entangled
start (ancilla check only, product-state start). Both are cheaper, and both still give
the right logical distribution without noise, so they are honest ablations.
"""

from __future__ import annotations

import math
from typing import Iterable

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from .layout import IcebergLayout, make_layout


class UnsupportedGateError(ValueError):
    """The bare circuit uses a gate the encoder has no logical implementation for."""


_PHASES = {"s": math.pi / 2, "sdg": -math.pi / 2, "t": math.pi / 4, "tdg": -math.pi / 4}
_IGNORED = {"barrier", "id", "delay"}


def _leading_plus(circuit: QuantumCircuit) -> tuple[set[int], int]:
    """Qubits whose first gate is ``H`` and the index where the body starts."""
    plus: set[int] = set()
    touched: set[int] = set()
    start = 0
    for pos, inst in enumerate(circuit.data):
        name = inst.operation.name
        qubits = [circuit.find_bit(q).index for q in inst.qubits]
        if name == "barrier":
            start = pos + 1
            continue
        if name == "h" and qubits[0] not in touched:
            plus.add(qubits[0])
            touched.add(qubits[0])
            start = pos + 1
            continue
        break
    return plus, start


def _check_measure_map(logical_clbits: list[int]) -> None:
    """Two qubits must not be measured into the same classical bit."""
    used = [c for c in logical_clbits if c >= 0]
    if len(used) != len(set(used)):
        raise UnsupportedGateError("two qubits are measured into the same classical bit")


def encode_direct(
    circuit: QuantumCircuit,
    *,
    checks: str = "both",
    init_flag: bool = True,
    sx_flag: bool = True,
    barriers: bool = True,
    readout: bool | None = None,
) -> QuantumCircuit:
    """Textbook Iceberg encoding: hub gates act directly on ``t`` and ``b``.

    This is the mapping used on trapped ions (all-to-all connectivity). On a grid it
    needs SWAP chains, and a single fault on a ``(data, b)`` gate can flip every logical
    bit but one without being detected. Prefer :func:`iceberg.encode_bus` on Sqale; this
    version is kept as the reference point and for the ``checks`` ablations.

    Args:
        circuit: bare circuit on ``k`` qubits. Supported: an optional leading ``H`` on
            every qubit, then ``rz, rzz, rx, rxx, p, s, sdg, t, tdg, x, y, z, cz``,
            then terminal measurements in the Z basis.
        checks: ``"both"`` (full code), ``"z"`` (parity only) or ``"x"`` (ancilla only).
        init_flag: verify the entangled start state with the flag qubit.
        sx_flag: guard the ``S_X`` measurement against hook errors with the flag qubit.
        barriers: separate blocks with barriers so a compiler cannot optimise the
            checks away (two of the flag CNOTs cancel in the noiseless circuit). With
            ``barriers=False`` an optimising transpile removes them and the flag
            protection is silently lost; leave this on.
        readout: append checks and measurements. Defaults to "the bare circuit measures".

    The layout is stored in ``result.metadata["iceberg"]`` for :func:`iceberg.postselect`.
    """
    k = circuit.num_qubits
    if k < 1:
        raise ValueError("the bare circuit has no qubits")
    plus, start = _leading_plus(circuit)
    if plus and len(plus) != k:
        raise UnsupportedGateError(
            "the leading H layer must cover every qubit or none of them "
            f"(found H on {sorted(plus)} of {k} qubits)"
        )
    init = "plus" if plus else "zero"

    # ---- terminal measurements of the bare circuit
    logical_clbits = [-1] * k
    measured_qubits: set[int] = set()
    for inst in circuit.data[start:]:
        if inst.operation.name == "measure":
            qi = circuit.find_bit(inst.qubits[0]).index
            if qi in measured_qubits:
                raise UnsupportedGateError(f"qubit {qi} is measured more than once")
            logical_clbits[qi] = circuit.find_bit(inst.clbits[0]).index
            measured_qubits.add(qi)
    _check_measure_map(logical_clbits)
    has_measure = bool(measured_qubits)
    do_readout = has_measure if readout is None else bool(readout)
    if do_readout and not has_measure:
        logical_clbits = list(range(k))
    num_bare_clbits = circuit.num_clbits if has_measure else k

    lay = make_layout(
        k, checks=checks, init=init, init_flag=init_flag, sx_flag=sx_flag,
        logical_clbits=tuple(logical_clbits), num_bare_clbits=num_bare_clbits, measured=do_readout,
    )
    K, t_i, b_i = lay.k_code, lay.top, lay.bottom

    regs: list = [QuantumRegister(K, "d"), QuantumRegister(1, "t"), QuantumRegister(1, "b")]
    if lay.sx_ancilla is not None:
        regs.append(QuantumRegister(1, "a"))
    if lay.flag is not None:
        regs.append(QuantumRegister(1, "f"))
    enc = QuantumCircuit(*regs, name=f"iceberg_{circuit.name}")
    q = enc.qubits
    d = [q[i] for i in range(K)]
    t, b = q[t_i], q[b_i]
    a = q[lay.sx_ancilla] if lay.sx_ancilla is not None else None
    f = q[lay.flag] if lay.flag is not None else None
    code = d + [t, b]

    labels: list[str] = []
    terms: list[list] = []      # [kind, logical qubits, index of the encoded rotation]

    def tag(block: str) -> None:
        """Label every instruction emitted since the last call."""
        labels.extend([block] * (len(enc.data) - len(labels)))

    def term(kind: str, *logical: int) -> None:
        """Record that the gate just emitted implements one logical rotation."""
        terms.append([kind, list(logical), len(enc.data) - 1])

    def bar(qubits: Iterable | None = None) -> None:
        if barriers:
            enc.barrier(*(list(qubits) if qubits is not None else enc.qubits))

    # ---- 1. start state
    if lay.ghz_init:
        chain = [t] + d + [b]
        enc.h(chain[0])
        for u, v in zip(chain[:-1], chain[1:]):
            enc.cx(u, v)
        if lay.init_flag:
            enc.cx(chain[0], f)
            enc.cx(chain[-1], f)
        bar()
    if init == "plus":
        for qb in code:
            enc.h(qb)
        bar()
    tag("init")

    # ---- 2. logical gates
    seen_measure: set[int] = set()
    for inst in circuit.data[start:]:
        op = inst.operation
        name = op.name
        qs = [circuit.find_bit(x).index for x in inst.qubits]
        if name == "measure":
            seen_measure.add(qs[0])
            continue
        if name in _IGNORED:
            if name == "barrier":
                bar(code)
                tag("barrier")
            continue
        if seen_measure & set(qs):
            raise UnsupportedGateError("gates after a measurement are not supported")
        if name == "rz":
            enc.rzz(op.params[0], d[qs[0]], b)
            term("z", qs[0])
        elif name == "p":
            enc.rzz(op.params[0], d[qs[0]], b)          # equal to rz up to a global phase
            term("z", qs[0])
        elif name in _PHASES:
            enc.rzz(_PHASES[name], d[qs[0]], b)
            term("z", qs[0])
        elif name == "rzz":
            enc.rzz(op.params[0], d[qs[0]], d[qs[1]])
            term("zz", qs[0], qs[1])
        elif name == "rx":
            enc.rxx(op.params[0], t, d[qs[0]])
            term("x", qs[0])
        elif name == "rxx":
            enc.rxx(op.params[0], d[qs[0]], d[qs[1]])
            term("xx", qs[0], qs[1])
        elif name == "x":
            enc.x(t)
            enc.x(d[qs[0]])
        elif name == "z":
            enc.z(d[qs[0]])
            enc.z(b)
        elif name == "y":
            enc.x(t)
            enc.y(d[qs[0]])
            enc.z(b)
        elif name == "cz":
            # CZ = exp(-i pi/4 (1 - Z_i - Z_j + Z_i Z_j)) up to a global phase
            enc.rzz(math.pi / 2, d[qs[0]], d[qs[1]])
            term("zz", qs[0], qs[1])
            enc.rzz(-math.pi / 2, d[qs[0]], b)
            term("z", qs[0])
            enc.rzz(-math.pi / 2, d[qs[1]], b)
            term("z", qs[1])
        else:
            raise UnsupportedGateError(
                f"no logical implementation for gate {name!r}; supported gates are "
                "rz, rzz, rx, rxx, p, s, sdg, t, tdg, x, y, z, cz and a leading H layer"
            )
        if name in ("x", "y", "z"):
            terms.append(["pauli_" + name, [qs[0]], len(enc.data) - 1])
        tag("mixer" if name in ("rx", "rxx") else "pauli" if name in ("x", "y", "z") else "cost")

    # ---- 3. checks and 4. readout
    if do_readout:
        creg = ClassicalRegister(lay.num_qubits, "m")
        enc.add_register(creg)
        bar()
        if lay.check_x:
            order = [t] + d + [b]
            enc.h(a)
            if lay.sx_flag:
                enc.cx(a, order[0])
                enc.cx(a, f)
                bar([a, f])
                for qb in order[1:-1]:
                    enc.cx(a, qb)
                bar([a, f])
                enc.cx(a, f)
                enc.cx(a, order[-1])
            else:
                for qb in order:
                    enc.cx(a, qb)
            enc.h(a)
            bar()
        tag("syndrome")
        for j, qb in enumerate(enc.qubits):
            enc.measure(qb, creg[j])
        tag("measure")

    rle: list[list] = []
    for name in labels:
        if rle and rle[-1][0] == name:
            rle[-1][1] += 1
        else:
            rle.append([name, 1])
    enc.metadata = {**(circuit.metadata or {}), "tier": "iceberg", "iceberg": lay.to_dict(),
                    "iceberg_blocks": rle, "iceberg_terms": terms}
    return enc


def layout_from_circuit(circuit: QuantumCircuit) -> IcebergLayout:
    """Layout stored by :func:`encode`."""
    return IcebergLayout.from_dict(circuit.metadata["iceberg"])
