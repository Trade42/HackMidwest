"""Grid-native Iceberg encoding: hub gates routed through a row of bus ancillas.

Why this exists. In the textbook mapping every single-paragraph term is a gate between a
data qubit and ``b``, and every mixer term is a gate between ``t`` and a data qubit. On a
nearest-neighbour grid those two hubs need SWAP chains, and a single fault on a hub gate
can leave an undetected two-qubit error such as ``X_i X_b``, which flips every logical
bit except ``i``.

Layout (``K`` data qubits, ``C = K/2 + 1`` columns, nearest-neighbour gates only; ``K`` is
the number of paragraphs rounded up to an even number, and at least 4):

    row 0   d  d  d  ...  d  t        code qubits
    row 1   c0 c1 c2 ...     c_{C-1}  bus ancillas
    row 2   d  d  d  ...  d  b        code qubits

How each block uses the bus:

* **Start state.** The GHZ chain snakes along row 0, crosses through ``c_{C-1}``
  (computed and uncomputed), and returns along row 2. Its two ends sit above and below
  ``c_0``, which records their parity as a flag.
* **Pair terms** ``exp(-i theta/2 Z_i Z_j)``. Neighbours in a row get a direct gate. Any other
  pair writes its parity onto the bus (CNOTs along the lane), rotates one bus qubit,
  and uncomputes. Terms that share a source are emitted back to back so the shared
  CNOTs cancel.
* **Single-paragraph terms** ``exp(-i theta/2 Z_i Z_b)``. The bit of ``b`` is fanned out along
  the lane (a chain of CNOTs). Each data qubit then needs only its own bus neighbour:
  CNOT, ``RZ`` on the bus qubit, CNOT. The fan-out is undone afterwards.
* **Mixer** ``exp(-i theta/2 X_t X_i)``. The same construction in the X basis, fanned out
  from ``t``.
* **S_X check.** The lane is turned into a cat state, each bus qubit applies a CNOT to
  the code qubits above and below it, and the cat is decoded onto one bus qubit.

Outside state preparation and the final check, no data qubit shares a gate with ``b`` or
``t``, so a single fault in the logical gates cannot create ``X_i X_b`` or ``Z_t Z_i``.
Bus qubits must all read 0 at the end. A bus qubit flipped by a single fault stays flipped
and discards the shot; ``tests/test_iceberg_faults.py`` checks this for every single
fault at one, two and three layers. Two earlier versions had gaps (a single witness copy
at two or more layers; a two-qubit lane), both found by an independent review and both
covered by regression tests: see ``precopy`` below.
"""

from __future__ import annotations

import itertools
import math
from typing import Iterable, Sequence

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from .encode import _IGNORED, _PHASES, UnsupportedGateError, _check_measure_map, _leading_plus
from .layout import IcebergLayout


# ----------------------------------------------------------------- placement
def edge_cost(site_a: tuple[int, int], site_b: tuple[int, int], edge_mode: str = "direct") -> int:
    """CNOT count of one pair term between two data sites ``(row, col)``."""
    (ra, ma), (rb, mb) = site_a, site_b
    if edge_mode == "direct" and ra == rb and abs(ma - mb) == 1:
        return 2
    return 4 + 2 * abs(ma - mb)


def place(k_code: int, edges: Iterable[tuple[int, int]], *, edge_mode: str = "direct",
          seed: int = 0, restarts: int = 20) -> tuple[int, ...]:
    """Assign logical qubits to data sites so pair terms are cheap.

    Site ``s`` is row ``s // (K/2)`` (0 = top, 1 = bottom) and column ``s % (K/2)``.
    Returns ``site_of[i]`` for each logical qubit. Small problems are solved by
    pair-swap hill climbing from several random starts, which is deterministic for a
    given seed.
    """
    half = k_code // 2
    edges = [tuple(e) for e in edges]
    sites = [(s // half, s % half) for s in range(k_code)]

    def total(assign: Sequence[int]) -> int:
        return sum(edge_cost(sites[assign[i]], sites[assign[j]], edge_mode) for i, j in edges)

    if not edges:
        return tuple(range(k_code))
    rng = np.random.default_rng(seed)
    best, best_cost = list(range(k_code)), None
    for trial in range(restarts):
        cur = list(range(k_code)) if trial == 0 else [int(v) for v in rng.permutation(k_code)]
        cur_cost = total(cur)
        improved = True
        while improved:
            improved = False
            for a, b in itertools.combinations(range(k_code), 2):
                cur[a], cur[b] = cur[b], cur[a]
                c = total(cur)
                if c < cur_cost:
                    cur_cost, improved = c, True
                else:
                    cur[a], cur[b] = cur[b], cur[a]
        if best_cost is None or cur_cost < best_cost:
            best, best_cost = list(cur), cur_cost
    return tuple(best)


# -------------------------------------------------------------------- parsing
def _blocks(circuit: QuantumCircuit, start: int) -> tuple[list[tuple], list[int], set[int]]:
    """Group the bare circuit body into commuting blocks.

    Returns ``(blocks, logical_clbits, measured)`` where each block is one of
    ``("z", pair_terms, fields)``, ``("x", fields)`` or ``("pauli", name, qubit)``.
    Gates inside a "z" or "x" block commute, so the encoder may reorder them.
    """
    k = circuit.num_qubits
    blocks: list[tuple] = []
    logical_clbits = [-1] * k
    measured: set[int] = set()

    def z_block() -> tuple:
        if not blocks or blocks[-1][0] != "z":
            blocks.append(("z", [], {}))
        return blocks[-1]

    def x_block() -> tuple:
        if not blocks or blocks[-1][0] != "x":
            blocks.append(("x", {}))
        return blocks[-1]

    for inst in circuit.data[start:]:
        op = inst.operation
        name = op.name
        qs = [circuit.find_bit(x).index for x in inst.qubits]
        if name == "measure":
            if qs[0] in measured:
                raise UnsupportedGateError(f"qubit {qs[0]} is measured more than once")
            logical_clbits[qs[0]] = circuit.find_bit(inst.clbits[0]).index
            measured.add(qs[0])
            continue
        if name in _IGNORED:
            continue
        if measured & set(qs):
            raise UnsupportedGateError("gates after a measurement are not supported")
        if name in ("rz", "p") or name in _PHASES:
            angle = _PHASES[name] if name in _PHASES else op.params[0]
            fields = z_block()[2]
            fields[qs[0]] = fields.get(qs[0], 0.0) + angle
        elif name == "rzz":
            z_block()[1].append((qs[0], qs[1], op.params[0]))
        elif name == "cz":
            blk = z_block()
            blk[1].append((qs[0], qs[1], math.pi / 2))
            for qb in qs:
                blk[2][qb] = blk[2].get(qb, 0.0) - math.pi / 2
        elif name == "rx":
            fields = x_block()[1]
            fields[qs[0]] = fields.get(qs[0], 0.0) + op.params[0]
        elif name in ("x", "y", "z"):
            blocks.append(("pauli", name, qs[0]))
        else:
            raise UnsupportedGateError(
                f"no bus implementation for gate {name!r}; supported gates are rz, rzz, rx, "
                "p, s, sdg, t, tdg, x, y, z, cz and a leading H layer"
            )
    return blocks, logical_clbits, measured


# -------------------------------------------------------------------- encoder
def encode_bus(
    circuit: QuantumCircuit,
    *,
    placement: Sequence[int] | None = None,
    edge_mode: str = "direct",
    init_flag: bool = True,
    precopy: bool | int = True,
    barriers: bool = True,
    readout: bool | None = None,
    seed: int = 0,
) -> QuantumCircuit:
    """Iceberg-encode ``circuit`` for a nearest-neighbour grid; see the module docstring.

    Args:
        circuit: bare circuit (leading ``H`` layer, then ``rz, rzz, rx, ...``, then
            terminal measurements), as produced by :func:`qsel.qaoa_circuit`.
        placement: ``site_of[i]`` for each logical qubit; chosen by :func:`place` if omitted.
        edge_mode: ``"direct"`` uses a direct gate for row neighbours, ``"bus"`` routes
            every pair term through the bus.
        init_flag: record the parity of the GHZ chain's two ends on ``c_0``.
        precopy: before the S_X check, copy the neighbour's bit one step along the lane and
            the S_X bus qubit's bit onto the neighbour (``True``, depth 2). Without it
            (``False``), a single fault that flips both that bus qubit and S_X cancels
            itself in the readout. Depth 1 (``precopy=1``, only the second copy) is enough
            for one layer but leaves a gap from two layers on; it is kept so the tests can
            show the gap.
        barriers: keep blocks apart so a compiler cannot merge across them. Leave this on
            for anything that goes through an optimising compiler.
        readout: append the S_X check and measurements (default: the bare circuit measures).

    The qubit order is data ``0..K-1``, ``t``, ``b``, then bus ``c_0..c_{C-1}``.
    ``metadata["iceberg"]["positions"]`` gives the ``(row, col)`` of every qubit.
    """
    if edge_mode not in ("direct", "bus"):
        raise ValueError("edge_mode must be 'direct' or 'bus'")
    if precopy not in (False, True, 0, 1, 2):
        raise ValueError("precopy must be True (two-step witness), False, or the depth 0, 1 or 2")
    depth = 2 if precopy is True else int(precopy)
    k = circuit.num_qubits
    if k < 1:
        raise ValueError("the bare circuit has no qubits")
    plus, start = _leading_plus(circuit)
    if plus and len(plus) != k:
        raise UnsupportedGateError("the leading H layer must cover every qubit or none of them")
    init = "plus" if plus else "zero"
    blocks, logical_clbits, measured = _blocks(circuit, start)
    _check_measure_map(logical_clbits)
    has_measure = bool(measured)
    do_readout = has_measure if readout is None else bool(readout)
    if do_readout and not has_measure:
        logical_clbits = list(range(k))
    num_bare_clbits = circuit.num_clbits if has_measure else k

    # At least 4 code-logical qubits, so the lane has 3 bus qubits and the S_X qubit has the
    # two-step witness. One or two paragraphs are padded with idle logical qubits.
    K = max(4, k + (k % 2))
    half = K // 2
    C = half + 1
    all_edges = sorted({(min(i, j), max(i, j)) for blk in blocks if blk[0] == "z" for i, j, _ in blk[1]})
    if placement is None:
        placement = place(K, all_edges, edge_mode=edge_mode, seed=seed)
    placement = tuple(int(s) for s in placement)
    if sorted(placement) != list(range(K)):
        raise ValueError("placement must be a permutation of the data sites")

    d_reg, t_reg, b_reg = QuantumRegister(K, "d"), QuantumRegister(1, "t"), QuantumRegister(1, "b")
    c_reg = QuantumRegister(C, "c")
    enc = QuantumCircuit(d_reg, t_reg, b_reg, c_reg, name=f"iceberg_bus_{circuit.name}")
    d = list(d_reg)
    t, b = t_reg[0], b_reg[0]
    c = list(c_reg)
    code = d + [t, b]

    row = [placement[i] // half for i in range(K)]
    col = [placement[i] % half for i in range(K)]
    top: list = [None] * C
    bot: list = [None] * C
    for i in range(K):
        (top if row[i] == 0 else bot)[col[i]] = d[i]
    top[C - 1], bot[C - 1] = t, b
    root = C - 1

    labels: list[str] = []
    terms: list[list] = []      # [kind, logical qubits, index of the encoded rotation]

    def tag(block: str) -> None:
        labels.extend([block] * (len(enc.data) - len(labels)))

    def term(kind: str, *logical: int) -> None:
        terms.append([kind, list(logical), len(enc.data) - 1])

    def bar() -> None:
        if barriers:
            enc.barrier(*enc.qubits)

    # ---- 1. start state: GHZ snake, flag on c_0, then H on the code qubits
    enc.h(top[0])
    for m in range(C - 1):
        enc.cx(top[m], top[m + 1])
    enc.cx(t, c[C - 1])
    enc.cx(c[C - 1], b)
    enc.cx(t, c[C - 1])
    for m in range(C - 1, 0, -1):
        enc.cx(bot[m], bot[m - 1])
    if init_flag:
        enc.cx(top[0], c[0])
        enc.cx(bot[0], c[0])
    bar()
    if init == "plus":
        for qb in code:
            enc.h(qb)
        bar()
    tag("init")

    # ---- 2. logical gates
    def cx_cancel(control, target) -> None:
        """Emit a CNOT, or cancel it against an identical CNOT emitted just before."""
        if enc.data:
            last = enc.data[-1]
            if last.operation.name == "cx" and last.qubits[0] == control and last.qubits[1] == target:
                enc.data.pop()
                return
        enc.cx(control, target)

    def pair_terms(terms: list[tuple[int, int, float]]) -> None:
        """Pair terms ``exp(-i theta/2 Z_i Z_j)``; they commute, so the order is ours to choose.

        Row neighbours get a direct gate. The rest go through the bus: the source's bit is
        written on its bus qubit, carried along the lane to the partner's column, the
        partner is added, one ``RZ`` is applied, and everything is undone. Terms are
        grouped by a shared source and sorted by lane distance, so the undo of one term
        and the setup of the next cancel wherever they coincide.
        """
        direct, routed = [], []
        for i, j, theta in terms:
            if edge_mode == "direct" and row[i] == row[j] and abs(col[i] - col[j]) == 1:
                direct.append((i, j, theta))
            else:
                routed.append((i, j, theta))
        for i, j, theta in direct:
            enc.rzz(theta, d[i], d[j])
            term("zz", i, j)

        remaining = list(routed)
        while remaining:
            degree: dict[int, int] = {}
            for i, j, _ in remaining:
                degree[i] = degree.get(i, 0) + 1
                degree[j] = degree.get(j, 0) + 1
            src = max(degree, key=lambda q: (degree[q], -q))
            mine = [(j if i == src else i, theta, (i, j)) for i, j, theta in remaining if src in (i, j)]
            remaining = [t for t in remaining if src not in (t[0], t[1])]
            # one direction along the lane at a time, nearest partner first
            mine.sort(key=lambda t: (col[t[0]] < col[src], abs(col[t[0]] - col[src]), t[0]))
            for partner, theta, (i, j) in mine:
                step = 1 if col[partner] >= col[src] else -1
                path = list(range(col[src], col[partner], step))
                cx_cancel(d[src], c[col[src]])
                for m in path:
                    cx_cancel(c[m], c[m + step])
                enc.cx(d[partner], c[col[partner]])
                enc.rz(theta, c[col[partner]])
                term("zz", i, j)
                enc.cx(d[partner], c[col[partner]])
                for m in reversed(path):
                    enc.cx(c[m], c[m + step])
                enc.cx(d[src], c[col[src]])

    def z_fields(fields: dict[int, float]) -> None:
        if not fields:
            return
        lo = min(col[i] for i in fields)
        enc.cx(b, c[C - 1])
        for m in range(C - 1, lo, -1):
            enc.cx(c[m], c[m - 1])
        for i in sorted(fields, key=lambda q: (col[q], row[q])):
            enc.cx(d[i], c[col[i]])
            enc.rz(fields[i], c[col[i]])
            term("z", i)
            enc.cx(d[i], c[col[i]])
        for m in range(lo + 1, C):
            enc.cx(c[m], c[m - 1])
        enc.cx(b, c[C - 1])

    def x_fields(fields: dict[int, float]) -> None:
        if not fields:
            return
        lo = min(col[i] for i in fields)
        lane = c[lo:]
        for qb in lane:
            enc.h(qb)
        enc.cx(c[C - 1], t)
        for m in range(C - 1, lo, -1):
            enc.cx(c[m - 1], c[m])
        for i in sorted(fields, key=lambda q: (col[q], row[q])):
            enc.cx(c[col[i]], d[i])
            enc.rx(fields[i], c[col[i]])
            term("x", i)
            enc.cx(c[col[i]], d[i])
        for m in range(lo + 1, C):
            enc.cx(c[m - 1], c[m])
        enc.cx(c[C - 1], t)
        for qb in lane:
            enc.h(qb)

    for blk in blocks:
        if blk[0] == "z":
            pair_terms(blk[1])
            if blk[1]:
                bar()
                tag("pairs")
            z_fields(blk[2])
            bar()
            tag("fields")
        elif blk[0] == "x":
            x_fields(blk[1])
            bar()
            tag("mixer")
        else:
            _, name, i = blk
            if name in ("x", "y"):
                enc.x(t)
            if name in ("z", "y"):
                enc.z(b)
            getattr(enc, name)(d[i])
            terms.append(["pauli_" + name, [i], len(enc.data) - 1])
            tag("pauli")

    # ---- 3. S_X check with a cat state on the lane, 4. readout
    if do_readout:
        creg = ClassicalRegister(K + 2 + C, "m")
        enc.add_register(creg)
        if depth:
            # Two-step witness for the S_X bus qubit. Its readout is (S_X flip) XOR (its own
            # earlier flip), so a fault causing both would cancel. Copying its bit onto its
            # neighbour exposes that, unless the same fault also flipped the neighbour
            # (a Y on the S_X qubit during a fan-out does exactly this). So the neighbour's
            # bit is first copied one step further along the lane.
            if C >= 3 and depth >= 2:
                enc.cx(c[root - 1], c[root - 2])
            if C >= 2:
                enc.cx(c[root], c[root - 1])
        enc.h(c[root])
        for m in range(C - 1, 0, -1):
            enc.cx(c[m], c[m - 1])
        for m in range(C):
            enc.cx(c[m], top[m])
            enc.cx(c[m], bot[m])
        for m in range(1, C):
            enc.cx(c[m], c[m - 1])
        enc.h(c[root])
        bar()
        tag("syndrome")
        for j, qb in enumerate(enc.qubits):
            enc.measure(qb, creg[j])
        tag("measure")

    positions: list[tuple[int, int]] = [(0, 0)] * (K + 2 + C)
    for i in range(K):
        positions[i] = (0 if row[i] == 0 else 2, col[i])
    positions[K] = (0, C - 1)
    positions[K + 1] = (2, C - 1)
    for m in range(C):
        positions[K + 2 + m] = (1, m)

    lane_idx = [K + 2 + m for m in range(C)]
    lay = IcebergLayout(
        k=k, k_code=K, checks="both", init=init, ghz_init=True, top=K, bottom=K + 1,
        sx_ancilla=lane_idx[root], flags=tuple(q for q in lane_idx if q != lane_idx[root]),
        init_flag=bool(init_flag), sx_flag=True, logical_clbits=tuple(logical_clbits),
        num_bare_clbits=num_bare_clbits, measured=do_readout, style="bus",
        positions=tuple(positions),
    )
    rle: list[list] = []
    for name in labels:
        if rle and rle[-1][0] == name:
            rle[-1][1] += 1
        else:
            rle.append([name, 1])
    enc.metadata = {**(circuit.metadata or {}), "tier": "iceberg", "iceberg": lay.to_dict(),
                    "iceberg_blocks": rle, "iceberg_terms": terms, "placement": list(placement)}
    return enc


def grid_coupling(layout: IcebergLayout) -> list[tuple[int, int]]:
    """Nearest-neighbour pairs (by qubit index) of the bus layout, for transpilers."""
    if layout.positions is None:
        raise ValueError("layout has no grid positions")
    pos = {q: p for q, p in enumerate(layout.positions)}
    pairs = []
    for a, b in itertools.combinations(sorted(pos), 2):
        (ra, ca), (rb, cb) = pos[a], pos[b]
        if abs(ra - rb) + abs(ca - cb) == 1:
            pairs.append((a, b))
    return pairs
