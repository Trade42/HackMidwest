"""Exact Pauli-noise simulation of (encoded) QAOA circuits in the logical space.

The circuits here contain only Clifford gates (``h``, ``cx``, ``cz``, ...) and rotations
about Pauli axes (``rz``, ``rx``, ``rzz``, ``rxx``). Two facts make noise cheap to simulate:

1. A Pauli fault moves through a Clifford gate as another Pauli (a "frame" update).
2. A Pauli fault moves through a rotation unchanged, except that the rotation angle is
   negated when the fault anticommutes with the rotation axis:
   ``R(theta) P = P R(-theta)``.

So a shot with any number of Pauli faults equals the noiseless circuit with some
rotation angles negated, followed by one final Pauli. The final Pauli only flips
measured bits. The sign-flipped circuit is still a valid encoded computation, so its
outcome distribution is the *logical* circuit's distribution with the same signs
flipped, which lives on ``2**k`` amplitudes instead of ``2**(physical qubits)``.

This gives two tools that agree exactly with brute-force statevector simulation
(see ``tests/test_iceberg_frames.py``) but scale to the full 18-qubit bus layout:

* :meth:`FrameSimulator.single_faults`: every single fault, scored exactly;
* :meth:`FrameSimulator.sample`: Monte-Carlo shots under circuit-level Pauli noise,
  returned as physical bitstrings so they go through :func:`iceberg.postselect`
  exactly like hardware counts.

``level`` chooses how two-qubit operations are counted as fault locations:
``"native"`` (each ``rzz``/``rxx`` is one noisy gate, as on trapped ions), ``"cx"``
(rotations decomposed into two CNOTs) or ``"cz"`` (every CNOT is ``H CZ H``, the
neutral-atom gate set). The number of two-qubit fault locations at ``"cz"`` level is
the CZ count of the circuit.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Sequence

import numpy as np
from qiskit import QuantumCircuit

from .faults import Fault, FaultReport
from .layout import IcebergLayout, layout_of

_CLIFFORD_1Q = {"h", "x", "y", "z", "s", "sdg", "sx", "sxdg", "id"}
_TWO_QUBIT_PAULIS = tuple(p for p in product("IXYZ", repeat=2) if p != ("I", "I"))
_BITS = {"I": (0, 0), "X": (1, 0), "Y": (1, 1), "Z": (0, 1)}


@dataclass(frozen=True)
class PauliNoise:
    """Circuit-level Pauli noise. These are assumptions, not device calibration data.

    Attributes:
        p1: error probability after each one-qubit gate (uniform over X, Y, Z).
        p2: error probability after each two-qubit gate.
        p_meas: probability that a measured bit is flipped.
        two_qubit: ``"depolarizing"`` (uniform over the 15 two-qubit Paulis),
            ``"dephasing"`` (only IZ, ZI, ZZ: phase errors, the kind a stray-field
            shift of the Rydberg level produces on a CZ), or 15 explicit weights in
            the order of ``itertools.product("IXYZ", repeat=2)`` without ``II``.
        noisy_rz: if False, ``rz`` is treated as an error-free frame change.
    """

    p1: float = 5e-4
    p2: float = 5e-3
    p_meas: float = 5e-3
    two_qubit: Any = "depolarizing"
    noisy_rz: bool = False

    def weights(self) -> np.ndarray:
        if isinstance(self.two_qubit, str):
            if self.two_qubit == "depolarizing":
                w = np.ones(15)
            elif self.two_qubit == "dephasing":
                w = np.array([1.0 if set(p) <= {"I", "Z"} else 0.0 for p in _TWO_QUBIT_PAULIS])
            else:
                raise ValueError(f"unknown two-qubit noise {self.two_qubit!r}")
        else:
            w = np.asarray(self.two_qubit, dtype=float)
            if w.shape != (15,):
                raise ValueError("two_qubit weights must have length 15")
        return w / w.sum()

    def to_dict(self) -> dict[str, Any]:
        tq = self.two_qubit if isinstance(self.two_qubit, str) else list(map(float, self.two_qubit))
        return {"p1": self.p1, "p2": self.p2, "p_meas": self.p_meas, "two_qubit": tq,
                "noisy_rz": self.noisy_rz}


@dataclass(frozen=True)
class _Op:
    name: str
    qubits: tuple[int, ...]
    term: int = -1          #: logical rotation implemented by this gate, -1 for Clifford gates
    block: str = "circuit"


# --------------------------------------------------------------- logical model
class LogicalSimulator:
    """Statevector of the logical circuit with a chosen sign on every rotation."""

    def __init__(self, k: int, terms: Sequence[tuple[str, tuple[int, ...], float]], init: str = "plus"):
        if init != "plus":
            raise NotImplementedError("frame simulation supports circuits that start in |+>^k")
        if k > 22:
            raise ValueError("logical simulation is limited to 22 logical qubits")
        self.k = k
        self.terms = list(terms)
        idx = np.arange(1 << k, dtype=np.int64)
        self._diag: list[np.ndarray | None] = []
        for kind, qs, _ in self.terms:
            if kind == "z":
                self._diag.append(1.0 - 2.0 * ((idx >> qs[0]) & 1))
            elif kind == "zz":
                self._diag.append(1.0 - 2.0 * (((idx >> qs[0]) ^ (idx >> qs[1])) & 1))
            elif kind == "x":
                self._diag.append(None)
            else:
                raise NotImplementedError(f"logical term {kind!r} is not supported by the frame simulator")
        self._cache: dict[bytes, np.ndarray] = {}

    def probabilities(self, flipped: np.ndarray | None = None) -> np.ndarray:
        """Outcome distribution; ``flipped[j]`` negates the angle of term ``j``."""
        if flipped is None:
            flipped = np.zeros(len(self.terms), dtype=bool)
        key = np.packbits(flipped).tobytes()
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        k = self.k
        psi = np.full(1 << k, (1 << k) ** -0.5, dtype=complex)
        phase = None
        for j, (kind, qs, angle) in enumerate(self.terms):
            theta = -angle if flipped[j] else angle
            if kind in ("z", "zz"):
                phase = theta * self._diag[j] if phase is None else phase + theta * self._diag[j]
                continue
            if phase is not None:
                psi = psi * np.exp(-0.5j * phase)
                phase = None
            c, s = np.cos(theta / 2), -1j * np.sin(theta / 2)
            view = psi.reshape(-1, 2, 1 << qs[0])
            a = view[:, 0, :].copy()
            b = view[:, 1, :]
            view[:, 0, :] = c * a + s * b
            view[:, 1, :] = s * a + c * b
        probs = np.abs(psi) ** 2          # a trailing diagonal phase does not change probabilities
        probs = probs / probs.sum()
        if len(self._cache) < 20000:
            self._cache[key] = probs
        return probs


# ------------------------------------------------------------------- program
def _terms_of(circuit: QuantumCircuit) -> tuple[dict[int, int], list[tuple[str, tuple[int, ...], float]]]:
    """Map instruction index -> term id, and the logical term list."""
    meta = circuit.metadata or {}
    by_index: dict[int, int] = {}
    terms: list[tuple[str, tuple[int, ...], float]] = []
    if "iceberg_terms" in meta:
        for kind, qs, idx in meta["iceberg_terms"]:
            if kind.startswith("pauli"):
                raise NotImplementedError("explicit logical Pauli gates are not supported by the frame simulator")
            by_index[int(idx)] = len(terms)
            terms.append((kind, tuple(int(q) for q in qs), float(circuit.data[int(idx)].operation.params[0])))
        return by_index, terms
    # Bare circuit, possibly routed: follow each logical wire through the SWAPs so a
    # rotation is attributed to the logical qubit that sits on the physical qubit then.
    wire = list(range(circuit.num_qubits))
    for pos, inst in enumerate(circuit.data):
        name = inst.operation.name
        qs = tuple(circuit.find_bit(q).index for q in inst.qubits)
        if name == "swap":
            wire[qs[0]], wire[qs[1]] = wire[qs[1]], wire[qs[0]]
        elif name in ("rz", "rzz", "rx"):
            by_index[pos] = len(terms)
            terms.append(({"rz": "z", "rzz": "zz", "rx": "x"}[name], tuple(wire[q] for q in qs),
                          float(inst.operation.params[0])))
    return by_index, terms


def _bare_wires(circuit: QuantumCircuit) -> tuple[dict[int, int], dict[int, int]]:
    """For a bare (possibly routed) circuit: wire -> compact logical index, and logical -> clbit."""
    wire = list(range(circuit.num_qubits))
    used: set[int] = set()
    clbit_of_wire: dict[int, int] = {}
    for inst in circuit.data:
        name = inst.operation.name
        qs = tuple(circuit.find_bit(q).index for q in inst.qubits)
        if name == "swap":
            wire[qs[0]], wire[qs[1]] = wire[qs[1]], wire[qs[0]]
        elif name == "measure":
            clbit_of_wire[wire[qs[0]]] = circuit.find_bit(inst.clbits[0]).index
            used.add(wire[qs[0]])
        elif name in ("rz", "rzz", "rx"):
            used.update(wire[q] for q in qs)
    compact = {w: i for i, w in enumerate(sorted(used))}
    return compact, {compact[w]: c for w, c in clbit_of_wire.items()}


def _cancel_h(ops: list[_Op]) -> list[_Op]:
    """Remove pairs of adjacent ``h`` gates on the same qubit."""
    out: list[_Op | None] = []
    last: dict[int, int] = {}
    for op in ops:
        if op.name == "h":
            q = op.qubits[0]
            prev = last.get(q)
            if prev is not None and out[prev] is not None and out[prev].name == "h":
                out[prev] = None
                # the gate before the cancelled pair is unknown; forget it for this qubit
                last.pop(q, None)
                continue
        out.append(op)
        for q in op.qubits:
            last[q] = len(out) - 1
    return [op for op in out if op is not None]


class FrameSimulator:
    """Pauli-frame simulator for a bare or Iceberg-encoded QAOA circuit."""

    def __init__(self, circuit: QuantumCircuit, level: str = "cz"):
        if level not in ("native", "cx", "cz"):
            raise ValueError("level must be 'native', 'cx' or 'cz'")
        self.level = level
        self.num_qubits = circuit.num_qubits
        if self.num_qubits > 63:
            raise ValueError("frame simulation supports at most 63 physical qubits")
        try:
            self.layout: IcebergLayout | None = layout_of(circuit)
        except ValueError:
            self.layout = None
        by_index, terms = _terms_of(circuit)
        if self.layout is None:
            compact, self._bare_clbits = _bare_wires(circuit)
            terms = [(kind, tuple(compact[w] for w in ws), angle) for kind, ws, angle in terms]
        labels = _block_labels(circuit)

        ops: list[_Op] = []
        self.measures: list[tuple[int, int]] = []
        for pos, inst in enumerate(circuit.data):
            name = inst.operation.name
            qs = tuple(circuit.find_bit(q).index for q in inst.qubits)
            blk = labels[pos]
            if name in ("barrier", "delay"):
                continue
            if name == "measure":
                self.measures.append((qs[0], circuit.find_bit(inst.clbits[0]).index))
                continue
            term = by_index.get(pos, -1)
            if name in ("rz", "rx"):
                if term < 0:
                    raise ValueError(f"rotation at instruction {pos} is not a tracked logical term")
                ops.append(_Op(name, qs, term, blk))
            elif name in ("rzz", "rxx"):
                if term < 0:
                    raise ValueError(f"rotation at instruction {pos} is not a tracked logical term")
                if level == "native":
                    ops.append(_Op(name, qs, term, blk))
                else:
                    a, b = qs
                    pre = [_Op("h", (a,), -1, blk), _Op("h", (b,), -1, blk)] if name == "rxx" else []
                    ops.extend(pre)
                    ops.extend([_Op("cx", (a, b), -1, blk), _Op("rz", (b,), term, blk), _Op("cx", (a, b), -1, blk)])
                    ops.extend(pre)
            elif name == "swap":
                a, b = qs
                ops.extend([_Op("cx", (a, b), -1, blk), _Op("cx", (b, a), -1, blk), _Op("cx", (a, b), -1, blk)])
            elif name in ("cx", "cz") or name in _CLIFFORD_1Q:
                if name != "id":
                    ops.append(_Op(name, qs, -1, blk))
            else:
                raise ValueError(f"gate {name!r} is outside the Clifford + Pauli-rotation set")
        if level == "cz":
            expanded: list[_Op] = []
            for op in ops:
                if op.name == "cx":
                    c, t = op.qubits
                    expanded.extend([_Op("h", (t,), -1, op.block), _Op("cz", (c, t), -1, op.block),
                                     _Op("h", (t,), -1, op.block)])
                else:
                    expanded.append(op)
            ops = _cancel_h(expanded)
        self.ops = ops
        self.num_clbits = circuit.num_clbits

        # ---- logical model
        if self.layout is not None:
            lay = self.layout
            if lay.init != "plus":
                raise NotImplementedError("frame simulation supports circuits that start in |+>^k")
            self.k_logical = lay.k_code
            self.logical_clbits = list(lay.logical_clbits) + [-1] * (lay.k_code - lay.k)
            self.num_bare_clbits = lay.num_bare_clbits
        else:
            self.k_logical = len(compact)
            self.logical_clbits = [self._bare_clbits.get(i, -1) for i in range(self.k_logical)]
            self.num_bare_clbits = circuit.num_clbits
        self.logical = LogicalSimulator(self.k_logical, terms)
        self.num_terms = len(terms)
        idx = np.arange(1 << self.k_logical, dtype=np.int64)
        to_cl = np.zeros_like(idx)
        for i, cb in enumerate(self.logical_clbits):
            if cb >= 0:
                to_cl |= ((idx >> i) & 1) << cb
        self._to_clbits = to_cl

    # ---------------------------------------------------------------- counts
    @property
    def two_qubit_gates(self) -> int:
        return sum(1 for op in self.ops if len(op.qubits) == 2)

    @property
    def one_qubit_gates(self) -> int:
        return sum(1 for op in self.ops if len(op.qubits) == 1)

    def ideal_distribution(self) -> np.ndarray:
        """Noiseless logical outcome distribution over the bare classical bits."""
        return np.bincount(self._to_clbits, weights=self.logical.probabilities(),
                           minlength=1 << self.num_bare_clbits)

    # ------------------------------------------------------------ propagation
    def _propagate(self, rows: int, inject) -> tuple[np.ndarray, np.ndarray]:
        """Run frames for ``rows`` shots. ``inject(pos, op, X, Z)`` adds faults after op ``pos``.

        Returns ``(flipped, mask)``: which rotations were negated per row, and which
        classical bits were flipped per row (before readout errors).
        """
        X = np.zeros(rows, dtype=np.uint64)
        Z = np.zeros(rows, dtype=np.uint64)
        flipped = np.zeros((rows, max(self.num_terms, 1)), dtype=bool)
        one = np.uint64(1)

        def bit(arr: np.ndarray, q: int) -> np.ndarray:
            return (arr >> np.uint64(q)) & one

        for pos, op in enumerate(self.ops):
            name = op.name
            if name == "h":
                q = np.uint64(op.qubits[0])
                diff = ((X >> q) ^ (Z >> q)) & one
                X ^= diff << q
                Z ^= diff << q
            elif name == "cx":
                c, t = op.qubits
                X ^= bit(X, c) << np.uint64(t)
                Z ^= bit(Z, t) << np.uint64(c)
            elif name == "cz":
                a, b = op.qubits
                Z ^= bit(X, a) << np.uint64(b)
                Z ^= bit(X, b) << np.uint64(a)
            elif name == "rz":
                flipped[:, op.term] ^= bit(X, op.qubits[0]).astype(bool)
            elif name == "rx":
                flipped[:, op.term] ^= bit(Z, op.qubits[0]).astype(bool)
            elif name == "rzz":
                a, b = op.qubits
                flipped[:, op.term] ^= (bit(X, a) ^ bit(X, b)).astype(bool)
            elif name == "rxx":
                a, b = op.qubits
                flipped[:, op.term] ^= (bit(Z, a) ^ bit(Z, b)).astype(bool)
            elif name in ("s", "sdg"):
                q = op.qubits[0]
                Z ^= bit(X, q) << np.uint64(q)
            elif name in ("sx", "sxdg"):
                q = op.qubits[0]
                X ^= bit(Z, q) << np.uint64(q)
            # x, y, z: a Pauli gate leaves the frame unchanged
            inject(pos, op, X, Z)

        mask = np.zeros(rows, dtype=np.uint64)
        for qb, cb in self.measures:
            mask |= bit(X, qb) << np.uint64(cb)
        return flipped, mask

    # ------------------------------------------------------------- decoding
    def _checks(self, mask: np.ndarray) -> np.ndarray:
        """True where the flip pattern makes a check fire."""
        if self.layout is None:
            return np.zeros(len(mask), dtype=bool)
        lay = self.layout
        one = np.uint64(1)
        bad = np.zeros(len(mask), dtype=bool)
        if lay.check_z:
            par = np.zeros(len(mask), dtype=np.uint64)
            for j in lay.code_qubits:
                par ^= (mask >> np.uint64(j)) & one
            bad |= par.astype(bool)
        if lay.check_x and lay.sx_ancilla is not None:
            bad |= ((mask >> np.uint64(lay.sx_ancilla)) & one).astype(bool)
        for j in lay.flags:
            bad |= ((mask >> np.uint64(j)) & one).astype(bool)
        return bad

    def _logical_flips(self, mask: np.ndarray) -> np.ndarray:
        """Flip pattern on the logical qubits (bit ``i`` = logical qubit ``i``)."""
        one = np.uint64(1)
        out = np.zeros(len(mask), dtype=np.int64)
        if self.layout is None:
            for i, cb in enumerate(self.logical_clbits):
                if cb >= 0:
                    out |= ((mask >> np.uint64(cb)) & one).astype(np.int64) << i
            return out
        zb = (mask >> np.uint64(self.layout.bottom)) & one
        for i in range(self.k_logical):
            out |= (((mask >> np.uint64(i)) & one) ^ zb).astype(np.int64) << i
        return out

    # --------------------------------------------------------- single faults
    def _locations(self, noisy_rz: bool = True) -> list[tuple[int, _Op, str, tuple[tuple[int, int], ...]]]:
        """Every (op position, op, Pauli label, per-qubit (x, z) bits) single fault."""
        out = []
        for pos, op in enumerate(self.ops):
            if len(op.qubits) == 1:
                if op.name == "rz" and not noisy_rz:
                    continue
                for letter in "XYZ":
                    out.append((pos, op, letter, (_BITS[letter],)))
            else:
                for pa, pb in _TWO_QUBIT_PAULIS:
                    out.append((pos, op, pa + pb, (_BITS[pa], _BITS[pb])))
        return out

    def single_faults(self, energies: np.ndarray | None = None, *, noisy_rz: bool = True,
                      include_readout: bool = True) -> FaultReport:
        """Score every single fault exactly; same report type as the statevector version."""
        locs = self._locations(noisy_rz)
        rows = len(locs)
        fx = np.zeros(rows, dtype=np.uint64)
        fz = np.zeros(rows, dtype=np.uint64)
        at: dict[int, list[int]] = {}
        for r, (pos, op, _, bits) in enumerate(locs):
            for qb, (bx, bz) in zip(op.qubits, bits):
                fx[r] |= np.uint64(bx) << np.uint64(qb)
                fz[r] |= np.uint64(bz) << np.uint64(qb)
            at.setdefault(pos, []).append(r)
        at_idx = {pos: np.array(rs) for pos, rs in at.items()}

        def inject(pos: int, op: _Op, X: np.ndarray, Z: np.ndarray) -> None:
            rs = at_idx.get(pos)
            if rs is not None:
                X[rs] ^= fx[rs]
                Z[rs] ^= fz[rs]

        flipped, mask = self._propagate(rows, inject)
        ideal = self.ideal_distribution()
        e0 = float(ideal @ energies) if energies is not None else 0.0
        detected = self._checks(mask)
        lflip = self._logical_flips(mask)
        n_idx = np.arange(1 << self.k_logical, dtype=np.int64)

        def score(flip_row: np.ndarray, lf: int) -> tuple[float, float]:
            probs = self.logical.probabilities(flip_row)
            dist = np.bincount(self._to_clbits[n_idx ^ lf], weights=probs, minlength=len(ideal))
            tvd = 0.5 * float(np.abs(dist - ideal).sum())
            d_e = float(dist @ energies) - e0 if energies is not None else 0.0
            return tvd, d_e

        faults: list[Fault] = []
        counts = {"gate1": 0, "gate2": 0, "readout": 0}
        seen_pos: set[int] = set()
        memo: dict[tuple[bytes, int], tuple[float, float]] = {}
        for r, (pos, op, label, _) in enumerate(locs):
            kind = "gate1" if len(op.qubits) == 1 else "gate2"
            if pos not in seen_pos:
                seen_pos.add(pos)
                counts[kind] += 1
            if detected[r]:
                faults.append(Fault(kind, pos, op.name, op.block, op.qubits, label, 1.0, 0.0, 0.0))
                continue
            key = (np.packbits(flipped[r]).tobytes(), int(lflip[r]))
            if key not in memo:
                memo[key] = score(flipped[r], int(lflip[r]))
            tvd, d_e = memo[key]
            faults.append(Fault(kind, pos, op.name, op.block, op.qubits, label, 0.0, tvd, d_e))

        if include_readout:
            none = np.zeros(max(self.num_terms, 1), dtype=bool)
            for qb, cb in self.measures:
                counts["readout"] += 1
                m = np.array([np.uint64(1) << np.uint64(cb)], dtype=np.uint64)
                if self._checks(m)[0]:
                    faults.append(Fault("readout", -1, "measure", "measure", (qb,), "X", 1.0, 0.0, 0.0))
                else:
                    tvd, d_e = score(none, int(self._logical_flips(m)[0]))
                    faults.append(Fault("readout", -1, "measure", "measure", (qb,), "X", 0.0, tvd, d_e))
        return FaultReport(faults, ideal, self.num_qubits, counts, e0)

    # ---------------------------------------------------------------- sampling
    def sample(self, shots: int, noise: PauliNoise, seed: int | None = None) -> dict[str, int]:
        """Sample ``shots`` noisy shots; returns physical counts (Qiskit bit order)."""
        rng = np.random.default_rng(seed)
        weights = noise.weights()
        px = np.array([_BITS[a][0] for a, _ in _TWO_QUBIT_PAULIS], dtype=np.uint64)
        pz = np.array([_BITS[a][1] for a, _ in _TWO_QUBIT_PAULIS], dtype=np.uint64)
        qx = np.array([_BITS[b][0] for _, b in _TWO_QUBIT_PAULIS], dtype=np.uint64)
        qz = np.array([_BITS[b][1] for _, b in _TWO_QUBIT_PAULIS], dtype=np.uint64)

        def inject(pos: int, op: _Op, X: np.ndarray, Z: np.ndarray) -> None:
            if len(op.qubits) == 1:
                if noise.p1 <= 0 or (op.name == "rz" and not noise.noisy_rz):
                    return
                hit = np.flatnonzero(rng.random(shots) < noise.p1)
                if len(hit):
                    which = rng.integers(0, 3, size=len(hit))        # 0 X, 1 Y, 2 Z
                    q = np.uint64(op.qubits[0])
                    X[hit] ^= (which != 2).astype(np.uint64) << q
                    Z[hit] ^= (which != 0).astype(np.uint64) << q
            else:
                if noise.p2 <= 0:
                    return
                hit = np.flatnonzero(rng.random(shots) < noise.p2)
                if len(hit):
                    which = rng.choice(15, size=len(hit), p=weights)
                    a, b = np.uint64(op.qubits[0]), np.uint64(op.qubits[1])
                    X[hit] ^= (px[which] << a) | (qx[which] << b)
                    Z[hit] ^= (pz[which] << a) | (qz[which] << b)

        flipped, mask = self._propagate(shots, inject)
        if noise.p_meas > 0:
            for _, cb in self.measures:
                hit = rng.random(shots) < noise.p_meas
                mask[hit] ^= np.uint64(1) << np.uint64(cb)

        # ---- noiseless outcome of the sign-flipped logical circuit, per shot
        doomed = self._checks(mask)                     # discarded whatever the logical outcome
        logical = np.zeros(shots, dtype=np.int64)
        live = np.flatnonzero(~doomed)
        if len(live):
            packed = np.packbits(flipped[live], axis=1)
            _, first, inverse = np.unique(packed, axis=0, return_index=True, return_inverse=True)
            inverse = np.asarray(inverse).reshape(-1)
            for g, rep in enumerate(first):
                members = live[inverse == g]
                probs = self.logical.probabilities(flipped[live[rep]])
                logical[members] = rng.choice(len(probs), size=len(members), p=probs)

        # ---- physical bitstring of that outcome, then the frame's bit flips
        phys = np.zeros(shots, dtype=np.uint64)
        if self.layout is None:
            for i, cb in enumerate(self.logical_clbits):
                if cb >= 0:
                    phys |= ((logical >> i) & 1).astype(np.uint64) << np.uint64(cb)
        else:
            lay = self.layout
            gauge = rng.integers(0, 2, size=shots).astype(np.uint64)     # value of the b bit
            par = gauge.copy()
            for i in range(lay.k_code):
                bit_i = ((logical >> i) & 1).astype(np.uint64) ^ gauge
                phys |= bit_i << np.uint64(i)
                par ^= bit_i
            phys |= gauge << np.uint64(lay.bottom)
            top_bit = par if lay.ghz_init else rng.integers(0, 2, size=shots).astype(np.uint64)
            phys |= top_bit << np.uint64(lay.top)
        phys ^= mask
        values, counts = np.unique(phys, return_counts=True)
        width = self.num_clbits
        return {format(int(v), f"0{width}b"): int(c) for v, c in zip(values, counts)}


def _block_labels(circuit: QuantumCircuit) -> list[str]:
    from .sim import blocks_of

    return blocks_of(circuit)
