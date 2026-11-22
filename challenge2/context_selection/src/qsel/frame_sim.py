"""Exact, fast sampling of Clifford + Pauli-rotation circuits under Pauli noise.

Why this exists: a 14-qubit Iceberg circuit with 4,000 noisy shots takes about a
minute as Aer statevector trajectories, because every shot re-simulates the whole
circuit. For the circuits used here that work is avoidable.

The compiled circuits contain only Clifford gates (``cz, sx, x, h, s, cx, swap``, and
rotations by multiples of pi/2) and Pauli rotations with arbitrary angles (``rz``,
also ``rx, ry, rzz, rxx``). For such circuits a Pauli fault does exactly two things:

1. it flips the sign of the angle of every later rotation whose generator it
   anticommutes with, and
2. it survives to the end of the circuit as a Pauli, whose X part flips measured bits.

So a noisy shot is the noiseless circuit with some rotation signs flipped, followed
by bit flips. Both effects are linear over GF(2) in the fault, which gives a lookup
table per fault location (one backward pass over the circuit).

The sign-flipped circuits are simulated without the full ``2**N`` state. Moving every
rotation generator to the end of the circuit turns the state into

    prod_j exp(-i s_j theta_j Q_j / 2) |S>,

with ``|S>`` the stabilizer state made by the Clifford gates alone and ``Q_j`` Paulis.
The state never leaves the span of the basis strings reachable from the support of
``|S>`` by the X parts of the ``Q_j``. For an Iceberg-encoded circuit on ``k`` logical
qubits that span has ``2**(k+1)`` strings however many physical qubits are used.

Nothing is approximated: sampling is exact for Pauli noise (depolarizing gate errors
and readout flips). Tests compare it with Aer's density-matrix simulator.

Relation to ``iceberg.frames.FrameSimulator``: that one is the project's main sampler.
It knows which logical term every rotation of an encoder-built circuit belongs to and
simulates at the logical level. This one assumes nothing about how the circuit was
built, so it also handles circuits a general transpiler has rewritten. It is used only
as an independent cross-check, through ``LocalNoisyBackend(method="frame")``, at sizes
Aer's trajectories are too slow for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit

_HALF_PI = math.pi / 2
_ROTATIONS = {"rz": "Z", "p": "Z", "rx": "X", "ry": "Y", "rzz": "ZZ", "rxx": "XX", "ryy": "YY"}
_CLIFFORD_1Q = {"h", "x", "y", "z", "s", "sdg", "sx", "sxdg", "id"}
_CLIFFORD_2Q = {"cx", "cz", "swap"}
# Clifford-angle rotations as primitive sequences (circuit order), index = k mod 4
_QUARTER_TURNS = {
    "Z": ((), ("s",), ("z",), ("sdg",)),
    "X": ((), ("sx",), ("x",), ("sxdg",)),
    "Y": ((), ("z", "h"), ("y",), ("h", "z")),
}
_ONE_QUBIT_PAULIS = ((1, 0), (1, 1), (0, 1))                                  # X, Y, Z as (x, z)
_TWO_QUBIT_PAULIS = tuple(p for p in product((0, 1), repeat=4) if any(p))     # (xa, za, xb, zb)


class UnsupportedCircuit(ValueError):
    """The circuit has a gate outside the Clifford + Pauli-rotation set."""


@dataclass(frozen=True)
class _Step:
    kind: str                         # "clifford" or "rotation"
    name: str
    qubits: tuple[int, ...]
    prims: tuple[tuple[str, tuple[int, ...]], ...] = ()   # primitive Clifford ops
    letters: str = ""                 # rotation generator, one letter per qubit
    theta: float = 0.0
    rotation_index: int = -1
    source: int = -1                  # index into circuit.data


# --------------------------------------------------------------- Pauli tableau
def _conj(prim: str, qs: tuple[int, ...], X: np.ndarray, Z: np.ndarray, r: np.ndarray) -> None:
    """Conjugate rows of Hermitian Paulis (sign ``(-1)**r``) by a primitive Clifford, in place."""
    a = qs[0]
    if prim == "h":
        r ^= X[:, a] & Z[:, a]
        X[:, a], Z[:, a] = Z[:, a].copy(), X[:, a].copy()
    elif prim == "s":
        r ^= X[:, a] & Z[:, a]
        Z[:, a] ^= X[:, a]
    elif prim == "z":
        r ^= X[:, a]
    elif prim == "x":
        r ^= Z[:, a]
    elif prim == "y":
        r ^= X[:, a] ^ Z[:, a]
    elif prim == "sdg":
        _conj("s", qs, X, Z, r)
        _conj("z", qs, X, Z, r)
    elif prim == "sx":
        _conj("h", qs, X, Z, r)
        _conj("s", qs, X, Z, r)
        _conj("h", qs, X, Z, r)
    elif prim == "sxdg":
        _conj("h", qs, X, Z, r)
        _conj("sdg", qs, X, Z, r)
        _conj("h", qs, X, Z, r)
    elif prim == "cx":
        b = qs[1]
        r ^= X[:, a] & Z[:, b] & (X[:, b] ^ Z[:, a] ^ 1)
        X[:, b] ^= X[:, a]
        Z[:, a] ^= Z[:, b]
    elif prim == "cz":
        b = qs[1]
        _conj("h", (b,), X, Z, r)
        _conj("cx", (a, b), X, Z, r)
        _conj("h", (b,), X, Z, r)
    elif prim == "swap":
        b = qs[1]
        X[:, [a, b]] = X[:, [b, a]]
        Z[:, [a, b]] = Z[:, [b, a]]
    elif prim == "id":
        pass
    else:  # pragma: no cover - guarded by the parser
        raise UnsupportedCircuit(prim)


def _pull_back(prim: str, qs: tuple[int, ...], M: np.ndarray, n: int) -> None:
    """Right-multiply the GF(2) map ``M`` by the symplectic action of a primitive Clifford.

    Columns ``0..n-1`` are X bits, ``n..2n-1`` are Z bits. After the call ``M`` maps a
    Pauli located before the gate to the same outputs.
    """
    a = qs[0]
    xa, za = a, n + a
    if prim == "h":
        M[:, [xa, za]] = M[:, [za, xa]]
    elif prim in ("s", "sdg"):
        M[:, xa] ^= M[:, za]
    elif prim in ("sx", "sxdg"):
        M[:, za] ^= M[:, xa]
    elif prim == "cx":
        xb, zb = qs[1], n + qs[1]
        M[:, xa] ^= M[:, xb]
        M[:, zb] ^= M[:, za]
    elif prim == "cz":
        xb, zb = qs[1], n + qs[1]
        M[:, xb] ^= M[:, za]
        M[:, xa] ^= M[:, zb]
    elif prim == "swap":
        b = qs[1]
        M[:, [a, b]] = M[:, [b, a]]
        M[:, [n + a, n + b]] = M[:, [n + b, n + a]]
    # x, y, z, id change signs only


def _parity(values: np.ndarray, mask: int) -> np.ndarray:
    """Parity of ``values & mask`` for an int64 array."""
    out = np.zeros(values.shape, dtype=np.int64)
    bit = 0
    while mask >> bit:
        if (mask >> bit) & 1:
            out ^= (values >> bit) & 1
        bit += 1
    return out


class GenericFrameSampler:
    """Pauli-noise sampler for one circuit. Build once, sample many times."""

    def __init__(self, circuit: QuantumCircuit, *, noiseless: Sequence[str] = ("rz", "p", "id"),
                 max_support: int = 1 << 20, max_cells: int = 1 << 25):
        self.num_qubits = n = circuit.num_qubits
        self.max_cells = int(max_cells)
        self.num_clbits = circuit.num_clbits
        if n > 62:
            raise UnsupportedCircuit("more than 62 qubits")
        self.noiseless = set(noiseless)
        self.steps: list[_Step] = []
        self.measures: list[tuple[int, int]] = []
        self._parse(circuit)
        self.num_rotations = sum(1 for s in self.steps if s.kind == "rotation")
        self.thetas = np.array([s.theta for s in self.steps if s.kind == "rotation"], dtype=float)
        self._fault_tables()
        self._final_frame_generators()
        self._stabilizer_support(circuit, max_support)

    # ---------------------------------------------------------------- parsing
    def _parse(self, circuit: QuantumCircuit) -> None:
        measured: set[int] = set()
        rot = 0
        for pos, inst in enumerate(circuit.data):
            op = inst.operation
            name = op.name
            qs = tuple(circuit.find_bit(q).index for q in inst.qubits)
            if name in ("barrier", "delay"):
                continue
            if name == "measure":
                self.measures.append((qs[0], circuit.find_bit(inst.clbits[0]).index))
                measured.add(qs[0])
                continue
            if measured & set(qs):
                raise UnsupportedCircuit("gates after a measurement")
            if name in _CLIFFORD_1Q or name in _CLIFFORD_2Q:
                self.steps.append(_Step("clifford", name, qs, prims=((name, qs),), source=pos))
            elif name in _ROTATIONS:
                letters = _ROTATIONS[name]
                try:
                    theta = float(op.params[0])
                except TypeError as exc:
                    raise UnsupportedCircuit("unbound parameter") from exc
                turns = theta / _HALF_PI
                if abs(turns - round(turns)) < 1e-9:
                    k = int(round(turns)) % 4
                    if len(letters) == 1:
                        prims = tuple((p, qs) for p in _QUARTER_TURNS[letters][k])
                    elif k == 0:
                        prims = ()
                    elif k == 2:
                        prims = tuple((letters[0].lower(), (q,)) for q in qs)
                    else:
                        raise UnsupportedCircuit(f"{name} by an odd multiple of pi/2")
                    self.steps.append(_Step("clifford", name, qs, prims=prims, source=pos))
                else:
                    self.steps.append(_Step("rotation", name, qs, letters=letters, theta=theta,
                                            rotation_index=rot, source=pos))
                    rot += 1
            else:
                raise UnsupportedCircuit(f"gate {name!r}")
        if not self.measures:
            raise UnsupportedCircuit("no measurements")

    # ------------------------------------------------------- fault lookup table
    def _fault_tables(self) -> None:
        """For every noisy gate, the effect of each Pauli fault placed right after it.

        Effect vector layout: ``N`` bits of final X frame, then one bit per rotation
        saying whether its sign flips.
        """
        n, R = self.num_qubits, self.num_rotations
        M = np.zeros((n + R, 2 * n), dtype=np.uint8)
        M[np.arange(n), np.arange(n)] = 1
        self.loc_kind: list[int] = []          # 1 or 2 qubits
        self.loc_table: list[np.ndarray] = []  # (3 or 15, n + R)
        self.loc_step: list[int] = []
        for idx in range(len(self.steps) - 1, -1, -1):
            step = self.steps[idx]
            if step.name not in self.noiseless:
                cols = [c for q in step.qubits for c in (q, n + q)]
                sub = M[:, cols]
                paulis = _ONE_QUBIT_PAULIS if len(step.qubits) == 1 else _TWO_QUBIT_PAULIS
                table = np.zeros((len(paulis), n + R), dtype=np.uint8)
                for i, bits in enumerate(paulis):
                    for col, bit in enumerate(bits):
                        if bit:
                            table[i] ^= sub[:, col]
                self.loc_kind.append(len(step.qubits))
                self.loc_table.append(table)
                self.loc_step.append(idx)
            if step.kind == "rotation":
                row = n + step.rotation_index
                for letter, q in zip(step.letters, step.qubits):
                    if letter in "ZY":
                        M[row, q] ^= 1
                    if letter in "XY":
                        M[row, n + q] ^= 1
            else:
                for prim, qs in reversed(step.prims):
                    _pull_back(prim, qs, M, n)
        self.loc_kind.reverse()
        self.loc_table.reverse()
        self.loc_step.reverse()

    # ------------------------------------------- rotation generators at the end
    def _final_frame_generators(self) -> None:
        n, R = self.num_qubits, self.num_rotations
        X = np.zeros((R, n), dtype=np.uint8)
        Z = np.zeros((R, n), dtype=np.uint8)
        r = np.zeros(R, dtype=np.uint8)
        active = 0
        for step in self.steps:
            if step.kind == "rotation":
                j = step.rotation_index
                for letter, q in zip(step.letters, step.qubits):
                    X[j, q] = letter in "XY"
                    Z[j, q] = letter in "ZY"
                active = j + 1
            elif active:
                for prim, qs in step.prims:
                    _conj(prim, qs, X[:active], Z[:active], r[:active])
        weights = 1 << np.arange(n, dtype=np.int64)
        self.gen_x = (X.astype(np.int64) @ weights).astype(np.int64)
        self.gen_z = (Z.astype(np.int64) @ weights).astype(np.int64)
        n_y = (X & Z).sum(axis=1)
        self.gen_phase = ((-1.0) ** r) * (1j ** n_y)

    # ------------------------------------------------------- stabilizer support
    def _clifford_state(self, circuit: QuantumCircuit) -> np.ndarray:
        """Dense state of the circuit with every non-Clifford rotation removed."""
        from iceberg.sim import apply, zero_state

        n = self.num_qubits
        psi = zero_state(n)
        for step in self.steps:
            if step.kind == "clifford":
                op = circuit.data[step.source].operation
                psi = apply(psi, np.asarray(op.to_matrix(), dtype=complex), step.qubits, n)
        return psi

    def _stabilizer_support(self, circuit: QuantumCircuit, max_support: int) -> None:
        psi = self._clifford_state(circuit)
        support = np.flatnonzero(np.abs(psi) > 1e-9).astype(np.int64)
        masks = [int(m) for m in set(self.gen_x.tolist()) if m]
        changed = True
        while changed:
            changed = False
            for m in masks:
                moved = support ^ m
                if not np.isin(moved, support, assume_unique=True).all():
                    support = np.union1d(support, moved)
                    changed = True
                    if len(support) > max_support:
                        raise UnsupportedCircuit("reachable basis is too large")
        self.support = support
        self.amps0 = psi[support].astype(complex)
        self.perm = [np.searchsorted(support, support ^ int(m)) for m in self.gen_x]
        self.phase = [self.gen_phase[j] * (1 - 2 * _parity(support, int(self.gen_z[j])))
                      for j in range(self.num_rotations)]
        # classical bits of every reachable basis string, before any frame flip
        cl = np.zeros(len(support), dtype=np.int64)
        for q, c in self.measures:
            cl |= ((support >> q) & 1) << c
        self.support_clbits = cl
        self.meas_qubit_weights = [(q, c) for q, c in self.measures]

    # ------------------------------------------------------------- simulation
    def _kernels(self) -> None:
        """Per-rotation update vectors and the noiseless state after each rotation."""
        if hasattr(self, "_ideal_states"):
            return
        R, m = self.num_rotations, len(self.support)
        if (R + 1) * m > self.max_cells:
            raise UnsupportedCircuit("circuit is too large for the frame sampler")
        cos, sin = np.cos(self.thetas / 2), np.sin(self.thetas / 2)
        self._cos = cos.astype(np.float32)
        self._diag = [bool(self.gen_x[j] == 0) for j in range(R)]
        self._plus: list[np.ndarray | None] = []
        self._minus: list[np.ndarray | None] = []
        self._hop: list[np.ndarray | None] = []
        for j in range(R):
            if self._diag[j]:
                lam = np.real(self.phase[j])
                self._plus.append((cos[j] - 1j * sin[j] * lam).astype(np.complex64))
                self._minus.append((cos[j] + 1j * sin[j] * lam).astype(np.complex64))
                self._hop.append(None)
            else:
                # (Q psi)[t] = phase[perm[t]] * psi[perm[t]]  (perm is an involution)
                self._plus.append(None)
                self._minus.append(None)
                self._hop.append((-1j * sin[j] * self.phase[j][self.perm[j]]).astype(np.complex64))
        states = np.empty((R + 1, m), dtype=np.complex64)
        psi = self.amps0.astype(np.complex64)
        states[0] = psi
        for j in range(R):
            if self._diag[j]:
                psi = psi * self._plus[j]
            else:
                psi = self._cos[j] * psi + psi[self.perm[j]] * self._hop[j]
            states[j + 1] = psi
        self._ideal_states = states

    def probabilities(self, flips: np.ndarray | None = None, *, chunk: int | None = None) -> np.ndarray:
        """Distribution over the reachable basis for each row of rotation sign flips.

        ``flips`` has one row per pattern and one 0/1 column per rotation (1 = the
        rotation runs with its angle negated). Rows share the noiseless prefix up to
        their first flip, so a fault late in the circuit costs little.
        """
        self._kernels()
        R, m = self.num_rotations, len(self.support)
        if chunk is None:
            chunk = max(1, min(256, (1 << 21) // m))    # about 16 MB of state per chunk
        if flips is None:
            flips = np.zeros((1, R), dtype=np.uint8)
        flips = np.atleast_2d(np.asarray(flips, dtype=np.uint8))
        count = flips.shape[0]
        first = np.where(flips.any(axis=1), flips.argmax(axis=1), R) if R else np.zeros(count, dtype=int)
        order = np.argsort(first, kind="stable")
        out = np.empty((count, m), dtype=np.float64)
        for lo in range(0, count, chunk):
            rows = order[lo:lo + chunk]
            f = first[rows]
            psi = self._ideal_states[f].copy()
            for j in range(int(f[0]), R):
                a = int(np.searchsorted(f, j, side="right"))
                if a == 0:
                    continue
                flipped = flips[rows[:a], j].astype(bool)[:, None]
                sub = psi[:a]
                if self._diag[j]:
                    sub *= np.where(flipped, self._minus[j], self._plus[j])
                else:
                    hop = sub[:, self.perm[j]] * self._hop[j]
                    hop *= np.where(flipped, np.float32(-1.0), np.float32(1.0))
                    sub *= self._cos[j]
                    sub += hop
            p = (psi.real.astype(np.float64) ** 2) + (psi.imag.astype(np.float64) ** 2)
            out[rows] = p / p.sum(axis=1, keepdims=True)
        return out

    def ideal_probabilities(self) -> np.ndarray:
        """Noiseless distribution over the classical bits (index = integer value)."""
        p = self.probabilities()[0]
        return np.bincount(self.support_clbits, weights=p, minlength=1 << self.num_clbits)

    # ----------------------------------------------------------------- sampling
    def sample(self, shots: int, *, p1: float = 0.0, p2: float = 0.0, p_meas: float = 0.0,
               seed: int | None = None) -> dict[str, int]:
        """Draw ``shots`` noisy shots.

        ``p1`` / ``p2``: probability of a non-identity Pauli after each noisy one- or
        two-qubit gate (uniform over the 3 or 15 Paulis). ``p_meas``: probability that a
        measured bit is reported flipped.
        """
        rng = np.random.default_rng(seed)
        n, R = self.num_qubits, self.num_rotations
        shots = int(shots)
        acc = np.zeros((shots, n + R), dtype=np.uint8)
        for kind, table in zip(self.loc_kind, self.loc_table):
            p = p1 if kind == 1 else p2
            if p <= 0:
                continue
            hit = np.flatnonzero(rng.random(shots) < p)
            if len(hit):
                acc[hit] ^= table[rng.integers(0, len(table), size=len(hit))]

        # group shots by rotation-sign pattern and simulate each pattern once
        if R:
            patterns, inverse = np.unique(acc[:, n:], axis=0, return_inverse=True)
            inverse = np.asarray(inverse).reshape(-1)
            probs = self.probabilities(patterns)
        else:
            inverse = np.zeros(shots, dtype=np.int64)
            probs = self.probabilities()
        cdf = np.cumsum(probs, axis=1)
        cdf[:, -1] = 1.0
        u = rng.random(shots)
        picks = np.empty(shots, dtype=np.int64)
        order = np.argsort(inverse, kind="stable")
        bounds = np.searchsorted(inverse[order], np.arange(len(cdf) + 1))
        for g in range(len(cdf)):
            rows = order[bounds[g]:bounds[g + 1]]
            if len(rows):
                picks[rows] = np.searchsorted(cdf[g], u[rows], side="right")
        picks = np.minimum(picks, len(self.support) - 1)

        bits = self.support_clbits[picks]
        frame = acc[:, :n]
        for q, c in self.measures:
            flip = frame[:, q].astype(np.int64)
            if p_meas > 0:
                flip ^= (rng.random(shots) < p_meas).astype(np.int64)
            bits ^= flip << c
        values, counts = np.unique(bits, return_counts=True)
        width = self.num_clbits
        return {format(int(v), f"0{width}b"): int(c) for v, c in zip(values, counts)}
