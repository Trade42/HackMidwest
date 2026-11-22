"""Independent helpers for verifying the iceberg package.

Nothing here imports iceberg.sim / iceberg.faults / iceberg.frames.  Only the
encoders, layout_of and (for comparison only) postselect are taken from the package.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# ----------------------------------------------------------------- gate matrices
# convention: for a gate on qubits [q0, q1] the matrix index is 2*bit(q1) + bit(q0)
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
PAULI = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


def kron2(a_q0, b_q1):
    """Two-qubit matrix with a on q0 (LSB) and b on q1."""
    return np.kron(b_q1, a_q0)


def gate_matrix(name: str, params) -> np.ndarray:
    th = float(params[0]) if params else 0.0
    if name == "h":
        return _H
    if name == "x":
        return _X
    if name == "y":
        return _Y
    if name == "z":
        return _Z
    if name == "id":
        return _I
    if name == "s":
        return np.diag([1, 1j]).astype(complex)
    if name == "sdg":
        return np.diag([1, -1j]).astype(complex)
    if name == "t":
        return np.diag([1, np.exp(1j * math.pi / 4)])
    if name == "tdg":
        return np.diag([1, np.exp(-1j * math.pi / 4)])
    if name == "p":
        return np.diag([1, np.exp(1j * th)])
    if name == "rz":
        return np.diag([np.exp(-0.5j * th), np.exp(0.5j * th)])
    if name == "rx":
        c, s = math.cos(th / 2), math.sin(th / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    if name == "sx":
        return 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex)
    if name == "rzz":
        return np.diag(np.exp(-0.5j * th * np.array([1, -1, -1, 1])))
    if name == "rxx":
        c, s = math.cos(th / 2), math.sin(th / 2)
        return c * np.eye(4, dtype=complex) - 1j * s * kron2(_X, _X)
    if name == "cx":
        m = np.zeros((4, 4), dtype=complex)
        for c in (0, 1):
            for t in (0, 1):
                m[2 * (t ^ c) + c, 2 * t + c] = 1
        return m
    if name == "cz":
        return np.diag([1, 1, 1, -1]).astype(complex)
    if name == "swap":
        m = np.zeros((4, 4), dtype=complex)
        for a in (0, 1):
            for b in (0, 1):
                m[2 * a + b, 2 * b + a] = 1
        return m
    raise ValueError(f"no matrix for gate {name!r}")


# ------------------------------------------------------------- circuit parsing
def parse(circ: QuantumCircuit):
    """Split a circuit into (ops, meas).

    ops: list of (data_index, name, qubit indices, params) for every unitary gate
         (barriers dropped), in order.
    meas: list of (data_index, qubit, clbit).
    Raises if a unitary gate follows a measurement on the same qubit.
    """
    ops, meas = [], []
    measured = set()
    for idx, inst in enumerate(circ.data):
        name = inst.operation.name
        qs = [circ.find_bit(q).index for q in inst.qubits]
        if name == "barrier":
            continue
        if name == "measure":
            meas.append((idx, qs[0], circ.find_bit(inst.clbits[0]).index))
            measured.add(qs[0])
            continue
        if measured & set(qs):
            raise ValueError("gate after measurement")
        params = [float(p) for p in inst.operation.params]
        ops.append((idx, name, qs, params))
    return ops, meas


# ---------------------------------------------------------------- simulator
def apply(psi: np.ndarray, mat: np.ndarray, qubits, n: int) -> np.ndarray:
    """Apply mat to the given qubits of an n-qubit state tensor (axis a = qubit n-1-a)."""
    if len(qubits) == 1:
        ax = n - 1 - qubits[0]
        out = np.tensordot(mat, psi, axes=([1], [ax]))
        return np.moveaxis(out, 0, ax)
    q0, q1 = qubits
    a0, a1 = n - 1 - q0, n - 1 - q1
    m4 = mat.reshape(2, 2, 2, 2)                      # [o1, o0, i1, i0]
    out = np.tensordot(m4, psi, axes=([2, 3], [a1, a0]))
    return np.moveaxis(out, [0, 1], [a1, a0])


def zero_state(n: int) -> np.ndarray:
    psi = np.zeros((2,) * n, dtype=complex)
    psi[(0,) * n] = 1.0
    return psi


def run_ops(ops, n: int, psi: np.ndarray | None = None) -> np.ndarray:
    if psi is None:
        psi = zero_state(n)
    for _, name, qs, params in ops:
        psi = apply(psi, gate_matrix(name, params), qs, n)
    return psi


def qubit_probs(psi: np.ndarray) -> np.ndarray:
    """Probabilities indexed by integer whose bit q is qubit q."""
    return (np.abs(psi) ** 2).reshape(-1)


def clbit_dist(qprobs: np.ndarray, meas, n_qubits: int, n_clbits: int) -> np.ndarray:
    """Distribution over classical registers (bit c of the index is classical bit c)."""
    idx = np.arange(1 << n_qubits, dtype=np.int64)
    cl = np.zeros_like(idx)
    for _, q, c in meas:
        # a later measurement into the same clbit overwrites the earlier one
        cl = (cl & ~(1 << c)) | (((idx >> q) & 1) << c)
    return np.bincount(cl, weights=qprobs, minlength=1 << n_clbits)


def exact_dist(circ: QuantumCircuit) -> np.ndarray:
    """Exact distribution over classical bits, own simulator."""
    ops, meas = parse(circ)
    psi = run_ops(ops, circ.num_qubits)
    return clbit_dist(qubit_probs(psi), meas, circ.num_qubits, circ.num_clbits)


def exact_dist_qiskit(circ: QuantumCircuit) -> np.ndarray:
    """Same thing through qiskit.quantum_info.Statevector (cross-check of my simulator)."""
    ops, meas = parse(circ)
    qc = circ.remove_final_measurements(inplace=False)
    # remove_final_measurements may drop idle qubits' registers? it only drops clbits.
    assert qc.num_qubits == circ.num_qubits
    sv = Statevector(qc)
    return clbit_dist(np.asarray(sv.probabilities()), meas, circ.num_qubits, circ.num_clbits)


# ----------------------------------------------------- keep rule, written by me
def my_postselect_dist(dist: np.ndarray, lay, *, force_parity: bool | None = None):
    """Apply the keep rule and decoding to an exact distribution over encoded clbits.

    Classical bit j reads qubit j.  Returns (logical unnormalised dist over bare
    clbits, kept mass).  The parity test is applied when the layout says the code
    checks S_Z (checks in {"both","z"}), unless force_parity overrides it.
    """
    nq = int(round(math.log2(len(dist))))
    m = np.arange(len(dist), dtype=np.int64)
    code = list(lay.code_qubits)
    parity = np.zeros_like(m)
    for j in code:
        parity ^= (m >> j) & 1
    use_parity = (lay.checks in ("both", "z")) if force_parity is None else force_parity
    keep = np.ones(len(dist), dtype=bool)
    if use_parity:
        keep &= parity == 0
    if lay.sx_ancilla is not None:
        keep &= ((m >> lay.sx_ancilla) & 1) == 0
    for f in lay.flags:
        keep &= ((m >> f) & 1) == 0
    b = (m >> lay.bottom) & 1
    logical = np.zeros_like(m)
    for i, cl in enumerate(lay.logical_clbits):
        if cl >= 0:
            logical |= (((m >> i) & 1) ^ b) << cl
    out = np.bincount(logical[keep], weights=dist[keep], minlength=1 << lay.num_bare_clbits)
    return out, float(dist[keep].sum())


def my_postselect_counts(counts: dict, lay, *, force_parity: bool | None = None):
    out, shots, kept = {}, 0, 0
    use_parity = (lay.checks in ("both", "z")) if force_parity is None else force_parity
    for key, c in counts.items():
        bits = key.replace(" ", "")[::-1]       # bits[j] = classical bit j
        shots += c
        if use_parity and sum(int(bits[j]) for j in lay.code_qubits) % 2:
            continue
        if lay.sx_ancilla is not None and bits[lay.sx_ancilla] == "1":
            continue
        if any(bits[f] == "1" for f in lay.flags):
            continue
        kept += c
        b = int(bits[lay.bottom])
        logical = ["0"] * lay.num_bare_clbits
        for i, cl in enumerate(lay.logical_clbits):
            if cl >= 0:
                logical[cl] = str(int(bits[i]) ^ b)
        lk = "".join(logical)[::-1]
        out[lk] = out.get(lk, 0) + c
    return out, shots, kept


def sample_counts(dist: np.ndarray, shots: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(shots, dist / dist.sum())
    width = int(round(math.log2(len(dist))))
    return {format(int(i), f"0{width}b"): int(draws[i]) for i in np.flatnonzero(draws)}


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def block_labels(enc: QuantumCircuit) -> list[str]:
    labels = []
    for name, cnt in enc.metadata["iceberg_blocks"]:
        labels.extend([name] * cnt)
    return labels
