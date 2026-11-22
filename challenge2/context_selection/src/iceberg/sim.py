"""A small exact statevector engine used to check the encoder.

It exists so the checks do not depend on any simulator's conventions: circuits are
flattened to a list of unitaries, the state is stepped gate by gate, and prefix
states can be reused, which is what makes exhaustive fault enumeration cheap.

Index convention matches Qiskit: basis state ``sum_q bit_q << q``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit

PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


@dataclass(frozen=True)
class Op:
    matrix: np.ndarray
    qubits: tuple[int, ...]
    name: str
    block: str
    index: int          #: position in the source circuit's instruction list


@dataclass(frozen=True)
class Program:
    """A circuit flattened for simulation."""

    num_qubits: int
    ops: tuple[Op, ...]
    measures: tuple[tuple[int, int], ...]   #: (qubit, clbit) pairs
    num_clbits: int

    def clbit_index(self) -> np.ndarray:
        """For every basis state, the integer formed by the measured classical bits."""
        idx = np.arange(1 << self.num_qubits, dtype=np.int64)
        out = np.zeros_like(idx)
        for qb, cb in self.measures:
            out |= ((idx >> qb) & 1) << cb
        return out


def blocks_of(circuit: QuantumCircuit) -> list[str]:
    """Per-instruction block labels written by the encoder (run-length encoded)."""
    rle = (circuit.metadata or {}).get("iceberg_blocks")
    if not rle:
        return ["circuit"] * len(circuit.data)
    labels: list[str] = []
    for name, count in rle:
        labels.extend([name] * int(count))
    if len(labels) != len(circuit.data):
        return ["circuit"] * len(circuit.data)
    return labels


def to_program(circuit: QuantumCircuit) -> Program:
    """Flatten a circuit with terminal measurements into a :class:`Program`."""
    labels = blocks_of(circuit)
    ops: list[Op] = []
    measures: list[tuple[int, int]] = []
    for pos, inst in enumerate(circuit.data):
        name = inst.operation.name
        qubits = tuple(circuit.find_bit(q).index for q in inst.qubits)
        if name in ("barrier", "delay"):
            continue
        if name == "measure":
            measures.append((qubits[0], circuit.find_bit(inst.clbits[0]).index))
            continue
        if any(qb in {m[0] for m in measures} for qb in qubits):
            raise ValueError("only terminal measurements are supported")
        if len(qubits) > 2:
            raise ValueError(f"gate {name!r} acts on more than two qubits; transpile first")
        ops.append(Op(np.asarray(inst.operation.to_matrix(), dtype=complex), qubits, name, labels[pos], pos))
    return Program(circuit.num_qubits, tuple(ops), tuple(measures), circuit.num_clbits)


def compact(circuit: QuantumCircuit) -> QuantumCircuit:
    """Drop idle qubits (useful after transpiling onto a larger device)."""
    active = sorted({circuit.find_bit(q).index for inst in circuit.data for q in inst.qubits
                     if inst.operation.name != "barrier"})
    remap = {old: new for new, old in enumerate(active)}
    out = QuantumCircuit(len(active), circuit.num_clbits, name=circuit.name)
    for inst in circuit.data:
        if inst.operation.name == "barrier":
            continue
        qs = [out.qubits[remap[circuit.find_bit(q).index]] for q in inst.qubits]
        cs = [out.clbits[circuit.find_bit(c).index] for c in inst.clbits]
        out.append(inst.operation, qs, cs)
    out.metadata = dict(circuit.metadata or {})
    out.metadata.pop("iceberg_blocks", None)
    return out


def zero_state(n: int) -> np.ndarray:
    psi = np.zeros(1 << n, dtype=complex)
    psi[0] = 1.0
    return psi


def apply(psi: np.ndarray, matrix: np.ndarray, qubits: tuple[int, ...], n: int) -> np.ndarray:
    """Apply a 1- or 2-qubit unitary (Qiskit ordering: first qubit least significant)."""
    k = len(qubits)
    axes = [n - 1 - qubits[k - 1 - a] for a in range(k)]
    tens = np.tensordot(matrix.reshape((2,) * (2 * k)), psi.reshape((2,) * n),
                        axes=(list(range(k, 2 * k)), axes))
    return np.moveaxis(tens, list(range(k)), axes).reshape(-1)


def apply_pauli(psi: np.ndarray, pauli: str, qubits: tuple[int, ...], n: int) -> np.ndarray:
    """Apply a Pauli string (one letter per qubit in ``qubits``)."""
    for letter, qb in zip(pauli, qubits):
        if letter != "I":
            psi = apply(psi, PAULI[letter], (qb,), n)
    return psi


def run(program: Program, psi: np.ndarray | None = None, start: int = 0) -> np.ndarray:
    """Evolve ``psi`` through ``program.ops[start:]``."""
    n = program.num_qubits
    psi = zero_state(n) if psi is None else psi
    for op in program.ops[start:]:
        psi = apply(psi, op.matrix, op.qubits, n)
    return psi


def prefix_states(program: Program) -> list[np.ndarray]:
    """``states[g]`` is the state before op ``g``; the last entry is the final state."""
    n = program.num_qubits
    psi = zero_state(n)
    states = [psi]
    for op in program.ops:
        psi = apply(psi, op.matrix, op.qubits, n)
        states.append(psi)
    return states


def outcome_probabilities(program: Program, psi: np.ndarray) -> np.ndarray:
    """Distribution over the measured classical bits."""
    probs = np.abs(psi) ** 2
    return np.bincount(program.clbit_index(), weights=probs, minlength=1 << program.num_clbits)
