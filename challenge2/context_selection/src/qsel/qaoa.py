"""QAOA for the selection QUBO: the circuit and an exact ideal-tier simulator.

Cost Hamiltonian (Ising form of the QUBO, ``|1>`` = selected):

    H_C = sum_i h_i Z_i + sum_{(i,j) in E} J_ij Z_i Z_j

Circuit: ``H`` on every qubit, then ``p`` layers of ``exp(-i gamma H_C)`` followed by
``exp(-i beta sum_i X_i)``. Only gates ``h, rz, rzz, rx, measure`` are emitted, which
is exactly the set the Iceberg encoder maps to logical gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from .qubo import Qubo


@dataclass(frozen=True)
class Angles:
    """QAOA angles in problem units (``gammas`` multiply ``H_C`` as built)."""

    gammas: tuple[float, ...]
    betas: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "gammas", tuple(float(g) for g in self.gammas))
        object.__setattr__(self, "betas", tuple(float(b) for b in self.betas))
        if len(self.gammas) != len(self.betas):
            raise ValueError("gammas and betas must have the same length")

    @property
    def p(self) -> int:
        return len(self.gammas)

    def to_dict(self) -> dict:
        return {"gammas": list(self.gammas), "betas": list(self.betas)}

    @classmethod
    def from_dict(cls, d: dict) -> "Angles":
        return cls(tuple(d["gammas"]), tuple(d["betas"]))


def qaoa_circuit(qubo: Qubo, angles: Angles, *, measure: bool = True) -> QuantumCircuit:
    """Bare QAOA circuit. Qubit ``i`` and classical bit ``i`` are paragraph ``i``."""
    n = qubo.n
    h, J, _ = qubo.to_ising()
    q = QuantumRegister(n, "q")
    qc = QuantumCircuit(q, name=f"qaoa_p{angles.p}")
    if measure:
        qc.add_register(ClassicalRegister(n, "c"))
    qc.h(q)
    for gamma, beta in zip(angles.gammas, angles.betas):
        qc.barrier()
        for (i, j), jij in J.items():
            qc.rzz(2.0 * gamma * jij, q[i], q[j])
        for i in range(n):
            if h[i] != 0.0:
                qc.rz(2.0 * gamma * float(h[i]), q[i])
        qc.barrier()
        for i in range(n):
            qc.rx(2.0 * beta, q[i])
    if measure:
        qc.barrier()
        qc.measure(q, qc.clbits)
    qc.metadata = {"tier": "bare", "n": n, "p": angles.p, "edges": qubo.num_edges}
    return qc


class QaoaSimulator:
    """Exact statevector QAOA for one QUBO (the ideal tier).

    The state lives on ``2**n`` amplitudes indexed by the selection integer, so
    probabilities line up with :meth:`Qubo.energies_all`.
    """

    def __init__(self, qubo: Qubo):
        if qubo.n > 22:
            raise ValueError("ideal simulation is limited to 22 logical qubits")
        self.qubo = qubo
        self.n = qubo.n
        self.energies = qubo.energies_all()
        self.e_min = float(self.energies.min())
        self.e_max = float(self.energies.max())
        h, J, _ = qubo.to_ising()
        coeffs = np.concatenate([np.abs(h), np.abs(np.fromiter(J.values(), dtype=float, count=len(J)))])
        coeffs = coeffs[coeffs > 0]
        #: natural unit for gamma: 1 / (RMS Ising coefficient)
        self.scale = float(np.sqrt(np.mean(coeffs**2))) if len(coeffs) else 1.0

    # ------------------------------------------------------------- evolution
    def _mix(self, psi: np.ndarray, beta: float) -> np.ndarray:
        """Apply ``exp(-i beta X)`` to every qubit. ``psi`` may be batched on axis 0."""
        c, s = np.cos(beta), -1j * np.sin(beta)
        lead = psi.shape[:-1]
        for qb in range(self.n):
            view = psi.reshape(*lead, -1, 2, 1 << qb)
            a = view[..., 0, :].copy()
            b = view[..., 1, :]
            view[..., 0, :] = c * a + s * b
            view[..., 1, :] = s * a + c * b
        return psi

    def state(self, gammas: Sequence[float], betas: Sequence[float]) -> np.ndarray:
        psi = np.full(1 << self.n, (1 << self.n) ** -0.5, dtype=complex)
        for gamma, beta in zip(gammas, betas):
            psi = psi * np.exp(-1j * gamma * self.energies)
            psi = self._mix(psi, beta)
        return psi

    def probabilities(self, angles: Angles) -> np.ndarray:
        p = np.abs(self.state(angles.gammas, angles.betas)) ** 2
        return p / p.sum()

    def expectation(self, gammas: Sequence[float], betas: Sequence[float]) -> float:
        psi = self.state(gammas, betas)
        return float(np.real(np.vdot(psi, self.energies * psi)))

    def grid_expectation(self, gammas: np.ndarray, betas: np.ndarray) -> np.ndarray:
        """p = 1 expectation on a full ``(len(gammas), len(betas))`` grid."""
        base = np.exp(-1j * np.outer(gammas, self.energies)) * (1 << self.n) ** -0.5
        out = np.empty((len(gammas), len(betas)))
        for bi, beta in enumerate(betas):
            psi = self._mix(base.copy(), float(beta))
            out[:, bi] = np.real(np.sum(np.conj(psi) * self.energies[None, :] * psi, axis=1))
        return out

    # --------------------------------------------------------------- metrics
    def approx_ratio(self, mean_energy: float) -> float:
        """Normalised approximation ratio in [0, 1]: 1 is the optimum, 0 the worst."""
        span = self.e_max - self.e_min
        return float((self.e_max - mean_energy) / span) if span > 0 else 1.0

    def optimum_mask(self, tol: float = 1e-9) -> np.ndarray:
        return self.energies <= self.e_min + tol

    def sample(self, angles: Angles, shots: int, seed: int | None = None) -> dict[str, int]:
        """Draw ``shots`` samples from the exact distribution as Qiskit-style counts."""
        rng = np.random.default_rng(seed)
        probs = self.probabilities(angles)
        draws = rng.multinomial(shots, probs)
        nz = np.flatnonzero(draws)
        return {format(int(k), f"0{self.n}b"): int(draws[k]) for k in nz}
