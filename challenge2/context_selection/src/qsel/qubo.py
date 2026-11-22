"""The weighted independent-set QUBO that every solver tier sees.

    min_{x in {0,1}^n}   - sum_i w_i x_i  +  P * sum_{(i,j) in E} x_i x_j ,
    w_i = r_i - theta

``r_i`` is the relevance of paragraph ``i`` to the question, ``E`` holds the
pairs whose overlap ``s_ij`` exceeds ``tau``, and ``P`` is larger than any
single weight so selecting both ends of a redundant edge never pays off.
Two-qubit gates appear only on ``E``, so ``tau`` sets circuit depth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np



@dataclass(frozen=True, eq=False)
class Qubo:
    """Weighted independent-set problem in QUBO form.

    Instances compare and hash by identity (the weights are an array).
    """

    weights: np.ndarray
    edges: tuple[tuple[int, int], ...]
    penalty: float
    meta: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        w = np.asarray(self.weights, dtype=float)
        object.__setattr__(self, "weights", w)
        seen = set()
        for i, j in self.edges:
            if not (0 <= i < j < len(w)):
                raise ValueError(f"edge {(i, j)} must satisfy 0 <= i < j < n")
            if (i, j) in seen:
                raise ValueError(f"duplicate edge {(i, j)}")
            seen.add((i, j))

    # ------------------------------------------------------------------ basics
    @property
    def n(self) -> int:
        return int(len(self.weights))

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def adjacency(self) -> np.ndarray:
        """Boolean ``(n, n)`` adjacency matrix of the redundancy graph."""
        adj = np.zeros((self.n, self.n), dtype=bool)
        for i, j in self.edges:
            adj[i, j] = adj[j, i] = True
        return adj

    def degrees(self) -> np.ndarray:
        return self.adjacency().sum(axis=1)

    # ---------------------------------------------------------------- energies
    def energy(self, x: Sequence[int]) -> float:
        """Objective value of one selection."""
        xv = np.asarray(x, dtype=float)
        if xv.shape != (self.n,):
            raise ValueError(f"selection must have length {self.n}")
        value = -float(self.weights @ xv)
        for i, j in self.edges:
            value += self.penalty * xv[i] * xv[j]
        return value

    def energies_all(self) -> np.ndarray:
        """Objective for all ``2**n`` selections (index bit ``i`` = ``x_i``).

        Built by doubling, one paragraph at a time, so memory stays at ``2**n`` floats.
        """
        if self.n > 26:
            raise ValueError("energies_all is exhaustive; n > 26 is too large")
        lower: dict[int, list[int]] = {}
        for i, j in self.edges:
            lower.setdefault(j, []).append(i)
        e = np.zeros(1)
        for k in range(self.n):
            idx = np.arange(len(e), dtype=np.int64)
            conflicts = np.zeros(len(e))
            for j in lower.get(k, ()):
                conflicts += (idx >> j) & 1
            e = np.concatenate([e, e - self.weights[k] + self.penalty * conflicts])
        return e

    def violations(self, x: Sequence[int]) -> int:
        """Number of redundant edges with both ends selected."""
        return int(sum(1 for i, j in self.edges if x[i] and x[j]))

    def is_feasible(self, x: Sequence[int]) -> bool:
        return self.violations(x) == 0

    def feasible_mask_all(self) -> np.ndarray:
        """Boolean mask over all ``2**n`` selections: no redundant pair chosen."""
        idx = np.arange(1 << self.n, dtype=np.int64)
        ok = np.ones(1 << self.n, dtype=bool)
        for i, j in self.edges:
            ok &= ((idx >> i) & (idx >> j) & 1) == 0
        return ok

    # ------------------------------------------------------------------- forms
    def to_ising(self) -> tuple[np.ndarray, dict[tuple[int, int], float], float]:
        """Ising form with ``x_i = (1 - z_i) / 2`` (``|1>`` means selected).

        Returns ``(h, J, const)`` with
        ``f(x) = const + sum_i h_i z_i + sum_{i<j} J_ij z_i z_j``.
        """
        deg = self.degrees().astype(float)
        h = self.weights / 2.0 - self.penalty * deg / 4.0
        J = {(i, j): self.penalty / 4.0 for i, j in self.edges}
        const = -float(self.weights.sum()) / 2.0 + self.penalty * self.num_edges / 4.0
        return h, J, const

    def to_superstaq(self) -> dict[tuple[int, ...], float]:
        """Dictionary form accepted by Superstaq's ``submit_qubo``."""
        out: dict[tuple[int, ...], float] = {}
        for i, w in enumerate(self.weights):
            if w != 0.0:
                out[(i,)] = -float(w)
        for i, j in self.edges:
            out[(i, j)] = float(self.penalty)
        return out

    def to_matrix(self) -> np.ndarray:
        """Upper-triangular ``Q`` with ``f(x) = x^T Q x``."""
        q = np.diag(-self.weights)
        for i, j in self.edges:
            q[i, j] = self.penalty
        return q


def normalize_relevance(r: np.ndarray, mode: str = "zscore") -> np.ndarray:
    """Put relevance on a per-question scale so one ``theta`` fits all questions."""
    r = np.asarray(r, dtype=float)
    if mode == "none":
        return r.copy()
    if mode == "zscore":
        sd = float(r.std())
        return (r - r.mean()) / sd if sd > 1e-12 else np.zeros_like(r)
    if mode == "minmax":
        span = float(r.max() - r.min())
        return (r - r.min()) / span if span > 1e-12 else np.zeros_like(r)
    raise ValueError(f"unknown normalisation {mode!r}")


def build_qubo(
    r: Sequence[float],
    s: np.ndarray,
    *,
    tau: float,
    theta: float = 0.0,
    penalty_scale: float = 1.5,
    normalize: str = "zscore",
    max_edges: int | None = None,
) -> Qubo:
    """Turn relevance ``r`` and overlap ``s`` into the selection QUBO.

    Args:
        r: relevance of each paragraph to the question.
        s: symmetric pairwise overlap; ``(i, j)`` is redundant when ``s_ij > tau``.
        tau: redundancy threshold. Lower ``tau`` means more edges, deeper circuits.
        theta: relevance floor; paragraphs below it get negative weight.
        penalty_scale: ``P = penalty_scale * max_i w_i`` (must exceed 1).
        normalize: per-question scaling of ``r`` (``zscore``, ``minmax``, ``none``).
        max_edges: keep only the strongest overlaps if more edges qualify.
    """
    r = np.asarray(r, dtype=float)
    s = np.asarray(s, dtype=float)
    n = len(r)
    if s.shape != (n, n):
        raise ValueError(f"overlap matrix must be {n}x{n}, got {s.shape}")
    if penalty_scale <= 1.0:
        raise ValueError("penalty_scale must exceed 1 so redundant pairs never pay off")

    w = normalize_relevance(r, normalize) - float(theta)
    cand = [(float(s[i, j]), i, j) for i in range(n) for j in range(i + 1, n) if s[i, j] > tau]
    cand.sort(reverse=True)
    if max_edges is not None:
        cand = cand[: int(max_edges)]
    edges = tuple(sorted((i, j) for _, i, j in cand))

    w_max = float(w.max()) if n else 0.0
    penalty = penalty_scale * (w_max if w_max > 0 else 1.0)
    meta = {
        "tau": float(tau),
        "theta": float(theta),
        "penalty_scale": float(penalty_scale),
        "normalize": normalize,
        "relevance": r.tolist(),
    }
    return Qubo(weights=w, edges=edges, penalty=float(penalty), meta=meta)


def random_qubo(
    n: int,
    *,
    edge_fraction: float = 0.2,
    theta: float = 0.0,
    penalty_scale: float = 1.5,
    dim: int = 8,
    seed: int = 0,
) -> Qubo:
    """Synthetic instance with the same structure as the real ones.

    Paragraph and question vectors are random unit vectors in ``dim`` dimensions;
    relevance and overlap are their cosine similarities. ``edge_fraction`` picks
    ``tau`` as a quantile so the expected edge count is controlled.
    """
    rng = np.random.default_rng(seed)
    q = rng.normal(size=dim)
    p = rng.normal(size=(n, dim)) + 0.6 * q
    q /= np.linalg.norm(q)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    r = p @ q
    s = p @ p.T
    off = s[np.triu_indices(n, 1)]
    if len(off) == 0 or edge_fraction <= 0:
        tau = 2.0
    else:
        tau = float(np.quantile(off, 1.0 - min(edge_fraction, 1.0))) - 1e-12
    return build_qubo(r, s, tau=tau, theta=theta, penalty_scale=penalty_scale)
