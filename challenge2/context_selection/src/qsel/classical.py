"""Classical selectors and the greedy repair used by every sampling tier."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .bits import Selection, from_indices, index_to_x, selected
from .qubo import Qubo


# --------------------------------------------------------------------- repair
def repair(qubo: Qubo, x: Sequence[int], *, min_select: int = 1) -> Selection:
    """Make a sample valid without improving it.

    While a redundant pair is selected, drop its lower-weight end (ties drop the
    higher index). Nothing is added, so repair cannot hide a bad sampler, with
    one exception: if the result has fewer than ``min_select`` paragraphs, the
    heaviest unselected compatible paragraphs are added so the language model
    always has something to read.
    """
    xs = [int(b) for b in x]
    w = qubo.weights
    while True:
        bad = [(i, j) for i, j in qubo.edges if xs[i] and xs[j]]
        if not bad:
            break
        i, j = min(bad, key=lambda e: (min(w[e[0]], w[e[1]]), e))
        drop = i if (w[i], -i) < (w[j], -j) else j
        xs[drop] = 0
    if sum(xs) < min_select:
        adj = qubo.adjacency()
        for i in np.argsort(-w, kind="stable"):
            if sum(xs) >= min_select:
                break
            if not xs[i] and not any(xs[j] and adj[i, j] for j in range(qubo.n)):
                xs[int(i)] = 1
    return tuple(xs)


# ---------------------------------------------------------------- brute force
def brute_force(qubo: Qubo, top: int = 1) -> list[tuple[Selection, float]]:
    """Exact enumeration. Returns the ``top`` lowest-energy valid selections."""
    e = qubo.energies_all()
    ok = qubo.feasible_mask_all()
    idx = np.flatnonzero(ok)
    order = idx[np.lexsort((idx, e[idx]))][:top]
    return [(index_to_x(int(k), qubo.n), float(e[k])) for k in order]


def optimum(qubo: Qubo) -> tuple[Selection, float]:
    """The exact optimum (lowest index wins ties)."""
    return brute_force(qubo, top=1)[0]


# --------------------------------------------------------------------- greedy
def greedy(qubo: Qubo, *, forced_first: int | None = None) -> Selection:
    """MMR-style greedy: add the heaviest compatible paragraph while it helps."""
    adj = qubo.adjacency()
    chosen: list[int] = []
    order = [int(i) for i in np.argsort(-qubo.weights, kind="stable")]
    if forced_first is not None:
        order = [forced_first] + [i for i in order if i != forced_first]
    for pos, i in enumerate(order):
        forced = forced_first is not None and pos == 0
        if qubo.weights[i] <= 0 and not forced:
            continue
        if any(adj[i, j] for j in chosen):
            continue
        chosen.append(i)
    return from_indices(chosen, qubo.n)


def greedy_candidates(qubo: Qubo, top: int = 3) -> list[tuple[Selection, float]]:
    """Up to ``top`` distinct greedy selections, each forced to start elsewhere."""
    seen: dict[Selection, float] = {}
    x0 = greedy(qubo)
    seen[x0] = qubo.energy(x0)
    for i in np.argsort(-qubo.weights, kind="stable"):
        x = greedy(qubo, forced_first=int(i))
        seen.setdefault(x, qubo.energy(x))
    ranked = sorted(seen.items(), key=lambda kv: (kv[1], kv[0]))
    best_first = [(x0, seen[x0])] + [kv for kv in ranked if kv[0] != x0]
    return best_first[:top]


# ---------------------------------------------------------------------- top-k
def top_k(relevance: Sequence[float], k: int = 3) -> Selection:
    """The ``k`` most relevant paragraphs, ignoring redundancy."""
    r = np.asarray(relevance, dtype=float)
    return from_indices(np.argsort(-r, kind="stable")[:k].tolist(), len(r))


# -------------------------------------------------------- simulated annealing
def simulated_annealing(
    qubo: Qubo,
    *,
    reads: int = 64,
    sweeps: int = 200,
    t_start: float | None = None,
    t_end: float | None = None,
    seed: int = 0,
) -> dict[Selection, int]:
    """Local single-flip simulated annealing; returns sample counts.

    This is the offline stand-in for Superstaq's ``submit_qubo`` annealer.
    """
    rng = np.random.default_rng(seed)
    n = qubo.n
    w = qubo.weights
    adj = qubo.adjacency().astype(float)
    scale = float(max(np.abs(w).max(), qubo.penalty, 1e-9))
    t0 = t_start if t_start is not None else 2.0 * scale
    t1 = t_end if t_end is not None else 0.02 * scale
    temps = np.geomspace(t0, t1, sweeps)
    counts: dict[Selection, int] = {}
    for _ in range(reads):
        x = rng.integers(0, 2, size=n).astype(float)
        for temp in temps:
            for i in rng.permutation(n):
                # energy change of flipping bit i
                delta = (1.0 - 2.0 * x[i]) * (-w[i] + qubo.penalty * float(adj[i] @ x))
                if delta <= 0 or rng.random() < np.exp(-delta / temp):
                    x[i] = 1.0 - x[i]
        key = tuple(int(b) for b in x)
        counts[key] = counts.get(key, 0) + 1
    return counts


def describe(qubo: Qubo, x: Sequence[int]) -> dict:
    """Small summary of one selection, used in logs and the demo."""
    return {
        "selected": selected(x),
        "energy": qubo.energy(x),
        "violations": qubo.violations(x),
    }
