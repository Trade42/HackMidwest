"""From measured counts to ranked selections and solver-quality metrics.

Every sampling tier (ideal, physical, error-detected, simulated annealing) ends here:
samples are repaired, merged and ranked by energy, and the same statistics are
computed for all of them. Counts are logical counts, i.e. for the error-detected
tier they have already been through ``iceberg.postselect``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from .bits import Selection, all_bit_table, key_to_x, selected, x_to_key
from .classical import repair
from .qubo import Qubo

TOL = 1e-9


@dataclass(frozen=True)
class RankedSelection:
    """One distinct selection after repair, with the kept shots that led to it."""

    x: tuple[int, ...]
    indices: tuple[int, ...]
    energy: float
    shots: int

    def to_dict(self) -> dict[str, Any]:
        return {"x": list(self.x), "indices": list(self.indices), "energy": self.energy,
                "shots": self.shots}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RankedSelection":
        return cls(tuple(d["x"]), tuple(d["indices"]), float(d["energy"]), int(d["shots"]))


@dataclass(frozen=True)
class SampleStats:
    """Solver-quality metrics of one batch of samples."""

    shots: int                  #: raw shots, before any post-selection
    kept: int                   #: shots that survived post-selection
    kept_share: float
    feasibility: float          #: share of kept shots with no redundant pair, before repair
    mean_energy: float          #: mean objective over kept shots, before repair
    approx_ratio: float         #: (e_max - mean_energy) / (e_max - e_min), 1 is optimal
    p_optimum: float            #: share of kept shots that are an exact optimum, before repair
    distinct_valid: int         #: distinct valid selections among kept shots
    diversity_per_1000: float   #: distinct valid selections per 1,000 kept shots
    best_energy: float          #: lowest energy after repair
    hit_optimum: bool           #: the top-ranked selection is an exact optimum

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key, value in out.items():
            if isinstance(value, float) and not math.isfinite(value):
                out[key] = None
        return out


def selection_counts_to_counts(samples: Mapping[Selection, int]) -> dict[str, int]:
    """Selection-tuple counts (classical samplers) -> Qiskit-style counts."""
    out: dict[str, int] = {}
    for x, c in samples.items():
        key = x_to_key(x)
        out[key] = out.get(key, 0) + int(c)
    return out


def rank_selections(
    qubo: Qubo,
    counts: Mapping[str, int],
    *,
    top: int = 3,
    min_select: int = 1,
) -> list[RankedSelection]:
    """Repair every sample, merge duplicates and return the ``top`` best distinct ones.

    Order: energy ascending, then shots descending, then the selection itself, so the
    result is deterministic for a given counts dictionary.
    """
    merged: dict[Selection, int] = {}
    for key, c in counts.items():
        if int(c) <= 0:
            continue
        x = repair(qubo, key_to_x(key, qubo.n), min_select=min_select)
        merged[x] = merged.get(x, 0) + int(c)
    scored = [(round(qubo.energy(x), 12), -c, x) for x, c in merged.items()]
    scored.sort()
    return [
        RankedSelection(x=x, indices=tuple(selected(x)), energy=float(e), shots=-neg_c)
        for e, neg_c, x in scored[: max(int(top), 0)]
    ]


def _reference(qubo: Qubo, min_select: int) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """All energies, the valid mask, e_min, e_max and the best reachable energy.

    The reachable target differs from ``e_min`` only when the optimum selects fewer than
    ``min_select`` paragraphs (for example the empty set when no weight is positive).
    """
    e = qubo.energies_all()
    ok = qubo.feasible_mask_all()
    size = all_bit_table(qubo.n).sum(axis=1)
    allowed = ok & (size >= min(min_select, qubo.n))
    target = float(e[allowed].min()) if allowed.any() else float(e[ok].min())
    return e, ok, float(e.min()), float(e.max()), target


def sample_stats(
    qubo: Qubo,
    counts: Mapping[str, int],
    *,
    raw_shots: int | None = None,
    min_select: int = 1,
) -> SampleStats:
    """Solver-quality metrics for logical ``counts``.

    Args:
        qubo: the problem the samples were drawn for.
        counts: logical counts after post-selection (Qiskit-style keys).
        raw_shots: shots before post-selection; defaults to ``sum(counts)``.
        min_select: same floor that :func:`rank_selections` uses in repair.
    """
    kept = int(sum(int(c) for c in counts.values()))
    shots = int(raw_shots) if raw_shots is not None else kept
    nan = float("nan")
    if kept == 0:
        return SampleStats(shots, 0, 0.0, nan, nan, nan, nan, 0, nan, nan, False)

    e, ok, e_min, e_max, target = _reference(qubo, min_select)
    mask = (1 << qubo.n) - 1
    idx = np.fromiter((int(k.replace(" ", ""), 2) & mask for k in counts), dtype=np.int64,
                      count=len(counts))
    mult = np.fromiter((int(c) for c in counts.values()), dtype=np.int64, count=len(counts))
    # distinct keys can collide after masking only if keys are wider than n; merge to be safe
    weight = np.bincount(idx, weights=mult, minlength=1 << qubo.n)
    present = weight > 0

    mean_energy = float(weight @ e) / kept
    span = e_max - e_min
    best = rank_selections(qubo, counts, top=1, min_select=min_select)
    best_energy = best[0].energy if best else nan
    distinct_valid = int(np.count_nonzero(present & ok))
    return SampleStats(
        shots=shots,
        kept=kept,
        kept_share=kept / shots if shots else 0.0,
        feasibility=float(weight[ok].sum()) / kept,
        mean_energy=mean_energy,
        approx_ratio=float((e_max - mean_energy) / span) if span > 0 else 1.0,
        p_optimum=float(weight[e <= e_min + TOL].sum()) / kept,
        distinct_valid=distinct_valid,
        diversity_per_1000=1000.0 * distinct_valid / kept,
        best_energy=best_energy,
        hit_optimum=bool(best and best_energy <= target + TOL),
    )


def subsample_counts(counts: Mapping[str, int], shots: int, seed: int) -> dict[str, int]:
    """Draw ``shots`` samples without replacement from the multiset ``counts``.

    Used to compare tiers at equal shot budgets. If ``shots`` is at least the number of
    available samples, all of them are returned.
    """
    keys = sorted(counts)
    colors = np.array([int(counts[k]) for k in keys], dtype=np.int64)
    total = int(colors.sum())
    if shots >= total:
        return {k: int(c) for k, c in zip(keys, colors) if c > 0}
    if shots <= 0:
        return {}
    rng = np.random.default_rng(seed)
    draw = rng.multivariate_hypergeometric(colors, int(shots))
    return {k: int(c) for k, c in zip(keys, draw) if c > 0}


def expected_hit_curve(
    qubo: Qubo,
    counts: Mapping[str, int],
    budgets: Iterable[int],
    *,
    raw_shots: int | None = None,
) -> dict[int, float]:
    """Chance that ``S`` raw shots contain at least one exact optimum, for each budget ``S``.

    ``1 - (1 - p_opt * kept_share) ** S``: a raw shot helps only if it survives
    post-selection and is optimal, so discarded shots are charged to the tier that
    discards them. This is the equal-raw-shots comparison between tiers.
    """
    kept = int(sum(int(c) for c in counts.values()))
    shots = int(raw_shots) if raw_shots is not None else kept
    if shots <= 0 or kept == 0:
        return {int(s): 0.0 for s in budgets}
    e = qubo.energies_all()
    e_min = float(e.min())
    mask = (1 << qubo.n) - 1
    optimal = sum(int(c) for k, c in counts.items()
                  if e[int(k.replace(" ", ""), 2) & mask] <= e_min + TOL)
    q = optimal / shots
    return {int(s): float(1.0 - (1.0 - q) ** int(s)) for s in budgets}
