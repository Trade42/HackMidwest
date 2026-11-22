"""``select``: one question's QUBO in, ranked selections out, for every solver tier.

Quantum tiers:

* ``"ideal"``: bare QAOA on the exact simulator;
* ``"physical"``: the same bare circuit on a noisy backend;
* ``"iceberg"``: the Iceberg-encoded circuit on a noisy backend, flagged shots discarded.
  ``encoder_opts`` goes to ``iceberg.encode`` (for example ``{"style": "direct"}``).

Classical solvers go through the same entry point so that the evaluation harness
treats every lane alike: ``"brute_force"``, ``"greedy"``, ``"sim_anneal"``, ``"top_k"``.

All tiers see the same QUBO and the same tuned angles, so a difference in the final
answer traces back to the solver tier alone.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from qiskit import QuantumCircuit

from . import classical
from .angles import tune_angles
from .backends import Backend, IdealBackend, circuit_stats
from .bits import Selection, all_bit_table, index_to_x, selected
from .qaoa import Angles, qaoa_circuit
from .qubo import Qubo, build_qubo
from .sampling import (RankedSelection, SampleStats, rank_selections, sample_stats,
                       selection_counts_to_counts)

QUANTUM_TIERS = ("ideal", "physical", "iceberg")
CLASSICAL_SOLVERS = ("brute_force", "greedy", "sim_anneal", "top_k")


@dataclass
class SelectionResult:
    """What a solver tier hands to the language model, plus how it got there."""

    tier: str
    backend: str
    selections: list[RankedSelection]
    stats: SampleStats | None
    timings: dict[str, float] = field(default_factory=dict)
    circuit: dict[str, Any] = field(default_factory=dict)
    angles: dict[str, Any] | None = None
    postselect: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def best(self) -> RankedSelection | None:
        return self.selections[0] if self.selections else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "backend": self.backend,
            "selections": [s.to_dict() for s in self.selections],
            "stats": self.stats.to_dict() if self.stats is not None else None,
            "timings": {k: float(v) for k, v in self.timings.items()},
            "circuit": self.circuit,
            "angles": self.angles,
            "postselect": self.postselect,
            "extra": self.extra,
        }


# ------------------------------------------------------------------- circuits
def prepare_circuit(
    qubo: Qubo,
    tier: str,
    angles: Angles,
    *,
    encoder_opts: Mapping[str, Any] | None = None,
) -> QuantumCircuit:
    """The sampling circuit of a quantum tier (bare, or Iceberg-encoded)."""
    if tier not in QUANTUM_TIERS:
        raise ValueError(f"{tier!r} is not a quantum tier; expected one of {QUANTUM_TIERS}")
    bare = qaoa_circuit(qubo, angles)
    if tier != "iceberg":
        return bare
    import iceberg

    return iceberg.encode(bare, **dict(encoder_opts or {}))


def finish(
    qubo: Qubo,
    tier: str,
    circuit: QuantumCircuit,
    counts: Mapping[str, int],
    *,
    top: int = 3,
    min_select: int = 1,
    timings: Mapping[str, float] | None = None,
    backend_name: str = "",
) -> SelectionResult:
    """Turn a backend's raw counts into a :class:`SelectionResult`.

    For the Iceberg tier the counts are post-selected first; the share of shots kept
    is part of the result. If no shot survives, the selection list is empty and the
    caller decides what to fall back to.
    """
    t0 = time.perf_counter()
    post: dict[str, Any] | None = None
    raw_shots = int(sum(int(c) for c in counts.values()))
    logical: Mapping[str, int] = counts
    if tier == "iceberg":
        import iceberg

        logical, ps = iceberg.postselect(counts, circuit)
        post = ps.to_dict()
        raw_shots = ps.shots
    selections = rank_selections(qubo, logical, top=top, min_select=min_select) if logical else []
    stats = sample_stats(qubo, logical, raw_shots=raw_shots, min_select=min_select)
    out_timings = dict(timings or {})
    out_timings["postprocess"] = time.perf_counter() - t0
    return SelectionResult(
        tier=tier,
        backend=backend_name,
        selections=selections,
        stats=stats,
        timings=out_timings,
        circuit=circuit_stats(circuit),
        postselect=post,
    )


# ------------------------------------------------------------------ classical
def exact_top(qubo: Qubo, top: int = 3, min_select: int = 1) -> list[RankedSelection]:
    """The ``top`` lowest-energy valid selections with at least ``min_select`` paragraphs."""
    e = qubo.energies_all()
    ok = qubo.feasible_mask_all() & (all_bit_table(qubo.n).sum(axis=1) >= min(min_select, qubo.n))
    idx = np.flatnonzero(ok)
    order = idx[np.lexsort((idx, e[idx]))][:top]
    out = []
    for k in order:
        x = index_to_x(int(k), qubo.n)
        out.append(RankedSelection(x, tuple(selected(x)), float(e[k]), 0))
    return out


def _from_candidates(qubo: Qubo, candidates: list[Selection], top: int, min_select: int,
                     keep_order: bool = False) -> list[RankedSelection]:
    seen: dict[Selection, None] = {}
    for x in candidates:
        seen.setdefault(classical.repair(qubo, x, min_select=min_select), None)
    ranked = [RankedSelection(x, tuple(selected(x)), float(qubo.energy(x)), 0) for x in seen]
    if not keep_order:
        ranked.sort(key=lambda s: (round(s.energy, 12), s.x))
    return ranked[:top]


def _select_classical(qubo: Qubo, solver: str, backend: Any, *, top: int, seed: int,
                      min_select: int, k: int, sa_reads: int, sa_sweeps: int,
                      qubo_method: str) -> SelectionResult:
    t0 = time.perf_counter()
    stats: SampleStats | None = None
    extra: dict[str, Any] = {}
    name = "local"
    if solver == "brute_force":
        selections = exact_top(qubo, top, min_select)
    elif solver == "greedy":
        cands = [x for x, _ in classical.greedy_candidates(qubo, max(top, 1))]
        selections = _from_candidates(qubo, cands, top, min_select, keep_order=True)
    elif solver == "top_k":
        relevance = qubo.meta.get("relevance")
        if relevance is None:
            raise ValueError("top_k needs qubo.meta['relevance'] (build the QUBO with build_qubo)")
        sizes = [k] + [s for s in (k - 1, k + 1) if 1 <= s <= qubo.n]
        cands = [classical.top_k(relevance, s) for s in sizes]
        # top-k ignores redundancy on purpose, so no repair here
        seen: dict[Selection, None] = {}
        for x in cands:
            seen.setdefault(x, None)
        selections = [RankedSelection(x, tuple(selected(x)), float(qubo.energy(x)), 0)
                      for x in seen][:top]
        extra["k"] = k
    elif solver == "sim_anneal":
        if backend is not None and hasattr(backend, "submit_qubo"):
            samples = backend.submit_qubo(qubo, method=qubo_method)
            name = getattr(backend, "name", "superstaq")
            extra["qubo_method"] = qubo_method
        else:
            samples = classical.simulated_annealing(qubo, reads=sa_reads, sweeps=sa_sweeps, seed=seed)
            extra["reads"], extra["sweeps"] = sa_reads, sa_sweeps
        counts = selection_counts_to_counts(samples)
        selections = rank_selections(qubo, counts, top=top, min_select=min_select)
        stats = sample_stats(qubo, counts, min_select=min_select)
    else:
        raise ValueError(f"unknown classical solver {solver!r}")
    return SelectionResult(
        tier=solver, backend=name, selections=selections, stats=stats,
        timings={"solve": time.perf_counter() - t0}, extra=extra,
    )


# ----------------------------------------------------------------- main entry
def select_qubo(
    qubo: Qubo,
    tier: str,
    backend: Backend | None = None,
    *,
    p: int = 1,
    shots: int = 4000,
    top: int = 3,
    seed: int = 0,
    angles: Angles | None = None,
    encoder_opts: Mapping[str, Any] | None = None,
    min_select: int = 1,
    k: int = 3,
    sa_reads: int = 64,
    sa_sweeps: int = 200,
    qubo_method: str = "sim_anneal",
) -> SelectionResult:
    """Solve one QUBO with the given tier or classical solver.

    Args:
        qubo: the selection problem.
        tier: one of :data:`QUANTUM_TIERS` or :data:`CLASSICAL_SOLVERS`.
        backend: where quantum circuits run. ``"ideal"`` defaults to the exact
            simulator; ``"physical"`` and ``"iceberg"`` need a noisy backend. For
            ``"sim_anneal"`` a backend with ``submit_qubo`` sends the problem to
            Superstaq, otherwise the local annealer runs.
        p: QAOA depth.
        shots: raw shots per circuit.
        top: how many distinct selections to return.
        seed: seed for tuning restarts, sampling and the local annealer.
        angles: reuse tuned angles (tune once on the ideal tier, reuse in every tier).
        encoder_opts: keyword arguments for ``iceberg.encode``.
        min_select: floor on the number of selected paragraphs.
        k: size for the ``"top_k"`` baseline.
    """
    if tier in CLASSICAL_SOLVERS:
        return _select_classical(qubo, tier, backend, top=top, seed=seed, min_select=min_select,
                                 k=k, sa_reads=sa_reads, sa_sweeps=sa_sweeps, qubo_method=qubo_method)
    if tier not in QUANTUM_TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {QUANTUM_TIERS + CLASSICAL_SOLVERS}")
    if backend is None:
        if tier != "ideal":
            raise ValueError(f"tier {tier!r} needs a backend (local noisy simulator or Superstaq)")
        backend = IdealBackend(seed=seed)

    timings: dict[str, float] = {}
    tuned: dict[str, Any] | None = None
    if angles is None:
        t0 = time.perf_counter()
        result = tune_angles(qubo, p=p, seed=seed)
        timings["tune"] = time.perf_counter() - t0
        angles, tuned = result.angles, result.to_dict()
    t0 = time.perf_counter()
    circuit = prepare_circuit(qubo, tier, angles, encoder_opts=encoder_opts)
    timings["build"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    counts = backend.run([circuit], int(shots))[0]
    timings["execute"] = time.perf_counter() - t0

    out = finish(qubo, tier, circuit, counts, top=top, min_select=min_select, timings=timings,
                 backend_name=getattr(backend, "name", type(backend).__name__))
    out.angles = tuned if tuned is not None else angles.to_dict()
    if getattr(backend, "reports_compiled", False) and getattr(backend, "last_compiled", None):
        out.extra["compiled"] = backend.last_compiled[-1]
    elif getattr(backend, "last_stats", None):
        out.extra["compiled"] = dict(backend.last_stats[-1])
    return out


def load_scores(qid: str, scores_dir: str | Path = "scores") -> tuple[np.ndarray, np.ndarray]:
    """Relevance ``r`` and overlap ``s`` written by the scoring step for one question."""
    path = Path(scores_dir) / f"{qid}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"no scores for question {qid!r} at {path}")
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["r"], dtype=float), np.asarray(data["s"], dtype=float)


def select(
    qid: str,
    tier: str,
    backend: Backend | None = None,
    *,
    scores_dir: str | Path = "scores",
    qubo_params: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> SelectionResult:
    """The interface named in the team plan: question id, tier and backend in, selections out.

    Loads ``scores_dir/{qid}.npz`` (arrays ``r`` and ``s``), builds the QUBO with
    ``build_qubo(r, s, **qubo_params)`` and calls :func:`select_qubo`. If ``qubo_params``
    is omitted, ``scores_dir/qubo_params.json`` is used when it exists.
    """
    r, s = load_scores(qid, scores_dir)
    params = dict(qubo_params) if qubo_params is not None else None
    if params is None:
        default = Path(scores_dir) / "qubo_params.json"
        if default.is_file():
            with default.open(encoding="utf-8") as fh:
                params = json.load(fh)
    if not params or "tau" not in params:
        raise ValueError("qubo_params must include at least 'tau' (redundancy threshold); pass it "
                         f"or write {Path(scores_dir) / 'qubo_params.json'}")
    qubo = build_qubo(r, s, **params)
    out = select_qubo(qubo, tier, backend, **kwargs)
    out.extra["qid"] = qid
    out.extra["qubo"] = {"n": qubo.n, "edges": qubo.num_edges, "penalty": qubo.penalty, **params}
    return out
