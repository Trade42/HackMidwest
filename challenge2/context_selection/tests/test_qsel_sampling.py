"""Counts -> ranked selections and solver-quality statistics."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from qsel import QaoaSimulator, optimum, random_qubo, tune_angles
from qsel.bits import x_to_key
from qsel.qubo import Qubo
from qsel.sampling import (RankedSelection, expected_hit_curve, rank_selections, sample_stats,
                           selection_counts_to_counts, subsample_counts)


def toy() -> Qubo:
    # weights 3, 2, 1, -1; paragraphs 0 and 1 are redundant
    return Qubo(np.array([3.0, 2.0, 1.0, -1.0]), ((0, 1),), 4.5)


def test_rank_selections_repairs_merges_and_orders():
    q = toy()
    counts = {
        x_to_key((1, 1, 1, 0)): 5,     # invalid: repaired to (1, 0, 1, 0)
        x_to_key((1, 0, 1, 0)): 2,     # the optimum, merges with the line above
        x_to_key((0, 1, 1, 0)): 9,
        x_to_key((0, 0, 0, 1)): 1,
    }
    ranked = rank_selections(q, counts, top=3)
    assert [r.x for r in ranked] == [(1, 0, 1, 0), (0, 1, 1, 0), (0, 0, 0, 1)]
    assert ranked[0].shots == 7 and ranked[0].indices == (0, 2)
    assert ranked[0].energy == pytest.approx(-4.0)
    assert [r.energy for r in ranked] == sorted(r.energy for r in ranked)
    assert len(rank_selections(q, counts, top=1)) == 1
    assert RankedSelection.from_dict(ranked[0].to_dict()) == ranked[0]


def test_rank_selections_ties_prefer_more_shots():
    q = Qubo(np.array([1.0, 1.0, 0.5]), (), 2.0)
    counts = {x_to_key((1, 0, 0)): 3, x_to_key((0, 1, 0)): 8}
    assert [r.x for r in rank_selections(q, counts, top=2)] == [(0, 1, 0), (1, 0, 0)]


def test_empty_sample_is_filled_to_min_select():
    q = toy()
    ranked = rank_selections(q, {x_to_key((0, 0, 0, 0)): 4}, top=3, min_select=1)
    assert ranked[0].x == (1, 0, 0, 0)
    assert rank_selections(q, {x_to_key((0, 0, 0, 0)): 4}, min_select=0)[0].x == (0, 0, 0, 0)


def test_sample_stats_on_known_counts():
    q = toy()
    counts = {x_to_key((1, 1, 1, 0)): 25, x_to_key((1, 0, 1, 0)): 50, x_to_key((0, 0, 0, 1)): 25}
    st = sample_stats(q, counts, raw_shots=200)
    assert (st.shots, st.kept) == (200, 100)
    assert st.kept_share == pytest.approx(0.5)
    assert st.feasibility == pytest.approx(0.75)
    assert st.p_optimum == pytest.approx(0.5)
    mean = 0.25 * q.energy((1, 1, 1, 0)) + 0.5 * q.energy((1, 0, 1, 0)) + 0.25 * q.energy((0, 0, 0, 1))
    assert st.mean_energy == pytest.approx(mean)
    e = q.energies_all()
    assert st.approx_ratio == pytest.approx((e.max() - mean) / (e.max() - e.min()))
    assert st.distinct_valid == 2 and st.diversity_per_1000 == pytest.approx(20.0)
    assert st.best_energy == pytest.approx(-4.0) and st.hit_optimum
    json.dumps(st.to_dict())


def test_sample_stats_match_the_exact_simulator():
    q = random_qubo(8, edge_fraction=0.3, seed=5)
    tuned = tune_angles(q, p=1)
    sim = QaoaSimulator(q)
    counts = sim.sample(tuned.angles, 60000, seed=3)
    st = sample_stats(q, counts)
    assert st.approx_ratio == pytest.approx(tuned.approx_ratio, abs=0.01)
    assert st.p_optimum == pytest.approx(tuned.p_optimum, abs=0.01)
    probs = sim.probabilities(tuned.angles)
    assert st.feasibility == pytest.approx(float(probs[q.feasible_mask_all()].sum()), abs=0.01)
    assert st.hit_optimum and st.best_energy == pytest.approx(optimum(q)[1])


def test_sample_stats_without_kept_shots():
    st = sample_stats(toy(), {}, raw_shots=4000)
    assert st.shots == 4000 and st.kept == 0 and st.kept_share == 0.0 and not st.hit_optimum
    assert math.isnan(st.approx_ratio)
    assert st.to_dict()["approx_ratio"] is None


def test_hit_optimum_respects_min_select_when_nothing_is_relevant():
    q = Qubo(np.array([-1.0, -2.0, -0.5]), (), 1.5)       # optimum is the empty selection
    counts = {x_to_key((0, 0, 0)): 10}
    assert sample_stats(q, counts, min_select=0).hit_optimum
    st = sample_stats(q, counts, min_select=1)
    assert st.best_energy == pytest.approx(0.5) and st.hit_optimum


def test_subsample_counts():
    counts = {"00": 500, "01": 300, "11": 200}
    sub = subsample_counts(counts, 100, seed=1)
    assert sum(sub.values()) == 100
    assert all(sub[k] <= counts[k] for k in sub)
    assert sub == subsample_counts(counts, 100, seed=1)
    assert sub != subsample_counts(counts, 100, seed=2)
    assert subsample_counts(counts, 5000, seed=1) == counts
    assert subsample_counts(counts, 0, seed=1) == {}
    means = np.mean([subsample_counts(counts, 100, seed=s).get("00", 0) for s in range(300)])
    assert means == pytest.approx(50, abs=2.0)


def test_expected_hit_curve():
    q = toy()
    best = x_to_key((1, 0, 1, 0))
    counts = {best: 10, x_to_key((0, 1, 1, 0)): 90}
    curve = expected_hit_curve(q, counts, [1, 10, 100], raw_shots=200)
    assert curve[1] == pytest.approx(0.05)
    assert curve[10] == pytest.approx(1 - 0.95**10)
    assert curve[1] < curve[10] < curve[100] <= 1.0
    assert expected_hit_curve(q, {}, [4], raw_shots=10) == {4: 0.0}
    # discarded shots are charged to the tier: same counts, more raw shots, lower curve
    assert expected_hit_curve(q, counts, [10], raw_shots=400)[10] < curve[10]


def test_selection_counts_to_counts():
    assert selection_counts_to_counts({(1, 0, 0): 2, (0, 1, 1): 3}) == {"001": 2, "110": 3}
