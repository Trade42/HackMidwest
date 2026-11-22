"""QUBO, classical solvers, QAOA circuit and angle tuning."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qsel import (QaoaSimulator, brute_force, build_qubo, greedy, greedy_candidates, optimum,
                  qaoa_circuit, random_qubo, repair, simulated_annealing, top_k, tune_angles)
from qsel.bits import all_bit_table, from_indices, index_to_x, key_to_x, selected, x_to_index, x_to_key
from qsel.qubo import Qubo, normalize_relevance


def small_qubos():
    return [random_qubo(n, edge_fraction=f, seed=s) for n, f, s in
            [(4, 0.5, 0), (5, 0.3, 1), (6, 0.4, 2), (7, 0.2, 3), (8, 0.6, 4)]]


def test_bit_helpers_roundtrip():
    for n in (1, 3, 6):
        for k in range(1 << n):
            x = index_to_x(k, n)
            assert x_to_index(x) == k
            assert key_to_x(x_to_key(x), n) == x
            assert from_indices(selected(x), n) == x
            assert x_to_key(x) == format(k, f"0{n}b")
    assert (all_bit_table(3)[5] == np.array([1, 0, 1])).all()


def test_energy_matches_ising_and_enumeration():
    for q in small_qubos():
        h, J, const = q.to_ising()
        e_all = q.energies_all()
        for k in range(1 << q.n):
            x = index_to_x(k, q.n)
            z = 1 - 2 * np.array(x)
            ising = const + float(h @ z) + sum(j * z[a] * z[b] for (a, b), j in J.items())
            assert q.energy(x) == pytest.approx(ising, abs=1e-10)
            assert e_all[k] == pytest.approx(q.energy(x), abs=1e-10)


def test_superstaq_dictionary_is_the_same_objective():
    q = random_qubo(6, edge_fraction=0.4, seed=7)
    d = q.to_superstaq()
    for x in product((0, 1), repeat=q.n):
        value = sum(c * np.prod([x[i] for i in key]) for key, c in d.items())
        assert value == pytest.approx(q.energy(x), abs=1e-10)


def test_build_qubo_edges_and_penalty():
    r = np.array([0.9, 0.8, 0.1, 0.2])
    s = np.array([[1, .7, .1, .2], [.7, 1, .3, .1], [.1, .3, 1, .65], [.2, .1, .65, 1]])
    q = build_qubo(r, s, tau=0.6, theta=0.0)
    assert q.edges == ((0, 1), (2, 3))
    assert q.penalty > q.weights.max() > 0
    assert build_qubo(r, s, tau=0.6, max_edges=1).edges == ((0, 1),)
    assert build_qubo(r, s, tau=0.99).edges == ()
    assert np.allclose(normalize_relevance(r, "zscore").mean(), 0.0)
    with pytest.raises(ValueError):
        build_qubo(r, s, tau=0.5, penalty_scale=1.0)
    with pytest.raises(ValueError):
        Qubo(np.ones(3), ((1, 0),), 2.0)


def test_redundant_pair_never_pays_off():
    for q in small_qubos():
        x, _ = optimum(q)
        assert q.is_feasible(x)
        assert q.energies_all().min() == pytest.approx(q.energy(x))


def test_repair_makes_valid_without_adding():
    rng = np.random.default_rng(0)
    for q in small_qubos():
        for _ in range(40):
            x = tuple(int(b) for b in rng.integers(0, 2, q.n))
            fixed = repair(q, x, min_select=0)
            assert q.is_feasible(fixed)
            assert all(f <= b for f, b in zip(fixed, x))            # nothing added
            if q.is_feasible(x):
                assert fixed == x
            assert q.energy(fixed) <= q.energy(x) + 1e-12


def test_repair_min_select_floor():
    q = random_qubo(6, edge_fraction=0.4, seed=5)
    fixed = repair(q, (0,) * q.n, min_select=2)
    assert sum(fixed) == 2 and q.is_feasible(fixed)
    assert fixed[int(np.argmax(q.weights))] == 1


def test_brute_force_matches_exhaustive():
    for q in small_qubos():
        valid = [(q.energy(x), x) for x in product((0, 1), repeat=q.n) if q.is_feasible(x)]
        valid.sort(key=lambda ex: (round(ex[0], 12), x_to_index(ex[1])))
        got = brute_force(q, top=3)
        assert [x for x, _ in got] == [x for _, x in valid[:3]]
        assert [e for _, e in got] == pytest.approx([e for e, _ in valid[:3]])


def test_greedy_is_valid_and_positive():
    for q in small_qubos():
        x = greedy(q)
        assert q.is_feasible(x)
        assert all(q.weights[i] > 0 for i in selected(x))
        cands = greedy_candidates(q, top=3)
        assert cands[0][0] == x
        assert len({c for c, _ in cands}) == len(cands)
        assert all(q.is_feasible(c) for c, _ in cands)


def test_top_k():
    assert top_k([0.1, 0.9, 0.5, 0.7], 2) == (0, 1, 0, 1)


def test_simulated_annealing_finds_optimum():
    for q in small_qubos():
        samples = simulated_annealing(q, reads=24, sweeps=80, seed=1)
        best = min(q.energy(x) for x in samples)
        assert best == pytest.approx(optimum(q)[1], abs=1e-9)
        assert sum(samples.values()) == 24


@pytest.mark.parametrize("p", [1, 2])
def test_qaoa_circuit_matches_simulator(p):
    q = random_qubo(6, edge_fraction=0.4, seed=3)
    tuned = tune_angles(q, p=p)
    sim = QaoaSimulator(q)
    circuit = qaoa_circuit(q, tuned.angles, measure=False)
    probs = Statevector(circuit).probabilities()
    assert np.abs(probs - sim.probabilities(tuned.angles)).max() < 1e-10
    assert tuned.expectation == pytest.approx(float(probs @ q.energies_all()), abs=1e-8)
    assert set(qaoa_circuit(q, tuned.angles).count_ops()) <= {"h", "rz", "rzz", "rx", "measure", "barrier"}


def test_tuned_angles_beat_random_guess_and_p2_not_worse():
    q = random_qubo(8, edge_fraction=0.3, seed=11)
    t1 = tune_angles(q, p=1)
    t2 = tune_angles(q, p=2)
    uniform = float(q.energies_all().mean())
    assert t1.expectation < uniform
    assert t2.expectation <= t1.expectation + 1e-9
    assert 0.0 <= t1.approx_ratio <= 1.0 and 0.0 <= t1.p_optimum <= 1.0


def test_simulator_sampling_is_seeded():
    q = random_qubo(5, seed=2)
    sim = QaoaSimulator(q)
    angles = tune_angles(q, p=1).angles
    a, b = sim.sample(angles, 500, seed=4), sim.sample(angles, 500, seed=4)
    assert a == b and sum(a.values()) == 500
