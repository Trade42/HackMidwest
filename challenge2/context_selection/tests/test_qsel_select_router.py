"""select(), the router and the compile-statistics helpers."""

from __future__ import annotations

import json

import numpy as np
import pytest

import iceberg
from qsel import (IdealBackend, LocalNoisyBackend, brute_force, build_qubo, compile_table,
                  estimate_two_qubit_gates, make_backend, optimum, predicted_kept_share, qaoa_circuit,
                  random_qubo, roadmap_point, route, select, select_qubo, tune_angles)
from qsel.router import measured_qubits
from qsel.compile_stats import near_square_grid
from qsel.select import CLASSICAL_SOLVERS, QUANTUM_TIERS, exact_top, finish, prepare_circuit


@pytest.fixture(scope="module")
def qubo():
    return random_qubo(6, edge_fraction=0.4, seed=3)


def test_classical_solvers(qubo):
    exact = select_qubo(qubo, "brute_force", top=3)
    assert [s.x for s in exact.selections] == [x for x, _ in brute_force(qubo, top=3)]
    assert exact.stats is None and exact.backend == "local"
    greedy = select_qubo(qubo, "greedy", top=3)
    assert all(qubo.is_feasible(s.x) for s in greedy.selections)
    assert len({s.x for s in greedy.selections}) == len(greedy.selections)
    sa = select_qubo(qubo, "sim_anneal", top=3, seed=1)
    assert sa.selections[0].energy == pytest.approx(optimum(qubo)[1])
    assert sa.stats is not None and sa.stats.hit_optimum
    topk = select_qubo(qubo, "top_k", k=2)
    assert len(topk.selections[0].indices) == 2
    assert set(topk.selections[0].indices) == set(np.argsort(-np.array(qubo.meta["relevance"]))[:2])
    for result in (exact, greedy, sa, topk):
        json.dumps(result.to_dict())
    with pytest.raises(ValueError):
        select_qubo(qubo, "nonsense")


def test_exact_top_respects_min_select():
    from qsel.qubo import Qubo

    q = Qubo(np.array([-1.0, -2.0, -0.5]), (), 1.5)
    assert exact_top(q, top=1, min_select=0)[0].x == (0, 0, 0)
    assert exact_top(q, top=1, min_select=1)[0].x == (0, 0, 1)
    assert select_qubo(q, "brute_force").selections[0].indices == (2,)


def test_top_k_needs_relevance():
    from qsel.qubo import Qubo

    with pytest.raises(ValueError):
        select_qubo(Qubo(np.array([1.0, 0.5]), (), 2.0), "top_k")


def test_ideal_and_iceberg_tiers_agree_without_noise(qubo):
    ideal = select_qubo(qubo, "ideal", shots=4000, seed=2)
    assert ideal.backend == "ideal" and ideal.stats.kept_share == 1.0
    assert ideal.selections[0].energy == pytest.approx(optimum(qubo)[1])
    assert set(ideal.timings) == {"tune", "build", "execute", "postprocess"}
    assert ideal.circuit["qubits"] == qubo.n and ideal.postselect is None

    angles = tune_angles(qubo, p=1).angles
    sizes = {"bus": qubo.n + 2 + qubo.n // 2 + 1, "direct": qubo.n + 4}
    for style, qubits in sizes.items():
        enc = select_qubo(qubo, "iceberg", IdealBackend(seed=2), shots=4000, angles=angles,
                          encoder_opts={"style": style})
        assert enc.postselect["kept_share"] == 1.0 and enc.stats.shots == 4000
        assert enc.circuit["qubits"] == qubits
        assert "tune" not in enc.timings and enc.angles == angles.to_dict()
        assert enc.selections[0].x == ideal.selections[0].x
        assert enc.stats.approx_ratio == pytest.approx(ideal.stats.approx_ratio, abs=0.03)
        json.dumps(enc.to_dict())
    default = select_qubo(qubo, "iceberg", IdealBackend(seed=2), shots=500, angles=angles)
    assert default.circuit["qubits"] == sizes["bus"]                 # bus is the default style


def test_noisy_tiers_need_a_backend_and_report_compiled_circuits(qubo):
    for tier in ("physical", "iceberg"):
        with pytest.raises(ValueError):
            select_qubo(qubo, tier)
    backend = LocalNoisyBackend(coupling="grid", grid_shape=(3, 4), layout_trials=2, seed=0)
    angles = tune_angles(qubo, p=1).angles
    bare = select_qubo(qubo, "physical", backend, shots=2000, angles=angles)
    enc = select_qubo(qubo, "iceberg", backend, shots=2000, angles=angles,
                      encoder_opts={"style": "direct", "checks": "z"})
    assert bare.stats.kept == 2000 and bare.extra["compiled"]["two_qubit_gates"] > 0
    assert 0 < enc.stats.kept < 2000 and enc.stats.shots == 2000
    assert enc.postselect["fail_sx"] == 0                  # parity-only variant has no ancilla
    assert enc.extra["compiled"]["two_qubit_gates"] > bare.extra["compiled"]["two_qubit_gates"]

    # the default local noisy backend, with the default (bus) encoding
    fast = make_backend({"kind": "local-pauli", "grid_shape": [4, 6], "seed": 0})
    bus = select_qubo(qubo, "iceberg", fast, shots=4000, angles=angles)
    assert 0 < bus.stats.kept < 4000 and bus.backend.startswith("local-pauli")
    assert bus.extra["compiled"]["two_qubit_gates"] > 0
    assert select_qubo(qubo, "physical", fast, shots=4000, angles=angles).stats.kept == 4000


def test_prepare_and_finish_for_batched_use(qubo):
    angles = tune_angles(qubo, p=1).angles
    assert prepare_circuit(qubo, "physical", angles).num_qubits == qubo.n
    direct = prepare_circuit(qubo, "iceberg", angles,
                             encoder_opts={"style": "direct", "sx_flag": False, "init_flag": False})
    assert direct.metadata["iceberg"]["sx_flag"] is False and direct.num_qubits == qubo.n + 3
    enc = prepare_circuit(qubo, "iceberg", angles)
    assert enc.metadata["iceberg"]["style"] == "bus"
    with pytest.raises(ValueError):
        prepare_circuit(qubo, "greedy", angles)
    counts = IdealBackend().run([enc], 1000)[0]
    done = finish(qubo, "iceberg", enc, counts, timings={"execute": 1.5}, backend_name="x")
    assert done.timings["execute"] == 1.5 and done.backend == "x" and len(done.selections) == 3
    # every shot fails the parity check: nothing survives, and that is reported, not raised
    broken = {format(int(k, 2) ^ 1, f"0{enc.num_clbits}b"): v for k, v in counts.items()}
    empty = finish(qubo, "iceberg", enc, broken)
    assert empty.selections == [] and empty.stats.kept == 0 and empty.postselect["kept_share"] == 0.0
    json.dumps(empty.to_dict())
    assert set(QUANTUM_TIERS) == {"ideal", "physical", "iceberg"}
    assert set(CLASSICAL_SOLVERS) == {"brute_force", "greedy", "sim_anneal", "top_k"}


def test_sim_anneal_uses_the_backends_qubo_service(qubo):
    class Service:
        name = "fake-service"

        def submit_qubo(self, problem, method="sim_anneal"):
            self.method = method
            return {optimum(problem)[0]: 7, (1,) * problem.n: 3}

    service = Service()
    out = select_qubo(qubo, "sim_anneal", service, qubo_method="rqaoa")
    assert service.method == "rqaoa" and out.backend == "fake-service"
    assert out.selections[0].x == optimum(qubo)[0] and out.stats.kept == 10


def test_select_by_question_id(tmp_path):
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(6, 5))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    q = rng.normal(size=5)
    r, s = emb @ (q / np.linalg.norm(q)), emb @ emb.T
    np.savez(tmp_path / "q1.npz", r=r, s=s)
    params = {"tau": 0.3, "theta": 0.0}
    out = select("q1", "brute_force", scores_dir=tmp_path, qubo_params=params)
    want = build_qubo(r, s, **params)
    assert out.selections[0].x == optimum(want)[0]
    assert out.extra["qid"] == "q1" and out.extra["qubo"]["edges"] == want.num_edges
    ideal = select("q1", "ideal", scores_dir=tmp_path, qubo_params=params, shots=2000)
    assert ideal.selections[0].x == out.selections[0].x

    with pytest.raises(ValueError):
        select("q1", "brute_force", scores_dir=tmp_path)
    (tmp_path / "qubo_params.json").write_text(json.dumps(params), encoding="utf-8")
    assert select("q1", "brute_force", scores_dir=tmp_path).selections[0].x == out.selections[0].x
    with pytest.raises(FileNotFoundError):
        select("missing", "brute_force", scores_dir=tmp_path, qubo_params=params)


# --------------------------------------------------------------------- router
def test_gate_estimate_matches_the_compiled_circuits():
    q = random_qubo(10, edge_fraction=0.2, seed=3)
    angles = tune_angles(q, p=1).angles
    full = LocalNoisyBackend(coupling="full")
    bare = qaoa_circuit(q, angles)
    assert estimate_two_qubit_gates(q, 1, encoded=False) == full.compiled_stats(bare)["two_qubit_gates"]
    assert estimate_two_qubit_gates(q, 1, encoded=True, style="direct") == \
        full.compiled_stats(iceberg.encode(bare, style="direct"))["two_qubit_gates"]
    assert estimate_two_qubit_gates(q, 2, encoded=False) == 2 * estimate_two_qubit_gates(q, 1, encoded=False)
    # bus style: K = 10, C = 6 -> 4*9 + 2*(12 + 20) per layer, 15 to start, 23 for the S_X check
    assert estimate_two_qubit_gates(q, 1, encoded=True) == 4 * 9 + 64 + 15 + 23
    assert estimate_two_qubit_gates(q, 1, encoded=True, style="bus") == estimate_two_qubit_gates(q, 1, encoded=True)
    actual = LocalNoisyBackend(coupling="grid").compiled_stats(iceberg.encode(bare))["two_qubit_gates"]
    assert 0.85 * estimate_two_qubit_gates(q, 1, encoded=True) <= actual <= estimate_two_qubit_gates(q, 1, encoded=True)
    assert measured_qubits(10) == 18 and measured_qubits(10, "direct") == 14 and measured_qubits(7) == 15
    with pytest.raises(ValueError):
        estimate_two_qubit_gates(q, 1, encoded=True, style="other")


def test_predicted_kept_share():
    assert predicted_kept_share(0, 0.01) == 1.0
    assert predicted_kept_share(100, 0.005) == pytest.approx((1 - 0.004) ** 100)
    assert predicted_kept_share(10, 0.0, p_meas=0.01, measured_qubits=14) == pytest.approx(0.99 ** 14)
    assert predicted_kept_share(200, 0.01) < predicted_kept_share(100, 0.01)


def test_route_decisions():
    q = random_qubo(10, edge_fraction=0.2, seed=3)
    assert route(q).tier == "iceberg"
    assert route(q, keep_floor=0.9).tier == "physical"
    assert route(q, p2=0.05).tier == "physical"
    assert route(q, max_two_qubit_gates=10).tier == "brute_force"
    assert route(q, max_two_qubit_gates=10, max_brute_force_n=8).tier == "greedy"
    decision = route(q)
    assert decision.estimates["bare_two_qubit_gates"] == 2 * q.num_edges
    assert decision.estimates["style"] == "bus"
    direct = route(q, style="direct")
    assert direct.estimates["iceberg_two_qubit_gates"] < decision.estimates["iceberg_two_qubit_gates"]
    assert route(q, encoder_opts={"style": "direct"}).estimates == direct.estimates
    assert "kept" in decision.reason or "keep" in decision.reason
    json.dumps(decision.to_dict())
    counted = route(q, gate_counter=lambda circuit: 500)
    assert counted.tier == "brute_force" and counted.estimates["gate_count_source"] == "compiled"
    grid = LocalNoisyBackend(coupling="grid", layout_trials=1)
    counter = lambda c: grid.compiled_stats(c)["two_qubit_gates"]          # noqa: E731
    routed = route(q, gate_counter=counter, style="direct")
    assert routed.estimates["iceberg_two_qubit_gates"] > direct.estimates["iceberg_two_qubit_gates"]
    in_place = route(q, gate_counter=counter)                              # bus: nothing to route
    assert in_place.estimates["iceberg_two_qubit_gates"] <= decision.estimates["iceberg_two_qubit_gates"]


# -------------------------------------------------------------- compile stats
def test_compile_table(qubo):
    angles = tune_angles(qubo, p=1).angles

    class Broken:
        name = "broken"

        def compile(self, circuit):
            raise RuntimeError("offline")

    table = compile_table(qubo, angles, {
        "all_to_all": LocalNoisyBackend(coupling="full"),
        "grid": LocalNoisyBackend(coupling="grid", grid_shape=(3, 4), layout_trials=1),
        "broken": Broken(),
    })
    assert table["n"] == qubo.n and table["edges"] == qubo.num_edges
    for tier in ("bare", "iceberg"):
        assert table[tier]["grid"]["two_qubit_gates"] >= table[tier]["all_to_all"]["two_qubit_gates"]
        assert "offline" in table[tier]["broken"]["error"]
    assert table["iceberg"]["logical"]["qubits"] == qubo.n + 2 + qubo.n // 2 + 1
    assert table["iceberg"]["all_to_all"]["two_qubit_gates"] > table["bare"]["all_to_all"]["two_qubit_gates"]
    assert table["iceberg"]["grid"]["two_qubit_gates"] == table["iceberg"]["all_to_all"]["two_qubit_gates"]
    json.dumps(table)
    direct = compile_table(qubo, angles, {"grid": LocalNoisyBackend(coupling="grid", grid_shape=(3, 4),
                                                                    layout_trials=1)},
                           encoder_opts={"style": "direct"})
    assert direct["iceberg"]["logical"]["qubits"] == qubo.n + 4


def test_roadmap_point():
    assert near_square_grid(14) == (3, 5) and near_square_grid(16) == (4, 4)
    point = roadmap_point(7, layout_trials=1)
    assert point["n"] == 7 and point["brute_force_seconds"] > 0
    assert point["bare"]["physical_qubits"] == 7
    assert point["style"] == "bus" and point["grid_shape"] == [3, 5]
    assert point["iceberg"]["physical_qubits"] == 8 + 2 + 5        # odd n is padded to even
    assert point["iceberg"]["two_qubit_gates"] == point["iceberg"]["grid"]["two_qubit_gates"]
    rows, cols = point["grid_shape"]
    assert rows * cols >= point["iceberg"]["physical_qubits"]
    direct = roadmap_point(7, layout_trials=1, encoder_opts={"style": "direct"})
    assert direct["style"] == "direct" and direct["iceberg"]["physical_qubits"] == 8 + 4
    assert direct["iceberg"]["grid"]["two_qubit_gates"] >= direct["iceberg"]["all_to_all"]["two_qubit_gates"]
    no_time = roadmap_point(8, layout_trials=1, max_brute_force_n=6, grid=False)
    assert no_time["brute_force_seconds"] is None and no_time["grid_shape"] is None
    assert no_time["iceberg"]["two_qubit_gates"] == no_time["iceberg"]["all_to_all"]["two_qubit_gates"]
    json.dumps(point)
