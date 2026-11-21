"""Small algebra, API, layout, and routing checks for the revised plan."""
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from evaluation.adapters import Problem, ShotResult
from evaluation.make_dem_dataset import sample_mechanisms
from evaluation.predecoder import (ComposedSession, bp_present_probability, combine_correction,
                                   combine_observables, residual_syndrome)
from evaluation.predecoder_layout import GrossLayout, circular_pad_spatial
from evaluation.predecoder_eval import promotion_gate
from evaluation.router import (FEATURE_NAMES, TieredSession, candidate_features, first_leg_converged,
                               fit_logistic, risk, select_threshold, threshold_report)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_actual_gross_layout_all_translations_and_roundtrips(basis):
    from circuit import GROSS_CIRCUITS, dem_matrices, load_circuit

    H, L, priors = dem_matrices(load_circuit(GROSS_CIRCUITS[basis], .004), basis)
    layout = GrossLayout(H, L, priors)
    assert layout.verify_roundtrips()
    assert layout.fault_indices.shape == (122, 12, 6)
    assert len(layout.fault_permutations) == 72
    for perm in layout.fault_permutations.values():
        assert np.array_equal(np.sort(perm), np.arange(8784))


def test_spatial_padding_does_not_wrap_time():
    grid = np.arange(13 * 12 * 6).reshape(13, 12, 6)
    padded = circular_pad_spatial(grid, 1)
    assert padded.shape == (13, 14, 8)
    assert padded[0, 1, 1] == grid[0, 0, 0]
    assert padded[0, 0, 0] == grid[0, -1, -1]
    assert padded[-1, 1, 1] == grid[-1, 0, 0]


def test_commit_residual_and_logical_frame_identities():
    H = sp.csr_matrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8))
    L = sp.csr_matrix(([1], ([11], [2])), shape=(12, 3), dtype=np.uint8)
    syndrome = np.array([1, 1], dtype=np.uint8)
    commits = np.array([0, 0, 1], dtype=np.uint8)
    residual = np.array([0, 1, 0], dtype=np.uint8)
    assert np.array_equal(residual_syndrome(H, syndrome, np.zeros(3, dtype=np.uint8)), syndrome)
    assert np.array_equal(residual_syndrome(H, syndrome, commits), [1, 0])
    total = combine_correction(commits, residual)
    residual_pred = np.asarray((L @ residual) % 2, dtype=np.uint8).reshape(-1)
    combined = combine_observables(L, commits, residual_pred)
    assert np.array_equal(combined, np.asarray((L @ total) % 2).reshape(-1))
    assert combined[-1] == 1


def test_wrong_commit_survives_fallback_and_is_scored(monkeypatch):
    from evaluation.evaluate import run_basis
    from evaluation.test_evaluation import fixture_data

    H = sp.csr_matrix(([1], ([0], [0])), shape=(936, 3), dtype=np.uint8)
    L = sp.csr_matrix(([1], ([11], [0])), shape=(12, 3), dtype=np.uint8)
    problem = Problem("X", H, L, np.full(3, .01), .004, None)
    session = object.__new__(ComposedSession)
    session.problem = problem

    class Committer:
        def commit(self, syndrome, full):
            return np.array([1 - int(full[1]), 0, 0], dtype=np.uint8), {"bp_context_iterations": 8}

    class Fallback:
        def decode_shot(self, full, residual):
            assert residual[0] == 1 - int(full[1])
            return ShotResult(np.zeros(12, dtype=np.uint8),
                              {"first_leg_converged": False, "fallback_used": True, "iterations": 4})

    session.committer = Committer()
    session.downstream = Fallback()
    data = fixture_data("X", [[0, 0, 0], [0, 1, 0]])
    result = run_basis(session, data)
    assert result["failed"].tolist() == [True, False]
    assert result["predictions"][:, -1].tolist() == [1, 0]
    assert [m["fallback_used"] for m in result["metrics"]] == [True, True]


def test_zero_commits_reproduce_downstream_prediction():
    from evaluation.evaluate import run_basis
    from evaluation.test_evaluation import fixture_data

    problem = Problem("X", sp.csr_matrix((936, 3)), sp.csr_matrix((12, 3)), np.full(3, .01), .004, None)
    session = object.__new__(ComposedSession)
    session.problem = problem

    class ZeroCommitter:
        def commit(self, syndrome, full):
            return np.zeros(3, dtype=np.uint8), {}

    class Downstream:
        def decode_shot(self, full, residual):
            pred = np.zeros(12, dtype=np.uint8)
            pred[-1] = full[0]
            return ShotResult(pred, {"first_leg_converged": True, "iterations": 1})

    session.committer = ZeroCommitter()
    session.downstream = Downstream()
    result = run_basis(session, fixture_data("X", [[0, 0, 0], [1, 0, 0]]))
    assert result["predictions"][:, -1].tolist() == [0, 1]


def test_tiered_router_preserves_order_and_reruns_only_hard_risky_shot(monkeypatch):
    import decoders
    import evaluation.router as module
    from evaluation.evaluate import run_basis
    from evaluation.test_evaluation import fixture_data

    calls = []

    class FakeRelay:
        def __init__(self, full):
            self.full = full

        def decode_detailed(self, syndrome):
            calls.append((self.full, int(syndrome[0])))
            errors = np.zeros(8784, dtype=np.uint8)
            errors[-1] = int(self.full)
            return SimpleNamespace(success=True, iterations=81 if syndrome[0] else 2,
                                   posterior_ratios=np.ones(8784), decoding=errors)

    monkeypatch.setattr(decoders, "relay_decoder", lambda H, priors, **changes: FakeRelay(changes.get("stop_nconv") != 1))
    monkeypatch.setattr(module, "load_frozen_router", lambda path, problem: ({"threshold": .5}, "fakehash"))
    monkeypatch.setattr(module, "risk", lambda model, features: .9)
    L = sp.csr_matrix(([1], ([11], [8783])), shape=(12, 8784), dtype=np.uint8)
    problem = Problem("X", sp.csr_matrix((936, 8784)), L, np.full(8784, .01), .004, None)
    session = TieredSession(problem, "unused.json")
    result = run_basis(session, fixture_data("X", [[0, 0, 0], [1, 0, 0]]))
    assert calls == [(False, 0), (False, 1), (True, 1)]
    assert result["predictions"][:, -1].tolist() == [0, 1]
    assert [m["fallback_used"] for m in result["metrics"]] == [False, True]


def test_installed_memory_bp_sign_and_detailed_field_meanings():
    import relay_bp

    H = sp.csr_matrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8))
    bp = relay_bp.MinSumBPDecoderF64(H, error_priors=np.full(3, .01), max_iter=8, alpha=1.0, gamma0=.15)
    quiet = bp.decode_detailed(np.array([0, 0], dtype=np.uint8))
    fault = bp.decode_detailed(np.array([1, 1], dtype=np.uint8))
    assert quiet.success and quiet.iterations >= 1 and (quiet.posterior_ratios > 0).all()
    assert fault.success and fault.decoding[1] == 1 and fault.posterior_ratios[1] < 0
    assert bp_present_probability(quiet.posterior_ratios).max() < .5
    assert bp_present_probability(fault.posterior_ratios)[1] > .5
    assert not first_leg_converged(SimpleNamespace(success=True, iterations=81))
    assert first_leg_converged(SimpleNamespace(success=True, iterations=80))


def test_router_features_and_logistic_threshold():
    result = SimpleNamespace(iterations=90, success=True, posterior_ratios=np.ones(8784),
                             decoding=np.zeros(8784, dtype=np.uint8))
    syndrome = np.zeros(936, dtype=np.uint8)
    syndrome[-1] = 1
    features = candidate_features(result, syndrome)
    assert len(features) == len(FEATURE_NAMES) == 17
    assert features[-1] == 1 and features[-2] == 0
    X = np.tile(features, (12, 1))
    X[:, 0] = np.arange(12)
    wrong = np.array([0] * 9 + [1] * 3, dtype=bool)
    model = fit_logistic(X, wrong)
    scores = risk(model, X)
    threshold = select_threshold(scores, wrong, 0)
    report = threshold_report(scores, wrong, threshold)
    assert report["wrong_accepted"] == 0
    assert report["wrong_escalated"] == 3


def test_dem_shard_sampling_is_deterministic_and_uses_matrix_parity():
    H = sp.csr_matrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8))
    L = sp.csr_matrix(np.array([[0, 0, 1]], dtype=np.uint8))
    priors = np.array([.2, .4, .6])
    a = sample_mechanisms(H, L, priors, 6, 123)
    b = sample_mechanisms(H, L, priors, 6, 123)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    e, s, o = a
    assert np.array_equal(s, (H @ e.T).T % 2)
    assert np.array_equal(o, (L @ e.T).T % 2)


def test_promotion_requires_measured_total_latency_improvement():
    base = {"first_leg_hard_fraction": .30, "failures": 10, "mean_total_ms": 100}
    bp = {"first_leg_hard_fraction": .25, "failures": 10, "mean_total_ms": 95}
    cnn = {"first_leg_hard_fraction": .18, "failures": 9, "mean_total_ms": 101}
    gate = promotion_gate(base, bp, cnn)
    assert all(value for key, value in gate.items() if key != "measured_total_latency_improves")
    assert gate["measured_total_latency_improves"] is False
    assert all(promotion_gate(base, bp, cnn | {"mean_total_ms": 90}).values())


def test_router_training_freezes_threshold_from_distinct_validation(tmp_path):
    import json
    from evaluation.router_train import train

    X = np.zeros((16, len(FEATURE_NAMES)))
    X[:, 0] = np.arange(16)
    wrong = np.array([0] * 12 + [1] * 4, dtype=bool)
    easy = np.zeros(16, dtype=bool)
    matrix_hash = "fixture-matrix"
    paths = {}
    for split, seed in (("train", 1), ("validation", 2)):
        path = tmp_path / f"{split}.npz"
        metadata = {"split": split, "basis": "X", "p": .004, "matrix_sha256": matrix_hash,
                    "feature_names": list(FEATURE_NAMES),
                    "dataset": {"sha256": f"fixture-{seed}", "seed": seed}}
        np.savez(path, features=X, wrong=wrong, first_leg_converged=easy, metadata=json.dumps(metadata))
        paths[split] = path
    output = tmp_path / "router.json"
    model = train(paths["train"], paths["validation"], 0, output)
    assert output.exists() and model["validation"]["wrong_accepted"] == 0
    assert model["train_dataset"]["seed"] != model["validation_dataset"]["seed"]


def test_router_threshold_plot_input_stays_labeled_validation_only(tmp_path):
    from evaluation.plot_results import load_rows

    folder = tmp_path / "validation_sweep"
    folder.mkdir()
    path = folder / "summary.csv"
    path.write_text("decoder,validation_only,p,threshold,per_cycle_rate,mean_pair_ms,p99_pair_ms\n"
                    "tiered-router-validation,True,0.004,0.2,0.001,20,50\n")
    row, = load_rows([path])
    assert row["validation_only"] is True
    assert "validation-only" in row["source"]
    assert row["threshold"] == "0.2"


def test_residual_router_training_label_restores_committed_logical_frame(monkeypatch):
    import decoders
    from evaluation.router_features import generate
    from evaluation.test_evaluation import fixture_data

    class FakeRelay:
        def decode_detailed(self, syndrome):
            assert syndrome[0] == 1
            return SimpleNamespace(success=True, iterations=81, posterior_ratios=np.ones(8784),
                                   decoding=np.zeros(8784, dtype=np.uint8))

    class Committer:
        def commit(self, syndrome, full):
            commits = np.zeros(8784, dtype=np.uint8)
            commits[-1] = 1
            return commits, {}

    monkeypatch.setattr(decoders, "relay_decoder", lambda H, priors, **changes: FakeRelay())
    H = sp.csc_matrix(([1], ([0], [8783])), shape=(936, 8784), dtype=np.uint8)
    L = sp.csr_matrix(([1], ([11], [8783])), shape=(12, 8784), dtype=np.uint8)
    data = fixture_data("X", [[0, 0, 0]])
    data.observables[0, -1] = 1
    result = generate(data, H, L, np.full(8784, .01), Committer())
    assert result["candidate_predictions"][0, -1] == 1
    assert result["wrong"].tolist() == [False]
