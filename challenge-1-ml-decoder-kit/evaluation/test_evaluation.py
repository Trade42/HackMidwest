"""Bounded correctness checks: no full decoding or data generation."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from evaluation.adapters import KitSession, Problem, ShotResult, _prediction
from evaluation.data import Dataset, load_dataset, validate_splits
from evaluation.evaluate import run_basis, score_pair
from evaluation.parse_baselines import parse


def fixture_data(basis, first_bits, p=0.004, seed=1):
    det = np.zeros((len(first_bits), 1728), dtype=np.uint8)
    det[:, :3] = np.array(first_bits, dtype=np.uint8)
    obs = np.zeros((len(first_bits), 12), dtype=np.uint8)
    return Dataset(Path(f"{basis}{seed}.npz"), f"hash-{basis}-{seed}-{p}", basis, p, seed,
                   det, obs, np.arange(936))


def test_last_observable_bit_is_scored_and_order_is_preserved():
    data = fixture_data("X", [[0, 0, 0], [1, 0, 0], [0, 1, 0]])

    class Router:
        def decode_shot(self, full, basis):
            assert np.array_equal(full[:3], basis[:3])
            pred = np.zeros(12, dtype=np.uint8)
            pred[-1] = full[0]
            return ShotResult(pred, {"fallback_used": bool(full[1])})

    result = run_basis(Router(), data)
    assert result["predictions"][:, -1].tolist() == [0, 1, 0]
    assert result["failed"].tolist() == [False, True, False]
    assert [m["fallback_used"] for m in result["metrics"]] == [False, False, True]
    assert len(result["seconds"]) == 3


def test_plugin_cannot_exclude_router_work_from_decode_timer():
    data = fixture_data("X", [[0, 0, 0]])

    class Timed:
        def decode_shot(self, full, basis):
            return ShotResult(np.zeros(12, dtype=np.uint8), decode_seconds=.125)

    with pytest.raises(ValueError, match="cannot override"):
        run_basis(Timed(), data)


def test_pair_failure_or_and_time_addition_match_kit():
    from decoders import per_cycle_rate

    x = {"failed": np.array([True, False, False]), "seconds": np.array([.1, .2, .3])}
    z = {"failed": np.array([False, True, False]), "seconds": np.array([.4, .5, .6])}
    failed, seconds, scores = score_pair(x, z, 12)
    assert failed.tolist() == [True, True, False]
    assert seconds == pytest.approx([.5, .7, .9])
    assert scores["failures_pair"] == 2
    assert (scores["per_cycle_rate"], scores["standard_error"]) == pytest.approx(per_cycle_rate(failed, 12))


def test_dataset_shapes_binary_and_split_mismatch(tmp_path):
    good = fixture_data("X", [[0, 0, 0], [1, 0, 0]])
    path = tmp_path / "x.npz"
    np.savez(path, detectors=good.detectors, observables=good.observables,
             basis_detectors=good.basis_detectors, p=.004, seed=1, basis="X")
    loaded = load_dataset(path, "X", np.arange(936))
    assert loaded.basis_input.shape == (2, 936)
    with pytest.raises(ValueError, match="basis must"):
        load_dataset(path, "Z")
    with pytest.raises(ValueError, match="shot counts differ"):
        validate_splits({"test": {"X": loaded, "Z": fixture_data("Z", [[0, 0, 0]])}})
    with pytest.raises(ValueError, match="probabilities differ"):
        validate_splits({"test": {"X": loaded, "Z": fixture_data("Z", [[0, 0, 0], [0, 0, 0]], p=.005)}})
    bad = tmp_path / "bad.npz"
    invalid = good.detectors.copy()
    invalid[0, -1] = 2
    np.savez(bad, detectors=invalid, observables=good.observables,
             basis_detectors=good.basis_detectors, p=.004, seed=1, basis="X")
    with pytest.raises(ValueError, match="other than 0 and 1"):
        load_dataset(bad)
    np.savez(bad, detectors=good.detectors[:, :-1], observables=good.observables,
             basis_detectors=good.basis_detectors, p=.004, seed=1, basis="X")
    with pytest.raises(ValueError, match="1728"):
        load_dataset(bad)


def test_sparse_error_to_all_observables_conversion():
    L = sp.csr_matrix(([1], ([11], [2])), shape=(12, 3), dtype=np.uint8)
    pred = _prediction(L, np.array([0, 0, 1], dtype=np.uint8))
    assert pred.shape == (12,) and pred[-1] == 1 and pred[:-1].sum() == 0


def test_partial_log_marks_missing_entry_without_making_up_measurements():
    lines = ("10000 shots at p=0.003, 12 cycles, failures listed as [X Z]\n"
             "bposd failures [36 23] of 10000  per cycle 4.93e-04 +- 6.4e-05  mean 71.9 ms  p99 467 ms\n"
             "relay-bp failures [3 2] of 10000  per cycle 4.17e-05 +- 1.9e-05  mean 42.5 ms  p99 293 ms\n"
             "10000 shots at p=0.004, 12 cycles, failures listed as [X Z]\n"
             "bposd failures [266 267] of 10000  per cycle 4.50e-03 +- 2.0e-04  mean 170.1 ms  p99 511 ms\n"
             "relay-bp failures [5")
    runs = parse(lines)
    assert runs[0]["status"] == "completed"
    assert runs[1]["status"] == "unfinished"
    assert runs[1]["partial_tail"] == "relay-bp failures [5"
    pending = next(e for e in runs[1]["entries"] if e["decoder"] == "relay-bp")
    assert pending == {"decoder": "relay-bp", "status": "unfinished"}


def test_first_leg_reruns_full_relay_only_after_failure(monkeypatch):
    import decoders

    calls = []

    class FakeDecoder:
        def __init__(self, first):
            self.first = first

        def decode_detailed(self, syndrome):
            calls.append((self.first, int(syndrome[0])))
            return SimpleNamespace(success=(not self.first or syndrome[0] == 0),
                                   decoding=np.array([0, 0, int(syndrome[0])], dtype=np.uint8), iterations=2)

    def fake_relay(H, priors, **changes):
        return FakeDecoder(changes.get("num_sets") == 0)

    monkeypatch.setattr(decoders, "relay_decoder", fake_relay)
    L = sp.csr_matrix(([1], ([11], [2])), shape=(12, 3), dtype=np.uint8)
    problem = Problem("X", sp.csr_matrix((1, 3)), L, np.full(3, .01), .004, None)
    session = KitSession(problem, "relay-first-leg-or-full")
    easy = session.decode_shot(np.zeros(1728), np.array([0], dtype=np.uint8))
    hard = session.decode_shot(np.zeros(1728), np.array([1], dtype=np.uint8))
    assert calls == [(True, 0), (True, 1), (False, 1)]
    assert easy.metrics["fallback_used"] is False
    assert hard.metrics["fallback_used"] is True
    assert hard.metrics["iterations"] == 4
    assert hard.predictions[-1] == 1


def test_five_solution_baseline_uses_plain_kit_decode(monkeypatch):
    import decoders

    called = []

    class FakeDecoder:
        def decode(self, syndrome):
            called.append("decode")
            return np.array([0, 0, 1], dtype=np.uint8)

        def decode_detailed(self, syndrome):
            raise AssertionError("baseline must use the kit's plain decode API")

    monkeypatch.setattr(decoders, "relay_decoder", lambda H, priors, **changes: FakeDecoder())
    L = sp.csr_matrix(([1], ([11], [2])), shape=(12, 3), dtype=np.uint8)
    problem = Problem("X", sp.csr_matrix((1, 3)), L, np.full(3, .01), .004, None)
    result = KitSession(problem, "relay-bp").decode_shot(np.zeros(1728), np.array([0], dtype=np.uint8))
    assert called == ["decode"]
    assert result.predictions[-1] == 1
    assert result.decode_seconds >= 0


def test_runner_writes_auditable_paired_shots_for_each_decoder(monkeypatch, tmp_path):
    import importlib
    import circuit

    runner = importlib.import_module("evaluation.evaluate")
    H = sp.csc_matrix((936, 8784), dtype=np.uint8)
    L = sp.csr_matrix((12, 8784), dtype=np.uint8)
    monkeypatch.setattr(circuit, "dem_matrices", lambda _circuit, _basis: (H, L, np.full(8784, .01)))

    class Dummy:
        setup_seconds = 0.0
        settings = {"kind": "dummy"}

        def decode_shot(self, full, basis):
            pred = np.zeros(12, dtype=np.uint8)
            pred[-1] = basis[0]
            return ShotResult(pred)

    monkeypatch.setattr(runner, "create_session", lambda spec, problem, options: Dummy())
    datasets = {"test": {"X": fixture_data("X", [[0, 0, 0], [1, 0, 0]], seed=1),
                         "Z": fixture_data("Z", [[1, 0, 0], [0, 0, 0]], seed=2)}}
    rows = runner.evaluate(datasets, {"X": None, "Z": None}, ["alpha", "beta"], tmp_path / "results")
    assert len(rows) == 2 and all(row["failures_pair"] == 2 for row in rows)
    for index, spec in enumerate(("alpha", "beta"), 1):
        with np.load(tmp_path / "results" / f"{index:02d}_{spec}_shots.npz") as artifact:
            assert artifact["predictions_X"][:, -1].tolist() == [0, 1]
            assert artifact["predictions_Z"][:, -1].tolist() == [1, 0]
            assert artifact["failed_pair"].tolist() == [True, True]
            assert np.allclose(artifact["seconds_pair"], artifact["seconds_X"] + artifact["seconds_Z"])
