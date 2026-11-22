"""Paired report checks on tiny saved-shot artifacts, with no real decoding."""
import hashlib
import json

import numpy as np
import pytest

from evaluation.compare_gain import compare, export


def _dataset(path, basis, n=4):
    np.savez(path, detectors=np.zeros((n, 1728), dtype=np.uint8),
             observables=np.zeros((n, 12), dtype=np.uint8),
             basis_detectors=np.arange(936), p=.004, seed=301 if basis == "X" else 302,
             basis=basis)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "basis": basis, "p": .004, "seed": 301 if basis == "X" else 302, "shots": n}


def _make_run(path, decoder, datasets, failures_x, failures_z, total_pair, machine="arm64"):
    from decoders import per_cycle_rate

    path.mkdir()
    n = len(failures_x)
    failures_x, failures_z = np.asarray(failures_x, bool), np.asarray(failures_z, bool)
    failed_pair = failures_x | failures_z
    pred_x, pred_z = np.zeros((n, 12), dtype=np.uint8), np.zeros((n, 12), dtype=np.uint8)
    pred_x[:, -1], pred_z[:, -1] = failures_x, failures_z
    total_pair = np.asarray(total_pair, dtype=np.float64)
    total_x, total_z = total_pair / 2, total_pair / 2
    decode_x, decode_z = total_x * .8, total_z * .8
    decode_pair = decode_x + decode_z
    artifact_name = "shots.npz"
    np.savez(path / artifact_name, predictions_X=pred_x, predictions_Z=pred_z,
             failed_X=failures_x, failed_Z=failures_z, failed_pair=failed_pair,
             seconds_X=decode_x, seconds_Z=decode_z, seconds_pair=decode_pair,
             seconds_total_X=total_x, seconds_total_Z=total_z, seconds_total_pair=total_pair,
             kit_seconds_X=decode_x, kit_seconds_Z=decode_z)
    rate, error = per_cycle_rate(failed_pair, 12)
    row = {"paired_shots": n, "failures_X": int(failures_x.sum()), "failures_Z": int(failures_z.sum()),
           "failures_pair": int(failed_pair.sum()), "per_cycle_rate": float(rate),
           "standard_error": float(error), "mean_pair_total_ms": float(total_pair.mean() * 1e3),
           "artifacts": artifact_name}
    metadata = {"p": .004, "cycles": 12, "datasets": {"test": datasets},
                "circuits": {"X": {"sha256": "circuit-x"}, "Z": {"sha256": "circuit-z"}},
                "machine": {"system": "Darwin", "machine": machine, "processor": "Apple M4",
                            "workers": 1, "thread_environment": {}},
                "package_versions": {name: "fixture" for name in ("numpy", "scipy", "stim", "ldpc", "relay-bp")},
                "decoders": [{"name": decoder, "summary": row}]}
    (path / "summary.json").write_text(json.dumps(metadata))
    return path


def test_paired_percent_gain_uses_same_shots_and_total_timers(tmp_path):
    datasets = {basis: _dataset(tmp_path / f"test_{basis}.npz", basis) for basis in "XZ"}
    reference = _make_run(tmp_path / "reference", "relay-bp", datasets,
                          [1, 0, 0, 0], [0, 1, 0, 0], [.2, .3, .4, .5])
    candidate = _make_run(tmp_path / "candidate", "evaluation.router:make_tiered_decoder", datasets,
                          [0, 1, 0, 0], [0, 0, 0, 0], [.1, .2, .3, .4])
    report = compare(reference, candidate, "evaluation.router:make_tiered_decoder", repetitions=200, seed=7)
    assert report["classification"] == "held_out_paired_saved_shots"
    assert report["reference"]["failures_pair"] == 2
    assert report["candidate"]["failures_pair"] == 1
    assert report["paired_failures"]["reference_only_fail"] == 1
    assert report["paired_failures"]["candidate_only_fail"] == 0
    assert report["per_cycle_rate_reduction_percent"] > 0
    assert report["mean_total_latency_reduction_percent"] == pytest.approx(100 * (1 - .25 / .35))
    assert report["mean_total_latency_reduction_bootstrap_95pct"] is not None
    export(report, tmp_path / "comparison")
    assert (tmp_path / "comparison" / "comparison.csv").exists()
    assert (tmp_path / "comparison" / "comparison.json").exists()


def test_comparison_rejects_log_and_mismatched_test_shots(tmp_path):
    datasets = {basis: _dataset(tmp_path / f"test_{basis}.npz", basis) for basis in "XZ"}
    reference = _make_run(tmp_path / "reference", "relay-bp", datasets,
                          [1, 0, 0, 0], [0, 0, 0, 0], [.2] * 4)
    candidate = _make_run(tmp_path / "candidate", "router", datasets,
                          [0, 0, 0, 0], [0, 0, 0, 0], [.1] * 4)
    with pytest.raises(ValueError, match="not baselines.log"):
        compare(tmp_path / "baselines.log", candidate, "router", repetitions=100)
    metadata = json.loads((candidate / "summary.json").read_text())
    metadata["datasets"]["test"]["X"]["sha256"] = "different-shots"
    (candidate / "summary.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="test datasets differ"):
        compare(reference, candidate, "router", repetitions=100)


def test_comparison_rejects_tampered_failure_mask_and_environment(tmp_path):
    datasets = {basis: _dataset(tmp_path / f"test_{basis}.npz", basis) for basis in "XZ"}
    reference = _make_run(tmp_path / "reference", "relay-bp", datasets,
                          [1, 0, 0, 0], [0, 0, 0, 0], [.2] * 4)
    candidate = _make_run(tmp_path / "candidate", "router", datasets,
                          [0, 0, 0, 0], [0, 0, 0, 0], [.1] * 4)
    artifact = candidate / "shots.npz"
    with np.load(artifact, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    arrays["failed_X"][0] = True
    np.savez(artifact, **arrays)
    with pytest.raises(ValueError, match="failure mask disagrees"):
        compare(reference, candidate, "router", repetitions=100)
    metadata = json.loads((candidate / "summary.json").read_text())
    metadata["machine"]["machine"] = "x86_64"
    (candidate / "summary.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="machine setting"):
        compare(reference, candidate, "router", repetitions=100)


def test_zero_reference_failures_has_no_finite_percent_gain(tmp_path):
    datasets = {basis: _dataset(tmp_path / f"test_{basis}.npz", basis) for basis in "XZ"}
    reference = _make_run(tmp_path / "reference", "relay-bp", datasets,
                          [0] * 4, [0] * 4, [.2] * 4)
    candidate = _make_run(tmp_path / "candidate", "router", datasets,
                          [1, 0, 0, 0], [0] * 4, [.1] * 4)
    report = compare(reference, candidate, "router", repetitions=100)
    assert report["per_cycle_rate_reduction_percent"] is None
    assert report["paired_failures"]["per_cycle_rate_reduction_bootstrap_95pct"] is None
