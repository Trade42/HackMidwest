"""Tests of predecoder.tcbp without a GPU: the kit's relay-BP stands in for TCBP-JOBS behind the same calls.

  python -m pytest -p no:cacheprovider predecoder/test_tcbp.py
"""
import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.data import load_dataset
from predecoder import tcbp
from predecoder.decode import decode_with_priors, kit_problem
from predecoder.priors import shot_priors

KIT = Path(__file__).resolve().parents[1]
VALIDATION = KIT / "data" / "p0.004" / "validation_X.npz"


def small_file(path, count, start=0):
    with np.load(VALIDATION) as file:
        np.savez_compressed(path, p=file["p"], seed=file["seed"], basis=file["basis"],
                            detectors=file["detectors"][start:start + count],
                            observables=file["observables"][start:start + count], basis_detectors=file["basis_detectors"])
    return path


def fake_logits(folder, data_path, seed=0):
    """A logits folder in predecoder.evaluate's format: the kit's log-odds plus noise (not a trained head)."""
    data = load_dataset(data_path)
    problem = kit_problem(data.basis, data.p)
    rng = np.random.default_rng(seed)
    logits = np.log(problem.priors / (1 - problem.priors))[None] + rng.normal(0, 2, (len(data.detectors), problem.H.shape[1]))
    folder.mkdir(parents=True)
    np.save(folder / "logits.npy", logits.astype(np.float16))
    meta = {"data": data.identity(), "shots": len(data.detectors), "columns": problem.H.shape[1],
            "model": {"head_sha256": "fake", "trunk_sha256": "fake", "device": "cpu"}}
    (folder / "meta.json").write_text(json.dumps(meta))
    return str(folder)


def test_choice_rule_and_grid_edges():
    def row(alpha, score, repr_=0.0, flag=0.0):
        return {"alpha": alpha, "score": score, "not_reproduced_weighted": repr_, "flag_weighted": flag}

    inside = [row(a, 10.0) for a in tcbp.GRID]
    inside[3]["score"] = 5.0
    chosen, extended = tcbp.choose(inside + [row(1.0, 0.0)])
    assert chosen["alpha"] == tcbp.GRID[3] and not extended          # the extension rows do not join: no edge
    edge = [row(a, 10.0 - a) for a in tcbp.GRID]                       # the best declared row is 0.7, an edge
    chosen, extended = tcbp.choose(edge + [row(0.8, 9.0), row(0.9, 1.0), row(1.0, 2.0), row(0.05, 0.0)])
    assert chosen["alpha"] == 0.9 and extended                         # the high side joins; the low side does not
    ties = [row(a, 3.0, repr_=1.0) for a in tcbp.GRID]
    ties[4]["not_reproduced_weighted"] = 0.0
    ties[5]["not_reproduced_weighted"] = 0.0
    assert tcbp.choose(ties)[0]["alpha"] == tcbp.GRID[4]               # then the smaller alpha
    grid = tcbp.settings_grid()
    assert grid[0]["alpha"] == 0 and [g["alpha"] for g in grid[1:8]] == tcbp.GRID


def test_decode_with_the_stand_in_gives_relay_bp_answers(tmp_path, monkeypatch):
    """fresh = relay-BP built per shot with the kit's priors; head = the same with shot_priors of the logits."""
    shots = small_file(tmp_path / "validation_X.npz", 3)
    logits = fake_logits(tmp_path / "logits", shots)
    settings = tmp_path / "fit.json"
    settings.write_text(json.dumps({"chosen": {"mode": "logodds", "alpha": 0.3, "eps": 1e-4}}))
    seen = []
    build = tcbp.Builder.build
    monkeypatch.setattr(tcbp.Builder, "build", lambda self, priors: seen.append(np.array(priors)) or build(self, priors))
    tcbp.main(["decode", "--decoder", "relay", "--data", str(shots), "--logits", logits, "--settings", str(settings),
               "--arms", "fresh", "head", "--batch-check", "--out", str(tmp_path / "out")])
    with np.load(tmp_path / "out" / "per_shot.npz") as file:
        result = {key: file[key] for key in file.files}
    data = load_dataset(shots)
    problem = kit_problem(data.basis, data.p)
    raw = np.load(Path(logits) / "logits.npy")
    for k in range(3):
        fresh = decode_with_priors(problem.H, problem.L, problem.priors, data.basis_input[k])
        priors = shot_priors(raw[k], problem.priors, 0.3, 1e-4, "logodds")
        head = decode_with_priors(problem.H, problem.L, priors, data.basis_input[k])
        assert np.array_equal(result["predicted_fresh"][k], fresh.predicted)
        assert np.array_equal(result["predicted_head"][k], head.predicted)
        assert result["flag_head"][k] == head.converged
        assert result["logits_hash_head"][k] == tcbp.row_hash(raw[k]) and result["logits_hash_fresh"][k] == 0
    # one warm-up, 3 fresh, 3 head, 1 for the batch check; the head's priors are the mixed and clipped ones
    assert len(seen) == 8 and np.array_equal(seen[4], shot_priors(raw[0], problem.priors, 0.3, 1e-4, "logodds"))
    assert np.array_equal(seen[1], problem.priors)
    assert (result["failed_head"] > 0).tolist() == (result["predicted_head"] != data.observables).any(axis=1).tolist()
    assert np.array_equal(result["predicted_batch"], result["predicted_fresh"])     # relay-BP built once, seed 0: the same

    # score: the stored TCBP file and a relay per-shot file, read beside it
    stored = tmp_path / "validation_X.tcbp.npz"
    np.savez(stored, predicted_T=result["predicted_fresh"], failed_T=result["failed_fresh"] > 0, data_sha256=data.sha256)
    relay = tmp_path / "relay_per_shot.npz"
    np.savez(relay, predicted_head=result["predicted_head"], failed_head=result["failed_head"],
             converged_head=np.array([1.0, 0.0, 1.0]), truth=data.observables)
    tcbp.main(["score", "--data", str(shots), "--tcbp-head", str(tmp_path / "out" / "per_shot.npz"), "--tcbp", str(stored),
               "--relay-head", str(relay), "--logits", logits, "--out", str(tmp_path / "score.json")])
    summary = json.loads((tmp_path / "score.json").read_text())
    assert summary["logits_rows_equal_to_local"] == 3
    assert summary["arms"]["TCBP-JOBS alone (stored)"]["failures"] == int((result["failed_fresh"] > 0).sum())


def test_fit_runs_and_resumes(tmp_path):
    failed = small_file(tmp_path / "train_X_T_failed.npz", 2)
    right = small_file(tmp_path / "train_X_T_right.npz", 2, start=2)
    folders = [fake_logits(tmp_path / "lf", failed), fake_logits(tmp_path / "lr", right, seed=1)]
    out = tmp_path / "fit.json"
    args = ["fit", "--decoder", "relay", "--data", str(failed), str(right), "--weights", "1", "3", "--logits", *folders,
            "--rows", "3", "--out", str(out)]
    tcbp.main(args)
    fit = json.loads(out.read_text())
    assert [row["alpha"] for row in fit["rows"]] == [0.0, 0.1, 0.2] and fit["chosen"] is None     # an incomplete grid chooses nothing
    assert len(list(out.with_suffix(".parts").iterdir())) == 6
    tcbp.main(args)                                                    # every part is reused
    assert json.loads(out.read_text())["rows"] == fit["rows"]
    with pytest.raises(SystemExit):
        tcbp.main(["fit", "--decoder", "relay", "--data", str(small_file(tmp_path / "validation_X.npz", 2)),
                   "--logits", folders[0], "--out", str(tmp_path / "no.json")])


def test_held_out_files_are_refused(tmp_path):
    held_out = small_file(tmp_path / "test_X.npz", 2)
    with pytest.raises(SystemExit):
        tcbp.main(["decode", "--decoder", "relay", "--data", str(held_out), "--arms", "fresh", "--out", str(tmp_path / "o")])
