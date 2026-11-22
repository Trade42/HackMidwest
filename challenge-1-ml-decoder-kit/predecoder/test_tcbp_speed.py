"""Tests of predecoder.tcbp_speed and of tcbp.py's --h-format, without a GPU.

The kit's relay-BP stands in for TCBP-JOBS (tcbp.RelayStandIn). It ignores the form of H, so these tests check the
plumbing only: that the check decodes the same shots with the same priors as tcbp.py's head arm, and that the
registered runs' part files keep their names. Whether a sparse H gives TCBP-JOBS the same answers is the server
job's question.

  python -m pytest -p no:cacheprovider predecoder/test_tcbp_speed.py
"""
import json

import numpy as np
import pytest

from evaluation.data import load_dataset
from predecoder import tcbp, tcbp_speed
from predecoder.decode import kit_problem
from predecoder.test_tcbp import VALIDATION, fake_logits, small_file


def test_h_matrix_holds_the_dense_ones_in_canonical_form():
    problem = kit_problem("X", 0.004)
    dense = np.ascontiguousarray(problem.H.toarray(), dtype=np.uint8)
    assert tcbp.h_matrix(dense, "dense") is dense                       # the registered path: the array itself
    for form in ("csr", "csc"):
        matrix = tcbp.h_matrix(dense, form)
        assert matrix.format == form and matrix.dtype == np.uint8 and matrix.has_canonical_format
        assert matrix.nnz == int(dense.sum()) and np.array_equal(matrix.toarray(), dense)
        assert tcbp.h_matrix(dense, form, np.int32).dtype == np.int32
    with pytest.raises(SystemExit):
        tcbp.h_matrix(dense, "coo")


def test_dense_part_files_keep_their_names(tmp_path):
    """--h-format dense (the default) names its part files as the registered runs did; another form does not."""
    shots = small_file(tmp_path / "validation_X.npz", 2)
    data = load_dataset(shots)
    for form in ("dense", "csr"):
        out = tmp_path / form
        tcbp.main(["decode", "--decoder", "relay", "--data", str(shots), "--arms", "fresh", "--h-format", form,
                   "--out", str(out)])
        registered = {"data": data.sha256, "arm": "fresh", "decoder": "relay", "preset": "tcbp-jobs", "opt": {}}
        name = f"fresh_{tcbp.tag(registered)}_000000_2.npz"
        assert (out / "parts" / name).exists() == (form == "dense")
        with open(out / "summary.json") as file:
            assert json.load(file)["h_format"] == form


def test_check_with_the_stand_in_matches_the_head_arm(tmp_path):
    """Every arm decodes the head arm's shots with its priors: the 12 bits and faults equal tcbp.py decode --arms head."""
    shots = small_file(tmp_path / "validation_X.npz", 4)
    logits = fake_logits(tmp_path / "logits", shots)
    settings = tmp_path / "fit.json"
    settings.write_text(json.dumps({"chosen": {"mode": "logodds", "alpha": 0.3, "eps": 1e-4}}))
    tcbp_speed.main(["run", "--decoder", "relay", "--data", str(shots), "--logits", logits, "--settings", str(settings),
                     "--limit", "4", "--block", "2", "--profile-builds", "2", "--out", str(tmp_path / "speed")])
    tcbp.main(["decode", "--decoder", "relay", "--data", str(shots), "--logits", logits, "--settings", str(settings),
               "--arms", "head", "--out", str(tmp_path / "head")])
    with open(tmp_path / "speed" / "summary.json") as file:
        summary = json.load(file)
    with np.load(tmp_path / "speed" / "check_per_shot.npz") as file:
        speed = {key: file[key] for key in file.files}
    with np.load(tmp_path / "head" / "per_shot.npz") as file:
        head = {key: file[key] for key in file.files}
    assert summary["arms_run"] == list(tcbp_speed.ARMS) and summary["repeatable"] is True
    for arm in ("csr", "csc", "dense_again"):
        row = summary["check"][arm]
        assert row["same_as_dense"] and row["bits_differ"] == row["faults_differ"] == row["corrections_differ"] == 0
    fields = list(tcbp_speed.FIELDS)
    predicted = ((speed["predicted_code_dense"][:, None] & tcbp.BITS) > 0).astype(np.uint8)
    assert np.array_equal(predicted, head["predicted_head"])
    assert np.array_equal(speed["rows_dense"][:, fields.index("n_faults")], head["n_faults_head"])
    assert np.array_equal(speed["logits_hash"], head["logits_hash_head"])
    assert summary["check"]["dense"]["failures"] == int(head["failed_head"].sum())
    total = speed["rows_dense"][:, fields.index("seconds_total")]
    parts = speed["rows_dense"][:, [fields.index(name) for name in ("seconds_priors", "seconds_build", "seconds_decode",
                                                                     "seconds_free")]].sum(axis=1)
    assert np.allclose(total, parts)                                     # the one timer is the sum of its parts
    assert (tmp_path / "speed" / "profile_dense.txt").read_text().strip()
    assert summary["profile"]["variants"].startswith("skipped")
    assert VALIDATION.exists()
    tcbp_speed.main(["report", "--summary", str(tmp_path / "speed" / "summary.json"), "--per-shot",
                     str(tmp_path / "speed" / "check_per_shot.npz"), "--reference", str(tmp_path / "head" / "per_shot.npz"),
                     "--out", str(tmp_path / "report.json")])
    report = json.loads((tmp_path / "report.json").read_text())["reference"]
    assert report["logits_rows_equal"] == 4
    assert all(report[arm] == {"bits_differ": 0, "faults_differ": 0} for arm in tcbp_speed.ARMS)
