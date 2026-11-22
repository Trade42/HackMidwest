"""Checks of the ML-first decoder's plumbing. Run from challenge-1-ml-decoder-kit/:

  python -m pytest -p no:cacheprovider ml_decoder

Needs torch (ml_decoder/requirements.txt); skipped without it. No test reads a dataset or needs a GPU.
"""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

torch = pytest.importorskip("torch")

from decoders import RELAY_SETTINGS  # noqa: E402
from ml_decoder.train_logical_cnn import LogicalCNN, load_checkpoint  # noqa: E402

KIT = Path(__file__).resolve().parent.parent


def save_model(path, **fields):
    model = LogicalCNN(width=8, depth=1)
    torch.save({"model": model.state_dict(), "config": dict(width=8, depth=1, kernel=5), **fields}, path)
    return model


def test_checkpoint_loads_with_weights_only_and_keeps_its_fields(tmp_path):
    path = tmp_path / "model_X.pt"
    model = save_model(path, basis="X", p_train=[0.0001], seed=3, shots=256)
    loaded, saved = load_checkpoint(path, "X", "cpu")
    assert saved["basis"] == "X" and saved["p_train"] == [0.0001] and saved["seed"] == 3
    assert not loaded.training
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), loaded.state_dict().values()))


def test_checkpoint_of_the_other_basis_is_refused(tmp_path):
    path = tmp_path / "model_X.pt"
    save_model(path, basis="X")
    with pytest.raises(ValueError, match="trained on X memory"):
        load_checkpoint(path, "Z", "cpu")


def test_older_checkpoint_without_basis_loads_with_a_warning(tmp_path, capsys):
    path = tmp_path / "old.pt"
    save_model(path)                                         # the format of the code at 0eab7dd: model and config only
    load_checkpoint(path, "Z", "cpu")
    assert "records no basis" in capsys.readouterr().err


def toy_problem(basis):
    """A problem with the kit's 936 basis detectors and a trivial check matrix, enough to build a session."""
    H = sp.identity(936, dtype=np.uint8, format="csc")
    return SimpleNamespace(basis=basis, H=H, L=sp.csr_matrix((12, 936), dtype=np.uint8), priors=np.full(936, 0.01))


def test_plugin_refuses_the_other_basis(tmp_path):
    from ml_decoder.plugin import make_decoder
    path = tmp_path / "model_X.pt"
    save_model(path, basis="X")
    with pytest.raises(ValueError, match="trained on X memory"):
        make_decoder(toy_problem("Z"), {"models": {"Z": str(path)}, "threshold": 1.0, "device": "cpu"})


def test_plugin_falls_back_to_the_kit_relay_settings_unless_told_otherwise(tmp_path):
    from ml_decoder.plugin import make_decoder
    path = tmp_path / "model_X.pt"
    save_model(path, basis="X")
    options = {"models": {"X": str(path)}, "threshold": 1e9, "device": "cpu"}
    session = make_decoder(toy_problem("X"), options)
    assert session.settings["relay"] == RELAY_SETTINGS | {"seed": 1}
    result = session.decode_shot(np.zeros(1728, np.uint8), np.zeros(936, np.uint8))
    assert result.metrics["fallback_used"] and result.predictions.tolist() == [0] * 12
    faster = {"stop_nconv": 1, "pre_iter": 40, "set_max_iter": 30}
    session = make_decoder(toy_problem("X"), options | {"relay": faster})
    assert session.settings["relay"] == RELAY_SETTINGS | {"seed": 1} | faster


def test_seeded_shot_budget_run_repeats_exactly(tmp_path):
    command = [sys.executable, str(KIT / "ml_decoder" / "train_logical_cnn.py"), "--basis", "X", "--p-train", "0.0001",
               "--p-report", "0.0001", "--width", "8", "--depth", "1", "--batch", "32", "--train-shots", "320",
               "--report-shots", "160", "--shots", "64", "--seed", "5", "--threads", "1", "--out", "model.pt"]
    outputs = []
    for run in ("a", "b"):
        (tmp_path / run).mkdir()
        outputs.append(subprocess.run(command, cwd=tmp_path / run, capture_output=True, text=True, check=True).stdout)
    assert outputs[0] == outputs[1] and "seed 5" in outputs[0].splitlines()[0]
    a, b = (torch.load(tmp_path / run / "model.pt", weights_only=True) for run in "ab")
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    assert a["seed"] == 5 and a["basis"] == "X" and a["shots"] == 320
