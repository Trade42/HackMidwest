"""Tests of tcbp_retrain.py without a GPU: plan, decode (the kit's relay-BP stands in for TCBP-JOBS) and score, end
to end on a few model shots, and the refusals that keep each arm's logits and base vector together.

From challenge-1-ml-decoder-kit/:  python -m pytest -p no:cacheprovider neutral_atom_noise/test_tcbp_retrain.py
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from neutral_atom_noise import tcbp_check, tcbp_retrain
from neutral_atom_noise.make_shots import make
from predecoder.test_predecoder import tiny_trunk
from predecoder.train import sha256_of

KIT = Path(__file__).resolve().parents[1]
KNOBS = str(KIT / "predecoder" / "logs" / "fit_tcbp_X_p0.005.json")
P = 0.0326371


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    """(root, logits root, plan path, heads): one task of 6 model shots, converted, and the three logits folders."""
    from predecoder import evaluate, train

    folder = tmp_path_factory.mktemp("retrain")
    trunk = tiny_trunk(folder / "trunk.pt")
    common = ["--trunk", trunk, "--steps", "2", "--batch", "2", "--report-every", "2", "--report-shots", "4",
              "--project", "8", "--hidden", "8", "--shape-width", "4", "--device", "cpu", "--threads", "2",
              "--warmup-steps", "1"]
    train.main(common + ["--out", str(folder / "kit")])
    train.main(common + ["--out", str(folder / "na"), "--noise-factor", "0.8"])
    heads = {"kit": str(folder / "kit" / "head.pt"), "retrained": str(folder / "na" / "head.pt")}
    shots = make("X", 0.8, None, False, 6, 99, folder / "shots")
    root = folder / "transfer"
    tcbp_check.main(["convert", "--shots", str(shots), "--p", str(P), "--out", str(root / "t")])
    data = next((root / "t").glob("*_kitformat_*.npz"))
    matched = str(root / "t" / "priors_matched.npy")
    logits = folder / "logits" / "small"
    run = ["--trunk", trunk, "--data", str(data), "--device", "cpu", "--time-shots", "0"]
    evaluate.main(["--head", heads["kit"], *run, "--out", str(logits / "kit")])
    evaluate.main(["--head", heads["kit"], *run, "--out", str(logits / "kit-matched"), "--base-priors", matched])
    evaluate.main(["--head", heads["retrained"], *run, "--out", str(logits / "retrained"), "--base-priors", matched])
    plan = folder / "plan.json"
    tcbp_retrain.main(["plan", "--root", str(root), "--task", "small:t:m,g,h,hm,r", "--kit-head", sha256_of(heads["kit"]),
                       "--retrained-head", sha256_of(heads["retrained"]), "--out", str(plan)])
    return root, folder / "logits", plan, heads


def decode(setup, out, limit=2):
    root, logits_root, plan, _ = setup
    tcbp_retrain.main(["decode", "--plan", str(plan), "--root", str(root), "--logits-root", str(logits_root),
                       "--settings", KNOBS, "--stop-at", "2099-01-01T00:00:00Z", "--out", str(out),
                       "--decoder", "relay", "--limit", str(limit)])


def test_plan_names_the_arms_in_the_declared_order(setup):
    plan = json.loads(setup[2].read_text())
    assert plan["tasks"][0]["arms"] == ["m", "g", "h", "hm", "r"]
    assert plan["knobs"] == {"mode": "logodds", "alpha": 0.4, "eps": 1e-4} and plan["prior_cap"] == 0.5


def test_decode_and_score_end_to_end(setup, tmp_path):
    root, _, plan, _ = setup
    decode(setup, tmp_path / "dec")
    summary = json.loads((tmp_path / "dec" / "small" / "summary.json").read_text())
    assert summary["shots_scored"] == 2 and set(summary["arms"]) == {"m", "g", "h", "hm", "r"}
    # relay-BP (the stand-in) may stop unconverged at this noise; TCBP-JOBS's OSD always reproduces the syndrome
    assert all(0 <= arm["not_reproduced"] <= 2 for arm in summary["arms"].values())
    assert summary["logits"]["r"]["base_priors"]["trained_with_this_base"] is True
    assert summary["logits"]["hm"]["base_priors"]["trained_with_this_base"] is False
    per_shot = np.load(tmp_path / "dec" / "small" / "per_shot.npz")
    assert (per_shot["logits_hash_m"] == 0).all() and (per_shot["logits_hash_r"] != 0).all()
    assert not np.array_equal(per_shot["logits_hash_h"], per_shot["logits_hash_hm"])     # another base, other logits
    tcbp_retrain.main(["score", "--plan", str(plan), "--root", str(root), "--decoded", str(tmp_path / "dec"),
                       "--out", str(tmp_path / "score.json")])
    score = json.loads((tmp_path / "score.json").read_text())["tasks"]["small"]
    assert score["shots_scored"] == 2 and [p["kind"] for p in score["pairs"]][:2] == ["primary", "primary"]
    assert (score["pairs"][0]["a"], score["pairs"][0]["b"]) == ("r", "m")


def test_an_arm_with_the_wrong_base_or_head_is_refused(setup, tmp_path):
    root, logits_root, plan, _ = setup
    swapped = tmp_path / "logits"
    shutil.copytree(logits_root, swapped)
    shutil.rmtree(swapped / "small" / "kit-matched")
    shutil.copytree(logits_root / "small" / "kit", swapped / "small" / "kit-matched")      # the kit head on generic
    with pytest.raises(SystemExit, match="base was not the matched"):
        tcbp_retrain.main(["decode", "--plan", str(plan), "--root", str(root), "--logits-root", str(swapped),
                           "--settings", KNOBS, "--stop-at", "2099-01-01T00:00:00Z", "--out", str(tmp_path / "o1"),
                           "--decoder", "relay", "--limit", "1"])
    shutil.rmtree(swapped / "small" / "retrained")
    shutil.copytree(logits_root / "small" / "kit-matched", swapped / "small" / "retrained")  # the kit head, not ours
    shutil.rmtree(swapped / "small" / "kit-matched")
    shutil.copytree(logits_root / "small" / "kit-matched", swapped / "small" / "kit-matched")
    with pytest.raises(SystemExit, match="not the retrained head"):
        tcbp_retrain.main(["decode", "--plan", str(plan), "--root", str(root), "--logits-root", str(swapped),
                           "--settings", KNOBS, "--stop-at", "2099-01-01T00:00:00Z", "--out", str(tmp_path / "o2"),
                           "--decoder", "relay", "--limit", "1"])
