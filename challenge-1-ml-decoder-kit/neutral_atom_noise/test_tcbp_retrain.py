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


def test_the_refit_then_the_rf_arm(setup, tmp_path):
    """fit on a train plan (arms m and r), then decode a plan with the arm rf at the chosen knobs."""
    root, logits_root, _, heads = setup
    common = ["--kit-head", sha256_of(heads["kit"]), "--retrained-head", sha256_of(heads["retrained"])]
    train_plan, eval_plan = tmp_path / "train.json", tmp_path / "eval.json"
    with pytest.raises(SystemExit, match="one train task"):
        tcbp_retrain.main(["plan", "--root", str(root), "--task", "small:t:m,r", *common, "--out", str(train_plan)])
        tcbp_retrain.main(["fit", "--plan", str(train_plan), "--root", str(root), "--logits-root", str(logits_root),
                           "--stop-at", "2099-01-01T00:00:00Z", "--out", str(tmp_path / "fit.json"), "--decoder", "relay"])
    tcbp_retrain.main(["plan", "--root", str(root), "--task", "small:t:m,r", *common, "--role", "train",
                       "--out", str(train_plan)])
    tcbp_retrain.main(["fit", "--plan", str(train_plan), "--root", str(root), "--logits-root", str(logits_root),
                       "--stop-at", "2099-01-01T00:00:00Z", "--out", str(tmp_path / "fit.json"), "--decoder", "relay",
                       "--limit", "1"])
    fit = json.loads((tmp_path / "fit.json").read_text())
    assert [row["arm"] for row in fit["rows"]] == ["m"] + ["r"] * len(tcbp_retrain.FIT_GRID)
    assert fit["complete"] and fit["chosen"] in [{k: g[k] for k in ("mode", "alpha", "eps")} for g in tcbp_retrain.FIT_GRID]
    assert {"alpha": 1.0, "eps": 1e-6, "mode": "logodds"} in tcbp_retrain.FIT_GRID
    best = min(row["failures"] for row in fit["rows"][1:])
    assert fit["rows"][1:][[row["failures"] for row in fit["rows"][1:]].index(best)]["failures"] == best
    tcbp_retrain.main(["plan", "--root", str(root), "--task", "small:t:m,r,rf", *common, "--primary", "rf:m",
                       "--out", str(eval_plan)])
    with pytest.raises(SystemExit, match="--refit"):
        decode((root, logits_root, eval_plan, heads), tmp_path / "no_refit", limit=1)
    tcbp_retrain.main(["decode", "--plan", str(eval_plan), "--root", str(root), "--logits-root", str(logits_root),
                       "--settings", KNOBS, "--stop-at", "2099-01-01T00:00:00Z", "--out", str(tmp_path / "dec"),
                       "--decoder", "relay", "--limit", "2", "--refit", str(tmp_path / "fit.json")])
    summary = json.loads((tmp_path / "dec" / "small" / "summary.json").read_text())
    assert summary["knobs_per_arm"]["rf"] == fit["chosen"] and summary["knobs_per_arm"]["r"]["alpha"] == 0.4
    tcbp_retrain.main(["score", "--plan", str(eval_plan), "--root", str(root), "--decoded", str(tmp_path / "dec"),
                       "--out", str(tmp_path / "score.json")])
    pairs = json.loads((tmp_path / "score.json").read_text())["tasks"]["small"]["pairs"]
    assert (pairs[0]["kind"], pairs[0]["a"], pairs[0]["b"]) == ("primary", "rf", "m")
    assert ("rf", "r") in [(q["a"], q["b"]) for q in pairs if q["kind"] == "secondary"]


def test_the_fit_rule_breaks_ties_as_declared():
    row = lambda alpha, eps, f, nr=0, fl=0: {"alpha": alpha, "eps": eps, "failures": f, "not_reproduced": nr, "flag_not_converged": fl}
    rows = [row(0.4, 1e-4, 9), row(0.2, 1e-6, 9), row(0.2, 1e-4, 9), row(1.0, 1e-4, 8, nr=1)]
    assert min(rows, key=tcbp_retrain.fit_key) == row(1.0, 1e-4, 8, nr=1)            # fewest failures first
    rows = rows[:3]
    assert min(rows, key=tcbp_retrain.fit_key) == row(0.2, 1e-4, 9)                  # then smaller alpha, then larger eps
    assert min([row(0.1, 1e-4, 9, fl=5), row(0.6, 1e-4, 9, fl=4)], key=tcbp_retrain.fit_key)["alpha"] == 0.6
