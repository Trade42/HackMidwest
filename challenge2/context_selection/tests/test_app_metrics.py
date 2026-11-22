"""Answer metrics and intervals."""

from __future__ import annotations

import numpy as np
import pytest

from qctx.metrics import (
    both_gold,
    bootstrap_ci,
    exact_match,
    f1_score,
    gold_recall,
    normalize_answer,
    paired_bootstrap_diff,
    wilson_interval,
)


def test_normalize_answer_matches_the_official_script():
    assert normalize_answer("The  Quick, Brown fox!") == "quick brown fox"
    assert normalize_answer("An apple a day") == "apple day"
    assert normalize_answer("U.S.A.") == "usa"
    assert normalize_answer("") == ""


def test_exact_match():
    assert exact_match("the Beatles.", "Beatles") == 1.0
    assert exact_match("Liga MX", "liga mx") == 1.0
    assert exact_match("Cruz Azul", "Liga MX") == 0.0
    assert exact_match("", "yes") == 0.0


def test_f1_partial_overlap():
    # prediction {jacobite, uprising}; gold {unsuccessful, jacobite, uprising, of, 1745}
    assert f1_score("Jacobite uprising", "the unsuccessful Jacobite uprising of 1745") == pytest.approx(2 * 1.0 * 0.4 / 1.4)
    assert f1_score("Sam Raimi", "Sam Raimi") == 1.0
    assert f1_score("nothing shared", "Sam Raimi") == 0.0
    assert f1_score("", "Sam Raimi") == 0.0


def test_f1_yes_no_special_case():
    assert f1_score("yes", "yes") == 1.0
    assert f1_score("yes", "no") == 0.0
    assert f1_score("yes it is", "yes") == 0.0      # official rule: yes/no must match exactly
    assert f1_score("noanswer", "no answer here") == 0.0


def test_gold_recall_and_both_gold():
    assert gold_recall([1, 5, 9], [5, 9]) == 1.0
    assert gold_recall([1, 5], [5, 9]) == 0.5
    assert gold_recall([], [5, 9]) == 0.0
    assert gold_recall([1], []) == 0.0
    assert both_gold([9, 5, 0], [5, 9]) == 1.0
    assert both_gold([5], [5, 9]) == 0.0


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    values = [1.0] * 30 + [0.0] * 20
    a = bootstrap_ci(values, n_boot=2000, seed=3)
    b = bootstrap_ci(values, n_boot=2000, seed=3)
    assert a == b
    mean, lo, hi = a
    assert mean == pytest.approx(0.6)
    assert lo < mean < hi
    assert 0.40 < lo < 0.52 and 0.68 < hi < 0.80        # about +-0.136 for n = 50
    assert bootstrap_ci([0.25]) == (0.25, 0.25, 0.25)
    assert all(np.isnan(x) for x in bootstrap_ci([]))
    assert bootstrap_ci([1.0, 1.0, 1.0], n_boot=200) == (1.0, 1.0, 1.0)


def test_paired_bootstrap_detects_a_clear_difference():
    rng = np.random.default_rng(0)
    b = rng.integers(0, 2, size=200).astype(float)
    a = np.maximum(b, (rng.random(200) < 0.3).astype(float))     # a is never worse than b
    res = paired_bootstrap_diff(a, b, n_boot=2000, seed=1)
    assert res["mean_diff"] == pytest.approx(float(a.mean() - b.mean()))
    assert res["lo"] > 0 and res["p_gt0"] == 1.0 and res["n"] == 200
    same = paired_bootstrap_diff(b, b, n_boot=500)
    assert same["mean_diff"] == 0.0 and same["lo"] == 0.0 and same["hi"] == 0.0 and same["p_gt0"] == 0.0
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1.0, 0.0], [1.0])


def test_paired_bootstrap_is_symmetric():
    a = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    b = [0, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    ab = paired_bootstrap_diff(a, b, n_boot=3000, seed=5)
    ba = paired_bootstrap_diff(b, a, n_boot=3000, seed=5)
    assert ab["mean_diff"] == pytest.approx(-ba["mean_diff"])
    assert ab["lo"] == pytest.approx(-ba["hi"]) and ab["hi"] == pytest.approx(-ba["lo"])


def test_wilson_interval():
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    lo0, hi0 = wilson_interval(0, 20)
    assert lo0 == 0.0 and 0.0 < hi0 < 0.2
    assert wilson_interval(0, 0) == (0.0, 1.0)
