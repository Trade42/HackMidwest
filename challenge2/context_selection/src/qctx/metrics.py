"""Answer and selection metrics, with bootstrap intervals over questions.

Exact match and F1 follow the official HotpotQA evaluation script: answers are
lower-cased, stripped of punctuation and English articles, and whitespace is
collapsed before comparison.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Iterable, Sequence

import numpy as np

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = set(string.punctuation)
_SPECIAL = ("yes", "no", "noanswer")


# ------------------------------------------------------------ answer metrics
def normalize_answer(text: str) -> str:
    """Official HotpotQA normalisation: lower, drop punctuation and articles, squeeze spaces."""
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in _PUNCTUATION)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> float:
    """1.0 when the normalised prediction equals the normalised gold answer."""
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    """Token-level F1 with the official yes/no/noanswer special case."""
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    if (p in _SPECIAL or g in _SPECIAL) and p != g:
        return 0.0
    p_tokens = p.split()
    g_tokens = g.split()
    common = Counter(p_tokens) & Counter(g_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    return 2.0 * precision * recall / (precision + recall)


# --------------------------------------------------------- selection metrics
def gold_recall(selected: Iterable[int], gold: Iterable[int]) -> float:
    """Share of the gold paragraphs that were selected."""
    gold_set = {int(i) for i in gold}
    if not gold_set:
        return 0.0
    return len(gold_set & {int(i) for i in selected}) / len(gold_set)


def both_gold(selected: Iterable[int], gold: Iterable[int]) -> float:
    """1.0 when every gold paragraph was selected."""
    gold_set = {int(i) for i in gold}
    if not gold_set:
        return 0.0
    return float(gold_set <= {int(i) for i in selected})


# ----------------------------------------------------------------- intervals
def bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Mean and percentile-bootstrap interval, resampling questions.

    Returns ``(mean, lo, hi)``. With one value or none the interval collapses.
    """
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(v.mean())
    if v.size == 1:
        return (mean, mean, mean)
    rng = np.random.default_rng(seed)
    means = _resampled_means(v, n_boot, rng)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return (mean, float(lo), float(hi))


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Paired bootstrap of ``mean(a) - mean(b)`` over the same questions.

    Returns ``mean_diff``, ``lo``, ``hi`` and ``p_gt0``, the share of resamples
    in which the difference is strictly positive.
    """
    av = np.asarray(list(a), dtype=float)
    bv = np.asarray(list(b), dtype=float)
    if av.shape != bv.shape:
        raise ValueError("paired bootstrap needs the same questions in both lanes")
    d = av - bv
    if d.size == 0:
        nan = float("nan")
        return {"mean_diff": nan, "lo": nan, "hi": nan, "p_gt0": nan, "n": 0}
    rng = np.random.default_rng(seed)
    means = _resampled_means(d, n_boot, rng) if d.size > 1 else np.full(n_boot, d.mean())
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "mean_diff": float(d.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "p_gt0": float(np.mean(means > 0.0)),
        "n": int(d.size),
    }


def _resampled_means(v: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Means of ``n_boot`` resamples of ``v``, drawn in chunks to bound memory."""
    n = v.size
    out = np.empty(n_boot, dtype=float)
    chunk = max(1, min(n_boot, 2_000_000 // max(n, 1)))
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(m, n))
        out[done:done + m] = v[idx].mean(axis=1)
        done += m
    return out


def wilson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes in ``n`` trials."""
    if n <= 0:
        return (0.0, 1.0)
    z = _normal_quantile(1.0 - alpha / 2.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _normal_quantile(q: float) -> float:
    """Standard normal quantile (scipy when available, otherwise bisection on erf)."""
    try:
        from scipy.stats import norm

        return float(norm.ppf(q))
    except Exception:  # pragma: no cover - scipy is a hard dependency of the project
        lo, hi = -10.0, 10.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < q:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
