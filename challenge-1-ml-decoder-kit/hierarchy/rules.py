"""How the hierarchy chooses among its candidates. Plain numpy, used by the plugin and by select.py alike.

A candidate is one decoder's answer to one shot: 12 predicted observable bits, and the weight
of the correction behind them (the sum of log((1 - prior) / prior) over its faults; a lower
weight is a more probable set of faults).

Every array here has the candidates on its first axis, in the order of ORDER, and the shots on
its second. Where two candidates score the same, the first in ORDER wins. Candidates with the
same 12 bits always score the same under the network, so that tie never changes the answer.
"""
import math

import numpy as np
from scipy.stats import binomtest

ORDER = ("R", "P1", "P2")   # relay-BP, PriorTilt under the first seed, PriorTilt under the second seed
P1 = ORDER.index("P1")


def log_sigmoid(x):
    """log(1 / (1 + exp(-x))), without overflow."""
    return -np.logaddexp(0.0, -x)


def network_score(predicted, logits):
    """How likely the network finds each candidate: [candidates, shots].

    predicted: [candidates, shots, 12] bits. logits: [shots, 12], positive = "this bit flipped".
    The score is the sum over the 12 bits of log sigmoid(logit) for a predicted 1 and
    log sigmoid(-logit) for a predicted 0: the log-probability of the 12 bits if the network's
    outputs were independent probabilities.
    """
    logits = np.asarray(logits, np.float64)
    return np.where(np.asarray(predicted) == 1, log_sigmoid(logits), log_sigmoid(-logits)).sum(axis=-1)


def lowest_weight(weight, allowed=None):
    """Index [shots] of the candidate with the lowest weight among the allowed ones.

    allowed: [candidates, shots] bool, or None for all. A shot with no allowed candidate
    chooses among all of them.
    """
    weight = np.asarray(weight, np.float64)
    if allowed is None:
        return weight.argmin(axis=0)
    allowed = np.asarray(allowed, bool)
    return np.where(allowed.any(axis=0), np.where(allowed, weight, np.inf).argmin(axis=0), weight.argmin(axis=0))


def most_likely(predicted, logits, weight=None, constant=None, allowed=None):
    """Index [shots] of the candidate the network finds most likely among the allowed ones (None: all).

    With a constant, the score is the network's score minus the candidate's weight divided by
    the constant. The constant is fitted, so it comes from other shots than the ones scored.
    """
    score = network_score(predicted, logits)
    if constant is not None:
        score = score - np.asarray(weight, np.float64) / constant
    if allowed is not None:
        score = np.where(np.asarray(allowed, bool), score, -np.inf)
    return score.argmax(axis=0)


def agree(predicted):
    """[shots] bool: P1 and P2 predict the same 12 bits."""
    return (predicted[ORDER.index("P1")] == predicted[ORDER.index("P2")]).all(axis=-1)


def tie_break(predicted, choice_where_they_differ):
    """Index [shots] of the hierarchy's answer: P1 where P1 and P2 agree, else the given choice."""
    return np.where(agree(predicted), P1, choice_where_they_differ)


def wilson_interval(failures, shots, z=1.959964):
    """The 95% Wilson score interval of a failure fraction."""
    if shots == 0:
        return 0.0, 1.0
    fraction = failures / shots
    centre = fraction + z * z / (2 * shots)
    half = z * math.sqrt(fraction * (1 - fraction) / shots + z * z / (4 * shots * shots))
    low = 0.0 if failures == 0 else (centre - half) / (1 + z * z / shots)
    high = 1.0 if failures == shots else (centre + half) / (1 + z * z / shots)
    return low, high


def paired_test(failed_a, failed_b):
    """Two decoders on the same shots: (both fail, a only, b only, exact two-sided p).

    The p-value is the exact binomial (McNemar) test on the discordant shots: under "no
    difference", each of them is a's failure or b's with probability one half.
    """
    failed_a, failed_b = np.asarray(failed_a, bool), np.asarray(failed_b, bool)
    both, only_a, only_b = int((failed_a & failed_b).sum()), int((failed_a & ~failed_b).sum()), int((~failed_a & failed_b).sum())
    p_value = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return both, only_a, only_b, float(p_value)
