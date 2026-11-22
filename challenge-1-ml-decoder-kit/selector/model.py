"""The selector's model: a logistic model that compares the answers of one shot (a conditional logit).

Every answer a of a shot gets a score s(a) = w . z(a), where z(a) are its features (features.py),
standardized with the training rows' mean and spread. The model says the chance that answer a
is the right one is exp(s(a)) / sum over the shot's answers of exp(s(b)), and the selector
picks the answer with the highest score (the earlier answer on a tie: TCBP-JOBS's first).

Fitting: maximize the log-chance of the right answer over the training shots where one answer
is right (a shot where none is right teaches nothing), minus lambda / 2 * |w|^2, by L-BFGS.
There is no intercept: a constant added to every answer of a shot changes nothing.

lambda is chosen by 5-fold cross-validation over the training shots (folds by shot, a fixed
seed): the lambda whose held-out folds choose the right answer most often; a tie goes to the
larger lambda. No validation or test shot takes part.

The fitted model is a small JSON file: the feature names, the mean and spread, the weights,
lambda, and the files it was fitted on. Applying it needs numpy only.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

LAMBDAS = (0.001, 0.01, 0.1, 1.0)
FOLDS = 5
FOLD_SEED = 20261122


def _groups(starts):
    return [slice(starts[i], starts[i + 1]) for i in range(len(starts) - 1)]


def _loss(weights, z, groups, right_rows, lam):
    """Mean negative log-chance of the right answer, plus lambda / 2 * |w|^2, and its gradient."""
    scores = z @ weights
    total, gradient = 0.0, np.zeros_like(weights)
    for rows, right in zip(groups, right_rows):
        s = scores[rows]
        top = s.max()
        chance = np.exp(s - top)
        chance /= chance.sum()
        total -= s[right] - top - np.log(np.exp(s - top).sum())
        gradient -= z[rows][right] - chance @ z[rows]
    n = max(len(groups), 1)
    return total / n + lam / 2 * weights @ weights, gradient / n + lam * weights


def fit_weights(z, starts, right, lam):
    """The weights on standardized features z, from the shots whose answers include the right one."""
    groups, right_rows = [], []
    for rows in _groups(starts):
        hit = np.flatnonzero(right[rows])
        if len(hit) == 1:
            groups.append(rows)
            right_rows.append(int(hit[0]))
        elif len(hit) > 1:
            raise ValueError("two different answers of one shot cannot both be right")
    result = minimize(_loss, np.zeros(z.shape[1]), args=(z, groups, right_rows, lam), jac=True, method="L-BFGS-B",
                      options={"maxiter": 2000, "gtol": 1e-9})
    return result.x, len(groups)


def pick(features, starts, model):
    """Index [choice shots] of the row each shot's highest-scoring answer is in (the earlier row on a tie)."""
    z = (features - np.asarray(model["mean"])) / np.asarray(model["scale"])
    scores = z @ np.asarray(model["weights"])
    return np.array([rows.start + int(np.argmax(scores[rows])) for rows in _groups(starts)], int), scores


def standardize(features):
    mean, scale = features.mean(axis=0), features.std(axis=0)
    return mean, np.where(scale > 0, scale, 1.0)


def cross_validate(features, starts, right, lambdas=LAMBDAS, folds=FOLDS, seed=FOLD_SEED, baseline=None):
    """Per lambda: held-out right choices over all folds, on the shots where an answer is right.

    baseline [rows] bool marks the row of a fixed choice per shot (TCBP-JOBS's answer); the result
    then also counts, on the held-out shots, where only the selector is wrong and where only the
    baseline is.
    """
    groups = _groups(starts)
    fold_of = np.random.default_rng(seed).permutation(len(groups)) % folds
    out = {}
    for lam in lambdas:
        counts = {"lambda": lam, "shots": 0, "right": 0, "only_selector_wrong": 0, "only_baseline_wrong": 0, "both_wrong": 0}
        for fold in range(folds):
            train_rows = np.concatenate([np.arange(g.start, g.stop) for g, f in zip(groups, fold_of) if f != fold] or [np.array([], int)])
            train_starts = np.concatenate([[0], np.cumsum([g.stop - g.start for g, f in zip(groups, fold_of) if f != fold])]).astype(int)
            mean, scale = standardize(features[train_rows])
            weights, _ = fit_weights((features[train_rows] - mean) / scale, train_starts, right[train_rows], lam)
            model = {"mean": mean, "scale": scale, "weights": weights}
            for g, f in zip(groups, fold_of):
                if f != fold or not right[g].any():
                    continue
                chosen, _ = pick(features[g], np.array([0, g.stop - g.start]), model)
                good = bool(right[g][chosen[0]])
                counts["shots"] += 1
                counts["right"] += good
                if baseline is not None:
                    base = bool(right[g][np.flatnonzero(baseline[g])[0]])
                    counts["only_selector_wrong"] += (not good) and base
                    counts["only_baseline_wrong"] += good and not base
                    counts["both_wrong"] += (not good) and not base
        out[lam] = counts
    return out


def choose_lambda(cv):
    """The lambda with the most held-out right choices; a tie goes to the larger lambda."""
    return max(cv, key=lambda lam: (cv[lam]["right"], lam))


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plain = {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in model.items()}
    Path(path).write_text(json.dumps(plain, indent=1) + "\n", encoding="utf-8")
    return sha256_of(path)


def load(path):
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("mean", "scale", "weights"):
        model[key] = np.asarray(model[key], float)
    return model
