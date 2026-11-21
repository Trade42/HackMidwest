"""Relay first-candidate features, logistic risk model, and tiered adapter."""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from evaluation.adapters import ShotResult, _prediction
from evaluation.predecoder_layout import matrix_identity

FEATURE_NAMES = ("first_solution_iterations", "candidate_converged", "mean_abs_posterior_ratio",
                 "correction_weight", *(f"syndrome_weight_cycle_{i}" for i in range(13)))
POSTERIOR_ABS_CLIP = 100.0


def candidate_features(result, syndrome):
    """The installed relay-bp 0.2.2 returns cumulative iterations and per-fault
    posterior log-odds ratios. Positive ratios favor absence; magnitude is used
    here so feature direction does not imply fault presence.
    """
    syndrome = np.asarray(syndrome, dtype=np.uint8)
    posterior = np.asarray(result.posterior_ratios, dtype=np.float64)
    correction = np.asarray(result.decoding, dtype=np.uint8)
    if syndrome.shape != (936,) or posterior.shape != (8784,) or correction.shape != (8784,):
        raise ValueError("Relay detailed result has an unexpected Gross-code shape")
    if not np.isin(correction, [0, 1]).all():
        raise ValueError("Relay correction is not binary")
    if np.isnan(posterior).any():
        raise ValueError("Relay posterior contains NaN")
    abs_mean = float(np.clip(np.abs(posterior), 0, POSTERIOR_ABS_CLIP).mean())
    cycle_weights = syndrome.reshape(13, 72).sum(axis=1)
    return np.array([result.iterations, int(bool(result.success)), abs_mean, int(correction.sum()),
                     *cycle_weights], dtype=np.float64)


def first_leg_converged(result, pre_iter=80):
    # decode_detailed.iterations is cumulative across legs; a tiny installed-API
    # check verified 2+2+2 iterations for an unsatisfiable two-leg example.
    return bool(result.success) and 0 < int(result.iterations) <= pre_iter


def fit_logistic(features, wrong, l2=1.0):
    features = np.asarray(features, dtype=np.float64)
    wrong = np.asarray(wrong, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES) or wrong.shape != (len(features),):
        raise ValueError("router features or labels have the wrong shape")
    if not np.isfinite(features).all() or not np.isin(wrong, [0, 1]).all() or len(np.unique(wrong)) != 2:
        raise ValueError("router training needs finite features and both correct and wrong candidates")
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0] = 1.0
    X = np.column_stack([np.ones(len(features)), (features - mean) / scale])

    def objective(w):
        z = X @ w
        loss = np.logaddexp(0, z).sum() - wrong @ z + .5 * l2 * np.dot(w[1:], w[1:])
        gradient = X.T @ (expit(z) - wrong)
        gradient[1:] += l2 * w[1:]
        return loss, gradient

    initial = np.zeros(X.shape[1])
    initial[0] = np.log(np.clip(wrong.mean(), 1e-9, 1 - 1e-9) / np.clip(1 - wrong.mean(), 1e-9, 1))
    result = minimize(objective, initial, jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"logistic fit failed: {result.message}")
    return {"weights": result.x.tolist(), "mean": mean.tolist(), "scale": scale.tolist(),
            "feature_names": list(FEATURE_NAMES), "l2": l2, "posterior_abs_clip": POSTERIOR_ABS_CLIP}


def risk(model, features):
    values = np.asarray(features, dtype=np.float64)
    if values.shape[-1] != len(FEATURE_NAMES) or model["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("router feature schema differs from frozen model")
    z = np.asarray(model["weights"])[0] + ((values - model["mean"]) / model["scale"]) @ np.asarray(model["weights"])[1:]
    return expit(z)


def threshold_report(scores, wrong, threshold):
    scores, wrong = np.asarray(scores), np.asarray(wrong, dtype=bool)
    escalated = scores > threshold
    return {"threshold": float(threshold), "shots": len(scores), "wrong_candidates": int(wrong.sum()),
            "escalated": int(escalated.sum()), "escalation_fraction": float(escalated.mean()),
            "wrong_escalated": int((wrong & escalated).sum()),
            "wrong_accepted": int((wrong & ~escalated).sum()),
            "wrong_detection_fraction": float((wrong & escalated).sum() / wrong.sum()) if wrong.any() else None}


def select_threshold(scores, wrong, max_missed_wrong):
    wrong = np.asarray(wrong, dtype=bool)
    if not wrong.any():
        raise ValueError("validation has no incorrect candidates; cannot select a risk threshold")
    if max_missed_wrong < 0 or max_missed_wrong >= wrong.sum():
        raise ValueError("max_missed_wrong must be below the number of validation errors")
    ordered = np.sort(np.asarray(scores)[wrong])
    return float(np.nextafter(ordered[max_missed_wrong], -np.inf))


def load_frozen_router(path, problem):
    path = Path(path)
    model = json.loads(path.read_text())
    if model.get("threshold") is None or model.get("basis") != problem.basis or model.get("p") != problem.p:
        raise ValueError("frozen router threshold, basis, or p does not match evaluation")
    if model.get("matrix_sha256") != matrix_identity(problem.H, problem.L, problem.priors):
        raise ValueError("frozen router matrix identity does not match evaluation")
    return model, hashlib.sha256(path.read_bytes()).hexdigest()


class TieredSession:
    def __init__(self, problem, model_path):
        from decoders import RELAY_SETTINGS, relay_decoder

        self.problem = problem
        self.model, digest = load_frozen_router(model_path, problem)
        start = perf_counter()
        self.first = relay_decoder(problem.H, problem.priors, stop_nconv=1)
        self.full = relay_decoder(problem.H, problem.priors)
        self.setup_seconds = perf_counter() - start
        self.settings = {"first_solution": RELAY_SETTINGS | {"stop_nconv": 1},
                         "fallback": dict(RELAY_SETTINGS), "router_sha256": digest,
                         "risk_threshold": self.model["threshold"], "routing_rule": "risk > threshold"}

    def decode_shot(self, detectors, basis_detectors):
        syndrome = np.asarray(basis_detectors, dtype=np.uint8)
        start = perf_counter()
        result = self.first.decode_detailed(syndrome)
        kit_seconds = perf_counter() - start
        easy = first_leg_converged(result)
        features = candidate_features(result, syndrome)
        probability = None
        fallback = False
        if not easy:
            probability = float(risk(self.model, features))
            fallback = probability > self.model["threshold"]
        if fallback:
            start = perf_counter()
            selected = self.full.decode_detailed(syndrome)
            kit_seconds += perf_counter() - start
        else:
            selected = result
        return ShotResult(_prediction(self.problem.L, selected.decoding),
                          {"converged": bool(selected.success), "candidate_converged": bool(result.success),
                           "first_leg_converged": easy, "fallback_used": fallback,
                           "first_solution_iterations": int(result.iterations),
                           "iterations": int(result.iterations) + (int(selected.iterations) if fallback else 0),
                           "kit_decode_seconds": kit_seconds,
                           "router_risk": probability,
                           "outcome": "converged" if selected.success else "unconverged"})


def make_tiered_decoder(problem, options):
    path = options.get("router_models", {}).get(problem.basis) or options.get("router_model")
    if not path:
        raise ValueError("tiered decoder requires plugin option router_model")
    return TieredSession(problem, path)
