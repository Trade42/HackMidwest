"""Evaluation plugin for the ML-first decoder: CNN, confidence threshold, relay-BP fallback.

Run from challenge-1-ml-decoder-kit:

  python -m evaluation.evaluate --test-x test_X.npz --test-z test_Z.npz \\
      --decoder ml_decoder.plugin:make_decoder --plugin-options ml_options.json --out results/ml_first

ml_options.json:

  {"models": {"X": "model_X.pt", "Z": "model_Z.pt"}, "threshold": 6.0}

Optional keys: "device" (default cuda when torch sees a GPU, else cpu) and "relay", a JSON object of relay-BP settings
to replace, e.g. {"stop_nconv": 1, "pre_iter": 40, "set_max_iter": 30} for a faster schedule. Without "relay" the
fallback is the kit's relay-BP (decoders.RELAY_SETTINGS: first leg 80, legs of 60, stop at 5 solutions) with seed 1.
The session's settings, which the evaluator saves in summary.json, list every relay-BP setting in use.

A shot keeps the CNN's 12 predicted bits when its smallest |logit| is at least `threshold`. Every other shot is decoded
by relay-BP. The evaluator times the whole decode_shot, so CNN inference and the fallback are both counted. Freeze the
threshold on validation shots (ml_first.py prints the sweep) before running the test set.
A checkpoint of the other memory basis is refused. A checkpoint that records no basis (older code) loads with a warning.
"""
from time import perf_counter

import numpy as np

RELAY_SEED = 1


class MLFirstSession:
    def __init__(self, problem, options):
        import torch
        from decoders import RELAY_SETTINGS, relay_decoder
        from ml_decoder.train_logical_cnn import load_checkpoint, shot_logits

        start = perf_counter()
        self.problem, self.shot_logits = problem, shot_logits
        self.device = options.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = float(options["threshold"])
        path = options["models"][problem.basis]
        self.model, saved = load_checkpoint(path, problem.basis, self.device)
        changes = {"seed": RELAY_SEED} | options.get("relay", {})
        self.relay = relay_decoder(problem.H, problem.priors, **changes)
        self.settings = {"threshold": self.threshold, "relay": RELAY_SETTINGS | changes, "relay_changes": changes,
                         "model": path, "device": self.device, "torch_threads": torch.get_num_threads(),
                         "checkpoint": {key: saved.get(key) for key in ("basis", "p_train", "seed", "shots")}}
        shot_logits(self.model, np.zeros(problem.H.shape[0], np.uint8), self.device)   # warm-up, outside every timer
        self.setup_seconds = perf_counter() - start

    def decode_shot(self, detectors, basis_detectors):
        from evaluation.adapters import ShotResult

        syndrome = np.asarray(basis_detectors, dtype=np.uint8)
        logits = self.shot_logits(self.model, syndrome, self.device)
        confidence = float(np.abs(logits).min())
        if confidence >= self.threshold:
            return ShotResult((logits > 0).astype(np.uint8),
                              {"ml_kept": True, "confidence": confidence, "fallback_used": False})
        result = self.relay.decode_detailed(syndrome)
        errors = np.asarray(result.decoding, dtype=np.uint8)
        predictions = np.asarray((self.problem.L @ errors) % 2, dtype=np.uint8).reshape(12)
        return ShotResult(predictions, {"ml_kept": False, "confidence": confidence, "fallback_used": True,
                                        "converged": bool(result.success), "iterations": int(result.iterations),
                                        "outcome": "converged" if result.success else "unconverged"})


def make_decoder(problem, options):
    """Factory for evaluation.adapters.create_session."""
    return MLFirstSession(problem, options)
