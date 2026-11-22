"""Evaluation plugin for the ML-first decoder: CNN, confidence threshold, relay-BP fallback.

Run from challenge-1-ml-decoder-kit:

  python -m evaluation.evaluate ... --decoder ml_decoder.plugin:make_decoder --plugin-options ml_options.json

ml_options.json:

  {"models": {"X": "model_X.pt", "Z": "model_Z.pt"}, "threshold": 6.0, "device": "cuda",
   "relay": {"stop_nconv": 1, "pre_iter": 40, "set_max_iter": 30}}

A shot keeps the CNN's 12 predicted bits when its smallest |logit| is at least `threshold`. Every other shot is decoded
by relay-BP at the given schedule. The evaluator times the whole decode_shot, so CNN inference and the fallback are
both counted. Freeze the threshold on validation shots (ml_first.py prints the sweep) before running the test set.
"""
from time import perf_counter

import numpy as np

DEFAULT_RELAY = {"stop_nconv": 1, "pre_iter": 40, "set_max_iter": 30}


class MLFirstSession:
    def __init__(self, problem, options):
        import torch
        from decoders import relay_decoder
        from ml_decoder.train_logical_cnn import NX, NY, ROUNDS, LogicalCNN

        start = perf_counter()
        self.torch, self.problem = torch, problem
        self.device = options.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = float(options["threshold"])
        path = options["models"][problem.basis]
        saved = torch.load(path, map_location=self.device)
        self.model = LogicalCNN(**saved["config"]).to(self.device).eval()
        self.model.load_state_dict(saved["model"])
        self.shape = (1, ROUNDS, NX, NY)
        relay = DEFAULT_RELAY | options.get("relay", {})
        self.relay = relay_decoder(problem.H, problem.priors, seed=1, **relay)
        self.settings = {"threshold": self.threshold, "relay": relay, "model": path, "device": self.device}
        with torch.no_grad():                                # warm-up, outside every decode timer
            self.model(torch.zeros(self.shape, device=self.device))
        self.setup_seconds = perf_counter() - start

    def decode_shot(self, detectors, basis_detectors):
        from evaluation.adapters import ShotResult

        torch = self.torch
        syndrome = np.asarray(basis_detectors, dtype=np.uint8)
        with torch.no_grad():
            x = torch.from_numpy(syndrome.astype(np.float32)).to(self.device).reshape(self.shape)
            logits = self.model(x)[0].float().cpu().numpy()
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
