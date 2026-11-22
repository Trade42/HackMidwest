"""The predecoder as a decoder for the team's evaluator.

Run from challenge-1-ml-decoder-kit/:

  python -m evaluation.evaluate --test-x <X shots> --test-z <Z shots> \\
      --decoder predecoder.plugin:make_decoder --plugin-options <options.json> --out <folder>

One shot of X memory, all inside decode_shot (the evaluator's timer wraps all of it):

  1. the network reads the shot's 1,728 detector bits; the head turns its vectors, the 936 basis
     bits and the kit's priors into one logit per column of the kit's H;
  2. the shot's priors: the head's probabilities, mixed with the kit's and clipped (priors.py);
  3. relay-BP at the kit's settings is built with those priors (seed 0) and decodes the basis bits.

A basis without a head (Z memory, for now) runs the kit's relay-BP, built once, as the
evaluator's own "relay-bp" does; the session's settings say so.

Options (a JSON object):

  heads        {"X": <head.pt of predecoder.train>}
  trunks       {"X": <the trunk checkpoint>} (default: the path stored in head.pt; its sha256 is checked)
  settings     the fit file of predecoder.priors, or {"mode": ..., "alpha": ..., "eps": ...}
  device       for the network: "cpu" (default), "cuda" or "auto"
  torch_threads  CPU threads for the network
  relay_changes  settings of relay-BP to replace, for quick tests only

The truth never reaches a session: decode_shot receives detector bits only.
"""
from time import perf_counter

import numpy as np

from evaluation.adapters import ShotResult
from predecoder.decode import decode_session, decode_with_priors, kit_session, settings as relay_settings
from predecoder.priors import load_settings, shot_priors

OPTIONS = {"heads", "trunks", "settings", "device", "torch_threads", "relay_changes"}


class PredecoderSession:
    """One memory basis: the network and head (if this basis has one), then relay-BP with the shot's priors."""

    def __init__(self, problem, options, predecoder=None):
        """predecoder replaces the loaded network and head (tests)."""
        start = perf_counter()
        unknown = set(options) - OPTIONS
        if unknown:
            raise ValueError(f"unknown options {sorted(unknown)}; known: {sorted(OPTIONS)}")
        self.problem = problem
        self.priors = np.asarray(problem.priors, np.float64)
        self.relay_changes = dict(options.get("relay_changes") or {})
        head = (options.get("heads") or {}).get(problem.basis)
        if predecoder is None and head:
            from predecoder.evaluate import Predecoder
            predecoder = Predecoder(head, (options.get("trunks") or {}).get(problem.basis), options.get("device", "cpu"),
                                    options.get("torch_threads"))
        self.predecoder = predecoder
        self.prior_settings = load_settings(options.get("settings"))
        self.kit_decoder = None if predecoder is not None else kit_session(problem.H, self.priors, **self.relay_changes)
        self.settings = {
            "decoder": ("network + head -> this shot's priors -> relay-BP built per shot" if predecoder is not None
                        else "no head for this basis: the kit's relay-BP, built once"),
            "head_priors": self.prior_settings if predecoder is not None else None,
            "predecoder": getattr(predecoder, "record", None),
            "relay": relay_settings() | self.relay_changes,
        }
        if predecoder is not None:      # so that the first real shot pays no one-time cost
            self.decode_shot(np.zeros(problem.circuit.num_detectors, np.uint8), np.zeros(problem.H.shape[0], np.uint8))
        self.setup_seconds = perf_counter() - start

    def shot_priors(self, full, basis):
        """(priors, seconds of the network and head) of one shot."""
        start = perf_counter()
        logits = self.predecoder(np.asarray(full, np.uint8)[None], np.asarray(basis, np.uint8)[None], self.problem.p,
                                 self.priors)[0]
        s = self.prior_settings
        return shot_priors(logits, self.priors, s["alpha"], s["eps"], s["mode"]), perf_counter() - start

    def decode_shot(self, full, basis):
        syndrome = np.ascontiguousarray(basis, dtype=np.uint8)
        if syndrome.shape != (self.problem.H.shape[0],):
            raise ValueError("the basis syndrome does not have one bit per row of H")
        if self.predecoder is None:
            result = decode_session(self.kit_decoder, self.problem.L, syndrome)
            return ShotResult(result.predicted, {"converged": result.converged, "iterations": result.iterations,
                                                 "stage": "kit relay-BP"})
        priors, seconds_network = self.shot_priors(full, syndrome)
        result = decode_with_priors(self.problem.H, self.problem.L, priors, syndrome, **self.relay_changes)
        return ShotResult(result.predicted, {"converged": result.converged, "iterations": result.iterations,
                                             "stage": "head priors", "seconds_network": seconds_network,
                                             "seconds_build": result.seconds_build, "seconds_decode": result.seconds_decode})


def make_decoder(problem, options):
    """The factory the evaluator calls, once per memory basis."""
    return PredecoderSession(problem, options)
