"""Evaluation-facing decoder sessions. Optional decoder packages load on demand."""
from dataclasses import dataclass, field
from importlib import import_module
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class Problem:
    basis: str
    H: object
    L: object
    priors: np.ndarray
    p: float
    circuit: object


@dataclass
class ShotResult:
    predictions: np.ndarray
    metrics: dict = field(default_factory=dict)
    decode_seconds: float | None = None


def _prediction(L, errors):
    errors = np.asarray(errors)
    if errors.shape != (L.shape[1],) or not np.isin(errors, [0, 1]).all():
        raise ValueError(f"decoder returned invalid error vector; expected ({L.shape[1]},) binary")
    return np.asarray((L @ errors.astype(np.uint8)) % 2, dtype=np.uint8).reshape(12)


class KitSession:
    """Wrap a kit decoder. `decode_shot` receives full and basis-only syndrome bits."""

    def __init__(self, problem, name):
        from decoders import BPOSD_SETTINGS, RELAY_SETTINGS, bposd_decoder, relay_decoder

        self.problem = problem
        self.name = name
        self.settings = dict(BPOSD_SETTINGS if name == "bposd" else RELAY_SETTINGS)
        start = perf_counter()
        if name == "bposd":
            self.decoder = bposd_decoder(problem.H, problem.priors)
            self.first_leg = None
        elif name == "relay-bp":
            self.decoder = relay_decoder(problem.H, problem.priors)
            self.first_leg = None
        elif name == "relay-first-solution":
            self.decoder = relay_decoder(problem.H, problem.priors, stop_nconv=1)
            self.settings["stop_nconv"] = 1
            self.first_leg = None
        elif name == "relay-first-leg-or-full":
            # In relay-bp 0.2.2, num_sets=0 executes only the ordered first leg.
            # Keep its memory BP parameters; failed first legs rerun the full kit decoder.
            self.first_leg = relay_decoder(problem.H, problem.priors, num_sets=0, stop_nconv=1)
            self.decoder = relay_decoder(problem.H, problem.priors)
            self.settings = {"first_leg": RELAY_SETTINGS | {"num_sets": 0, "stop_nconv": 1},
                             "fallback": dict(RELAY_SETTINGS)}
        else:
            raise ValueError(f"unknown kit decoder {name!r}")
        self.setup_seconds = perf_counter() - start

    def decode_shot(self, detectors, basis_detectors):
        syndrome = np.asarray(basis_detectors, dtype=np.uint8)
        if syndrome.shape != (self.problem.H.shape[0],):
            raise ValueError("basis syndrome width does not match H")
        if self.name == "bposd":
            start = perf_counter()
            errors = self.decoder.decode(syndrome)
            elapsed = perf_counter() - start
            return ShotResult(_prediction(self.problem.L, errors), decode_seconds=elapsed)
        if self.name == "relay-bp":
            # Preserve the kit baseline's plain decode call and timer boundary.
            start = perf_counter()
            errors = self.decoder.decode(syndrome)
            elapsed = perf_counter() - start
            return ShotResult(_prediction(self.problem.L, errors), decode_seconds=elapsed)
        if self.first_leg is not None:
            start = perf_counter()
            first = self.first_leg.decode_detailed(syndrome)
            elapsed = perf_counter() - start
            if first.success:
                result = first
                fallback = False
            else:
                start = perf_counter()
                result = self.decoder.decode_detailed(syndrome)
                elapsed += perf_counter() - start
                fallback = True
            return ShotResult(_prediction(self.problem.L, result.decoding),
                              {"converged": bool(result.success), "first_leg_converged": bool(first.success),
                               "fallback_used": fallback, "iterations": int(first.iterations) +
                               (int(result.iterations) if fallback else 0),
                               "outcome": "converged" if result.success else "unconverged"}, elapsed)
        start = perf_counter()
        result = self.decoder.decode_detailed(syndrome)
        elapsed = perf_counter() - start
        return ShotResult(_prediction(self.problem.L, result.decoding),
                          {"converged": bool(result.success), "iterations": int(result.iterations),
                           "outcome": "converged" if result.success else "unconverged"}, elapsed)


KIT_NAMES = ("bposd", "relay-bp", "relay-first-solution", "relay-first-leg-or-full")


def create_session(spec, problem, options=None):
    """Built-ins or `module:function` plugin factory(problem, options) -> session."""
    if spec in KIT_NAMES:
        return KitSession(problem, spec)
    if ":" not in spec:
        raise ValueError(f"unknown decoder {spec!r}; use a kit name or module:function")
    module, name = spec.split(":", 1)
    factory = getattr(import_module(module), name)
    start = perf_counter()
    session = factory(problem, options or {})
    if not hasattr(session, "decode_shot"):
        raise TypeError(f"{spec}: plugin session must implement decode_shot(full, basis)")
    if not hasattr(session, "setup_seconds"):
        session.setup_seconds = perf_counter() - start
    return session


def validate_shot_result(result):
    if not isinstance(result, ShotResult):
        raise TypeError("decode_shot must return ShotResult")
    pred = np.asarray(result.predictions)
    if pred.shape != (12,) or not np.isin(pred, [0, 1]).all():
        raise ValueError("decode_shot predictions must be binary shape (12,)")
    if not isinstance(result.metrics, dict):
        raise TypeError("decode_shot metrics must be a dict")
    if result.decode_seconds is not None and (not np.isfinite(result.decode_seconds) or result.decode_seconds < 0):
        raise ValueError("decode_seconds must be finite and nonnegative")
    return pred.astype(np.uint8, copy=False)
