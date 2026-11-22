"""PriorTilt decoding every shot once: a reference row for the team's evaluator.

Run from challenge-1-ml-decoder-kit/:

  python -m evaluation.evaluate --test-x <X shots> --test-z <Z shots> \
      --decoder hierarchy.priortilt_plugin:make_decoder --plugin-options <options.json> --out <folder>

It is the first stage of hierarchy.plugin alone: PriorTilt (Ye and Delfosse, arXiv:2611.13373)
with the paper's schedule for the shots' p, on the kit's H and priors, under the first of the
plugin's two seeds, on a decoder built once per basis. The options file of hierarchy.plugin
can be given as it is: only seeds, priortilt_repo and priortilt_max_rounds are read here.
"""
from time import perf_counter

import numpy as np

from evaluation.adapters import ShotResult
from hierarchy.stages import DEFAULT_SEEDS, Stage, decoder_settings, problem_sha256


class PriorTiltSession:
    """One memory basis, one PriorTilt decode per shot."""

    def __init__(self, problem, options):
        start = perf_counter()
        self.problem = problem
        seeds = list(options.get("seeds", DEFAULT_SEEDS))
        decoders = decoder_settings(problem.p, seeds, options.get("priortilt_repo"), options.get("priortilt_max_rounds"))
        self.stage = Stage("P1", problem, decoders)
        self.settings = {"decoder": "PriorTilt, one decode per shot, on a decoder built once (it keeps its random stream)",
                         "seed": seeds[0], "settings": decoders["P1"], "problem_sha256": problem_sha256(problem)}
        self.setup_seconds = perf_counter() - start

    def decode_shot(self, full, basis):
        syndrome = np.ascontiguousarray(basis, dtype=np.uint8)
        if syndrome.shape != (self.problem.H.shape[0],):
            raise ValueError("the basis syndrome does not have one bit per row of H")
        answer = self.stage.run(syndrome)
        return ShotResult(answer.predicted, {"converged": bool(answer.converged), "iterations": int(answer.iterations),
                                             "weight": float(answer.weight), "seconds_P1": float(answer.seconds)})


def make_decoder(problem, options):
    """The factory the evaluator calls, once per memory basis."""
    return PriorTiltSession(problem, options)
