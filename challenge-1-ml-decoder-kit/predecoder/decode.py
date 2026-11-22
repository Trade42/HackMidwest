"""Relay-BP with one shot's own priors, and the small helpers every script here shares.

relay-bp 0.2.2 takes its priors only when a decoder is built (`decode` takes none), so a
shot with its own priors gets a decoder of its own. That costs about 8 ms per shot, and the
time is counted in the shot's latency.

A decoder built for one shot also starts from the same random state on every shot (its seed),
while the kit's decoder is built once and carries its random memory strengths from one shot
to the next. To measure what the priors do, and nothing else, the comparison therefore has
three arms (compare.py):

  kit       the kit's relay-BP, built once with the kit's priors, decoding the shots in order
  fresh     relay-BP built per shot with the kit's priors (same seed every shot)
  head      relay-BP built per shot with the shot's priors from the head (same seed every shot)

"head" against "fresh" isolates the priors; "head" against "kit" is the claim a user sees.
"""
import time
from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest

from circuit import GROSS_CIRCUITS, dem_matrices, load_circuit
from decoders import RELAY_SETTINGS, relay_decoder

RELAY_SEED = 0          # relay-bp's own default; every per-shot decoder uses it


@dataclass
class Decoded:
    """One shot through relay-BP."""
    predicted: np.ndarray       # the 12 observable bits of the correction
    converged: bool
    iterations: int
    seconds_build: float        # building the decoder (0 for a decoder built once)
    seconds_decode: float


def kit_problem(basis, p):
    """The kit's decoding problem of one memory basis at noise strength p (as evaluation.adapters.Problem)."""
    from evaluation.adapters import Problem
    circuit = load_circuit(GROSS_CIRCUITS[basis], p)
    H, L, priors = dem_matrices(circuit, basis)
    return Problem(basis, H, L, np.asarray(priors, np.float64), p, circuit)


def decode_with_priors(H, L, priors, syndrome, seed=RELAY_SEED, **changes):
    """Build relay-BP at the kit's settings with these priors and decode one syndrome. Both steps are timed."""
    start = time.perf_counter()
    decoder = relay_decoder(H, np.asarray(priors, np.float64), seed=seed, **changes)
    built = time.perf_counter()
    result = decoder.decode_detailed(np.ascontiguousarray(syndrome, dtype=np.uint8))
    done = time.perf_counter()
    correction = np.asarray(result.decoding, np.uint8)
    return Decoded((L @ correction % 2).astype(np.uint8).reshape(-1), bool(result.success), int(result.iterations),
                   built - start, done - built)


def first_leg_converges(H, priors, syndrome, seed=RELAY_SEED):
    """Does relay-BP's first leg alone (80 iterations, num_sets=0) converge on this syndrome with these priors?

    The kit's relay-BP goes on after a converged first leg until it holds 5 solutions, so its
    iteration count does not say whether a shot was easy. This separate, untimed decode does:
    the shots whose first leg fails hold almost all of relay-BP's failures (gate G1).
    """
    decoder = relay_decoder(H, np.asarray(priors, np.float64), seed=seed, num_sets=0, stop_nconv=1)
    return bool(decoder.decode_detailed(np.ascontiguousarray(syndrome, dtype=np.uint8)).success)


def decode_session(decoder, L, syndrome):
    """One shot through a decoder that was built once (the kit's arm)."""
    start = time.perf_counter()
    result = decoder.decode_detailed(np.ascontiguousarray(syndrome, dtype=np.uint8))
    seconds = time.perf_counter() - start
    correction = np.asarray(result.decoding, np.uint8)
    return Decoded((L @ correction % 2).astype(np.uint8).reshape(-1), bool(result.success), int(result.iterations), 0.0, seconds)


def kit_session(H, priors, **changes):
    """The kit's relay-BP, built once (its default seed), as evaluation.adapters builds it."""
    return relay_decoder(H, np.asarray(priors, np.float64), **changes)


def paired(failed_a, failed_b):
    """(both fail, a only, b only, exact two-sided p): the binomial (McNemar) test on the shots only one gets wrong."""
    failed_a, failed_b = np.asarray(failed_a, bool), np.asarray(failed_b, bool)
    both = int((failed_a & failed_b).sum())
    only_a, only_b = int((failed_a & ~failed_b).sum()), int((~failed_a & failed_b).sum())
    p_value = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return both, only_a, only_b, float(p_value)


def wilson(failures, shots, z=1.959964):
    """95% Wilson interval of a share."""
    if shots == 0:
        return 0.0, 1.0
    share = failures / shots
    centre = (share + z * z / (2 * shots)) / (1 + z * z / shots)
    half = z / (1 + z * z / shots) * np.sqrt(share * (1 - share) / shots + z * z / (4 * shots * shots))
    return max(0.0, centre - half), min(1.0, centre + half)


def settings():
    """The relay-BP settings every arm uses, for the logs."""
    return dict(RELAY_SETTINGS) | {"seed_of_per_shot_decoders": RELAY_SEED}
