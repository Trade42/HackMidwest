"""Two relay-BP schedules on the same Stim shots, paired: the kit's (80 / 60 / stop at 5) and 40 / 30 / stop at 1.

  python ml_decoder/relay_schedule_paired.py --p 0.004 --shots 5000 --seed 1

40 / 30 / stop at 1 was the fallback of ml_first.py and plugin.py before the kit's schedule became their default.
The shots are ml_first.py's: sample(circuit, shots, seed) of the kit's circuit, basis detectors only.
Each schedule decodes every shot in order on one fresh decoder with seed 1. Relay-BP carries state from one decode
to the next, so its results depend on the order of the shots and on its seed: both are the same for the two schedules.
Two scoring rules:
  kit      decoders.score: a shot fails when any predicted logical bit is wrong; an unconverged decoding is scored
           like any other
  strict   an unconverged shot also fails, as ml_first.py and relay_knobs.py count it
Paired: shots that only one schedule gets wrong, and the exact two-sided McNemar p (binomial on the discordant shots).
Iteration counts do not depend on the machine. The ms per shot do.
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.stats import binomtest

SCHEDULES = {"kit 80 / 60 / stop at 5": {}, "40 / 30 / stop at 1": {"stop_nconv": 1, "pre_iter": 40, "set_max_iter": 30}}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--kit", default=os.path.dirname(here), help="folder with circuit.py")
    parser.add_argument("--basis", choices=["X", "Z"], default="X")
    parser.add_argument("--p", type=float, default=0.004)
    parser.add_argument("--shots", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1, help="Stim seed of the shots")
    parser.add_argument("--decoder-seed", type=int, default=1)
    parser.add_argument("--out", default=None, help="optional .npz with every shot's outcome per schedule")
    return parser.parse_args()


def wilson(k, n, z=1.96):
    """95% Wilson interval of a fraction k / n."""
    centre, half = (k + z * z / 2) / (n + z * z), z * np.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return max(0.0, centre - half), min(1.0, centre + half)


def decode_all(relay_decoder, H, L, priors, det, obs, changes, seed, name):
    """(wrong, converged, iterations, seconds), one entry per shot, all shots in order on one fresh decoder."""
    decoder = relay_decoder(H, priors, seed=seed, **changes)
    n = len(det)
    wrong, converged, iterations = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, int)
    start = time.perf_counter()
    for k in range(n):
        result = decoder.decode_detailed(det[k])
        converged[k], iterations[k] = bool(result.success), int(result.iterations)
        wrong[k] = bool((((L @ np.asarray(result.decoding, np.uint8)) % 2) != obs[k]).any())
        if (k + 1) % 1000 == 0:
            print(f"{name}: {k + 1} of {n} shots", file=sys.stderr, flush=True)
    return wrong, converged, iterations, time.perf_counter() - start


def main():
    args = parse_args()
    sys.path.insert(0, args.kit)
    from circuit import GROSS_CIRCUITS, dem_matrices, detector_ids, load_circuit, sample
    from decoders import RELAY_SETTINGS, relay_decoder

    circuit = load_circuit(GROSS_CIRCUITS[args.basis], args.p)
    H, L, priors = dem_matrices(circuit, args.basis)
    det, obs = sample(circuit, args.shots, args.seed)
    det = det[:, detector_ids(circuit, args.basis)]
    n = len(det)
    print(f"{args.basis} memory, p={args.p}, {n} Stim shots: sample(circuit, {n}, {args.seed}). "
          f"Each schedule: one fresh decoder, seed {args.decoder_seed}, shots in order.", flush=True)
    results = {}
    for name, changes in SCHEDULES.items():
        wrong, converged, iterations, seconds = decode_all(relay_decoder, H, L, priors, det, obs, changes,
                                                           args.decoder_seed, name)
        strict = wrong | ~converged
        results[name] = {"kit": wrong, "strict": strict, "converged": converged, "iterations": iterations}
        print(f"{name}: settings {RELAY_SETTINGS | changes}")
        for rule, failed in (("kit", wrong), ("strict", strict)):
            low, high = wilson(int(failed.sum()), n)
            print(f"  {rule:6s} scoring: {int(failed.sum()):4d} of {n} failures, 95% Wilson [{low:.2e}, {high:.2e}]")
        print(f"  unconverged {int((~converged).sum())} of {n} | iterations mean {iterations.mean():.0f}, median "
              f"{np.median(iterations):.0f}, p99 {np.percentile(iterations, 99):.0f} | {seconds / n * 1e3:.1f} ms per shot here",
              flush=True)
    first, second = SCHEDULES
    print(f"Paired, the same {n} shots: A = {first}, B = {second}")
    for rule in ("kit", "strict"):
        a, b = results[first][rule], results[second][rule]
        only_a, only_b = int((a & ~b).sum()), int((b & ~a).sum())
        p_value = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
        print(f"  {rule:6s} scoring: both fail {int((a & b).sum())}, only A {only_a}, only B {only_b}, "
              f"neither {int((~a & ~b).sum())}. Exact McNemar p = {p_value:.2g}")
    if args.out:
        np.savez(args.out, **{f"{key}_{rule}": value[rule] for key, value in zip("AB", results.values())
                              for rule in ("kit", "strict", "converged", "iterations")})


if __name__ == "__main__":
    main()
