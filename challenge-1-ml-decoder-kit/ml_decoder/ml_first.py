"""ML layer first, relay-BP as the fallback, scored on real kit shots.

  python ml_first.py --basis X --model model_X.pt --p 0.004 --shots 5000 --seed 1
  python ml_first.py --basis X --model model_X.pt --relay '{"stop_nconv": 1, "pre_iter": 40, "set_max_iter": 30}'

The CNN predicts the 12 logical bits. Its confidence is the smallest |logit| among the 12. Shots above the threshold
keep the CNN's answer. The rest go to relay-BP. By default that is the kit's relay-BP (decoders.RELAY_SETTINGS: a first
leg of 80, legs of 60, stop at 5 solutions) with seed 1. --relay replaces single settings, as a JSON object.
The table sweeps the threshold: the share of shots the CNN keeps, failures among them, total failures with a 95% Wilson
interval, and mean time per shot. Pick the threshold on one --seed and report on another.
A shot fails when any of its 12 predicted bits is wrong. A relay-BP decoding that did not converge also counts as failed.

The CNN time of every shot is measured on the path plugin.decode_shot runs: numpy in, numpy out, on cuda a synchronise
per shot (train_logical_cnn.shot_logits). The mean time of a row adds the relay-BP time of the shots it does not keep.

Relay-BP carries state from one decode to the next, so its results depend on the order of the shots and on its seed.
This script runs relay-BP on every shot, in order. The plugin runs it only on the shots the CNN does not keep, so the
sweep's fallback failures estimate the plugin's but are not exactly the same numbers.
It lives next to train_logical_cnn.py in challenge-1-ml-decoder-kit/ml_decoder/ and finds the kit one folder up.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

RELAY_SEED = 1


def default_kit():
    """The kit folder: this file's folder if circuit.py is in it, else the folder above."""
    here = os.path.dirname(os.path.abspath(__file__))
    return here if os.path.exists(os.path.join(here, "circuit.py")) else os.path.dirname(here)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kit", default=default_kit(), help="folder with circuit.py")
    parser.add_argument("--basis", choices=["X", "Z"], default="X")
    parser.add_argument("--model", required=True, help="checkpoint from train_logical_cnn.py")
    parser.add_argument("--p", type=float, default=0.004)
    parser.add_argument("--shots", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--relay", default="{}", help='JSON object of relay-BP settings to replace, e.g. {"stop_nconv": 1}')
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    return parser.parse_args()


def wilson(k, n, z=1.96):
    """95% Wilson interval of a fraction k / n."""
    if n == 0:
        return float("nan"), float("nan")
    centre, half = (k + z * z / 2) / (n + z * z), z * np.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return max(0.0, centre - half), min(1.0, centre + half)


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, args.kit)
    from circuit import GROSS_CIRCUITS, dem_matrices, detector_ids, load_circuit, sample
    from decoders import RELAY_SETTINGS, relay_decoder
    from train_logical_cnn import load_checkpoint, shot_logits

    model, saved = load_checkpoint(args.model, args.basis, args.device)
    changes = {"seed": RELAY_SEED} | json.loads(args.relay)
    circuit = load_circuit(GROSS_CIRCUITS[args.basis], args.p)
    H, L, priors = dem_matrices(circuit, args.basis)
    det, obs = sample(circuit, args.shots, args.seed)
    det = det[:, detector_ids(circuit, args.basis)]
    n = len(det)
    print(f"{args.basis} memory, p={args.p}, {n} Stim shots, seed {args.seed}. CNN on {args.device}, "
          f"{torch.get_num_threads()} torch threads. Checkpoint: basis {saved.get('basis')}, "
          f"train p {saved.get('p_train')}, seed {saved.get('seed')}, {saved.get('shots')} training shots.")
    print(f"Fallback relay-BP settings: {RELAY_SETTINGS | changes} (changes to the kit's: {changes})", flush=True)

    shot_logits(model, det[0], args.device)                  # warm-up, untimed
    logits, ml_ms = np.zeros((n, L.shape[0]), np.float32), np.zeros(n)
    for k in range(n):
        start = time.perf_counter()
        logits[k] = shot_logits(model, det[k], args.device)
        ml_ms[k] = (time.perf_counter() - start) * 1e3
    ml_wrong = ((logits > 0) != (obs > 0)).any(axis=1)
    confidence = np.abs(logits).min(axis=1)

    decoder = relay_decoder(H, priors, **changes)
    bp_wrong, bp_ms = np.zeros(n, bool), np.zeros(n)
    for k in range(n):
        start = time.perf_counter()
        result = decoder.decode_detailed(det[k])
        bp_ms[k] = (time.perf_counter() - start) * 1e3
        bp_wrong[k] = (not result.success) or bool((((L @ np.asarray(result.decoding, np.uint8)) % 2) != obs[k]).any())

    print(f"CNN alone: {int(ml_wrong.sum())} of {n} shots wrong, {ml_ms.mean():.2f} ms per shot. "
          f"Relay-BP alone: {int(bp_wrong.sum())} of {n} failures, {bp_ms.mean():.1f} ms per shot.")
    print("  CNN keeps | threshold | CNN failures among kept | total failures | 95% Wilson interval | mean ms per shot")
    for keep in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        threshold = np.inf if keep == 0 else -np.inf if keep == 1 else float(np.quantile(confidence, 1 - keep))
        kept = confidence >= threshold
        total = int(ml_wrong[kept].sum() + bp_wrong[~kept].sum())
        low, high = wilson(total, n)
        mean_ms = (ml_ms.mean() if keep > 0 else 0.0) + bp_ms[~kept].sum() / n
        print(f"  {kept.mean():8.1%} | {threshold:9.2f} | {int(ml_wrong[kept].sum()):6d} of {int(kept.sum()):6d}"
              f"        | {total:5d} of {n} | [{low:.2e}, {high:.2e}]    | {mean_ms:8.2f}")


if __name__ == "__main__":
    main()
