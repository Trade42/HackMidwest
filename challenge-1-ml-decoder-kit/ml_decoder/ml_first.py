"""ML layer first, relay-BP as the fallback, scored on real kit shots.

  python ml_first.py --basis X --model model_X.pt --p 0.004 --shots 5000

The CNN predicts the 12 logical bits. Its confidence is the smallest |logit| among the 12. Shots above the threshold
keep the CNN's answer. The rest go to relay-BP at the tuned schedule (first leg 40, legs of 30, first solution).
The table sweeps the threshold: the share of shots the CNN keeps, failures among them, total failures and mean time.
Pick the threshold on one --seed and report on another.
It lives next to train_logical_cnn.py in challenge-1-ml-decoder-kit/ml_decoder/ and finds the kit one folder up.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch


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
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, args.kit)
    from circuit import GROSS_CIRCUITS, dem_matrices, detector_ids, load_circuit, sample
    from decoders import relay_decoder
    from train_logical_cnn import NX, NY, ROUNDS, LogicalCNN

    circuit = load_circuit(GROSS_CIRCUITS[args.basis], args.p)
    H, L, priors = dem_matrices(circuit, args.basis)
    det, obs = sample(circuit, args.shots, args.seed)
    det = det[:, detector_ids(circuit, args.basis)]
    n = len(det)

    saved = torch.load(args.model, map_location=args.device)
    model = LogicalCNN(**saved["config"]).to(args.device).eval()
    model.load_state_dict(saved["model"])
    cuda = args.device.startswith("cuda")
    x = torch.tensor(det, dtype=torch.float32, device=args.device).reshape(n, ROUNDS, NX, NY)
    with torch.no_grad():
        logits = torch.cat([model(x[a:a + 1024]) for a in range(0, n, 1024)]).float().cpu().numpy()
        timed = min(200, n)                                  # latency of one shot at a time
        model(x[:1])
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        for k in range(timed):
            model(x[k:k + 1])
        if cuda:
            torch.cuda.synchronize()
        ml_ms = (time.perf_counter() - start) / timed * 1e3
    ml_wrong = ((logits > 0) != (obs > 0)).any(axis=1)
    confidence = np.abs(logits).min(axis=1)

    decoder = relay_decoder(H, priors, seed=1, stop_nconv=1, pre_iter=40, set_max_iter=30)
    bp_wrong, bp_ms = np.zeros(n, bool), np.zeros(n)
    for k in range(n):
        start = time.perf_counter()
        result = decoder.decode_detailed(det[k])
        bp_ms[k] = (time.perf_counter() - start) * 1e3
        bp_wrong[k] = (not result.success) or bool((((L @ np.asarray(result.decoding, np.uint8)) % 2) != obs[k]).any())

    print(f"{args.basis} memory, p={args.p}, {n} Stim shots, seed {args.seed}. CNN alone: {ml_wrong.mean():.2%} of shots wrong, "
          f"{ml_ms:.2f} ms per shot on {args.device}. Relay-BP alone: {bp_wrong.sum()} failures, {bp_ms.mean():.1f} ms per shot.")
    print("  CNN keeps | threshold | CNN failures among kept | total failures | mean ms per shot")
    for keep in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        threshold = np.inf if keep == 0 else -np.inf if keep == 1 else float(np.quantile(confidence, 1 - keep))
        kept = confidence >= threshold
        total = int(ml_wrong[kept].sum() + bp_wrong[~kept].sum())
        mean_ms = (ml_ms if keep > 0 else 0.0) + bp_ms[~kept].sum() / n
        print(f"  {kept.mean():8.1%} | {threshold:9.2f} | {int(ml_wrong[kept].sum()):6d} of {int(kept.sum()):6d}"
              f"        | {total:5d} of {n} | {mean_ms:8.2f}")


if __name__ == "__main__":
    main()
