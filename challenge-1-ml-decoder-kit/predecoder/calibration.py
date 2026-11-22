"""Is the head right about which faults fired? Held-out error-model shots, against two references.

Run from challenge-1-ml-decoder-kit/:

  python -m predecoder.calibration --head <head.pt> --trunk <the trunk checkpoint> --p 0.004 --shots 1000 --seed 77

Shots are drawn from the full error model with their fired faults (a seed the trainer never
uses for its own shots), mapped onto the kit's columns. For every column of every shot it
compares three probabilities that the column fired:

  kit      the kit's fixed prior
  BP       plain min-sum BP's own belief after 80 iterations (the first leg of relay-BP, with
           memory strength 0.15), started from the kit's priors: what BP already knows
  head     the head's probability

and prints, for each, the mean log-loss per column (lower is better) and a calibration table:
for columns put in each band of probability, the mean probability and the share that fired.
If the head is not clearly better than BP's own belief, handing it to BP cannot add much.
These are DEM-sampled shots: a diagnostic, not a decoder result.
"""
import argparse
import json
import math

import numpy as np
import torch

from circuit import dem_matrices
from decoders import RELAY_SETTINGS
from make_circuit import memory_experiment
from neural.train import load_decoder, stream_seed
from predecoder.train import BANDS, Problem, head_logits, load_head

CALIBRATION_STREAM = 21


def bands(q, y):
    band = np.clip(np.searchsorted(BANDS, q, side="right") - 1, 0, len(BANDS) - 2)
    counts = np.bincount(band, minlength=len(BANDS) - 1)
    sums_q = np.bincount(band, weights=q, minlength=len(BANDS) - 1)
    fired = np.bincount(band, weights=y, minlength=len(BANDS) - 1)
    return [{"low": BANDS[b], "high": min(BANDS[b + 1], 1.0), "columns": int(counts[b]), "mean_q": sums_q[b] / counts[b],
             "fired": fired[b] / counts[b]} for b in range(len(counts)) if counts[b]]


def log_loss(q, y, floor=1e-12):
    q = np.clip(q, floor, 1 - floor)
    return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())


def main(argv=None):
    import relay_bp

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", required=True)
    parser.add_argument("--trunk")
    parser.add_argument("--p", type=float, default=0.004)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--chunk", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--out", help="a JSON file for the numbers")
    args = parser.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    trunk, head, saved = load_head(args.head, device, args.trunk)
    _, trunk_saved = load_decoder(args.trunk or saved["trunk_file"], "cpu")
    problem = Problem(trunk_saved["config"], [args.p], device)
    H = problem.map.basis_H.tocsr()
    config = trunk_saved["config"]
    _, _, kit = dem_matrices(memory_experiment(config["code"], config["basis"], config["rounds"], args.p), config["basis"])
    kit = np.asarray(kit, np.float64)
    bp = relay_bp.MinSumBPDecoderF64(H, error_priors=kit, max_iter=RELAY_SETTINGS["pre_iter"],
                                     alpha=RELAY_SETTINGS["alpha"], gamma0=RELAY_SETTINGS["gamma0"])
    amp = torch.float16 if device.type == "cuda" else None
    generator = torch.Generator(device=device)
    q_head, q_bp, labels, converged = [], [], [], []
    with torch.no_grad():
        for index, start in enumerate(range(0, args.shots, args.chunk)):
            size = min(args.chunk, args.shots - start)
            generator.manual_seed(stream_seed(args.seed, CALIBRATION_STREAM, index))
            trunk_input, basis_bits, fired, detectors, observables = problem.batch(size, args.p, generator)
            problem.map.check(fired.cpu().numpy(), detectors.cpu().numpy(), observables.cpu().numpy())
            logits = head_logits(trunk, head, trunk_input, basis_bits, problem.prior_logit[args.p], amp)
            logits = logits - math.log(saved["config"]["pos_weight"])
            q_head.append(torch.sigmoid(logits.double()).cpu().numpy())
            labels.append(fired.cpu().numpy())
            for syndrome in basis_bits.cpu().numpy():
                result = bp.decode_detailed(syndrome)
                converged.append(bool(result.success))
                ratios = np.asarray(result.posterior_ratios, np.float64)
                q_bp.append(1 / (1 + np.exp(np.clip(np.nan_to_num(ratios, nan=0.0), -50, 50))))
    columns = len(kit)
    q_head, y = np.concatenate(q_head), np.concatenate(labels).astype(np.float64)
    q_bp = np.stack(q_bp)
    q_kit = np.tile(kit, (args.shots, 1))
    converged = np.array(converged)
    print(f"held-out error-model shots: {config['basis']} memory, p={args.p:g}, {args.shots} shots (seed {args.seed}, stream "
          f"{CALIBRATION_STREAM}), {y.sum() / args.shots:.1f} fired columns per shot; head {args.head} at step {saved['step']}")
    groups = {"all shots": np.ones(args.shots, bool), "BP's first leg converges": converged,
              "BP's first leg does not converge": ~converged}
    rows = {}
    for group, keep in groups.items():
        count = int(keep.sum())
        print(f"{group}: {count} shots")
        rows[group] = {"shots": count}
        if not count:
            continue
        for name, q in (("kit", q_kit), ("BP, 80 iterations", q_bp), ("head", q_head)):
            qs, ys = q[keep].ravel(), y[keep].ravel()
            row = {"log_loss": log_loss(qs, ys), "bands": bands(qs, ys),
                   "fired_per_shot": float(ys.sum() / count),
                   "fired_above_half_per_shot": float(((qs > 0.5) & (ys == 1)).sum() / count),
                   "not_fired_above_half_per_shot": float(((qs > 0.5) & (ys == 0)).sum() / count)}
            rows[group][name] = row
            print(f"  {name:18s} log-loss per column {row['log_loss']:.6f} | fired per shot {row['fired_per_shot']:.1f}, "
                  f"of them above 0.5: {row['fired_above_half_per_shot']:.1f}; not fired and above 0.5: "
                  f"{row['not_fired_above_half_per_shot']:.2f}")
            print("      " + " | ".join(f"[{b['low']:g},{b['high']:g}) n={b['columns']} q={b['mean_q']:.4f} "
                                        f"fired={b['fired']:.4f}" for b in row["bands"]))
    assert q_head.shape == (args.shots, columns)
    if args.out:
        with open(args.out, "w") as file:
            json.dump({"p": args.p, "shots": args.shots, "seed": args.seed, "head": args.head, "head_step": saved["step"],
                       "trunk_sha256": saved["trunk_sha256"], "rows": rows}, file, indent=1)


if __name__ == "__main__":
    main()
