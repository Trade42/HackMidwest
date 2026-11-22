"""Score a trained decoder on the kit's Stim circuit shots.

Run from challenge-1-ml-decoder-kit/. On a file saved by make_dataset.py:

  python -m neural.evaluate_stim --checkpoint runs/x1/checkpoint.pt --data data/p0.004/validation_X.npz

or on shots sampled here from the kit's circuit (a Stim seed gives different shots on
different machines, so use saved files to compare decoders):

  python -m neural.evaluate_stim --checkpoint runs/x1/checkpoint.pt --p 0.004 --shots 10000 --seed 12345

A shot fails when any of the 12 logical bits is predicted wrong. The numbers are for one
memory basis, the one the checkpoint was trained for. The kit's headline rate is per pair of
an X shot and a Z shot and needs both bases.

Files named test_* or go_* are the team's held-out shots. The script refuses them unless
--final is given: score them once, after every choice is frozen on validation shots.
"""
import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch

from circuit import detector_ids, sample
from make_circuit import memory_experiment
from neural.train import load_decoder, predict


def wilson_interval(failures, shots, z=1.959964):
    """The 95% Wilson score interval of a failure fraction."""
    if shots == 0:
        return 0.0, 1.0
    fraction = failures / shots
    centre = fraction + z * z / (2 * shots)
    half = z * math.sqrt(fraction * (1 - fraction) / shots + z * z / (4 * shots * shots))
    low = 0.0 if failures == 0 else (centre - half) / (1 + z * z / shots)        # exact ends, free of rounding
    high = 1.0 if failures == shots else (centre + half) / (1 + z * z / shots)
    return low, high


def per_cycle(fraction, cycles):
    """The rate per syndrome cycle of a failure fraction per shot: 1 - (1 - P)^(1 / cycles), as in the kit."""
    return 1 - (1 - fraction) ** (1 / cycles)


def load_shots(path, final=False):
    """(detectors, observables, basis, description) from a file of make_dataset.py."""
    path = Path(path)
    if path.name.startswith(("test_", "go_")) and not final:
        raise SystemExit(f"{path} is a held-out file. Pass --final to score it, once, after every choice is frozen.")
    with np.load(path) as data:
        description = f"{path} (Stim circuit shots, p={float(data['p']):g}, seed {int(data['seed'])}, basis {str(data['basis'])})"
        return data["detectors"], data["observables"], str(data["basis"]), description


def sample_shots(config, p, shots, seed):
    """(detectors, observables, description) sampled now from the kit's circuit."""
    circuit = memory_experiment(config["code"], config["basis"], config["rounds"], p)
    detectors, observables = sample(circuit, shots, seed)
    return detectors, observables, f"Stim circuit shots sampled here: p={p:g}, seed {seed}, basis {config['basis']}"


def decode(model, detectors, device, batch=250):
    """Predictions [shots, observables] as a uint8 array."""
    parts = [predict(model, torch.from_numpy(detectors[start:start + batch]).to(device)).cpu()
             for start in range(0, len(detectors), batch)]
    return torch.cat(parts).numpy()


def batch_one_latency(model, detectors, device, shots=200, warm_up=10):
    """Seconds per shot at batch size 1. The timer wraps: copy the shot to the device, run the network, copy the answer back."""
    seconds = []
    for index in range(min(len(detectors), shots + warm_up)):
        shot = torch.from_numpy(detectors[index:index + 1])
        start = time.perf_counter()
        predict(model, shot.to(device)).cpu()
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds.append(time.perf_counter() - start)
    return np.array(seconds[warm_up:])


def evaluate(checkpoint, detectors, observables, device="cpu", latency_shots=200):
    """Score the checkpoint's decoder. Returns a dict of numbers and the predictions."""
    device = torch.device(device)
    model, saved = load_decoder(checkpoint, device)
    config = saved["config"]
    if detectors.shape[1] != model.num_inputs:   # the file holds every detector; a "basis" model reads the kit's 936
        circuit = memory_experiment(config["code"], config["basis"], config["rounds"])
        detectors = detectors[:, detector_ids(circuit, config["basis"])]
    detectors, observables = np.ascontiguousarray(detectors, dtype=np.uint8), np.asarray(observables, dtype=np.uint8)
    predictions = decode(model, detectors, device)
    failed = (predictions != observables).any(axis=1)
    shots, failures, cycles = len(failed), int(failed.sum()), config["rounds"]
    low, high = wilson_interval(failures, shots)
    latency = batch_one_latency(model, detectors, device, latency_shots)
    numbers = {
        "basis": config["basis"], "shots": shots, "failures": failures,
        "failure_fraction": failures / shots, "failure_interval": [low, high],
        "per_cycle": per_cycle(failures / shots, cycles), "per_cycle_interval": [per_cycle(low, cycles), per_cycle(high, cycles)],
        "always_zero_failures": int(observables.any(axis=1).sum()),
        "bits_right": float((predictions == observables).mean()), "always_zero_bits_right": float((observables == 0).mean()),
        "latency_ms_mean": float(latency.mean() * 1e3), "latency_ms_median": float(np.median(latency) * 1e3),
        "latency_ms_p99": float(np.percentile(latency, 99) * 1e3), "latency_shots": len(latency),
        "device": device_name(device), "checkpoint_step": saved["step"], "checkpoint_commit": saved["commit"],
        "detectors_read": config["detectors"], "training_seed": config["seed"],
    }
    return numbers, predictions, failed


def device_name(device):
    if device.type == "cuda":
        return f"GPU {torch.cuda.get_device_name(device)}"
    return f"CPU {platform.processor() or platform.machine()}, {torch.get_num_threads()} torch threads"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", help="an .npz file of make_dataset.py")
    parser.add_argument("--final", action="store_true", help="allow a held-out test_* or go_* file")
    parser.add_argument("--p", type=float, help="without --data: sample at this noise strength")
    parser.add_argument("--shots", type=int, default=10000, help="without --data: shots to sample")
    parser.add_argument("--seed", type=int, help="without --data: the Stim seed")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--threads", type=int, help="CPU threads for torch")
    parser.add_argument("--latency-shots", type=int, default=200)
    parser.add_argument("--save", help="write the numbers (.json) and the predictions (.npz) with this path prefix")
    args = parser.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    config = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["config"]
    if args.data:
        detectors, observables, basis, description = load_shots(args.data, args.final)
        if basis != config["basis"]:
            raise SystemExit(f"the checkpoint decodes {config['basis']} memory, the file holds {basis} memory")
    elif args.p is not None and args.seed is not None:
        detectors, observables, description = sample_shots(config, args.p, args.shots, args.seed)
    else:
        raise SystemExit("give --data, or --p and --seed")

    numbers, predictions, failed = evaluate(args.checkpoint, detectors, observables, device, args.latency_shots)
    numbers["data"] = description
    low, high = numbers["failure_interval"]
    cycle_low, cycle_high = numbers["per_cycle_interval"]
    print(f"checkpoint {args.checkpoint}: step {numbers['checkpoint_step']}, commit {numbers['checkpoint_commit']}, "
          f"training seed {numbers['training_seed']}, reads detectors={numbers['detectors_read']}")
    print(f"data: {description}")
    print(f"{numbers['basis']} memory: failures {numbers['failures']} of {numbers['shots']} "
          f"(always-0 would fail {numbers['always_zero_failures']})")
    print(f"  logical error per shot {numbers['failure_fraction']:.4e}, 95% Wilson interval [{low:.4e}, {high:.4e}]")
    print(f"  per cycle, this basis alone {numbers['per_cycle']:.4e}, interval [{cycle_low:.4e}, {cycle_high:.4e}]")
    print(f"  bits right {numbers['bits_right']:.4f} (always-0 {numbers['always_zero_bits_right']:.4f})")
    print(f"batch-one inference on {numbers['device']}: mean {numbers['latency_ms_mean']:.2f} ms, "
          f"median {numbers['latency_ms_median']:.2f} ms, p99 {numbers['latency_ms_p99']:.2f} ms over "
          f"{numbers['latency_shots']} shots (the timer wraps the copy to the device, the network and the copy back)")
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save + ".json").write_text(json.dumps(numbers, indent=2) + "\n", encoding="utf-8")
        np.savez_compressed(args.save + ".npz", predictions=predictions, failed=failed)
    return numbers


if __name__ == "__main__":
    main()
