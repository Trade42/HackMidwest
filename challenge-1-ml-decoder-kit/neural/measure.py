"""Print the size of a model and what one training step costs on this machine's CPU.

  python -m neural.measure --width 128 --depth 8

The memory per shot decides the batch size on a GPU. It is the growth of the process's peak
memory between a step with a small batch and a step with a larger one, in 32-bit floats;
float16 mixed precision needs about half.
"""
import argparse
import time

import torch
from torch.nn import functional as F

from neural.model import CascadeDecoder, count_parameters

try:
    import resource   # not on Windows: the memory line is then left out
except ImportError:
    resource = None


def training_step(model, batch):
    """One forward and backward pass on random detector bits. Returns the seconds it took."""
    detectors = torch.rand(batch, model.num_inputs) < 0.1
    labels = torch.zeros(batch, model.pool.shape[0])
    start = time.perf_counter()
    F.binary_cross_entropy_with_logits(model(detectors), labels).backward()
    return time.perf_counter() - start


def peak_megabytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024   # Linux reports kilobytes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--code", default="gross")
    parser.add_argument("--basis", choices=["X", "Z"], default="X")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--small-batch", type=int, default=2)
    parser.add_argument("--large-batch", type=int, default=6)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    model = CascadeDecoder(args.code, args.basis, width=args.width, depth=args.depth)
    print(f"{args.code} code, {args.basis} memory, width {args.width}, depth {args.depth}: "
          f"{count_parameters(model)} parameters, {model.num_inputs} detector bits in, {model.pool.shape[0]} logits out")
    training_step(model, args.small_batch)
    small_peak = peak_megabytes() if resource else None
    seconds = training_step(model, args.large_batch)
    print(f"one training step on the CPU ({args.threads} threads): {seconds / args.large_batch:.3f} s per shot "
          f"at batch {args.large_batch}")
    if resource:
        per_shot = (peak_megabytes() - small_peak) / (args.large_batch - args.small_batch)
        print(f"memory of the activations: {per_shot:.0f} MB per shot in 32-bit floats "
              f"(batch {args.small_batch} against {args.large_batch})")


if __name__ == "__main__":
    main()
