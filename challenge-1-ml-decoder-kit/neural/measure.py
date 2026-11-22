"""Print the size of a model and where the time of one training step goes on this machine.

  python -m neural.measure                      (the default model; cuda if there is one)
  python -m neural.measure --device cpu --width 32 --depth 3 --batch 16

Three parts:
1. the parameters, and on Linux the memory of the activations per shot (it decides the batch
   size on a GPU; measured on the CPU in 32-bit floats, float16 needs about half);
2. one training step of train.py, with its parts timed separately;
3. the single operations of a residual block at the model's sizes, forward and backward, each
   with the number of times a step calls it. The gather is timed twice: with this code's
   backward pass and with PyTorch's own, which scatters.

Every time is the mean over `--repeats` calls after `--warm-up` calls, with the GPU
synchronised before the clock is read.
"""
import argparse
import time

import torch
from torch.nn import functional as F

from neural import train
from neural.model import CascadeDecoder, Gather, count_parameters

try:
    import resource   # not on Windows: the memory line is then left out
except ImportError:
    resource = None


def clock(function, device, repeats, warm_up):
    """Mean seconds of one call."""
    for _ in range(warm_up):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def memory_per_shot(model, small=2, large=6):
    """MB of activations per shot of a training step on the CPU, from the growth of the peak memory."""
    peaks = []
    for batch in (small, large):
        detectors = torch.rand(batch, model.num_inputs) < 0.1
        F.binary_cross_entropy_with_logits(model(detectors), torch.zeros(batch, model.pool.shape[0])).backward()
        peaks.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)   # Linux reports kilobytes
    return (peaks[1] - peaks[0]) / (large - small)


def step_parts(run, repeats, warm_up):
    """Seconds of the parts of one training step: the draw, the forward pass, the backward pass, the whole step."""
    config, device = run.config, run.device
    p = train.noise_at(0, config)
    run.generator.manual_seed(1)
    detectors, observables = run.sampler.sample(config["batch"], p, run.generator)

    def forward():
        with torch.autocast(device.type, dtype=run.amp_dtype, enabled=run.amp_dtype is not None):
            logits = run.model(detectors)
        return F.binary_cross_entropy_with_logits(logits.float(), observables.float())

    def forward_backward():
        run.optimizer.zero_grad(set_to_none=True)
        run.scaler.scale(forward()).backward()

    run.model.train()
    draw = clock(lambda: run.sampler.sample(config["batch"], p, run.generator), device, repeats, warm_up)
    forward_seconds = clock(forward, device, repeats, warm_up)
    both = clock(forward_backward, device, repeats, warm_up)
    whole = clock(lambda: train.train_step(run), device, repeats, warm_up)
    return {"draw the shots": draw, "forward pass": forward_seconds, "backward pass": both - forward_seconds,
            "optimizer, clipping, moving average": whole - draw - both, "whole step": whole}


def block_operations(run, repeats, warm_up):
    """[(name, seconds per call, calls per step)] for the operations of a residual block, forward and backward."""
    model, device, batch = run.model, run.device, run.config["batch"]
    width, depth = run.config["width"], run.config["depth"]
    block = model.blocks[-1]
    narrow = block.down.out_features
    dtype = run.amp_dtype or torch.float32

    def tensor(*shape):
        return torch.randn(*shape, device=device, dtype=dtype, requires_grad=True)

    checks_wide, checks = tensor(batch, model.rounds, model.checks, width), tensor(batch, model.rounds, model.checks, narrow)
    data = tensor(batch, model.rounds - 1, model.checks, narrow)
    side_by_side = tensor(batch, model.rounds, model.checks // 2, block.to_data.linears[0].in_features)
    neighbours, places = block.to_data.neighbours[0], block.to_data.places[0]
    operations = [
        ("normalisation at full width", lambda: block.norm_in(checks_wide), depth),
        ("SiLU at full width", lambda: F.silu(checks_wide), depth),
        (f"linear {width} -> {narrow}", lambda: block.down(checks_wide), depth),
        (f"linear {narrow} -> {width}", lambda: block.up(checks), depth),
        ("residual sum at full width", lambda: checks_wide + block.branch_scale * checks_wide, depth),
        ("normalisation at the bottleneck", lambda: block.norm_checks(checks), 3 * depth),
        ("SiLU at the bottleneck", lambda: F.silu(checks), 3 * depth),
        ("gather, this code's backward", lambda: Gather.apply(checks, neighbours, places), 4 * depth),
        ("gather, PyTorch's backward", lambda: checks.index_select(-2, neighbours), 0),
        (f"linear on six neighbours, {side_by_side.shape[-1]} -> {2 * narrow}",
         lambda: block.to_data.linears[0](side_by_side), 4 * depth),
        ("whole step checks -> data", lambda: block.to_data.checks_to_data(checks), depth),
        ("whole step data -> checks", lambda: block.to_checks.data_to_checks(data), depth),
        ("whole residual block", lambda: block(checks_wide), depth),
    ]
    rows = []
    for name, operation, calls in operations:
        def forward_backward(operation=operation):
            with torch.autocast(device.type, dtype=run.amp_dtype, enabled=run.amp_dtype is not None):
                out = operation()
            out.backward(torch.ones_like(out))
        rows.append((name, clock(forward_backward, device, repeats, warm_up), calls))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--code", default="gross")
    parser.add_argument("--basis", choices=["X", "Z"], default="X")
    parser.add_argument("--detectors", choices=["all", "basis"], default="all")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--batch", type=int, help="default: 128 on cuda, 4 on a CPU")
    parser.add_argument("--p", type=float, default=0.004, help="noise strength of the shots that are drawn")
    parser.add_argument("--amp", choices=["auto", "fp16", "bf16", "off"], default="auto")
    parser.add_argument("--repeats", type=int, help="default: 10 on cuda, 2 on a CPU")
    parser.add_argument("--warm-up", type=int, help="default: 3 on cuda, 1 on a CPU")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads for torch")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    on_gpu = device == "cuda"
    batch = args.batch or (128 if on_gpu else 4)
    repeats, warm_up = args.repeats or (10 if on_gpu else 2), args.warm_up if args.warm_up is not None else (3 if on_gpu else 1)
    torch.manual_seed(0)

    model = CascadeDecoder(args.code, args.basis, detectors=args.detectors, width=args.width, depth=args.depth)
    print(f"{args.code} code, {args.basis} memory, width {args.width}, depth {args.depth}: "
          f"{count_parameters(model)} parameters, {model.num_inputs} detector bits in, {model.pool.shape[0]} logits out")
    if resource:
        print(f"memory of the activations in training, on the CPU: {memory_per_shot(model):.0f} MB per shot in 32-bit floats")

    config = train.DEFAULTS | {"code": args.code, "basis": args.basis, "detectors": args.detectors, "width": args.width,
                               "depth": args.depth, "batch": batch, "p_start": args.p, "p_train": args.p,
                               "report_p": [args.p], "lr_decay_steps": 1000, "amp": args.amp}
    run = train.new_run(config, device)
    amp = {torch.float16: "fp16", torch.bfloat16: "bf16", None: "off"}[run.amp_dtype]
    name = torch.cuda.get_device_name(0) if on_gpu else f"CPU, {args.threads} threads"
    print(f"device: {name} | mixed precision {amp} | batch {batch} | shots drawn at p={args.p:g} from "
          f"{run.sampler.num_mechanisms} mechanisms | mean of {repeats} calls")
    parts = step_parts(run, repeats, warm_up)
    print("one training step:")
    for part, seconds in parts.items():
        print(f"  {part:38s} {seconds * 1e3:9.2f} ms")
    print(f"  -> {batch / parts['whole step']:.0f} shots/s")
    print("operations of one residual block, forward and backward:   ms per call x calls per step = ms per step")
    for operation, seconds, calls in block_operations(run, repeats, warm_up):
        total = f"x {calls:3d} = {seconds * calls * 1e3:8.2f}" if calls else "(for comparison)"
        print(f"  {operation:38s} {seconds * 1e3:9.3f} {total}")
    if on_gpu:
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB")


if __name__ == "__main__":
    main()
