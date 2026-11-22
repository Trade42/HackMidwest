"""Train a CNN that reads one syndrome of the Gross code memory experiment and predicts its 12 logical bits.

  python train_logical_cnn.py --basis X --hours 8                        overnight on the RTX 4060
  python train_logical_cnn.py --basis X --minutes 2 --width 32 --depth 2 --batch 256      quick check on any machine
  python train_logical_cnn.py --basis X --eval model_X.pt --shots 20000                   error rates of a saved model
  python train_logical_cnn.py --basis X --train-shots 500000 --report-shots 100000 --seed 1 --threads 4 ...   repeatable

It lives in challenge-1-ml-decoder-kit/ml_decoder/ and finds the kit one folder up (or pass --kit). Needs PyTorch.
The data never runs out: faults are drawn on the GPU from the kit's own error model at a random p per batch, with
syndrome = H e mod 2 and labels = L e mod 2. Nothing is read from disk, so the model cannot memorise a dataset.
The curriculum starts at low noise and moves up to the challenge's range, 0.003 to 0.005.
These are DEM-sampled shots. Final numbers must come from the kit's Stim shots through score().

--seed seeds torch, numpy and cuda. A run limited by --hours or --minutes still depends on the clock: how many steps
fit, the learning-rate schedule and the curriculum all follow the elapsed time. A run limited by --train-shots counts
training shots instead, so the same command, seed and thread count give the same model and the same lines on the
same CPU. On a GPU the seed fixes the data, but some kernels are not bit-for-bit repeatable.
Each report scores the model on the same held-out shots, drawn from their own seed (seed + 1,000,000).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

P_GRID = [0.0005, 0.001, 0.002, 0.003, 0.004, 0.005]      # curriculum, low noise first
P_REPORT = [0.003, 0.004, 0.005]
ROUNDS, NX, NY = 13, 12, 6                                # detector row = 72 * round + 6 * x + y


def default_kit():
    """The kit folder: this file's folder if circuit.py is in it, else the folder above."""
    here = os.path.dirname(os.path.abspath(__file__))
    return here if os.path.exists(os.path.join(here, "circuit.py")) else os.path.dirname(here)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kit", default=default_kit(), help="folder with circuit.py")
    parser.add_argument("--basis", choices=["X", "Z"], default="X")
    parser.add_argument("--hours", type=float, default=0.0)
    parser.add_argument("--minutes", type=float, default=0.0)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--kernel", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--report", type=float, default=10.0, help="minutes between reports and checkpoints")
    parser.add_argument("--shots", type=int, default=20_000, help="held-out shots per p in each report")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None, help="checkpoint file, default model_<basis>.pt")
    parser.add_argument("--resume", default=None, help="checkpoint to continue from")
    parser.add_argument("--eval", default=None, help="checkpoint to evaluate, no training")
    parser.add_argument("--p-train", default=None, help="comma list of p to train on instead of the curriculum")
    parser.add_argument("--p-report", default=None, help="comma list of p to report on")
    parser.add_argument("--seed", type=int, default=1, help="seeds torch, numpy and cuda; printed and saved")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads; pin it for a repeatable CPU run")
    parser.add_argument("--train-shots", type=int, default=0,
                        help="stop after this many training shots instead of --hours or --minutes; progress, curriculum "
                             "and reports then count shots, not seconds")
    parser.add_argument("--report-shots", type=int, default=0, help="with --train-shots: training shots between reports")
    return parser.parse_args()


class Block(nn.Module):
    def __init__(self, width, kernel):
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, kernel, padding=kernel // 2, padding_mode="circular", bias=False)
        self.norm1 = nn.GroupNorm(8, width)
        self.conv2 = nn.Conv2d(width, width, kernel, padding=kernel // 2, padding_mode="circular", bias=False)
        self.norm2 = nn.GroupNorm(8, width)

    def forward(self, x):
        y = F.gelu(self.norm1(self.conv1(x)))
        return F.gelu(x + self.norm2(self.conv2(y)))


class LogicalCNN(nn.Module):
    """Rounds are channels, the 12 x 6 torus is the image, padding wraps around."""

    def __init__(self, width=128, depth=6, kernel=5, outputs=12):
        super().__init__()
        self.stem = nn.Conv2d(ROUNDS, width, kernel, padding=kernel // 2, padding_mode="circular")
        self.blocks = nn.Sequential(*[Block(width, kernel) for _ in range(depth)])
        self.squeeze = nn.Conv2d(width, 16, 1)
        self.head = nn.Sequential(nn.Linear(16 * NX * NY, 512), nn.GELU(), nn.Linear(512, outputs))

    def forward(self, syndrome):                         # (batch, 13, 12, 6) of 0 and 1
        x = self.blocks(F.gelu(self.stem(syndrome * 2 - 1)))
        return self.head(F.gelu(self.squeeze(x)).flatten(1))


def sparse(matrix, device):
    coo = matrix.tocoo()
    index = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(index, torch.ones(coo.nnz), coo.shape, check_invariants=False).coalesce().to(device)


class Sampler:
    """Fresh (syndrome, logical bits) batches from the kit's error model."""

    def __init__(self, kit, basis, device, grid):
        sys.path.insert(0, kit)
        from circuit import GROSS_CIRCUITS, dem_matrices, load_circuit
        priors, shape = [], None
        self.grid = list(grid)
        for p in self.grid:
            H, L, pri = dem_matrices(load_circuit(GROSS_CIRCUITS[basis], p), basis)
            if shape is None:
                shape, H0, L0 = H.shape, H, L
            assert H.shape == shape and (H != H0).nnz == 0 and (L != L0).nnz == 0, "the fault set changes with p"
            priors.append(np.asarray(pri, np.float32))
        assert shape[0] == ROUNDS * NX * NY, f"expected {ROUNDS * NX * NY} detectors, got {shape[0]}"
        self.H, self.L = sparse(H0, device), sparse(L0, device)
        self.priors = torch.tensor(np.stack(priors), device=device)
        self.columns, self.device = shape[1], device

    @torch.no_grad()
    def batch(self, p_index, size, generator=None):
        faults = (torch.rand(size, self.columns, device=self.device, generator=generator) < self.priors[p_index]).float()
        syndrome = torch.sparse.mm(self.H, faults.T).T.remainder(2)
        logical = torch.sparse.mm(self.L, faults.T).T.remainder(2)
        return syndrome.reshape(size, ROUNDS, NX, NY), logical


def curriculum(progress, levels):
    """Index into the noise grid: a window of three levels that reaches the top after 40% of the run."""
    top = min(levels - 1, 1 + int((levels - 1) * min(1.0, progress / 0.4)))
    return int(np.random.randint(max(0, top - 2), top + 1))


REPORT_SEED_OFFSET = 1_000_000


@torch.no_grad()
def report(model, sampler, shots, batch, levels, seed):
    """Per p, on held-out DEM shots: shots with any wrong bit, bits right, and what an always-0 guess gets.

    The shots come from their own generator, seeded with seed + REPORT_SEED_OFFSET, so every report of a run sees the
    same shots and the training stream does not depend on when reports happen.
    """
    model.eval()
    generator = torch.Generator(device=sampler.device).manual_seed(seed + REPORT_SEED_OFFSET)
    parts = []
    for p in levels:
        wrong_bits = wrong_shots = ones = nonzero = seen = 0
        while seen < shots:
            syndrome, logical = sampler.batch(sampler.grid.index(p), batch, generator)
            miss = (model(syndrome) > 0) != (logical > 0.5)
            wrong_bits += miss.sum().item()
            wrong_shots += miss.any(dim=1).sum().item()
            ones += logical.sum().item()
            nonzero += (logical > 0.5).any(dim=1).sum().item()
            seen += batch
        bits = seen * logical.shape[1]
        parts.append(f"p={p}, {seen} DEM shots: {wrong_shots / seen:.2%} of shots wrong (always-0 guess {nonzero / seen:.2%}), "
                     f"{1 - wrong_bits / bits:.2%} of bits right (always-0 guess {1 - ones / bits:.2%})")
    model.train()
    return " | ".join(parts)


def load_checkpoint(path, basis, device):
    """(model in eval mode, the saved dict). Refuses a checkpoint trained on the other memory basis.

    Checkpoints written before the basis, p grid and seed were saved (the code at 0eab7dd) still load, with a warning.
    """
    saved = torch.load(path, map_location=device, weights_only=True)
    if "basis" not in saved:
        print(f"warning: {path} records no basis, training grid or seed (an older checkpoint). "
              f"Its memory basis cannot be checked against {basis}.", file=sys.stderr, flush=True)
    elif saved["basis"] != basis:
        raise ValueError(f"{path} was trained on {saved['basis']} memory shots; this run decodes {basis} memory")
    model = LogicalCNN(**saved["config"]).to(device)
    model.load_state_dict(saved["model"])
    return model.eval(), saved


def shot_logits(model, syndrome, device):
    """The 12 logits of one shot, numpy in and numpy out. This is the path plugin.decode_shot runs and ml_first.py times:
    the copy to the device, the model, the copy back, and on cuda a synchronise, so nothing is left running."""
    x = torch.from_numpy(np.asarray(syndrome, dtype=np.float32)).to(device).reshape(1, ROUNDS, NX, NY)
    with torch.no_grad():
        logits = model(x)[0].float().cpu().numpy()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return logits


def main():
    args = parse_args()
    torch.manual_seed(args.seed)                          # also seeds every cuda device
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    out = args.out or f"model_{args.basis}.pt"
    train_grid = [float(v) for v in args.p_train.split(",")] if args.p_train else P_GRID
    levels = [float(v) for v in args.p_report.split(",")] if args.p_report else P_REPORT
    print(f"{args.basis} memory on {args.device}, seed {args.seed}, {torch.get_num_threads()} torch threads, "
          f"train p {train_grid}, report p {levels}", flush=True)
    sampler = Sampler(args.kit, args.basis, args.device, sorted(set(train_grid) | set(levels)))
    train_index = [sampler.grid.index(p) for p in train_grid]
    config = dict(width=args.width, depth=args.depth, kernel=args.kernel)
    start_from = args.eval or args.resume
    earlier_shots = 0
    if start_from:
        model, saved = load_checkpoint(start_from, args.basis, args.device)
        config, earlier_shots = saved["config"], saved.get("shots", 0)
        if args.resume and saved.get("seed") == args.seed:
            print(f"warning: {start_from} was trained with seed {args.seed}; resuming with it replays the same "
                  f"training shots. Pass another --seed.", file=sys.stderr, flush=True)
    else:
        model = LogicalCNN(**config).to(args.device)
    model.train()
    print(f"{sampler.columns} faults, {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters", flush=True)
    if args.eval:
        print(report(model, sampler, args.shots, args.batch, levels, args.seed))
        return
    seconds = args.hours * 3600 + args.minutes * 60
    budget = args.train_shots
    assert (seconds > 0) != (budget > 0), "give --hours or --minutes, or --train-shots"

    def checkpoint(steps):
        """Plain Python values and tensors only, so torch.load(..., weights_only=True) reads it."""
        return {"model": model.state_dict(), "config": config, "basis": args.basis, "p_train": train_grid,
                "p_report": levels, "seed": args.seed, "batch": args.batch, "lr": args.lr,
                "shots": earlier_shots + steps * args.batch, "resumed_from": args.resume}

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    amp = args.device.startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=amp)
    start = last = time.time()
    step = shown = 0
    next_report = args.report_shots
    running = None
    while (step * args.batch < budget) if budget else (time.time() - start < seconds):
        progress = step * args.batch / budget if budget else (time.time() - start) / seconds
        for group in optimizer.param_groups:               # warm-up, then cosine decay
            group["lr"] = args.lr * min(1.0, (step + 1) / 500) * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * progress)))
        syndrome, logical = sampler.batch(train_index[curriculum(progress, len(train_index))], args.batch)
        with torch.autocast(device_type="cuda" if amp else "cpu", dtype=torch.float16 if amp else torch.bfloat16, enabled=amp):
            loss = F.binary_cross_entropy_with_logits(model(syndrome).float(), logical)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        step += 1
        running = loss.item() if running is None else 0.98 * running + 0.02 * loss.item()
        if budget:                                         # shot-counted reports: no clock in the line
            if args.report_shots and step * args.batch >= next_report and step * args.batch < budget:
                next_report += args.report_shots
                torch.save(checkpoint(step), out)
                print(f"{step * args.batch:9d} shots, loss {running:.4f} | "
                      f"{report(model, sampler, args.shots, args.batch, levels, args.seed)}", flush=True)
        elif time.time() - last > args.report * 60:
            last = time.time()
            torch.save(checkpoint(step), out)
            print(f"{(last - start) / 3600:5.2f} h, {step * args.batch / 1e6:7.1f} M shots, loss {running:.4f} | "
                  f"{report(model, sampler, args.shots, args.batch, levels, args.seed)}", flush=True)
            shown += 1
    torch.save(checkpoint(step), out)
    print(f"done: {step * args.batch} shots, loss {running:.4f} | "
          f"{report(model, sampler, args.shots, args.batch, levels, args.seed)}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
