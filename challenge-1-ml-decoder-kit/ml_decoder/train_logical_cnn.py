"""Train a CNN that reads one syndrome of the Gross code memory experiment and predicts its 12 logical bits.

  python train_logical_cnn.py --basis X --hours 8                        overnight on the RTX 4060
  python train_logical_cnn.py --basis X --minutes 2 --width 32 --depth 2 --batch 256      quick check on any machine
  python train_logical_cnn.py --basis X --eval model_X.pt --shots 20000                   error rates of a saved model

It lives in challenge-1-ml-decoder-kit/ml_decoder/ and finds the kit one folder up (or pass --kit). Needs PyTorch.
The data never runs out: faults are drawn on the GPU from the kit's own error model at a random p per batch, with
syndrome = H e mod 2 and labels = L e mod 2. Nothing is read from disk, so the model cannot memorise a dataset.
The curriculum starts at low noise and moves up to the challenge's range, 0.003 to 0.005.
These are DEM-sampled shots. Final numbers must come from the kit's Stim shots through score().
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
    def batch(self, p_index, size):
        faults = (torch.rand(size, self.columns, device=self.device) < self.priors[p_index]).float()
        syndrome = torch.sparse.mm(self.H, faults.T).T.remainder(2)
        logical = torch.sparse.mm(self.L, faults.T).T.remainder(2)
        return syndrome.reshape(size, ROUNDS, NX, NY), logical


def curriculum(progress, levels):
    """Index into the noise grid: a window of three levels that reaches the top after 40% of the run."""
    top = min(levels - 1, 1 + int((levels - 1) * min(1.0, progress / 0.4)))
    return int(np.random.randint(max(0, top - 2), top + 1))


@torch.no_grad()
def report(model, sampler, shots, batch, levels):
    """Per p: shots with any wrong bit, bits right, and the bits an always-0 guess would get right."""
    model.eval()
    parts = []
    for p in levels:
        wrong_bits = wrong_shots = ones = seen = 0
        while seen < shots:
            syndrome, logical = sampler.batch(sampler.grid.index(p), batch)
            miss = (model(syndrome) > 0) != (logical > 0.5)
            wrong_bits += miss.sum().item()
            wrong_shots += miss.any(dim=1).sum().item()
            ones += logical.sum().item()
            seen += batch
        bits = seen * logical.shape[1]
        parts.append(f"p={p}: {wrong_shots / seen:.2%} of shots wrong, {1 - wrong_bits / bits:.2%} of bits right "
                     f"(always-0 guess {1 - ones / bits:.2%})")
    model.train()
    return " | ".join(parts)


def main():
    args = parse_args()
    out = args.out or f"model_{args.basis}.pt"
    train_grid = [float(v) for v in args.p_train.split(",")] if args.p_train else P_GRID
    levels = [float(v) for v in args.p_report.split(",")] if args.p_report else P_REPORT
    sampler = Sampler(args.kit, args.basis, args.device, sorted(set(train_grid) | set(levels)))
    train_index = [sampler.grid.index(p) for p in train_grid]
    config = dict(width=args.width, depth=args.depth, kernel=args.kernel)
    start_from = args.eval or args.resume
    if start_from:
        saved = torch.load(start_from, map_location=args.device)
        config = saved["config"]
    model = LogicalCNN(**config).to(args.device)
    if start_from:
        model.load_state_dict(saved["model"])
    print(f"{args.basis} memory on {args.device}: {sampler.columns} faults, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters", flush=True)
    if args.eval:
        print(report(model, sampler, args.shots, args.batch, levels))
        return
    seconds = args.hours * 3600 + args.minutes * 60
    assert seconds > 0, "give --hours or --minutes"
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    amp = args.device.startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=amp)
    start = last = time.time()
    step = shown = 0
    running = None
    while time.time() - start < seconds:
        progress = (time.time() - start) / seconds
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
        if time.time() - last > args.report * 60:
            last = time.time()
            torch.save({"model": model.state_dict(), "config": config}, out)
            print(f"{(last - start) / 3600:5.2f} h, {step * args.batch / 1e6:7.1f} M shots, loss {running:.4f} | "
                  f"{report(model, sampler, args.shots, args.batch, levels)}", flush=True)
            shown += 1
    torch.save({"model": model.state_dict(), "config": config}, out)
    print(f"done: {step * args.batch / 1e6:.1f} M shots, loss {running:.4f} | {report(model, sampler, args.shots, args.batch, levels)}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
