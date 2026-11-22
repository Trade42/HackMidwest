"""Step 4: a learned tracker. A small recurrent network per flagged data atom, to compare with the model of step 2.

usage (from challenge-1-ml-decoder-kit/):
    python -m loss_tracker.learned train --data TRAIN.npz --out loss_tracker/tables/learned_X_p0.004_s1e-3.pt
    python -m loss_tracker.learned track --data VAL.npz --model loss_tracker/tables/learned_X_p0.004_s1e-3.pt --out POST.npz

Inputs of one atom, for each of the 13 detector rounds (what the brief lists, and nothing from the probes):
  the 3 detector bits of its X checks and the 3 of its Z checks (in its CX order), and whether those rounds exist;
  for each of those 6 checks, whether its check atom was flagged in this cycle or the one before;
  whether a check atom of the other type next to it was flagged then (per type);
  constant over the rounds: left or right data atom, and for each of its 6 checks how many OTHER flagged data
  atoms sit on it.
The network: a bidirectional GRU (64 units) over the rounds, then for each of the 12 cycles a linear layer to the
6 loss slots of that cycle; a softmax over the 72 slots. Trained with cross-entropy on the true loss slot of
TRAINING shots, on the CPU, seed 0.

`train` reads the truth of the training file. `track` reads detectors and heralds only (track.read_inputs) and
writes the same file format as track.py, so accuracy.py and decode.py can use it.
"""
import argparse
import hashlib
import json
import time

import numpy as np

from loss.lossmodel import CYCLES, load_dataset
from loss_tracker.track import read_inputs
from loss_tracker.tracker import ROUNDS, SLOTS, TYPES, Geometry, features

HIDDEN = 64


def atom_inputs(geo, detectors, heralds, batch=500):
    """(shot, atom, x) for every flagged data atom: x has shape (entries, 13 rounds, 23 features)."""
    shots_, atoms_, xs = [], [], []
    for lo in range(0, len(detectors), batch):
        feat = features(geo, detectors[lo:lo + batch], heralds[lo:lo + batch])
        flagged = feat["flagged"]
        s, a = np.nonzero(flagged)
        cols = []
        for T in TYPES:
            y = feat["pattern"][T][s, a]                                              # (n, 13)
            cols += [((y >> m) & 1) for m in range(3)]
            cols += [np.broadcast_to(np.isin(np.arange(ROUNDS), geo.rounds[T])[None, :], y.shape)]
            cols += [feat["own_herald"][T][s, a][:, m] for m in range(3)]
            cols += [feat["hook"][T][s, a]]
        cols += [np.broadcast_to(geo.side[a][:, None], (len(a), ROUNDS))]
        for T in TYPES:
            others = flagged[s[:, None, None], geo.co_atom[T][a]].sum(axis=2)         # (n, 3): other flagged atoms per check
            cols += [np.broadcast_to(others[:, m, None] / 2.0, (len(a), ROUNDS)) for m in range(3)]
        xs.append(np.stack(cols, axis=2).astype(np.float32))
        shots_.append(s + lo)
        atoms_.append(a)
    return np.concatenate(shots_), np.concatenate(atoms_), np.concatenate(xs)


def build_model(n_features):
    import torch

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = torch.nn.GRU(n_features, HIDDEN, batch_first=True, bidirectional=True)
            self.head = torch.nn.Linear(2 * HIDDEN, 6)

        def forward(self, x):                       # x: (n, 13, features) -> (n, 72) logits over the loss slots
            h, _ = self.gru(x)
            return self.head(h[:, :CYCLES]).reshape(len(x), SLOTS)

    return Net()


def train(a):
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    data = load_dataset(a.data)
    meta = data["meta"]
    n = a.shots or meta["shots"]
    geo = Geometry(meta["basis"], meta["p"])
    shot, atom, x = atom_inputs(geo, data["detectors"][:n], data["heralds"][:n])
    label = data["data_slot"][:n][shot, atom].astype(np.int64) - 1                    # the truth: training shots only
    assert (label >= 0).all()
    X, Y = torch.from_numpy(x), torch.from_numpy(label)
    model = build_model(x.shape[2])
    steps = a.epochs * ((len(X) + a.batch - 1) // a.batch)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)
    gen = torch.Generator().manual_seed(0)
    t0 = time.perf_counter()
    for epoch in range(a.epochs):
        perm = torch.randperm(len(X), generator=gen)
        total = 0.0
        for lo in range(0, len(X), a.batch):
            idx = perm[lo:lo + a.batch]
            loss = torch.nn.functional.cross_entropy(model(X[idx]), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        print(f"epoch {epoch + 1}/{a.epochs}  mean cross-entropy over the 72 slots {total / len(X):.4f}  {time.perf_counter() - t0:.0f}s", flush=True)
    info = {"basis": meta["basis"], "p": meta["p"], "s": meta["s"], "train_file": a.data.split("/")[-1], "train_seed": meta["seed"],
            "train_shots": n, "train_atoms": int(len(X)), "features": int(x.shape[2]), "hidden": HIDDEN, "epochs": a.epochs,
            "batch": a.batch, "lr": a.lr, "parameters": int(sum(p.numel() for p in model.parameters())),
            "train_sha256": hashlib.sha256(open(a.data, "rb").read()).hexdigest()}
    torch.save({"state": model.state_dict(), "info": info}, a.out)
    print(json.dumps({**info, "out": a.out, "seconds": round(time.perf_counter() - t0, 1)}), flush=True)


def track(a):
    import torch
    torch.set_num_threads(a.threads)
    detectors, heralds, meta = read_inputs(a.data)                                     # detectors and heralds only
    n = a.shots or len(detectors)
    saved = torch.load(a.model, weights_only=True)
    info = saved["info"]
    geo = Geometry(info["basis"], info["p"])
    shot, atom, x = atom_inputs(geo, detectors[:n], heralds[:n])
    model = build_model(x.shape[2])
    model.load_state_dict(saved["state"])
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        logp = torch.cat([torch.log_softmax(model(torch.from_numpy(x[lo:lo + 4096])), dim=1) for lo in range(0, len(x), 4096)]).numpy()
    seconds = time.perf_counter() - t0
    # the file format of track.py: posterior = softmax(log_prior + loglik); the network gives the posterior itself
    w = (1 - info["s"]) ** np.arange(SLOTS)
    log_prior = np.log(w / w.sum())
    out = {"data": a.data.split("/")[-1], "data_sha256": hashlib.sha256(open(a.data, "rb").read()).hexdigest(), "data_seed": meta["seed"],
           "basis": meta["basis"], "p": meta["p"], "s": meta["s"], "shots": n, "model": a.model, "mode": "learned",
           "model_sha256": hashlib.sha256(open(a.model, "rb").read()).hexdigest(), "flagged_data_atoms": int(len(shot))}
    np.savez_compressed(a.out, shot=shot, atom=atom, loglik=(logp - log_prior[None, :]).astype(np.float32), log_prior=log_prior,
                        meta=np.array(json.dumps(out)))
    print(json.dumps({**out, "out": a.out, "network_seconds": round(seconds, 2)}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--shots", type=int)
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch", type=int, default=512)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--threads", type=int, default=4)
    t.add_argument("--out", required=True)
    k = sub.add_parser("track")
    k.add_argument("--data", required=True)
    k.add_argument("--model", required=True)
    k.add_argument("--shots", type=int)
    k.add_argument("--threads", type=int, default=4)
    k.add_argument("--out", required=True)
    a = ap.parse_args()
    (train if a.cmd == "train" else track)(a)


if __name__ == "__main__":
    main()
