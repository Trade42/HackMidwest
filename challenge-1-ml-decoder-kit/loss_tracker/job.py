"""One self-contained job for another machine: sample a dataset, run the tracker, decode, print the tables.

usage (from challenge-1-ml-decoder-kit/, any Python with loss/requirements.txt):
    python -m loss_tracker.job --seed 7300 --shots 4000 --workers 10 --out OUTDIR \
        --tables loss_tracker/tables/X_p0.004_s1e-3.npz --conditions starter,exact,tracker,oracle-round

- It needs no lane lock, no scratch folder and no particular venv: every step runs `sys.executable -m ...`.
- OUTDIR/data.npz: the dataset (Clifft, chunks of 2,000 shots, chunk k uses seed + k). Its sha256 is printed.
- OUTDIR/post.npz: the tracker's output (track.py: it reads detectors and heralds only), with the committed
  tables and the version given by --mode/--sweeps/--order/--explain (default: order 1, coupled, 2 sweeps).
- OUTDIR/accuracy.{md,json}: the tracker against the probes of this dataset (accuracy.py).
- OUTDIR/parts/START-STOP.npz: per-shot results of every condition, in parts of --part shots. A finished part
  is kept, so a rerun of the same command resumes. Every condition is decoded on every shot of a part, so any
  prefix of the parts is a complete paired comparison.
- OUTDIR/tables.{md,json}: summarize.py on the parts. Everything is also printed.
Results depend only on the shots and the conditions, not on the number of workers or parts.
"""
import argparse
import json
import os
import subprocess
import sys
import time

from loss.lossmodel import make_dataset, save_dataset


def run(*args):
    subprocess.run([sys.executable, "-m", *args], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", default="X")
    ap.add_argument("--p", type=float, default=0.004)
    ap.add_argument("--s", type=float, default=1e-3)
    ap.add_argument("--shots", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--part", type=int, default=500)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--mode", default="coupled")
    ap.add_argument("--sweeps", type=int, default=2)
    ap.add_argument("--order", type=int, default=1)
    ap.add_argument("--explain", action="store_true", help="the tracker version that explains shared bits")
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--decode-shots", type=int, help="decode the first N shots only (default: all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.perf_counter()
    data, post = os.path.join(a.out, "data.npz"), os.path.join(a.out, "post.npz")
    if not os.path.exists(data):
        d = make_dataset(a.basis, a.p, a.s, a.shots, a.seed, threads=a.workers)
        sha = save_dataset(d, data + ".tmp.npz")
        os.replace(data + ".tmp.npz", data)
        lost = (d["data_slot"] > 0).sum(axis=1)
        print(json.dumps({"dataset": data, "sha256": sha, "basis": a.basis, "p": a.p, "s": a.s, "shots": a.shots,
                          "seed": a.seed, "lost_data_atoms_per_shot": round(float(lost.mean()), 3),
                          "check_heralds_per_shot": round(float(d["heralds"][:, :1728].sum(axis=1).mean()), 3),
                          "sample_seconds": round(time.perf_counter() - t0, 1)}), flush=True)
        del d
    if not os.path.exists(post):
        run("loss_tracker.track", "--data", data, "--tables", a.tables, "--mode", a.mode, "--sweeps", str(a.sweeps),
            "--order", str(a.order), *(["--explain"] if a.explain else []), "--out", post + ".tmp.npz")
        os.replace(post + ".tmp.npz", post)
    run("loss_tracker.accuracy", "--data", data, "--post", f"tracker={post}", "--main", "tracker",
        "--out", os.path.join(a.out, "accuracy"))
    n = a.decode_shots or a.shots
    for start in range(0, n, a.part):
        stop = min(start + a.part, n)
        out = os.path.join(a.out, "parts", f"{start:06d}-{stop:06d}.npz")
        if os.path.exists(out):
            print(f"keep {out}", flush=True)
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        run("loss_tracker.decode", "--data", data, "--post", post, "--conditions", a.conditions, "--start", str(start),
            "--stop", str(stop), "--workers", str(a.workers), "--out", out + ".tmp.npz")
        os.replace(out + ".tmp.npz", out)
    run("loss_tracker.summarize", "--parts", os.path.join(a.out, "parts", "*.npz"), "--order", a.conditions,
        "--title", f"{a.basis} memory, p = {a.p}, s = {a.s:g}, seed {a.seed}", "--out", os.path.join(a.out, "tables"))
    print(f"job done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
