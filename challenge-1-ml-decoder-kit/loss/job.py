"""One self-contained loss job for another machine: sample a dataset, decode condition groups, print the tables.

usage (from challenge-1-ml-decoder-kit/, any Python with the folder's requirements.txt):
    python -m loss.job --basis X --p 0.004 --s 1e-3 --shots 4000 --seed 5300 --workers 10 --out OUTDIR \\
        --group blind:300 --group starter,exact,oracle-round:4000 --table starter:starter,exact,oracle-round

- It needs no lane lock, no scratch folder and no particular venv: it runs `sys.executable -m loss.decode`.
- The dataset is OUTDIR/data.npz (Clifft, chunks of 2,000 shots, chunk k uses seed + k). Its sha256 is printed.
- --group CONDITIONS:N decodes the first N shots under each condition, in parts of --part shots
  (OUTDIR/parts/GROUP/START-STOP.npz). A finished part is kept, so a rerun of the same command resumes.
- --table REFERENCE:CONDITIONS prints a table with paired comparisons against REFERENCE (summarize.py).
- The tables are printed to stdout at the end and written to OUTDIR/tables.md.
Results depend only on the shots and the conditions, not on the number of workers or parts.
"""
import argparse
import json
import os
import subprocess
import sys
import time

from loss.lossmodel import make_dataset, save_dataset
from loss.summarize import load_parts, markdown, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", default="X")
    ap.add_argument("--p", type=float, default=0.004)
    ap.add_argument("--s", type=float, required=True)
    ap.add_argument("--shots", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--part", type=int, default=500)
    ap.add_argument("--group", action="append", required=True)
    ap.add_argument("--table", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.perf_counter()
    data = os.path.join(a.out, "data.npz")
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
    for g in a.group:
        conds, _, n = g.rpartition(":")
        n = int(n)
        assert n <= a.shots
        name = conds.replace(",", "+").replace("/", "_").replace("@", "cap")
        for start in range(0, n, a.part):
            stop = min(start + a.part, n)
            out = os.path.join(a.out, "parts", name, f"{start:06d}-{stop:06d}.npz")
            if os.path.exists(out):
                print(f"keep {out}", flush=True)
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            subprocess.run([sys.executable, "-m", "loss.decode", "--data", data, "--conditions", conds,
                            "--start", str(start), "--stop", str(stop), "--workers", str(a.workers),
                            "--out", out + ".tmp.npz"], check=True)
            os.replace(out + ".tmp.npz", out)
    per = load_parts([os.path.join(a.out, "parts", "*", "*.npz")])
    tables = []
    for spec in a.table or [f"{a.group[0].rpartition(':')[0].split(',')[0]}:{','.join(per)}"]:
        ref, _, conds = spec.partition(":")
        rows = summarize(per, ref, conds.split(","))
        tables.append(markdown(rows, ref, f"{a.basis} memory, p = {a.p}, s = {a.s:g}, seed {a.seed}: reference {ref}"))
    text = "\n".join(tables)
    with open(os.path.join(a.out, "tables.md"), "w") as f:
        f.write(text)
    print(text, flush=True)
    print(f"job done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
