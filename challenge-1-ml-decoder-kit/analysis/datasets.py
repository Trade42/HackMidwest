"""The datasets of the gate measurements: make the exploration sets, and record what every file is.

  python -m analysis.datasets make --out-dir DIR [--shots-003 100000] [--shots-005 20000]
  python -m analysis.datasets manifest FILE... --out analysis/results/datasets.json

`make` calls the kit's make_dataset.py with the brief's seeds: p = 0.003 with seeds 9103 (X) and
9104 (Z), p = 0.005 with seeds 9105 (X) and 9106 (Z). A Stim seed does not give the same shots on
every machine, so a dataset is identified by its sha256, never by its seed alone.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import stim

from analysis.common import refuse_held_out, sha256, write_json

EXPLORATION = [(0.003, "X", 9103), (0.003, "Z", 9104), (0.005, "X", 9105), (0.005, "Z", 9106)]


def exploration_path(out_dir, p, basis):
    return Path(out_dir) / f"p{p}" / f"explore_{basis}.npz"


def make(args):
    shots = {0.003: args.shots_003, 0.005: args.shots_005}
    for p, basis, seed in EXPLORATION:
        out = exploration_path(args.out_dir, p, basis)
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "make_dataset.py", "--basis", basis, "--p", str(p), "--shots", str(shots[p]),
               "--seed", str(seed), "--out", str(out)]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    manifest_files([exploration_path(args.out_dir, p, b) for p, b, _ in EXPLORATION], args.manifest)


def describe(path):
    refuse_held_out(path)
    with np.load(path) as z:
        return dict(file=Path(path).name, path=str(path), sha256=sha256(path), basis=str(z["basis"]),
                    p=float(z["p"]), seed=int(z["seed"]), shots=int(z["detectors"].shape[0]),
                    detectors=int(z["detectors"].shape[1]))


def manifest_files(files, out):
    rows = [describe(f) for f in files]
    for r in rows:
        print(f"{r['file']:20s} p={r['p']} {r['basis']} seed {r['seed']} shots {r['shots']} sha256 {r['sha256']}")
    if out:
        write_json(out, dict(stim_version_here=stim.__version__, datasets=rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--out-dir", required=True)
    m.add_argument("--shots-003", type=int, default=100_000)
    m.add_argument("--shots-005", type=int, default=20_000)
    m.add_argument("--manifest", help="also write the manifest json here")
    f = sub.add_parser("manifest")
    f.add_argument("files", nargs="+")
    f.add_argument("--out")
    args = ap.parse_args()
    if args.cmd == "make":
        make(args)
    else:
        manifest_files(args.files, args.out)


if __name__ == "__main__":
    main()
