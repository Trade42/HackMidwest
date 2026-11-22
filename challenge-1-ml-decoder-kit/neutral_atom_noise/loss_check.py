"""Check that our loss loop equals loss/'s own job when the noise is the kit's.

  python -m neutral_atom_noise.loss_check --shots 100 --p 0.004 --s 1e-3 --seed 5300 --workers 4 --out OUTDIR

With every rate at the kit's uniform p and atom loss s per atom per CX:

  theirs   `python -m loss.job` samples the shots with Clifft and decodes them under its `starter` condition
           (the published flag-aware recipe).
  ours     `make_shots.sample_loss` samples our rewrite of the circuit at the same seed, and `run.decode_shots`
           decodes with --priors uniform:p --flags starter.

They must agree bit for bit on the shots (detectors, observables, heralds) and shot for shot on the decoding
(failed, converged, iterations, wrong observables). The exit code is 0 only if everything is equal.
"""
import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from neutral_atom_noise.make_shots import load_shots, make
from neutral_atom_noise.model import kit_uniform
from neutral_atom_noise.run import decode_shots

KIT = Path(__file__).resolve().parent.parent     # challenge-1-ml-decoder-kit/


def theirs(basis, p, s, shots, seed, workers, out):
    """Run loss.job as a program. Returns its dataset and its `starter` result."""
    from loss.lossmodel import load_dataset
    subprocess.run([sys.executable, "-m", "loss.job", "--basis", basis, "--p", repr(p), "--s", repr(s), "--shots", str(shots),
                    "--seed", str(seed), "--workers", str(workers), "--part", str(shots), "--out", str(out),
                    "--group", f"starter:{shots}"], check=True, cwd=KIT, stdout=subprocess.DEVNULL)
    with np.load(out / "parts" / "starter" / f"{0:06d}-{shots:06d}.npz") as f:
        result = {k: f[k][:, 0] for k in ("failed", "converged", "iterations", "wrong_mask")}   # one column per condition
    return load_dataset(out / "data.npz"), result


def ours(basis, p, s, shots, seed, workers, out):
    """Our sampler and our decode loop on the kit's uniform noise plus loss. Returns the shots and the result."""
    noise = replace(kit_uniform(p), loss=s)
    path = make(basis, None, p, True, shots, seed, out, noise=noise, name=f"{basis}_p{p:g}_s{s:g}_loss_seed{seed}")
    return load_shots(path), decode_shots(str(path), "relay", f"uniform:{p!r}", 0, shots, workers, flags="starter")


def compare(basis="X", p=0.004, s=1e-3, shots=100, seed=5300, workers=4, out="loss_check"):
    """A report: for each compared array, whether theirs equals ours; and the failure counts."""
    out = Path(out)
    their_shots, their_result = theirs(basis, p, s, shots, seed, workers, out / "theirs")
    our_shots, our_result = ours(basis, p, s, shots, seed, workers, out / "ours")
    equal = {name: bool(np.array_equal(their_shots[name], our_shots[name])) for name in ("detectors", "observables", "heralds")}
    for name in their_result:
        if their_result[name].shape != our_result[name].shape:
            raise SystemExit(f"{name}: shapes {their_result[name].shape} and {our_result[name].shape}")
        equal[name] = bool((their_result[name] == our_result[name]).all())
    return {"basis": basis, "p": p, "s": s, "shots": shots, "seed": seed, "equal": equal, "all_equal": all(equal.values()),
            "failures_theirs": int(their_result["failed"].sum()), "failures_ours": int(our_result["failed"].sum()),
            "lost_data_atoms_per_shot": float(our_shots["heralds"][:, 1728:].sum(axis=1).mean()),
            "mean_iterations": float(our_result["iterations"].mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--basis", choices=["X", "Z"], default="X")
    parser.add_argument("--p", type=float, default=0.004)
    parser.add_argument("--s", type=float, default=1e-3)
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--seed", type=int, default=5300)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = compare(args.basis, args.p, args.s, args.shots, args.seed, args.workers, args.out)
    print(json.dumps(report), flush=True)
    sys.exit(0 if report["all_equal"] else 1)


if __name__ == "__main__":
    main()
