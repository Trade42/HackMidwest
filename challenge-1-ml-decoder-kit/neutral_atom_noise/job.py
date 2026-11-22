"""One self-contained job: sample the shots, decode every row, fit the generic p, write the tables and the plot.

  python -m neutral_atom_noise.job --basis X --factors 0.6,0.7,0.8,0.9 --shots 2000 --seed 4100 \
      --decoders relay,bposd --rows matched,generic-2q,generic-fit --workers 16 --out OUTDIR

Run it from challenge-1-ml-decoder-kit/ with any Python that has this folder's requirements.txt. It needs no
path of a particular machine: everything goes under --out, and --workers sets the number of processes.

- Shots: OUTDIR/shots/. Test shots use --seed, train shots --train-seed. Their sha256 is printed.
- Rows, for every factor and decoder (the train shots are a multiple of both part sizes):
    matched       priors from the circuit that made the shots
    generic-2q    the kit's uniform model at the p with the same total two-qubit error
    generic-fit   the kit's uniform model at the p that decodes best on TRAIN shots: p = multiplier x (the
                  generic-2q p) for every multiplier of --grid; the fewest failures win, ties go to the fewest
                  mean iterations. The winner then decodes the test shots. Train results: OUTDIR/train/, the
                  choice: OUTDIR/fit.json. If the winner is multiplier 1 the row is the generic-2q row.
  With --loss the shots also lose atoms, and a row is written FLAGS/PRIORS, for example starter/matched or
  blind/generic-2q (FLAGS is a condition of loss/lossmap.py).
- Results: OUTDIR/results/, in parts of --part shots. A finished part is kept, so the same command resumes.
  --budget-minutes stops before a new part once the time is spent (exit code 3: more to do).
  --first N decodes only the first N test shots of every row; a later call without it (or with a larger N)
  widens the same rows. Keep --part and --relay-part the same in every call on one OUTDIR, and N a multiple
  of both.
- At the end: OUTDIR/tables.md, summary.json, rates.csv and the plot (summarize.py).
"""
import argparse
import json
import sys
import time
from pathlib import Path

from neutral_atom_noise.make_shots import make
from neutral_atom_noise.model import neutral_atom
from neutral_atom_noise.run import open_pool, result_name, run_part
from neutral_atom_noise.summarize import load_results, write_all


class OutOfTime(Exception):
    pass


def decode_all(shots_path, shots, decoder, priors, flags, args, folder, deadline):
    """Decode `shots` shots of a file in parts, keeping finished parts. One worker pool serves the whole row."""
    part = args.part if decoder == "bposd" else args.relay_part
    todo = [(start, min(start + part, shots)) for start in range(0, shots, part)
            if not (Path(folder) / result_name(shots_path, decoder, priors, flags, start, min(start + part, shots))).exists()]
    if not todo:
        return
    command = "python -m neutral_atom_noise.job " + " ".join(sys.argv[1:])
    with open_pool(str(shots_path), decoder, priors, flags, args.workers) as pool:
        for start, stop in todo:
            if deadline and time.monotonic() > deadline:
                raise OutOfTime
            run_part(str(shots_path), decoder, priors, flags, start, stop, args.workers, folder, command=command, pool=pool)


EDGE_STEP, EDGE_STEPS = 1.4, 4      # a best p at the edge of the grid moves the grid outward, at most 4 steps


def fit_generic(factor, decoder, flags, args, out, deadline):
    """The uniform p that decodes the TRAIN shots best. Returns its fit.json entry.

    The grid is --grid times the generic-2q p. If the best point is at an edge of the grid, the grid is extended
    beyond that edge by a factor 1.4 per step (at most 4 steps) until the best point is inside it.
    """
    train = make(args.basis, factor, None, args.loss, args.train_shots, args.train_seed, out / "shots")
    base = neutral_atom(factor).two_qubit_error()
    multipliers, grid = sorted(args.grid), {}
    for _ in range(EDGE_STEPS + 1):
        for multiplier in multipliers:
            if multiplier not in grid:
                grid[multiplier] = _train_point(train, base, multiplier, decoder, flags, args, out, deadline)
        best = min(grid.values(), key=lambda g: (g["failures"], g["mean_iterations"]))
        ordered = sorted(grid)
        if best["multiplier"] == ordered[-1]:
            multipliers = [round(ordered[-1] * EDGE_STEP, 4)]
        elif best["multiplier"] == ordered[0]:
            multipliers = [round(ordered[0] / EDGE_STEP, 4)]
        else:
            break
    return {"p": best["p"], "multiplier": best["multiplier"], "priors": best["priors"], "train_file": train.name,
            "train_seed": args.train_seed, "grid": [grid[m] for m in sorted(grid)],
            "inside_grid": sorted(grid)[0] < best["multiplier"] < sorted(grid)[-1],
            "rule": "fewest failures, then fewest mean iterations; a best point at the edge extends the grid by 1.4 per step"}


def _train_point(train, base, multiplier, decoder, flags, args, out, deadline):
    """Decode the train shots with the kit's uniform model at multiplier x base. Returns its grid entry."""
    priors = f"uniform:{multiplier * base:.6g}"
    shots = min(args.train_first or args.train_shots, args.train_shots)
    decode_all(train, shots, decoder, priors, flags, args, out / "train", deadline)
    row = load_results(out / "train")[(train.name, decoder, priors, flags)]
    return {"multiplier": multiplier, "p": float(priors.removeprefix("uniform:")), "priors": priors,
            "failures": int(row["failed"].sum()), "shots": len(row["failed"]),
            "unconverged": int((~row["converged"]).sum()), "mean_iterations": float(row["iterations"].mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--basis", choices=["X", "Z"], default="X")
    parser.add_argument("--factors", required=True, help="comma-separated noise factors")
    parser.add_argument("--shots", type=int, default=2000, help="test shots per factor")
    parser.add_argument("--seed", type=int, required=True, help="seed of the test shots")
    parser.add_argument("--train-shots", type=int, default=250, help="a multiple of --part")
    parser.add_argument("--train-seed", type=int, default=None, help="seed of the train shots (default: seed + 1000)")
    parser.add_argument("--train-first", type=int, default=None, help="fit on only the first N train shots (for a "
                        "costly decoder); a multiple of the part size")
    parser.add_argument("--grid", default="0.35,0.5,0.7,1", help="multipliers of the generic-2q p tried by generic-fit")
    parser.add_argument("--decoders", default="relay,bposd")
    parser.add_argument("--rows", default="matched,generic-2q,generic-fit")
    parser.add_argument("--loss", action="store_true", help="shots with atom loss (Clifft); rows are FLAGS/PRIORS")
    parser.add_argument("--workers", type=int, default=4, help="worker processes")
    parser.add_argument("--part", type=int, default=125, help="shots per result file of BP-OSD")
    parser.add_argument("--relay-part", type=int, default=50, help="shots per result file of relay-BP (its slow shots "
                        "take up to 18,080 iterations: small parts keep a hold of the lane short)")
    parser.add_argument("--first", type=int, default=None, help="decode only the first N test shots of every row")
    parser.add_argument("--budget-minutes", type=float, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    args.grid = [float(g) for g in args.grid.split(",")]
    args.train_seed = args.seed + 1000 if args.train_seed is None else args.train_seed
    if args.train_seed == args.seed:
        parser.error("the train shots need a seed of their own: nothing may be fitted on the shots it is scored on")
    out = Path(args.out)
    deadline = time.monotonic() + 60 * args.budget_minutes if args.budget_minutes else None
    fit_path = out / "fit.json"
    fit = json.loads(fit_path.read_text()) if fit_path.exists() else {}
    began, finished = time.monotonic(), True
    try:
        for factor in [float(f) for f in args.factors.split(",")]:
            test = make(args.basis, factor, None, args.loss, args.shots, args.seed, out / "shots")
            for decoder in args.decoders.split(","):
                for row in args.rows.split(","):
                    flags, _, priors = row.rpartition("/")
                    flags = flags or None
                    if priors == "generic-fit":
                        entry = fit_generic(factor, decoder, flags, args, out, deadline)
                        fit.setdefault(test.name, {})[decoder + ("/" + flags if flags else "")] = entry
                        fit_path.write_text(json.dumps(fit, indent=1))
                        print(json.dumps({"fit": test.name, "decoder": decoder, "flags": flags, **entry}), flush=True)
                        priors = "generic-2q" if entry["multiplier"] == 1 else entry["priors"]
                    decode_all(test, min(args.first or args.shots, args.shots), decoder, priors, flags, args,
                               out / "results", deadline)
    except OutOfTime:
        finished = False
        print(f"budget of {args.budget_minutes} minutes spent: rerun the same command to go on", flush=True)
    if (out / "results").exists():
        print(write_all(out / "results", fit_path, out), flush=True)
    print(f"job {'done' if finished else 'paused'} after {time.monotonic() - began:.0f} s", flush=True)
    sys.exit(0 if finished else 3)


if __name__ == "__main__":
    main()
