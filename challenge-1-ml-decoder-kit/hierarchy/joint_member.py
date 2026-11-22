"""Joint relay-BP as a member of the ensemble: both check types decoded together, on listed shots.

The baselines decode X memory from its 936 detectors on X checks. The joint problem keeps all
1,728 detectors of the shot and every error mechanism (the kit's dem_matrices(circuit) without
a basis: about 68,000 mechanisms), so it also uses what the Z checks saw. The decoder is the
kit's relay-BP at the kit's RELAY_SETTINGS, on a fresh decoder for every shot, and the answer
is the same 12 observables of the file's basis. This is the "joint" variant of gate G2
(analysis/g2_headroom.py, by the gates worker), rebuilt here; nothing is imported from analysis/.

It is slow (tens of seconds to minutes per shot), so it runs only on the shots a selection
file lists: the shots where the fast members of the ensemble do not all agree
(`python -m hierarchy.ensemble --write-selection`).

Run from challenge-1-ml-decoder-kit/:

  python -m hierarchy.joint_member --data data/p0.004/validation_X.npz \\
      --selection runs/ensemble/validation_X.selection.npz --out runs/joint_p0.004 --processes 12

Several --data files take as many --selection files, in the same order. --shots 3 17 42 lists
shots of one file directly. A selection made from another data file (sha256) is refused, and
so is a file named test_* or go_* without --final.

Every finished shot is saved under <out>/parts/, so a stopped run continues where it stopped.
A shot's answer does not depend on the other shots or on the number of processes: its
decoder is fresh. When all listed shots are done, <out>/<name>.joint.npz holds, one row per
listed shot in rising shot order:

  shot           the shot's index in the data file
  truth          its 12 observable bits, for scoring only
  predicted_J, failed_J, converged_J, iterations_J
  n_faults_J, weight_J   in the joint problem: not comparable with the weights of the basis problem
  seconds_J      the decode call alone
  build_seconds_J        building the shot's fresh decoder, which the decode time leaves out
  data_sha256, shots_in_file, p, basis, settings_J, problem_sha256, machine, processes
"""
import argparse
import hashlib
import json
import multiprocessing
import os
import time
from importlib import metadata
from pathlib import Path

import numpy as np

from circuit import GROSS_CIRCUITS, dem_matrices, load_circuit
from decoders import RELAY_SETTINGS, relay_decoder
from evaluation.adapters import Problem
from hierarchy.candidates import read
from hierarchy.rules import wilson_interval
from hierarchy.stages import load_shots, machine, problem_sha256

NAME = "J"


def joint_problem(basis, p):
    """The kit's problem of one memory experiment with all its detectors: H (1,728 x mechanisms), L, priors."""
    circuit = load_circuit(GROSS_CIRCUITS[basis], p)
    H, L, priors = dem_matrices(circuit)
    return Problem(basis, H, L, priors, p, circuit)


def decode_shot(problem, llr, detectors, changes):
    """One shot on a fresh decoder. The timer wraps the decode call alone, as in gate G2."""
    start = time.perf_counter()
    decoder = relay_decoder(problem.H, problem.priors, **changes)
    built = time.perf_counter()
    result = decoder.decode_detailed(np.ascontiguousarray(detectors, dtype=np.uint8))
    seconds = time.perf_counter() - built
    errors = np.asarray(result.decoding, np.uint8)
    on = errors.astype(bool)
    return {"predicted_J": np.asarray((problem.L @ errors) % 2, np.uint8), "converged_J": bool(result.success),
            "iterations_J": int(result.iterations), "n_faults_J": int(on.sum()), "weight_J": float(llr[on].sum()),
            "seconds_J": seconds, "build_seconds_J": built - start}


_PREPARED = {}   # per process: the shots and the joint problem of each data file


def prepared(job):
    if job["data"] not in _PREPARED:
        shots = load_shots(job["data"], job["final"])
        problem = joint_problem(shots.basis, shots.p)
        if shots.detectors.shape[1] != problem.H.shape[0]:
            raise ValueError(f"{job['data']}: {shots.detectors.shape[1]} detector bits per shot, the joint problem has {problem.H.shape[0]}")
        _PREPARED[job["data"]] = (shots, problem, np.log((1 - problem.priors) / problem.priors), problem_sha256(problem))
    return _PREPARED[job["data"]]


def run_job(job):
    """Decode one listed shot, save its part file, and return a line for the log."""
    shots, problem, llr, problem_hash = prepared(job)
    shot = job["shot"]
    row = decode_shot(problem, llr, shots.detectors[shot], job["changes"])
    truth = shots.observables[shot]
    row |= {"shot": shot, "truth": truth, "failed_J": bool((row["predicted_J"] != truth).any()),
            "fingerprint": job["fingerprint"], "problem_sha256": problem_hash, "processes": job["processes"]}
    part = Path(job["part"])
    scratch = part.with_name(part.name + ".unfinished")
    with open(scratch, "wb") as file:
        np.savez(file, **row)
    os.replace(scratch, part)       # a part file exists only when it is complete
    return (f"{Path(job['data']).name} shot {shot}: {'failed' if row['failed_J'] else 'right'}, "
            f"{'converged' if row['converged_J'] else 'NOT converged'}, {row['iterations_J']} iterations, {row['seconds_J']:.0f} s")


def listed_shots(selection, shots, path):
    """The shot indices of a selection file of hierarchy.ensemble, after checking it was made from this data file."""
    with np.load(selection, allow_pickle=False) as file:
        if str(file["data_sha256"]) != shots.sha256:
            raise SystemExit(f"{selection} was made from a data file with sha256 {file['data_sha256']}; {path} has {shots.sha256}")
        return [int(shot) for shot in file["shot"]]


def plan(path, wanted, args):
    """What one shot file needs: its settings, its fingerprint and the jobs still to run."""
    shots = load_shots(path, args.final)                      # refuses a held-out file before opening it
    wanted = listed_shots(wanted, shots, path) if isinstance(wanted, (str, Path)) else [int(shot) for shot in wanted]
    wanted = sorted(set(wanted))
    if wanted and not 0 <= wanted[0] <= wanted[-1] < len(shots.detectors):
        raise SystemExit(f"{path} has shots 0 to {len(shots.detectors) - 1}; asked for {wanted[0]} to {wanted[-1]}")
    changes = {"num_sets": args.relay_sets} if args.relay_sets is not None else {}
    settings = {"decoder": "joint relay-BP: the kit's relay_decoder on dem_matrices(circuit) with every detector, "
                           "a fresh decoder per shot, called through decode_detailed",
                "settings": json.loads(json.dumps(RELAY_SETTINGS | changes)), "changes_to_the_kit": changes,
                "relay_bp": metadata.version("relay-bp")}
    fingerprint = hashlib.sha256(json.dumps([shots.sha256, settings], sort_keys=True).encode()).hexdigest()
    out = Path(args.out)
    parts = {shot: out / "parts" / f"{path.stem}.joint.s{shot:06d}.npz" for shot in wanted}
    for part in parts.values():
        if part.is_file() and str(read(part)["fingerprint"]) != fingerprint:
            raise SystemExit(f"{part} belongs to another run (other data or settings). Use another --out folder.")
    jobs = [{"data": str(path), "final": args.final, "shot": shot, "changes": changes, "fingerprint": fingerprint,
             "part": str(part), "processes": args.processes} for shot, part in parts.items() if not part.is_file()]
    return {"path": path, "shots": shots, "settings": settings, "fingerprint": fingerprint, "parts": parts, "jobs": jobs,
            "final_file": out / f"{path.stem}.joint.npz"}


def merge(item):
    """Join the part files of one shot file, in rising shot order, into its output file. Returns the path."""
    parts = [read(part) for _, part in sorted(item["parts"].items())]
    shots = item["shots"]
    hashes = {str(part["problem_sha256"]) for part in parts}
    if len(hashes) > 1:
        raise SystemExit(f"{item['path']}: the parts were decoded on different problems")
    merged = {key: np.array([part[key] for part in parts]) for key in
              ("shot", "truth", "predicted_J", "failed_J", "converged_J", "iterations_J", "n_faults_J", "weight_J",
               "seconds_J", "build_seconds_J", "processes")}
    if not parts:
        merged["truth"], merged["predicted_J"] = np.zeros((0, 12), np.uint8), np.zeros((0, 12), np.uint8)
    merged |= {"data_sha256": shots.sha256, "data_name": item["path"].name, "shots_in_file": len(shots.detectors),
               "p": shots.p, "basis": shots.basis, "dataset_seed": shots.seed,
               "settings_J": json.dumps(item["settings"], sort_keys=True), "fingerprint": item["fingerprint"],
               "problem_sha256": hashes.pop() if hashes else "", "machine": json.dumps(machine())}
    np.savez_compressed(item["final_file"], **merged)
    return item["final_file"]


def print_file(path):
    with np.load(path, allow_pickle=False) as file:
        listed, failures = len(file["shot"]), int(file["failed_J"].sum())
        print(f"{file['data_name']}: {file['basis']} memory, p = {float(file['p']):g}, {listed} listed shots of "
              f"{int(file['shots_in_file'])}, data sha256 {file['data_sha256']}")
        if not listed:
            return
        low, high = wilson_interval(failures, listed)
        seconds = file["seconds_J"]
        print(f"  joint relay-BP on the listed shots: failures {failures} of {listed}  {failures / listed:.1%} "
              f"[{low:.1%}, {high:.1%}] (a hard subset: not a rate for the whole file); unconverged {int((~file['converged_J']).sum())}; "
              f"iterations mean {file['iterations_J'].mean():.0f}")
        print(f"  time per listed shot, the decode call alone on a fresh decoder: mean {seconds.mean():.1f} s, median "
              f"{np.median(seconds):.1f} s, largest {seconds.max():.1f} s; building the decoder: mean "
              f"{file['build_seconds_J'].mean():.2f} s; {'/'.join(str(n) for n in np.unique(file['processes']))} process(es) "
              f"side by side on {json.loads(str(file['machine']))['processor']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="+", required=True, help="shot files of make_dataset.py")
    parser.add_argument("--selection", nargs="+", help="per data file: the selection file of hierarchy.ensemble --write-selection")
    parser.add_argument("--shots", type=int, nargs="+", help="in place of --selection, for one data file: the shots to decode")
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument("--processes", type=int, default=4, help="processes side by side (default 4)")
    parser.add_argument("--final", action="store_true", help="allow a held-out test_* or go_* file")
    parser.add_argument("--relay-sets", type=int, help="for quick tests only: relay-BP's number of legs after the first (the kit's: 300)")
    args = parser.parse_args(argv)

    files = [Path(path) for path in args.data]
    if (args.selection is None) == (args.shots is None):
        raise SystemExit("give --selection (one file per data file) or --shots (for one data file)")
    lists = args.selection if args.selection else [args.shots]
    if len(lists) != len(files):
        raise SystemExit("give one selection file per data file, in the same order (or one data file with --shots)")
    if len({path.stem for path in files}) != len(files):
        raise SystemExit("two shot files have the same name: decode them in separate runs with separate --out folders")

    start = time.perf_counter()
    Path(args.out, "parts").mkdir(parents=True, exist_ok=True)
    items = [plan(path, wanted, args) for path, wanted in zip(files, lists, strict=True)]
    jobs = [job for item in items for job in item["jobs"]]
    for job in jobs:
        job["processes"] = max(1, min(args.processes, len(jobs)))    # how many really run side by side
    print(f"{len(jobs)} shot(s) to decode on {max(1, min(args.processes, len(jobs)))} process(es); "
          f"{sum(part.is_file() for item in items for part in item['parts'].values())} kept from an earlier run", flush=True)
    if args.processes > 1 and len(jobs) > 1:
        with multiprocessing.get_context("spawn").Pool(min(args.processes, len(jobs))) as pool:
            for line in pool.imap_unordered(run_job, jobs):
                print(line, flush=True)
    else:
        for job in jobs:
            print(run_job(job), flush=True)
    for item in items:
        print(f"\nsaved {merge(item)}")
        print_file(item["final_file"])
    print(f"\nwall time of this run: {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
