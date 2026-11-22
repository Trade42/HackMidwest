"""The candidates of a shot file: every shot decoded by relay-BP and by PriorTilt under two seeds.

Run from challenge-1-ml-decoder-kit/:

  python -m hierarchy.candidates --data data/p0.004/validation_X.npz --out runs/candidates --processes 4

--data takes shot files of make_dataset.py, or folders (every .npz inside that is not held
out). The problem of a file is the kit's: H, L and priors of the file's basis at the file's p.
The output folder receives <name>.chunks.npz per file, with one row per shot in file order:

  truth                          the file's 12 observable bits, for scoring only
  for each of R, P1, P2:
    predicted_<d>                the 12 predicted observable bits
    converged_<d>, iterations_<d>
    n_faults_<d>, weight_<d>     the faults of the correction, and the sum of log((1 - prior) / prior) over them
    seconds_<d>                  the decode call alone
    failed_<d>                   any predicted bit differs from the truth
    settings_<d>                 the decoder's settings, as JSON
  ran_R                          relay-BP decoded this shot (always, in chunk mode)
  data_sha256, p, basis, mode, seed_scheme, seeds, chunk_size, problem_sha256, machine,
  processes (per chunk: how many ran side by side)

Two modes:

  chunks   (default) The file is cut into chunks of 500 shots in file order. Every chunk is
           decoded on fresh decoders: relay-BP, PriorTilt with seed 1 + chunk index and PriorTilt
           with seed 1001 + chunk index. The chunks are spread over the processes, and every
           chunk is saved as soon as it is done, so a stopped run continues where it stopped.
  session  The file is decoded the way the plugin decodes it: decoders built once, the shots
           in order, PriorTilt under seeds 1 and 1001 on every shot, and relay-BP only on the
           shots where the two predictions differ. Output: <name>.session.npz.

The answers do not depend on the number of processes. Only seconds_<d>, and the records
machine and processes, change from run to run.

A file whose name starts with test_ or go_ is refused unless --final is given.
"""
import argparse
import hashlib
import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np

from hierarchy import select
from hierarchy.rules import ORDER, wilson_interval
from hierarchy.stages import (DEFAULT_SEEDS, HELD_OUT, Hierarchy, Stage, decoder_settings, kit_problem, load_shots,
                              machine, problem_sha256, syndromes)

CHUNK_SIZE = 500
PER_SHOT = {"predicted": np.uint8, "converged": bool, "iterations": np.int64, "n_faults": np.int32,
            "weight": np.float64, "seconds": np.float64}


# ---------- decoding ----------

def empty_rows(shots):
    """The per-shot arrays of `shots` shots, filled with "this decoder did not run"."""
    rows = {}
    for name in ORDER:
        for field, dtype in PER_SHOT.items():
            rows[f"{field}_{name}"] = np.zeros((shots, 12) if field == "predicted" else shots, dtype)
        rows[f"weight_{name}"][:] = np.nan
    rows["ran_R"] = np.zeros(shots, bool)
    return rows


def store(rows, shot, candidate):
    for field in PER_SHOT:
        rows[f"{field}_{candidate.name}"][shot] = getattr(candidate, field)
    if candidate.name == "R":
        rows["ran_R"][shot] = True


def decode_chunk(syndrome_rows, problem, settings, chunk):
    """One chunk on fresh decoders: every shot by R, P1 and P2. PriorTilt's seeds are its two seeds plus the chunk index."""
    stages = [Stage(name, problem, settings, seed_offset=chunk) for name in ORDER]
    rows = empty_rows(len(syndrome_rows))
    for shot, syndrome in enumerate(syndrome_rows):
        for stage in stages:
            store(rows, shot, stage.run(syndrome))
    return rows


def decode_session(syndrome_rows, problem, settings):
    """A whole file the way the plugin decodes it: decoders built once, relay-BP only where P1 and P2 differ."""
    hierarchy = Hierarchy(problem, settings)
    rows = empty_rows(len(syndrome_rows))
    for shot, syndrome in enumerate(syndrome_rows):
        first, second = hierarchy.first_two(syndrome)
        store(rows, shot, first)
        store(rows, shot, second)
        if not np.array_equal(first.predicted, second.predicted):
            store(rows, shot, hierarchy.third(syndrome))
    return rows


# ---------- one job: one chunk, or one file in session mode ----------

_PREPARED = {}   # per process: the shots and the problem of each data file


def prepared(job):
    if job["data"] not in _PREPARED:
        shots = load_shots(job["data"], job["final"])
        problem = kit_problem(shots.basis, shots.p)
        _PREPARED[job["data"]] = (shots, problem, syndromes(shots, problem), problem_sha256(problem))
    return _PREPARED[job["data"]]


def run_job(job):
    """Decode one job, save its part file, and return a line for the log."""
    shots, problem, syndrome_rows, problem_hash = prepared(job)
    start = time.perf_counter()
    if job["mode"] == "chunks":
        first = job["chunk"] * job["chunk_size"]
        last = min(first + job["chunk_size"], len(syndrome_rows))
        rows = decode_chunk(syndrome_rows[first:last], problem, job["settings"], job["chunk"])
    else:
        first, last = 0, len(syndrome_rows)
        rows = decode_session(syndrome_rows, problem, job["settings"])
    truth = shots.observables[first:last]
    for name in ORDER:
        ran = rows["ran_R"] if name == "R" else True
        rows[f"failed_{name}"] = (rows[f"predicted_{name}"] != truth).any(axis=1) & ran
    rows |= {"shot": np.arange(first, last), "truth": truth, "fingerprint": job["fingerprint"],
             "problem_sha256": problem_hash, "processes": job["processes"]}
    part = Path(job["part"])
    scratch = part.with_name(part.name + ".unfinished")
    with open(scratch, "wb") as file:
        np.savez(file, **rows)
    os.replace(scratch, part)       # a part file exists only when it is complete
    failures = ", ".join(f"{name} {int(rows[f'failed_{name}'].sum())}" for name in ORDER)
    label = f"chunk {job['chunk']}" if job["mode"] == "chunks" else "session"
    return (f"{Path(job['data']).name} {label}: shots {first} to {last - 1}, failures {failures}, "
            f"relay-BP ran on {int(rows['ran_R'].sum())}, {time.perf_counter() - start:.0f} s")


# ---------- planning, merging, printing ----------

def expand(paths):
    """The shot files named by --data. A folder stands for the .npz files in it that are not held out."""
    files = []
    for path in map(Path, paths):
        if path.is_dir():
            inside = sorted(path.glob("*.npz"))
            skipped = [file.name for file in inside if file.name.startswith(HELD_OUT)]
            if skipped:
                print(f"{path}: left alone, held out: {', '.join(skipped)}")
            files += [file for file in inside if not file.name.startswith(HELD_OUT)]
        else:
            files.append(path)
    names = [file.stem for file in files]
    if len(set(names)) != len(names):
        raise SystemExit("two shot files have the same name: decode them in separate runs with separate --out folders")
    return files


def plan(path, args):
    """What one shot file needs: its output file, its fingerprint and the jobs still to run."""
    shots = load_shots(path, args.final)                       # refuses a held-out file before opening it
    settings = decoder_settings(shots.p, args.seeds, args.priortilt_repo, args.priortilt_max_rounds,
                                {"num_sets": args.relay_sets} if args.relay_sets is not None else None)
    chunk_size = args.chunk_size if args.mode == "chunks" else 0
    fingerprint = hashlib.sha256(json.dumps(
        {"data_sha256": shots.sha256, "mode": args.mode, "chunk_size": chunk_size, "settings": settings},
        sort_keys=True).encode()).hexdigest()
    out = Path(args.out)
    count = -(-len(shots.detectors) // chunk_size) if chunk_size else 1
    parts = [out / "parts" / f"{path.stem}.{args.mode}.{index:04d}.npz" for index in range(count)]
    wanted = range(count)
    if args.chunks and args.mode == "chunks":
        first, last = (int(value) for value in args.chunks.split(":"))
        wanted = range(max(first, 0), min(last, count))
    for part in parts:
        if part.is_file() and str(read(part)["fingerprint"]) != fingerprint:
            raise SystemExit(f"{part} belongs to another run (other data, settings or seeds). Use another --out folder.")
    jobs = [{"data": str(path), "final": args.final, "mode": args.mode, "chunk": index, "chunk_size": chunk_size,
             "settings": settings, "fingerprint": fingerprint, "part": str(parts[index]), "processes": args.processes}
            for index in wanted if not parts[index].is_file()]
    if args.mode == "chunks":
        scheme = (f"chunks of {chunk_size} shots in file order, fresh decoders per chunk, PriorTilt seeds "
                  f"{args.seeds[0]} + chunk index (P1) and {args.seeds[1]} + chunk index (P2)")
    else:
        scheme = (f"session: decoders built once, shots in file order, PriorTilt seeds {args.seeds[0]} (P1) and "
                  f"{args.seeds[1]} (P2), relay-BP only where P1 and P2 differ")
    return {"path": path, "shots": shots, "settings": settings, "fingerprint": fingerprint, "parts": parts, "jobs": jobs,
            "chunk_size": chunk_size, "seed_scheme": scheme, "final_file": out / f"{path.stem}.{args.mode}.npz"}


def read(path):
    """Every array of an .npz file, as a dict."""
    with np.load(path, allow_pickle=False) as file:
        return {key: file[key] for key in file.files}


def merge(item, args):
    """Join the part files of one shot file, in shot order, into its output file. Returns the path."""
    parts = [read(part) for part in item["parts"]]
    shot = np.concatenate([part["shot"] for part in parts])
    shots = item["shots"]
    if not np.array_equal(shot, np.arange(len(shots.detectors))):
        raise SystemExit(f"{item['path']}: the parts do not cover every shot once")
    merged = {"truth": np.concatenate([part["truth"] for part in parts])}
    for name in ORDER:
        for field in (*PER_SHOT, "failed"):
            merged[f"{field}_{name}"] = np.concatenate([part[f"{field}_{name}"] for part in parts])
        merged[f"settings_{name}"] = json.dumps(item["settings"][name], sort_keys=True)
    hashes = {str(part["problem_sha256"]) for part in parts}
    if len(hashes) != 1:
        raise SystemExit(f"{item['path']}: the parts were decoded on different problems")
    merged |= {"ran_R": np.concatenate([part["ran_R"] for part in parts]),
               "data_sha256": shots.sha256, "data_name": item["path"].name, "p": shots.p, "basis": shots.basis,
               "dataset_seed": shots.seed, "mode": args.mode, "chunk_size": item["chunk_size"],
               "seeds": np.array(args.seeds), "seed_scheme": item["seed_scheme"], "fingerprint": item["fingerprint"],
               "problem_sha256": hashes.pop(), "machine": json.dumps(machine()),
               "processes": np.array([int(part["processes"]) for part in parts])}
    np.savez_compressed(item["final_file"], **merged)
    return item["final_file"]


def print_decoders(path):
    """One line per decoder of one output file."""
    with np.load(path, allow_pickle=False) as file:
        shots = len(file["truth"])
        print(f"{file['data_name']}: {file['basis']} memory, p = {float(file['p']):g}, {shots} shots, data sha256 {file['data_sha256']}")
        print(f"  {file['seed_scheme']}")
        print(f"  problem sha256 {file['problem_sha256']} (H, L and priors from the kit's circuit at this p)")
        print(f"  {'decoder':8s} {'ran on':>7s} {'failures':>8s}  {'rate':>7s}  {'95% Wilson interval':20s} {'unconverged':>11s} "
              f"{'iterations':>10s}  {'ms/shot: mean':>13s} {'median':>8s} {'p99':>8s}")
        for name in ORDER:
            ran = file["ran_R"] if name == "R" else np.ones(shots, bool)
            if not ran.any():
                print(f"  {name:8s} {0:7d}")
                continue
            failures, ms = int(file[f"failed_{name}"][ran].sum()), file[f"seconds_{name}"][ran] * 1e3
            low, high = wilson_interval(failures, int(ran.sum()))
            print(f"  {name:8s} {int(ran.sum()):7d} {failures:8d}  {failures / ran.sum():7.3%}  [{low:.3%}, {high:.3%}]{'':3s} "
                  f"{int((~file[f'converged_{name}'][ran]).sum()):11d} {file[f'iterations_{name}'][ran].mean():10.1f}  "
                  f"{ms.mean():13.1f} {np.median(ms):8.1f} {np.percentile(ms, 99):8.0f}")
        processes = "/".join(str(n) for n in np.unique(file["processes"]))
        print(f"  times: the decode call alone, {processes} process(es) side by side on {json.loads(str(file['machine']))['processor']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="+", required=True, help="shot files of make_dataset.py, or folders of them")
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument("--processes", type=int, default=4, help="processes side by side (default 4)")
    parser.add_argument("--mode", choices=["chunks", "session"], default="chunks")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="shots per chunk in chunk mode (default 500)")
    parser.add_argument("--chunks", help="a:b decodes only the chunks a <= index < b now; the rest stays for a later run")
    parser.add_argument("--seeds", type=int, nargs=2, default=list(DEFAULT_SEEDS), help="PriorTilt's two seeds (default 1 1001)")
    parser.add_argument("--priortilt-repo", help="a clone of PriorTilt's repository (default: the environment variable PRIORTILT_REPO)")
    parser.add_argument("--final", action="store_true", help="allow a held-out test_* or go_* file")
    parser.add_argument("--priortilt-max-rounds", type=int, help="for quick tests only: PriorTilt's budget of rounds (the paper's: 3000)")
    parser.add_argument("--relay-sets", type=int, help="for quick tests only: relay-BP's number of legs after the first (the kit's: 300)")
    args = parser.parse_args(argv)

    start = time.perf_counter()
    Path(args.out, "parts").mkdir(parents=True, exist_ok=True)
    items = [plan(path, args) for path in expand(args.data)]
    jobs = [job for item in items for job in item["jobs"]]
    for job in jobs:
        job["processes"] = max(1, min(args.processes, len(jobs)))    # how many really run side by side
    print(f"{len(jobs)} job(s) to run on {max(1, min(args.processes, len(jobs)))} process(es); "
          f"{sum(part.is_file() for item in items for part in item['parts'])} part(s) kept from an earlier run", flush=True)
    if args.processes > 1 and len(jobs) > 1:
        with multiprocessing.get_context("spawn").Pool(min(args.processes, len(jobs))) as pool:
            for line in pool.imap_unordered(run_job, jobs):
                print(line, flush=True)
    else:
        for job in jobs:
            print(run_job(job), flush=True)

    for item in items:
        missing = sum(not part.is_file() for part in item["parts"])
        if missing:
            print(f"\n{item['path'].name}: {missing} of {len(item['parts'])} chunks are still to decode. Run again to finish.")
            continue
        final_file = merge(item, args)
        print(f"\nsaved {final_file}")
        print_decoders(final_file)
        print("")
        pool_, sources = select.load_candidates([final_file])
        select.report(pool_, sources, select.score(pool_))
    print(f"\nwall time of this run: {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
