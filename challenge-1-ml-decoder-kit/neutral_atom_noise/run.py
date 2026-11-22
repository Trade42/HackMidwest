"""Decode one shot file with one decoder and one picture of the noise.

  python -m neutral_atom_noise.run --shots SHOTS.npz --decoder relay --priors matched --workers 4 --out OUTDIR
  python -m neutral_atom_noise.run --shots SHOTS.npz --decoder bposd --priors generic-2q --stop 500 --out OUTDIR

Run it from challenge-1-ml-decoder-kit/. Every row decodes the same problem (the kit's H and L of the shots'
memory basis, `priors.kit_problem`); only the priors differ (`priors.parse_priors`):

  matched       the error model of the circuit that made the shots
  generic-2q    the kit's uniform model at the p with the same total two-qubit error
  uniform:P     the kit's uniform model at p = P

For shots with atom loss (make_shots --loss), --flags adds `loss/`'s flag-aware bumps on top of those priors,
shot by shot: `--flags starter` is the published recipe, `--flags blind` ignores the heralds. The decoder gets
the detector bits and the heralds, never where or when an atom was really lost: the file does not hold it.

Decoders are the kit's, at the kit's settings (`decoders.py`). Relay-BP gets a fresh decoder with seed 1 for
every shot, so a result depends only on the shot and the priors, not on --workers or on how a run is split.
A shot fails when any of the 12 predicted observables is wrong; an unconverged shot is scored on the decoder's
output like any other. The true observables are used for scoring only.

The result is OUTDIR/<shots>__<decoder>__<priors>[__<flags>]__<start>-<stop>.npz with one entry per shot
(failed, converged, iterations, wrong_mask, ms) and its meta. --start and --stop split a long run into parts.
"""
import argparse
import json
import multiprocessing
import platform
import sys
import time
from pathlib import Path

import numpy as np

from circuit import GROSS_CIRCUITS, detector_ids, load_circuit
from decoders import bposd_decoder, relay_decoder
from neutral_atom_noise.make_shots import load_shots, sha256
from neutral_atom_noise.priors import kit_problem, parse_priors

_STATE = {}


def _setup(shots_path, decoder, priors_spec, flags):
    """Build, once per worker process, everything a shot needs."""
    shots = load_shots(shots_path)
    basis, noise = shots["meta"]["basis"], shots["noise"]
    H, L, _ = kit_problem(basis)
    priors, uniform_p = parse_priors(priors_spec, basis, noise)
    rows = detector_ids(load_circuit(GROSS_CIRCUITS[basis]), basis)
    _STATE.update(decoder=decoder, H=H, L=L, priors=priors, uniform_p=uniform_p, flags=None,
                  detectors=shots["detectors"][:, rows], observables=shots["observables"])
    if flags:
        _STATE.update(_flag_setup(shots, basis, priors, flags))
    if decoder == "bposd" and not flags:
        _STATE["bposd"] = bposd_decoder(H, priors)


def _flag_setup(shots, basis, priors, flags):
    """loss/'s map from heralds to columns, with OUR base priors in place of its lossless kit priors.

    LossMap reads the kit's circuit, so its columns are the kit's: the same H and L as every other row.
    """
    from loss.lossmap import LossMap, PriorBuilder, parse_condition
    if "heralds" not in shots:
        raise SystemExit("--flags needs shots made with make_shots --loss")
    loss_map = LossMap(basis, 0.001)
    if (loss_map.H != _STATE["H"]).nnz or (loss_map.L != _STATE["L"]).nnz:
        raise SystemExit("loss/'s matrices are not the kit's")
    loss_map.pri = priors
    return {"flags": parse_condition(flags), "builder": PriorBuilder(loss_map, shots["noise"].loss),
            "heralds": shots["heralds"]}


def _decode(k):
    """Decode shot k. Returns (k, failed, converged, iterations, wrong observables as a bit mask, milliseconds)."""
    s = _STATE
    priors = s["priors"] if s["flags"] is None else s["builder"].priors(s["flags"], s["heralds"][k])
    start = time.perf_counter()
    if s["decoder"] == "relay":
        settings = s["flags"]["relay"] if s["flags"] else {}     # loss/'s 'stop1' conditions change one setting
        result = relay_decoder(s["H"], priors, seed=1, **settings).decode_detailed(s["detectors"][k])
        error, converged, iterations = np.asarray(result.decoding, np.uint8), bool(result.success), int(result.iterations)
    else:
        decoder = s["bposd"] if s["flags"] is None else bposd_decoder(s["H"], priors)
        error = np.asarray(decoder.decode(s["detectors"][k]), np.uint8)
        converged, iterations = bool(decoder.converge), int(decoder.iter)
    ms = 1e3 * (time.perf_counter() - start)
    wrong = ((s["L"] @ error) % 2).astype(np.uint8) != s["observables"][k]
    return k, bool(wrong.any()), converged, iterations, int(np.packbits(wrong, bitorder="little").view(np.uint16)[0]), ms


def open_pool(shots_path, decoder, priors_spec, flags, workers):
    """Worker processes that hold one row's decoding problem. One pool serves every part of a row."""
    return multiprocessing.get_context("fork").Pool(workers, initializer=_setup,
                                                    initargs=(shots_path, decoder, priors_spec, flags))


def decode_shots(shots_path, decoder, priors_spec, start, stop, workers, flags=None, progress=None, pool=None):
    """Decode shots start..stop-1 of a file. Returns a dict of per-shot arrays, in shot order.

    `pool` must come from open_pool with the same file, decoder, priors and flags; without it a pool is made.
    """
    own = pool is None
    pool = open_pool(shots_path, decoder, priors_spec, flags, workers) if own else pool
    rows, began = {}, time.perf_counter()
    try:
        for n, row in enumerate(pool.imap_unordered(_decode, range(start, stop), chunksize=1), 1):
            rows[row[0]] = row[1:]
            if progress and (n % progress == 0 or n == stop - start):
                failures = sum(r[0] for r in rows.values())
                print(f"  {n}/{stop - start} shots, {time.perf_counter() - began:.0f} s, {failures} failures", flush=True)
    finally:
        if own:
            pool.close()
            pool.join()
    table = np.array([rows[k] for k in range(start, stop)], dtype=np.float64)
    return {"shots": np.arange(start, stop), "failed": table[:, 0].astype(bool), "converged": table[:, 1].astype(bool),
            "iterations": table[:, 2].astype(np.int64), "wrong_mask": table[:, 3].astype(np.uint16), "ms": table[:, 4]}


def result_name(shots_path, decoder, priors_spec, flags, start, stop):
    parts = [Path(shots_path).stem, decoder, priors_spec.replace(":", "")] + ([flags.replace("/", "_")] if flags else [])
    return "__".join(parts) + f"__{start:06d}-{stop:06d}.npz"


def run_part(shots_path, decoder, priors_spec, flags, start, stop, workers, out, command=None, pool=None):
    """Decode one part of a shot file and save it in `out`. A part that is already there is kept.

    Returns the file's path and a one-line summary (None for a kept part).
    """
    out = Path(out)
    path = out / result_name(shots_path, decoder, priors_spec, flags, start, stop)
    if path.exists():
        print(f"keep {path.name}", flush=True)
        return path, None
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} UTC {path.name}: {decoder}, priors {priors_spec}, flags {flags}, "
          f"{workers} workers", flush=True)
    began = time.perf_counter()
    result = decode_shots(shots_path, decoder, priors_spec, start, stop, workers, flags, progress=250, pool=pool)
    seconds = time.perf_counter() - began
    run = {"shots_file": Path(shots_path).name, "shots_sha256": sha256(shots_path), "shots_meta": load_shots(shots_path)["meta"],
           "decoder": decoder, "priors": priors_spec, "flags": flags, "start": start, "stop": stop, "workers": workers,
           "machine": platform.node(), "seconds": round(seconds, 1),
           "command": command or "python -m neutral_atom_noise.run " + " ".join(sys.argv[1:])}
    out.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npz")
    np.savez_compressed(partial, meta=np.array(json.dumps(run)), **result)
    partial.replace(path)
    summary = {"result": path.name, "failures": int(result["failed"].sum()), "shots": stop - start,
               "unconverged": int((~result["converged"]).sum()), "mean_iterations": round(float(result["iterations"].mean()), 1),
               "mean_ms": round(float(result["ms"].mean()), 1), "seconds": round(seconds, 1)}
    print(json.dumps(summary), flush=True)
    return path, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shots", required=True, help="a file written by make_shots")
    parser.add_argument("--decoder", choices=["relay", "bposd"], required=True)
    parser.add_argument("--priors", required=True, help="matched, generic-2q or uniform:P")
    parser.add_argument("--flags", default=None, help="for loss shots: a condition of loss/lossmap.py, e.g. starter or blind")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4, help="worker processes")
    parser.add_argument("--out", required=True, help="output folder")
    args = parser.parse_args()
    stop = load_shots(args.shots)["meta"]["shots"] if args.stop is None else args.stop
    run_part(args.shots, args.decoder, args.priors, args.flags, args.start, stop, args.workers, args.out)


if __name__ == "__main__":
    main()
