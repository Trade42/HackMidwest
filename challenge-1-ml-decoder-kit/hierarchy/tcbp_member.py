"""TCBP-JOBS as a member of the ensemble: its answer to every shot of saved shot files.

TCBP-JOBS is NVIDIA's GPU decoder (the nv-qldpc-decoder of cudaq-qec), which the teammates run
as a reference in ../tcbp_jobs_baseline.py. This tool uses their two calls (tcbp_jobs and hard,
imported from that file) on saved shots, so that its answers can be paired, shot for shot, with
the other decoders' answers on the same files.

Run from challenge-1-ml-decoder-kit/, in the teammates' TCBP environment (Linux, an NVIDIA
GPU, `pip install cudaq-qec`; neither torch nor PriorTilt is needed):

  python -m hierarchy.tcbp_member --data data/p0.004 --out runs/tcbp_p0.004 --batch
  python -m hierarchy.tcbp_member --data data/p0.004/validation_X.npz --out runs/tcbp_p0.004
  python -m hierarchy.tcbp_member --data <file> --out <folder> --decoder relay     # no GPU: the plumbing only

--data takes shot files of make_dataset.py, or folders (every .npz inside that is not held
out). A file named test_* or go_* is refused unless --final is given. The problem of a file is
the kit's: H, L and priors of the file's basis at the file's p, as in hierarchy.candidates.

Two modes:

  serial   (default) every shot is decoded alone and timed alone. The first decode of a basis
           runs once more before the loop, untimed, as in the kit's baseline: GPU start-up
           stays out of the latency.
  --batch  decode_batch in batches of 4,096 shots: fast, and no time per shot (seconds is NaN).

Per shot file the output folder receives <name>.tcbp.npz, one row per shot in file order:

  truth          the file's 12 observable bits, for scoring only
  predicted_T    the 12 predicted observable bits (L e mod 2 of the returned error vector e)
  failed_T       any predicted bit differs from the truth
  converged_T    the error vector reproduces the shot's syndrome (H e = s mod 2), checked here
  n_faults_T, weight_T   the faults of e, and the sum of log((1 - prior) / prior) over them
  seconds_T      the decode call alone; NaN in batch mode
  decoder, preset, overrides, mode, fit_seconds, build_seconds, version, data_sha256, p, basis,
  problem_sha256, machine, repeat_shots, repeat_differ

repeat_differ answers "does the decoder give the same answer twice": the first shots of the
file are decoded a second time, and the shots whose 12 bits changed are counted.

--decoder relay puts the kit's relay-BP behind the same two calls. Its output is named
<name>.relay-plumbing.npz, so it cannot be taken for TCBP-JOBS. Everything but the decoder
itself is then tested without a GPU; --opt then changes relay-BP's settings.
"""
import argparse
import ast
import hashlib
import json
import subprocess
import time
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hierarchy.candidates import expand
from hierarchy.rules import wilson_interval
from hierarchy.stages import kit_problem, load_shots, machine, problem_sha256, syndromes

BATCH = 4096          # shots per decode_batch call, as in tcbp_jobs_baseline.py
REPEAT_SHOTS = 256    # shots decoded a second time, to see whether the answers repeat
SUFFIX = {"tcbp-jobs": "tcbp", "relay": "relay-plumbing"}


class RelayStandIn:
    """The kit's relay-BP behind the two calls this tool makes of a TCBP-JOBS decoder."""

    def __init__(self, problem, overrides):
        from decoders import relay_decoder

        self.decoder = relay_decoder(problem.H, problem.priors, **overrides)

    def decode(self, syndrome):
        return SimpleNamespace(result=self.decoder.decode(syndrome))

    def decode_batch(self, batch):
        return SimpleNamespace(result=np.stack([self.decoder.decode(syndrome) for syndrome in batch]))


def check_environment(decoder):
    """Fail now, with a plain message, if the decoder cannot be imported. Returns the decoder's version."""
    try:
        import tcbp_jobs_baseline   # noqa: F401  the teammates' file, in challenge-1-ml-decoder-kit/
    except ImportError as error:
        raise SystemExit(f"cannot import tcbp_jobs_baseline.py: run from challenge-1-ml-decoder-kit/ ({error})") from error
    if decoder == "relay":
        try:
            return f"relay-bp {metadata.version('relay-bp')} (plumbing stand-in, not TCBP-JOBS)"
        except metadata.PackageNotFoundError as error:
            raise SystemExit("--decoder relay needs the kit's relay-bp package, which is not installed here") from error
    try:
        import cudaq_qec
        from cudaq_qec.plugins import nv_qldpc   # noqa: F401
    except ImportError as error:
        raise SystemExit(f"cudaq-qec with its nv_qldpc plugin is not importable in this Python: {error}. "
                         "Use the teammates' TCBP environment, or --decoder relay for a plumbing run.") from error
    for package in ("cudaq-qec", "cudaq_qec"):
        try:
            return f"cudaq-qec {metadata.version(package)}"
        except metadata.PackageNotFoundError:
            continue
    return f"cudaq-qec {getattr(cudaq_qec, '__version__', 'of unknown version')}"


def build_decoder(problem, args, overrides):
    """(decoder, seconds to fit the TCBP model, seconds to build the decoder)."""
    if args.decoder == "relay":
        start = time.perf_counter()
        return RelayStandIn(problem, overrides), 0.0, time.perf_counter() - start
    from tcbp_jobs_baseline import tcbp_jobs     # the teammates' calls, used as they are

    return tcbp_jobs(problem.H, problem.priors, args.preset, overrides)


def error_vectors(result, shape, call):
    """The 0/1 error vectors of a decoder result as an array of the given shape; anything else stops the run."""
    from tcbp_jobs_baseline import hard

    if not hasattr(result, "result") and isinstance(result, (list, tuple)):
        result = SimpleNamespace(result=[item.result for item in result])    # a list of per-shot results
    if not hasattr(result, "result"):
        raise SystemExit(f"{call} returned a {type(result).__name__} without a `result`: the decoder's interface is not "
                         f"the one tcbp_jobs_baseline.py uses. Attributes: {[name for name in dir(result) if not name.startswith('_')]}")
    errors = hard(result.result)
    if errors.shape != shape:
        raise SystemExit(f"{call} returned error vectors of shape {errors.shape}; the problem needs {shape} "
                         "(shots x error mechanisms of H)")
    return errors


def answers(errors, syndrome_rows, problem, llr):
    """Per shot, from error vectors [shots, mechanisms]: predicted bits, syndrome reproduced, faults, weight."""
    predicted = np.asarray((problem.L @ errors.T) % 2, np.uint8).T          # uint8 sums wrap at 256, which keeps the parity
    reproduced = (np.asarray((problem.H @ errors.T) % 2, np.uint8).T == syndrome_rows).all(axis=1)
    return {"predicted": predicted, "converged": reproduced, "n_faults": errors.sum(axis=1, dtype=np.int32),
            "weight": errors.astype(np.float64) @ llr}


def decode_batches(decoder, syndrome_rows, problem, llr):
    """Every shot through decode_batch. No time per shot."""
    parts = []
    for first in range(0, len(syndrome_rows), BATCH):
        batch = np.ascontiguousarray(syndrome_rows[first:first + BATCH], dtype=np.uint8)
        errors = error_vectors(decoder.decode_batch(batch), (len(batch), problem.H.shape[1]), "decode_batch")
        parts.append(answers(errors, batch, problem, llr))
        print(f"    shots {first} to {first + len(batch) - 1} decoded", flush=True)
    rows = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    return rows | {"seconds": np.full(len(syndrome_rows), np.nan)}


def decode_serially(decoder, syndrome_rows, problem, llr, warm_up):
    """Every shot alone, timed alone. The timer wraps the decode call and the cut to 0/1, as in the teammates' script."""
    from tcbp_jobs_baseline import hard

    shape = (1, problem.H.shape[1])
    if warm_up:
        error_vectors(decoder.decode(syndrome_rows[0]), (problem.H.shape[1],), "decode")    # untimed: start-up, and a check of the result
    seconds, errors = np.zeros(len(syndrome_rows)), np.zeros((len(syndrome_rows), problem.H.shape[1]), np.uint8)
    for shot, syndrome in enumerate(syndrome_rows):
        start = time.perf_counter()
        error = hard(decoder.decode(syndrome).result)
        seconds[shot] = time.perf_counter() - start
        if error.shape != shape[1:]:
            raise SystemExit(f"decode returned an error vector of shape {error.shape} on shot {shot}; the problem needs {shape[1:]}")
        errors[shot] = error
        if (shot + 1) % 1000 == 0:
            print(f"    {shot + 1} shots decoded, mean {seconds[:shot + 1].mean() * 1e3:.1f} ms", flush=True)
    return answers(errors, syndrome_rows, problem, llr) | {"seconds": seconds}


def gpu_name():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True,
                              check=True, timeout=20).stdout.strip().replace("\n", "; ") or "none reported"
    except (OSError, subprocess.SubprocessError):
        return "unknown (nvidia-smi did not answer)"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="+", required=True, help="shot files of make_dataset.py, or folders of them")
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument("--batch", action="store_true", help="decode_batch: fast, predictions only, no time per shot")
    parser.add_argument("--decoder", choices=sorted(SUFFIX), default="tcbp-jobs", help="relay: a plumbing run without a GPU")
    parser.add_argument("--preset", default="tcbp-jobs", help="nv-qldpc preset, as in tcbp_jobs_baseline.py (default tcbp-jobs)")
    parser.add_argument("--opt", action="append", default=[], metavar="NAME=VALUE", help="override one decoder option")
    parser.add_argument("--final", action="store_true", help="allow a held-out test_* or go_* file")
    args = parser.parse_args(argv)

    # Everything that can fail without the decoder fails here, before the two-minute model fit.
    try:
        overrides = {name: ast.literal_eval(value) for name, value in (item.split("=", 1) for item in args.opt)}
    except (ValueError, SyntaxError) as error:
        raise SystemExit(f"--opt takes NAME=VALUE with a Python literal as the value: {error}") from error
    version = check_environment(args.decoder)
    files = expand(args.data)
    if not files:
        raise SystemExit("no shot file to decode")
    shot_files = [load_shots(path, args.final) for path in files]           # refuses a held-out file before opening it
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mode = "batch" if args.batch else "serial"
    where = machine() | {"gpu": gpu_name() if args.decoder == "tcbp-jobs" else "not used"}
    print(f"{args.decoder} ({version}), preset {args.preset}, overrides {overrides}, mode {mode}; {len(files)} shot file(s)", flush=True)

    built = {}     # one decoder per (basis, p): the model fit takes about two minutes
    for path, shots in zip(files, shot_files, strict=True):
        target = out / f"{path.stem}.{SUFFIX[args.decoder]}.npz"
        fingerprint = hashlib.sha256(json.dumps([shots.sha256, args.decoder, args.preset, overrides, mode], sort_keys=True).encode()).hexdigest()
        print(f"{path.name}: {shots.basis} memory, p = {shots.p:g}, {len(shots.detectors)} shots, data sha256 {shots.sha256}", flush=True)
        if target.is_file():
            with np.load(target, allow_pickle=False) as file:
                if str(file["fingerprint"]) != fingerprint:
                    raise SystemExit(f"{target} exists and was made with other data or settings. Use another --out folder.")
            print(f"  kept {target} from an earlier run", flush=True)
            continue
        key = (shots.basis, shots.p)
        if key not in built:
            problem = kit_problem(shots.basis, shots.p)
            print(f"  H is {problem.H.shape[0]} x {problem.H.shape[1]}; building the decoder ...", flush=True)
            decoder, fit, build = build_decoder(problem, args, overrides)
            print(f"  model fitted in {fit:.1f} s, decoder built in {build * 1e3:.0f} ms", flush=True)
            built[key] = {"problem": problem, "decoder": decoder, "fit": fit, "build": build, "warm": False,
                          "llr": np.log((1 - problem.priors) / problem.priors), "problem_sha256": problem_sha256(problem)}
        made = built[key]
        problem, decoder, llr = made["problem"], made["decoder"], made["llr"]
        syndrome_rows = syndromes(shots, problem)
        start = time.perf_counter()
        if args.batch:
            rows = decode_batches(decoder, syndrome_rows, problem, llr)
        else:
            rows = decode_serially(decoder, syndrome_rows, problem, llr, warm_up=not made["warm"])
        made["warm"] = True
        wall = time.perf_counter() - start

        again = min(REPEAT_SHOTS, len(syndrome_rows))     # the same shots once more: does the answer repeat?
        if args.batch:
            repeat = error_vectors(decoder.decode_batch(np.ascontiguousarray(syndrome_rows[:again], dtype=np.uint8)),
                                   (again, problem.H.shape[1]), "decode_batch")
        else:
            repeat = np.stack([error_vectors(decoder.decode(syndrome), (problem.H.shape[1],), "decode") for syndrome in syndrome_rows[:again]])
        repeat_differ = int((answers(repeat, syndrome_rows[:again], problem, llr)["predicted"] != rows["predicted"][:again]).any(axis=1).sum())

        failed = (rows["predicted"] != shots.observables).any(axis=1)
        failures = int(failed.sum())
        low, high = wilson_interval(failures, len(failed))
        print(f"  failures {failures} of {len(failed)}  {failures / len(failed):.3%}  95% Wilson interval [{low:.3%}, {high:.3%}]; "
              f"the correction does not reproduce the syndrome on {int((~rows['converged']).sum())} shots")
        if args.batch:
            print(f"  decode_batch: no time per shot; {wall:.1f} s for the file")
        else:
            ms = rows["seconds"] * 1e3
            print(f"  serial decode, the first decode of the basis untimed: mean {ms.mean():.1f} ms, median {np.median(ms):.1f} ms, "
                  f"p99 {np.percentile(ms, 99):.0f} ms per shot, on {where['processor']}, GPU {where['gpu']}")
        print(f"  decoded a second time, {repeat_differ} of the first {again} shots change their 12 bits")
        np.savez_compressed(
            target, truth=shots.observables, failed_T=failed, **{f"{key_}_T": value for key_, value in rows.items()},
            data_sha256=shots.sha256, data_name=path.name, p=shots.p, basis=shots.basis, dataset_seed=shots.seed,
            decoder=args.decoder, preset=args.preset, overrides=json.dumps(overrides, sort_keys=True), mode=mode,
            batch_size=BATCH if args.batch else 1, fit_seconds=made["fit"], build_seconds=made["build"], version=version,
            problem_sha256=made["problem_sha256"], machine=json.dumps(where), repeat_shots=again, repeat_differ=repeat_differ,
            fingerprint=fingerprint)
        print(f"  saved {target}", flush=True)


if __name__ == "__main__":
    main()
