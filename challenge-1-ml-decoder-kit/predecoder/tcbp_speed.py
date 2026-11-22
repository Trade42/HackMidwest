"""tcbp-speed: where the 42 ms of TCBP-JOBS's per-shot build go, and a cheaper build that gives the same answers.

The TCBP version (tcbp.py) builds one decoder per shot, because a decoder takes its priors only when it is built:

    get_decoder("nv-qldpc-decoder", H, error_rate_vec=<the shot's priors>, **preset)      about 42 ms on the A100

H is handed over as a dense uint8 array (936 x 8784). cudaq-qec's own docstring says a dense array is first copied
into a dense tensor and then converted to sparse storage, on every call, and that a scipy sparse matrix is "the
preferred input - no dense allocation occurs". This tool changes that one argument and nothing else: one model fit,
one options dict and the same priors for every arm; only the form of H differs (tcbp.Builder.with_h_format).

One command, run once on the server in the teammates' TCBP environment (cudaq-qec 0.9.0, an NVIDIA GPU):

  python -m predecoder.tcbp_speed run --data data/p0.005/validation_X.npz --logits <logits folder> \\
      --settings predecoder/logs/fit_tcbp_X_p0.005.json --limit 1000 --out <folder>

  1. Facts: the package version, the TCBP model's size in bytes, H's shape and number of ones, the GPU.
  2. Profile (--profile-builds, 50): cProfile of 50 builds of the registered path (tcbp.decode_one with a dense H).
     cProfile cannot see inside the C++ of get_decoder, so the time is then located by subtraction: 50 builds each,
     timed one by one (build, then freeing the decoder), of: dense, CSR and CSC H (and CSR with other dtypes); the
     model given as a file path; and other presets with fewer parts: bp10-minllr-jobs (JOBS, no model), tcbp (the
     model, no OSD or JOBS), bp60 (bare BP). Each variant is tried on its own: one that fails is reported and skipped.
     The other presets are timing probes only: they are different decoders, and none of them is a candidate.
  3. Check (--limit shots, the head's priors of --settings, capped at 0.5 as in tcbp.py): arms dense, csr, csc and
     dense again. Every shot is decoded by every arm, one shot at a time, in blocks of --block shots; the arms take
     turns block by block, in a rotating order, so a busy GPU slows all of them alike. Per shot and arm: the 12
     predicted bits, the number of faults of the correction, a hash of the whole correction (8784 bits), whether it
     reproduces the syndrome, the decoder's flag, and the times. One timer wraps the shot's whole cost (the priors and
     the cap, the build, the decode, and freeing the decoder); its parts are kept too. The registered runs' "build"
     and "decode" timers did not include freeing the decoder, so the old path is timed again here, under this timer.
     An arm is the same as dense when its 12 bits, its faults and its whole correction equal dense's on every shot.
     If dense and dense again differ on any shot, the decoder is not repeatable in this run and no arm can be called
     the same: the summary says so.

Not tried (the brief's other ideas):
  - a built decoder whose priors are changed in place: the Python Decoder of 0.9.0 offers decode, decode_async,
    decode_batch, get_block_size, get_syndrome_size and get_version only (tcbp.py probe, job 1). The plugin's binary
    (PyPI wheel cudaq_qec_cu13 0.9.0, read on luminaty, never run) holds the messages "set_priors: one rate per column
    is required" and "the JOBS column costs are fixed at construction", but no exported set_priors: unreachable here;
  - options that change the decoder (batch sizes, thread counts): a different decoder, so not a candidate.

--decoder relay puts the kit's relay-BP behind the same calls (no GPU): it ignores the form of H, so it tests the
plumbing only, not the claim.
"""
import argparse
import cProfile
import hashlib
import io
import json
import os
import pstats
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np

from predecoder import tcbp
from predecoder.files import read_logits

ARMS = ("dense", "csr", "csc", "dense_again")         # dense_again: the registered path a second time (repeatability)
FIELDS = ("predicted_code", "n_faults", "failed", "reproduced", "flag",
          "seconds_priors", "seconds_build", "seconds_decode", "seconds_free", "seconds_total")
SPARSE_DTYPES = (np.uint8, np.int32, np.float64, np.bool_)     # tried in this order if get_decoder refuses one


def error_hash(error):
    """The first 8 bytes of the sha256 of the packed correction, as an unsigned integer."""
    return int.from_bytes(hashlib.sha256(np.packbits(np.asarray(error, np.uint8)).tobytes()).digest()[:8], "little")


def stats(values_ms):
    values = np.asarray(values_ms, np.float64)
    return {"n": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "p99": float(np.percentile(values, 99)), "max": float(values.max())}


def head_priors(problem, logits_row, setting):
    from predecoder.priors import shot_priors

    return tcbp.capped(shot_priors(logits_row, problem.priors, setting["alpha"], setting["eps"], setting["mode"]))


# ---------- 1. facts ----------

def facts(builder, version):
    out = {"version": version, "machine": tcbp.where(builder.decoder), "h_shape": list(builder.H.shape),
           "h_ones": int(builder.H.nnz)}
    if builder.decoder != "relay":
        from cudaq_qec.plugins import nv_qldpc

        out["model_bytes"] = len(builder.model)
        out["model_option"] = f"{type(builder.options['tcbp_model']).__name__} of {builder.options['tcbp_model'].nbytes} bytes"
        out["options"] = {key: (value if key != "tcbp_model" else "<model>") for key, value in builder.options.items()}
        out["dense_bytes"] = int(builder.dense.nbytes)
        try:
            report = nv_qldpc.inspect(builder.model)
            out["model_inspect"] = {key: report[key] for key in list(report)[:40]}
        except Exception as error:          # noqa: BLE001  a note only
            out["model_inspect"] = f"failed: {error!r}"
    tools = {}
    for tool in ("nsys", "ncu", "perf", "py-spy"):
        found = subprocess.run(["bash", "-c", f"command -v {tool}"], capture_output=True, text=True).stdout.strip()
        tools[tool] = found or None
    out["profilers_on_path"] = tools
    return out


# ---------- 2. profile ----------

def profile(builder, priors, syndromes, out, builds):
    """cProfile of `builds` shots of the registered path, then timing-only builds of each variant."""
    result = {}
    profiler = cProfile.Profile()
    seconds = []
    tcbp.decode_one(builder, priors[0], syndromes[0])                        # warm, untimed
    profiler.enable()
    for k in range(builds):
        seconds.append(tcbp.decode_one(builder, priors[k % len(priors)], syndromes[k % len(priors)])[2:])
    profiler.disable()
    profiler.dump_stats(str(out / "profile_dense.pstats"))
    text = io.StringIO()
    report = pstats.Stats(profiler, stream=text)
    report.sort_stats("cumulative").print_stats(25)
    report.sort_stats("tottime").print_stats(15)
    (out / "profile_dense.txt").write_text(text.getvalue())
    print(f"cProfile of {builds} shots of the registered path (tcbp.decode_one, dense H; build {np.mean([s[0] for s in seconds]) * 1e3:.1f} ms, "
          f"decode {np.mean([s[1] for s in seconds]) * 1e3:.1f} ms mean under the profiler):", flush=True)
    print("\n".join(line for line in text.getvalue().splitlines()[:60] if line.strip()), flush=True)
    result["cprofile"] = {"shots": builds, "file": "profile_dense.txt",
                          "build_ms_mean_under_profiler": float(np.mean([s[0] for s in seconds]) * 1e3),
                          "decode_ms_mean_under_profiler": float(np.mean([s[1] for s in seconds]) * 1e3)}
    if builder.decoder == "relay":
        result["variants"] = "skipped: the relay stand-in has no get_decoder"
        return result
    result["python_parts"] = python_parts(builder, priors)
    result["variants"] = {}
    for name, matrix, options in variants(builder, out):
        try:
            result["variants"][name] = time_builds(builder, matrix, options, priors, builds)
            row = result["variants"][name]
            print(f"  {name:58s} build {row['build_ms']['mean']:6.1f} ms (median {row['build_ms']['median']:6.1f}, "
                  f"p90 {row['build_ms']['p90']:6.1f}), free {row['free_ms']['mean']:5.1f} ms", flush=True)
        except Exception as error:          # noqa: BLE001  one variant failing must not stop the others
            result["variants"][name] = {"error": repr(error), "traceback": traceback.format_exc()[-2000:]}
            print(f"  {name:58s} FAILED: {error!r}", flush=True)
    return result


def variants(builder, out):
    """(name, H, options) of the timing probes. Every get_decoder call gets error_rate_vec from the shot's priors."""
    from cudaq_qec.plugins import nv_qldpc
    from scipy import sparse

    dense, options = builder.dense, builder.options
    csr, csc = tcbp.h_matrix(dense, "csr"), tcbp.h_matrix(dense, "csc")
    model_path = out / "tcbp_model.bin"
    model_path.write_bytes(bytes(builder.model))
    by_path = dict(options, tcbp_model=str(model_path))
    rows = [("tcbp-jobs, H dense uint8 (the registered path)", dense, options),
            ("tcbp-jobs, H scipy CSR uint8", csr, options),
            ("tcbp-jobs, H scipy CSC uint8", csc, options)]
    for dtype in (np.int32, np.float64, np.bool_):
        rows.append((f"tcbp-jobs, H scipy CSR {np.dtype(dtype).name}", sparse.csr_matrix(dense, dtype=dtype), options))
    rows += [("tcbp-jobs, model as a file path, H dense", dense, by_path),
             ("tcbp-jobs, model as a file path, H CSR", csr, by_path)]
    for preset, what in (("bp10-minllr-jobs", "JOBS, no model"), ("tcbp", "the model, no OSD/JOBS"), ("bp60", "bare BP")):
        other = nv_qldpc.preset(preset, **({"tcbp_model": builder.model} if preset.startswith("tcbp") else {}))
        rows += [(f"{preset} ({what}), H dense", dense, other), (f"{preset} ({what}), H CSR", csr, other)]
    return rows


def time_builds(builder, matrix, options, priors, builds):
    """Build `builds` decoders one at a time (one untimed first), each with one shot's priors; time the build and the free."""
    build, free = [], []
    for k in range(builds + 1):
        rates = np.asarray(priors[k % len(priors)], float)
        start = time.perf_counter()
        decoder = builder.qec.get_decoder("nv-qldpc-decoder", matrix, error_rate_vec=rates, **options)
        built = time.perf_counter()
        del decoder
        done = time.perf_counter()
        if k:
            build.append((built - start) * 1e3)
            free.append((done - built) * 1e3)
    return {"builds": builds, "build_ms": stats(build), "free_ms": stats(free)}


def python_parts(builder, priors):
    """Times of the pieces that run in Python, for scale: none of them is inside get_decoder."""
    from cudaq_qec.plugins import nv_qldpc
    from scipy import sparse

    def timed(function, repeat=20):
        values = []
        for _ in range(repeat):
            start = time.perf_counter()
            function()
            values.append((time.perf_counter() - start) * 1e3)
        return float(np.median(values))

    csc = tcbp.h_matrix(builder.dense, "csc")
    return {"what": "median ms of 20 calls; for scale only",
            "np_asarray_priors": timed(lambda: np.asarray(priors[0], float)),
            "copy_of_dense_H": timed(lambda: builder.dense.copy()),
            "dense_H_to_scipy_csr": timed(lambda: sparse.csr_matrix(builder.dense)),
            "scipy_csc_to_csr": timed(lambda: csc.tocsr()),
            "nv_qldpc_preset_with_model": timed(lambda: nv_qldpc.preset(builder.preset, tcbp_model=builder.model)),
            "np_nonzero_of_dense_H": timed(lambda: np.nonzero(builder.dense), 5)}


# ---------- 3. check ----------

def arm_builders(builder, problem, syndromes, logits, setting, arms):
    """{arm: builder}: dense arms use the registered builder itself; a sparse arm shares its model and options and
    takes the first dtype of SPARSE_DTYPES that get_decoder accepts (one untimed shot each). An arm none of them
    builds is dropped, and said so."""
    builders, dtypes = {}, {}
    priors = head_priors(problem, logits[0], setting)
    for arm in arms:
        if arm.startswith("dense"):
            builders[arm] = builder
            continue
        for dtype in SPARSE_DTYPES:
            try:
                other = builder.with_h_format(arm, dtype)
                tcbp.decode_one(other, priors, syndromes[0])
            except BaseException as error:  # noqa: BLE001  SystemExit of h_matrix included
                print(f"  arm {arm} with dtype {np.dtype(dtype).name} refused: {error!r}", flush=True)
                continue
            builders[arm], dtypes[arm] = other, np.dtype(dtype).name
            break
        else:
            print(f"  arm {arm} dropped: no dtype of {[np.dtype(d).name for d in SPARSE_DTYPES]} built", flush=True)
    return builders, dtypes


def check(builders, problem, syndromes, truth, logits, setting, block, out):
    """Every shot by every arm, blocks of `block` shots, arms in a rotating order. Returns per-shot rows per arm."""
    arms = list(builders)
    shots = len(syndromes)
    rows = {arm: np.zeros((shots, len(FIELDS)), np.float64) for arm in arms}
    codes = {arm: np.zeros(shots, np.int64) for arm in arms}
    hashes = {arm: np.zeros(shots, np.uint64) for arm in arms}
    logits_hashes = np.array([tcbp.row_hash(logits[k]) for k in range(shots)], np.uint64)
    for arm in arms:                                                        # each arm's first build, untimed
        tcbp.decode_one(builders[arm], head_priors(problem, logits[0], setting), syndromes[0])
    start = time.time()
    for index, first in enumerate(range(0, shots, block)):
        last = min(first + block, shots)
        turn = index % len(arms)
        for arm in arms[turn:] + arms[:turn]:
            for k in range(first, last):
                begin = time.perf_counter()
                priors = head_priors(problem, logits[k], setting)
                ready = time.perf_counter()
                error, flag, build, decode = tcbp.decode_one(builders[arm], priors, syndromes[k])
                end = time.perf_counter()
                predicted = (problem.L @ error) % 2
                codes[arm][k] = int(np.asarray(predicted, np.int64) @ tcbp.BITS)
                hashes[arm][k] = error_hash(error)
                rows[arm][k] = (codes[arm][k], int(error.sum()), bool((predicted != truth[k]).any()),
                                np.array_equal((problem.H @ error) % 2, syndromes[k]), flag,
                                ready - begin, build, decode, (end - ready) - build - decode, end - begin)
        save_check(out, rows, codes, hashes, logits_hashes, truth, last)
        print(f"  shots {first}-{last - 1} done by {', '.join(arms[turn:] + arms[:turn])}; {time.time() - start:.0f} s; "
              + "; ".join(f"{arm} {rows[arm][:last, FIELDS.index('seconds_total')].mean() * 1e3:.1f} ms" for arm in arms),
              flush=True)
    return rows, codes, hashes, logits_hashes


def save_check(out, rows, codes, hashes, logits_hashes, truth, done):
    arrays = {"truth": truth[:done], "logits_hash": logits_hashes[:done], "fields": np.array(FIELDS)}
    for arm in rows:
        arrays[f"rows_{arm}"] = rows[arm][:done]
        arrays[f"predicted_code_{arm}"] = codes[arm][:done]
        arrays[f"error_hash_{arm}"] = hashes[arm][:done]
    np.savez_compressed(out / "check_per_shot.npz", **arrays)


def compare(rows, codes, hashes, arms, reference="dense"):
    """Per arm against the reference: shots whose 12 bits, faults or whole correction differ; failures; times."""
    faults = FIELDS.index("n_faults")
    result = {}
    for arm in arms:
        r = rows[arm]
        timing = {name: stats(r[:, FIELDS.index(name)] * 1e3) for name in FIELDS if name.startswith("seconds_")}
        row = {"failures": int(r[:, FIELDS.index("failed")].sum()),
               "not_reproduced": int((r[:, FIELDS.index("reproduced")] == 0).sum()),
               "flag_not_converged": int((r[:, FIELDS.index("flag")] == 0).sum()),
               "ms": timing}
        if arm != reference:
            row["bits_differ"] = int((codes[arm] != codes[reference]).sum())
            row["faults_differ"] = int((r[:, faults] != rows[reference][:, faults]).sum())
            row["corrections_differ"] = int((hashes[arm] != hashes[reference]).sum())
            row["same_as_dense"] = row["bits_differ"] == row["faults_differ"] == row["corrections_differ"] == 0
        result[arm] = row
    return result


# ---------- the command ----------

def run(args):
    from predecoder.priors import load_settings

    if "dense" not in args.arms or len(set(args.arms)) != len(args.arms):
        raise SystemExit("--arms must hold dense (the reference) and no arm twice")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = tcbp.load(args.data, args.final)
    problem = tcbp.problem_of(data)
    shots = min(args.limit, len(data.detectors))
    meta, logits_path = read_logits(args.logits, data)
    if meta["shots"] < shots:
        raise SystemExit(f"{args.logits} holds the logits of {meta['shots']} shots only")
    setting = load_settings(args.settings)
    logits = np.load(logits_path, mmap_mode="r")
    version = tcbp.check_environment(args.decoder)
    syndromes = np.ascontiguousarray(data.basis_input[:shots], dtype=np.uint8)
    truth = data.observables[:shots]
    summary = {"data": data.identity(), "shots": shots, "decoder": args.decoder, "preset": args.preset,
               "overrides": args.opt, "head_priors": setting, "prior_cap": tcbp.PRIOR_CAP, "block": args.block,
               "arms": list(args.arms), "logits": {"folder": str(args.logits), "model": meta.get("model")},
               "threads": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
    started = time.time()

    def save():
        summary["seconds"] = time.time() - started
        with open(out / "summary.json", "w") as file:
            json.dump(summary, file, indent=1, default=str)

    print(f"{data.path.name}: {shots} shots, p={data.p:g}, decoder {args.decoder} ({version}), preset {args.preset}, "
          f"overrides {args.opt}; head priors {setting}; arms {list(args.arms)}, blocks of {args.block}", flush=True)
    builder = tcbp.Builder(problem, args.decoder, args.preset, args.opt)
    summary["model_seconds"] = builder.model_seconds
    summary["facts"] = facts(builder, version)
    brief = {key: value for key, value in summary["facts"].items() if key not in ("model_inspect", "options")}
    print(f"  model fitted in {builder.model_seconds:.1f} s; facts: {brief}", flush=True)
    model_before = hashlib.sha256(bytes(builder.options["tcbp_model"])).hexdigest() if args.decoder != "relay" else None
    save()
    priors = [head_priors(problem, logits[k], setting) for k in range(min(args.profile_builds, shots) or 1)]
    if args.profile_builds:
        try:
            summary["profile"] = profile(builder, priors, syndromes, out, args.profile_builds)
        except Exception as error:          # noqa: BLE001  the check still runs
            summary["profile"] = {"error": repr(error), "traceback": traceback.format_exc()[-2000:]}
            print(f"profile FAILED: {error!r}", flush=True)
        save()
    builders, dtypes = arm_builders(builder, problem, syndromes, logits, setting, args.arms)
    arms = list(builders)
    summary["arms_run"], summary["sparse_dtypes"] = arms, dtypes
    tcbp.CAPPED.update(shots=0, columns=0)
    rows, codes, hashes, _ = check(builders, problem, syndromes, truth, logits, setting, args.block, out)
    summary["check"] = compare(rows, codes, hashes, arms)
    summary["capped_in_check"] = {"what": "counted over every arm's decodes", **tcbp.CAPPED}
    if model_before is not None:
        summary["model_option_unchanged"] = hashlib.sha256(bytes(builder.options["tcbp_model"])).hexdigest() == model_before
    repeatable = summary["check"].get("dense_again", {}).get("same_as_dense")
    summary["repeatable"] = repeatable
    summary["timer"] = ("seconds_total: one perf_counter timer around the shot's priors (shot_priors and the 0.5 cap), "
                        "tcbp.decode_one (get_decoder with the shot's priors, decode(), the cut to 0/1) and freeing the "
                        "decoder; one process, shots one at a time, the GPU shared with whatever else runs on it")
    save()
    print(f"\nRESULT ({shots} shots of {data.path.name}, head priors, one model; reference: dense):", flush=True)
    for arm in arms:
        row = summary["check"][arm]
        ms = row["ms"]
        same = "" if arm == "dense" else (f"; differs from dense on bits {row['bits_differ']}, faults {row['faults_differ']}, "
                                          f"corrections {row['corrections_differ']} -> {'SAME' if row['same_as_dense'] else 'NOT THE SAME'}")
        print(f"  {arm:12s} failures {row['failures']}, not reproduced {row['not_reproduced']}; total {ms['seconds_total']['mean']:.1f} ms "
              f"mean (median {ms['seconds_total']['median']:.1f}, p99 {ms['seconds_total']['p99']:.1f}) = priors "
              f"{ms['seconds_priors']['mean']:.2f} + build {ms['seconds_build']['mean']:.1f} + decode {ms['seconds_decode']['mean']:.1f} "
              f"+ free {ms['seconds_free']['mean']:.1f}{same}", flush=True)
    if repeatable is False:
        print("  dense and dense_again differ: the decoder is not repeatable in this run, so no arm can be called the same",
              flush=True)
    print(f"saved {out}; {time.time() - started:.0f} s", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser("run")
    sub.add_argument("--data", required=True)
    sub.add_argument("--logits", required=True, help="a folder of predecoder.evaluate for --data")
    sub.add_argument("--settings", required=True, help="the fit file of tcbp.py (the knobs)")
    sub.add_argument("--limit", type=int, default=1000)
    sub.add_argument("--block", type=int, default=100)
    sub.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    sub.add_argument("--profile-builds", type=int, default=50, help="0: no profile")
    sub.add_argument("--decoder", choices=("tcbp-jobs", "relay"), default="tcbp-jobs")
    sub.add_argument("--preset", default="tcbp-jobs")
    sub.add_argument("--opt", type=json.loads, default={})
    sub.add_argument("--final", action="store_true", help="allow a held-out file (test_*, go_*)")
    sub.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
