"""tcbp-speed: where the 42 ms of TCBP-JOBS's per-shot build go, and a cheaper build that gives the same answers.

The TCBP version (tcbp.py) builds one decoder per shot, because a decoder takes its priors only when it is built:

    get_decoder("nv-qldpc-decoder", H, error_rate_vec=<the shot's priors>, **preset)      about 42 ms on the A100

H is handed over as a dense uint8 array (936 x 8784). cudaq-qec's own docstring says a dense array is first copied
into a dense tensor and then converted to sparse storage, on every call, and that a scipy sparse matrix is "the
preferred input - no dense allocation occurs". This tool changes that one argument and nothing else: one model fit,
one options dict and the same priors for every arm; only the form of H differs (tcbp.Builder.with_h_format).

`run`, once on the server in the teammates' TCBP environment (cudaq-qec 0.9.0, an NVIDIA GPU); `report` on any machine:

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

  Job 2 adds two arms that move the decoder's free (about 4.6 ms with JOBS) off the answer's path, both with H CSR:
     csr_after   tcbp.decode_one(keep=True) hands the decoder back; it is freed once the answer is recorded;
     csr_thread  it is handed to a Releaser thread that frees it while the loop goes on (1 process, 2 Python threads;
                 a bounded queue; the thread checks it holds the decoder's last reference, and is joined at the end).
     Each is timed to its answer (seconds_answer) and by its blocks' wall time per shot (a thread arm's block ends
     when its thread has freed every decoder). The rule, fixed before the run: an arm is a gain over csr only if its
     wall time per shot is below csr's in the same process. Run the thread arm in a process of its own, beside dense
     and csr, so that a crash in it cannot take the other arms with it. --profile-set options times builds with
     decoder options changed (OPTION_PROBES): other decoders, timing probes only.

  python -m predecoder.tcbp_speed report --summary <folder>/summary.json --per-shot <folder>/check_per_shot.npz \\
      --reference <tcbp.py decode folder>/per_shot.npz            the tables, and the arms against another run's head arm

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

ARMS = ("dense", "csr", "csc", "dense_again")         # the default (job 1); dense_again: the registered path again
FREE_ARMS = ("csr_after", "csr_thread")              # job 2: H CSR, the decoder freed after the answer, or on a thread
ALL_ARMS = ARMS + FREE_ARMS
FIELDS = ("predicted_code", "n_faults", "failed", "reproduced", "flag",
          "seconds_priors", "seconds_build", "seconds_decode", "seconds_free", "seconds_total", "seconds_answer")
# seconds_total: from the shot's start until the loop is free for the next shot; seconds_answer: until the answer is in
# the caller's hands. For the inline arms they are the same (tcbp.decode_one frees before it returns).
SPARSE_DTYPES = (np.uint8, np.int32, np.float64, np.bool_)     # tried in this order if get_decoder refuses one


def error_hash(error):
    """The first 8 bytes of the sha256 of the packed correction, as an unsigned integer."""
    return int.from_bytes(hashlib.sha256(np.packbits(np.asarray(error, np.uint8)).tobytes()).digest()[:8], "little")


def stats(values_ms):
    """Mean, median, p90, p99 and max of the values that were measured (a NaN is a value not measured)."""
    values = np.asarray(values_ms, np.float64)
    values = values[~np.isnan(values)]
    if not len(values):
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "p90": float("nan"), "p99": float("nan"),
                "max": float("nan")}
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

# Job 2's timing probes: decoder options seen in the plugin's binary (PyPI wheel strings, so their meaning is a guess).
# Each makes a different decoder; none is a candidate without a check of its answers.
OPTION_PROBES = ({"osd_batch_size": 1}, {"bp_batch_size": 1}, {"n_threads": 1}, {"repeatable": False})


def profile(builder, priors, syndromes, out, builds, which="all"):
    """cProfile of `builds` shots of the registered path, then timing-only builds of each variant of `which`."""
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
    if which == "all":
        result["python_parts"] = python_parts(builder, priors)
    result["variants"] = {}
    for name, matrix, options in variants(builder, out, which):
        try:
            result["variants"][name] = time_builds(builder, matrix, options, priors, builds)
            row = result["variants"][name]
            print(f"  {name:58s} build {row['build_ms']['mean']:6.1f} ms (median {row['build_ms']['median']:6.1f}, "
                  f"p90 {row['build_ms']['p90']:6.1f}), free {row['free_ms']['mean']:5.1f} ms", flush=True)
        except Exception as error:          # noqa: BLE001  one variant failing must not stop the others
            result["variants"][name] = {"error": repr(error), "traceback": traceback.format_exc()[-2000:]}
            print(f"  {name:58s} FAILED: {error!r}", flush=True)
    return result


def variants(builder, out, which="all"):
    """(name, H, options) of the timing probes. Every get_decoder call gets error_rate_vec from the shot's priors."""
    from cudaq_qec.plugins import nv_qldpc
    from scipy import sparse

    dense, options = builder.dense, builder.options
    csr, csc = tcbp.h_matrix(dense, "csr"), tcbp.h_matrix(dense, "csc")
    if which == "options":
        rows = [("tcbp-jobs, H CSR (the probes' reference)", csr, options)]
        for change in OPTION_PROBES:
            name = ", ".join(f"{key}={value}" for key, value in change.items())
            rows.append((f"tcbp-jobs, H CSR, {name} (another decoder: timing only)", csr, dict(options, **change)))
        return rows
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

def h_form(arm):
    return "dense" if arm.startswith("dense") else arm.split("_")[0]


def free_mode(arm):
    """inline: tcbp.decode_one frees the decoder before it returns (the registered path); after: the caller frees it
    once the answer is recorded; thread: a Releaser thread frees it while the loop goes on."""
    return "after" if arm.endswith("_after") else "thread" if arm.endswith("_thread") else "inline"


class Releaser:
    """A thread that frees the decoders handed to it (the lists of tcbp.decode_one(keep=True)) and times each free.

    The queue holds at most `depth` decoders, so they cannot pile up on the GPU if the thread falls behind: put()
    waits instead, and that wait is inside the shot's time. The thread checks that its list is the decoder's only
    holder before it clears it (not_last_owner counts the shots where it was not, so the free happened elsewhere)."""

    def __init__(self, depth=2):
        import queue
        import threading

        self.queue = queue.Queue(maxsize=depth)
        self.seconds, self.not_last_owner, self.errors = {}, 0, []
        self.thread = threading.Thread(target=self._work, name="tcbp-free", daemon=False)
        self.thread.start()

    def _work(self):
        import sys

        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                key, held = item
                del item
                if sys.getrefcount(held[0]) > 2:            # the list's own reference and getrefcount's argument
                    self.not_last_owner += 1
                start = time.perf_counter()
                held.clear()
                self.seconds[key] = time.perf_counter() - start
                del held
            except BaseException as error:  # noqa: BLE001  reported in the summary
                self.errors.append(repr(error))
            finally:
                self.queue.task_done()

    def put(self, key, held):
        self.queue.put((key, held))

    def drain(self):
        self.queue.join()

    def stop(self):
        if self.thread.is_alive():
            self.queue.put(None)
            self.thread.join()


def arm_builders(builder, problem, syndromes, logits, setting, arms):
    """{arm: builder}: dense arms use the registered builder itself; a sparse arm shares its model and options and
    takes the first dtype of SPARSE_DTYPES that get_decoder accepts (one untimed shot each). An arm none of them
    builds is dropped, and said so. Arms of the same form (csr, csr_after, csr_thread) share one builder."""
    builders, dtypes, forms = {}, {}, {}
    priors = head_priors(problem, logits[0], setting)
    for arm in arms:
        form = h_form(arm)
        if form == "dense":
            builders[arm] = builder
            continue
        if form in forms:
            builders[arm], dtypes[arm] = forms[form]
            continue
        for dtype in SPARSE_DTYPES:
            try:
                other = builder.with_h_format(form, dtype)
                tcbp.decode_one(other, priors, syndromes[0])
            except BaseException as error:  # noqa: BLE001  SystemExit of h_matrix included
                print(f"  arm {arm} with dtype {np.dtype(dtype).name} refused: {error!r}", flush=True)
                continue
            builders[arm], dtypes[arm] = other, np.dtype(dtype).name
            forms[form] = (other, np.dtype(dtype).name)
            break
        else:
            print(f"  arm {arm} dropped: no dtype of {[np.dtype(d).name for d in SPARSE_DTYPES]} built", flush=True)
    return builders, dtypes


def check(builders, problem, syndromes, truth, logits, setting, block, out, depth=2):
    """Every shot by every arm, blocks of `block` shots, arms in a rotating order. Returns per-shot rows per arm, and
    per arm the wall time of its blocks (a thread arm's block ends when its thread has freed every decoder)."""
    arms = list(builders)
    shots = len(syndromes)
    rows = {arm: np.zeros((shots, len(FIELDS)), np.float64) for arm in arms}
    codes = {arm: np.zeros(shots, np.int64) for arm in arms}
    hashes = {arm: np.zeros(shots, np.uint64) for arm in arms}
    walls = {arm: 0.0 for arm in arms}
    logits_hashes = np.array([tcbp.row_hash(logits[k]) for k in range(shots)], np.uint64)
    releasers = {arm: Releaser(depth) for arm in arms if free_mode(arm) == "thread"}
    column = {name: FIELDS.index(name) for name in FIELDS}
    try:
        for arm in arms:                                                    # each arm's first build, untimed
            tcbp.decode_one(builders[arm], head_priors(problem, logits[0], setting), syndromes[0])
        start = time.time()
        for index, first in enumerate(range(0, shots, block)):
            last = min(first + block, shots)
            turn = index % len(arms)
            for arm in arms[turn:] + arms[:turn]:
                mode = free_mode(arm)
                block_start = time.perf_counter()
                for k in range(first, last):
                    begin = time.perf_counter()
                    priors = head_priors(problem, logits[k], setting)
                    ready = time.perf_counter()
                    if mode == "inline":
                        error, flag, build, decode = tcbp.decode_one(builders[arm], priors, syndromes[k])
                        answered = end = time.perf_counter()
                        free = (end - ready) - build - decode
                    else:
                        error, flag, build, decode, held = tcbp.decode_one(builders[arm], priors, syndromes[k], keep=True)
                        answered = time.perf_counter()
                        if mode == "after":
                            held.clear()
                            end = time.perf_counter()
                            free = end - answered
                        else:
                            releasers[arm].put(k, held)
                            end = time.perf_counter()
                            free = np.nan                           # the thread's own timer, filled in below
                        del held
                    predicted = (problem.L @ error) % 2
                    codes[arm][k] = int(np.asarray(predicted, np.int64) @ tcbp.BITS)
                    hashes[arm][k] = error_hash(error)
                    rows[arm][k] = (codes[arm][k], int(error.sum()), bool((predicted != truth[k]).any()),
                                    np.array_equal((problem.H @ error) % 2, syndromes[k]), flag,
                                    ready - begin, build, decode, free, end - begin, answered - begin)
                if mode == "thread":
                    releasers[arm].drain()
                    for k in range(first, last):
                        rows[arm][k, column["seconds_free"]] = releasers[arm].seconds.get(k, np.nan)
                walls[arm] += time.perf_counter() - block_start
            save_check(out, rows, codes, hashes, logits_hashes, truth, last)
            print(f"  shots {first}-{last - 1} done by {', '.join(arms[turn:] + arms[:turn])}; {time.time() - start:.0f} s; "
                  + "; ".join(f"{arm} {rows[arm][:last, column['seconds_total']].mean() * 1e3:.1f} ms" for arm in arms),
                  flush=True)
    finally:
        for releaser in releasers.values():
            releaser.stop()
    threads = {arm: {"frees": len(r.seconds), "not_last_owner": r.not_last_owner, "errors": r.errors[:20],
                     "queue_depth": depth} for arm, r in releasers.items()}
    return rows, codes, hashes, logits_hashes, walls, threads


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
            summary["profile"] = profile(builder, priors, syndromes, out, args.profile_builds, args.profile_set)
        except Exception as error:          # noqa: BLE001  the check still runs
            summary["profile"] = {"error": repr(error), "traceback": traceback.format_exc()[-2000:]}
            print(f"profile FAILED: {error!r}", flush=True)
        save()
    builders, dtypes = arm_builders(builder, problem, syndromes, logits, setting, args.arms)
    arms = list(builders)
    summary["arms_run"], summary["sparse_dtypes"] = arms, dtypes
    tcbp.CAPPED.update(shots=0, columns=0)
    summary["free_modes"] = {arm: free_mode(arm) for arm in arms}
    rows, codes, hashes, _, walls, threads = check(builders, problem, syndromes, truth, logits, setting, args.block, out,
                                                   args.queue_depth)
    summary["check"] = compare(rows, codes, hashes, arms)
    for arm in arms:
        summary["check"][arm]["wall_ms_per_shot"] = walls[arm] / shots * 1e3
    summary["threads"] = threads
    summary["rule"] = ("fixed before job 2 ran: an arm that moves the free is a gain over csr only if its wall time per "
                       "shot (its blocks' wall time over its shots, a thread arm's block ending when its thread has freed "
                       "every decoder) is below csr's in the same process; its time to answer is reported beside it")
    summary["capped_in_check"] = {"what": "counted over every arm's decodes", **tcbp.CAPPED}
    if model_before is not None:
        summary["model_option_unchanged"] = hashlib.sha256(bytes(builder.options["tcbp_model"])).hexdigest() == model_before
    repeatable = summary["check"].get("dense_again", {}).get("same_as_dense")
    summary["repeatable"] = repeatable
    summary["timer"] = ("seconds_total: one perf_counter timer around the shot's priors (shot_priors and the 0.5 cap), "
                        "tcbp.decode_one (get_decoder with the shot's priors, decode(), the cut to 0/1) and freeing the "
                        "decoder; one process, shots one at a time, the GPU shared with whatever else runs on it. "
                        "seconds_answer: the same timer stopped when the answer is in the caller's hands (inline arms: "
                        "after the free; csr_after and csr_thread: before it). wall_ms_per_shot: the arm's blocks' wall "
                        "time over its shots, scoring included (the same for every arm), a thread arm's drain included")
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
              f"+ free {ms['seconds_free']['mean']:.1f} ({free_mode(arm)}); answer {ms['seconds_answer']['mean']:.1f} ms "
              f"(p99 {ms['seconds_answer']['p99']:.1f}); wall {row['wall_ms_per_shot']:.1f} ms per shot{same}", flush=True)
    if "csr" in arms:
        base = summary["check"]["csr"]
        for arm in [arm for arm in arms if free_mode(arm) != "inline"]:
            row = summary["check"][arm]
            moved = {part: row["ms"][f"seconds_{part}"]["mean"] - base["ms"][f"seconds_{part}"]["mean"]
                     for part in ("priors", "build", "decode")}
            verdict = "GAIN" if row["wall_ms_per_shot"] < base["wall_ms_per_shot"] else "NO GAIN"
            row["against_csr"] = {"wall_ms_per_shot_difference": row["wall_ms_per_shot"] - base["wall_ms_per_shot"],
                                  "answer_ms_difference": row["ms"]["seconds_answer"]["mean"] - base["ms"]["seconds_answer"]["mean"],
                                  "parts_ms_difference": moved, "verdict": verdict}
            print(f"  {arm} against csr (same process): wall {row['wall_ms_per_shot'] - base['wall_ms_per_shot']:+.2f} ms per shot "
                  f"-> {verdict} by the rule; answer {row['against_csr']['answer_ms_difference']:+.2f} ms; the next parts "
                  f"moved by {', '.join(f'{part} {value:+.2f}' for part, value in moved.items())} ms"
                  + (f"; thread: {threads[arm]}" if arm in threads else ""), flush=True)
        save()
    if repeatable is False:
        print("  dense and dense_again differ: the decoder is not repeatable in this run, so no arm can be called the same",
              flush=True)
    print(f"saved {out}; {time.time() - started:.0f} s", flush=True)


# ---------- report (this PC; no decoder) ----------

def report_command(args):
    """The tables of a run's summary.json, and its arms against another run's per-shot file of tcbp.py decode."""
    with open(args.summary) as file:
        summary = json.load(file)
    result = {"summary": str(args.summary)}
    profile_result = summary.get("profile", {})
    if isinstance(profile_result.get("variants"), dict):
        print("timed builds (each: one untimed, then the rest one at a time; get_decoder, then freeing the decoder), ms:")
        for name, row in profile_result["variants"].items():
            if "error" in row:
                print(f"  {name:58s} FAILED: {row['error']}")
            else:
                print(f"  {name:58s} build mean {row['build_ms']['mean']:6.1f}  median {row['build_ms']['median']:6.1f}  "
                      f"p90 {row['build_ms']['p90']:6.1f};  free mean {row['free_ms']['mean']:5.1f}")
        print("python parts (median ms):", {k: (round(v, 3) if isinstance(v, float) else v)
                                           for k, v in profile_result.get("python_parts", {}).items()})
    print(f"check: {summary['shots']} shots, arms {summary.get('arms_run')}, sparse dtypes {summary.get('sparse_dtypes')}, "
          f"repeatable {summary.get('repeatable')}")
    for arm, row in summary.get("check", {}).items():
        ms = row["ms"]
        print(f"  {arm:12s} failures {row['failures']:3d}  total {ms['seconds_total']['mean']:6.1f} ms (median "
              f"{ms['seconds_total']['median']:6.1f}, p99 {ms['seconds_total']['p99']:6.1f}) = priors {ms['seconds_priors']['mean']:.2f} "
              f"+ build {ms['seconds_build']['mean']:.1f} + decode {ms['seconds_decode']['mean']:.1f} + free {ms['seconds_free']['mean']:.1f}"
              + ("" if arm == "dense" else f"; vs dense: bits {row['bits_differ']}, faults {row['faults_differ']}, "
                 f"corrections {row['corrections_differ']}"))
    if args.reference:
        with np.load(args.per_shot) as file:
            ours = {key: file[key] for key in file.files}
        with np.load(args.reference) as file:
            theirs = {key: file[key] for key in file.files}
        shots = len(ours["truth"])
        if not np.array_equal(theirs["truth"][:shots], ours["truth"]):
            raise SystemExit(f"{args.reference} was not made from the same shots")
        fields = list(ours["fields"])
        arm = args.reference_arm
        result["reference"] = {"file": str(args.reference), "arm": arm, "shots": shots,
                               "logits_rows_equal": int((theirs[f"logits_hash_{arm}"][:shots] == ours["logits_hash"]).sum())}
        print(f"against {args.reference} ({arm} arm, another run and another model fit), first {shots} shots: logits rows "
              f"equal on {result['reference']['logits_rows_equal']}")
        for name in summary.get("arms_run", []):
            bits = int((ours[f"predicted_code_{name}"] != theirs[f"predicted_code_{arm}"][:shots]).sum())
            faults = int((ours[f"rows_{name}"][:, fields.index("n_faults")] != theirs[f"n_faults_{arm}"][:shots]).sum())
            result["reference"][name] = {"bits_differ": bits, "faults_differ": faults}
            print(f"  {name:12s} bits differ on {bits}, faults on {faults}")
    if args.out:
        with open(args.out, "w") as file:
            json.dump(result, file, indent=1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser("run")
    sub.add_argument("--data", required=True)
    sub.add_argument("--logits", required=True, help="a folder of predecoder.evaluate for --data")
    sub.add_argument("--settings", required=True, help="the fit file of tcbp.py (the knobs)")
    sub.add_argument("--limit", type=int, default=1000)
    sub.add_argument("--block", type=int, default=100)
    sub.add_argument("--arms", nargs="+", choices=ALL_ARMS, default=list(ARMS))
    sub.add_argument("--profile-builds", type=int, default=50, help="0: no profile")
    sub.add_argument("--profile-set", choices=("all", "options"), default="all",
                     help="all: job 1's variants; options: H CSR with decoder options changed (timing probes only)")
    sub.add_argument("--queue-depth", type=int, default=2, help="the most decoders waiting for csr_thread's thread")
    sub.add_argument("--decoder", choices=("tcbp-jobs", "relay"), default="tcbp-jobs")
    sub.add_argument("--preset", default="tcbp-jobs")
    sub.add_argument("--opt", type=json.loads, default={})
    sub.add_argument("--final", action="store_true", help="allow a held-out file (test_*, go_*)")
    sub.add_argument("--out", required=True)
    sub = commands.add_parser("report")
    sub.add_argument("--summary", required=True, help="summary.json of run")
    sub.add_argument("--per-shot", help="check_per_shot.npz of the same run")
    sub.add_argument("--reference", help="per_shot.npz of tcbp.py decode on the same shot file (another run)")
    sub.add_argument("--reference-arm", default="head")
    sub.add_argument("--out")
    args = parser.parse_args(argv)
    {"run": run, "report": report_command}[args.command](args)


if __name__ == "__main__":
    main()
