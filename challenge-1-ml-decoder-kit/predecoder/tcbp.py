"""TCBP-JOBS with one shot's own priors from the head: the network tells NVIDIA's decoder which faults to suspect.

The head (head.py) gives every column of H a probability that it fired in this shot. priors.py
turns that into one shot's priors with two knobs (alpha, eps). This tool hands those priors to
TCBP-JOBS (the nv-qldpc-decoder of cudaq-qec, the teammates' strongest decoder,
../tcbp_jobs_baseline.py) in place of relay-BP.

TCBP-JOBS takes its priors only when a decoder is built (`error_rate_vec`), so every shot gets
a decoder of its own. Its model (`nv_qldpc.init_tcbp_model(H)`) does not depend on the priors:
it is fitted once per run and shared by every shot's decoder. Building one decoder takes about
51 ms on the server's A100, and that time is counted in the shot's latency.

Run from challenge-1-ml-decoder-kit/, in the teammates' TCBP environment (Linux, an NVIDIA GPU,
cudaq-qec 0.9.0). Torch is not needed: the logits come from predecoder.evaluate, run before in
an environment with torch, and are read from its output folder.

  decode   one shot file, one decoder built per shot, arms:
             fresh   the kit's priors           (TCBP-JOBS alone, but built per shot and decoded one shot at a time)
             head    the head's priors, mixed and clipped by the knobs of --settings
           python -m predecoder.tcbp decode --data data/p0.005/validation_X.npz --logits <folder> \\
               --settings predecoder/logs/fit_tcbp_X_p0.004.json --arms fresh head --out <folder>

  fit      the knobs, on TRAIN shot files only (the subsets of predecoder.subset made from
           TCBP-JOBS's own failures: --per-shot <train_X.tcbp.npz> --arm T):
           python -m predecoder.tcbp fit --data <train_X_T_failed.npz> <train_X_T_right.npz> --weights 1 <weight> \\
               --logits <folder> <folder> --out <fit.json>

  score    on this PC, from the files decode wrote and the other decoders' per-shot files
           (no decoder runs): the failures, the paired tests, and the fallback arm.
           python -m predecoder.tcbp score --data <shot file> --tcbp-head <decode folder>/per_shot.npz \\
               --tcbp <stored .tcbp.npz of TCBP-JOBS> --relay-head <compare folder>/per_shot.npz --out <summary.json>

--decoder relay puts the kit's relay-BP behind the same calls (no GPU: the plumbing only; its
output says so). --probe prints the public attributes of a built decoder and of a decode result
(the search for a way to change a built decoder's priors) and stops.

Every shot records: the 12 predicted bits, whether they are wrong, whether the correction
reproduces the syndrome (H e = s, checked here), the decoder's own `converged` flag, the faults
of the correction, the seconds of the build and of the decode (separately), and a hash of the
float16 logits row it used, so that two runs can be shown to have used the same priors.
Files named test_* or go_* are refused without --final.
"""
import argparse
import gc
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from predecoder.files import HELD_OUT, read_logits

BITS = 1 << np.arange(12)             # the 12 predicted bits as one number, bit i = observable i
FIELDS = ("failed", "reproduced", "flag", "n_faults", "seconds_build", "seconds_decode", "predicted_code", "logits_hash")
CHUNK = 100                           # shots per part file of the decode command
# TCBP-JOBS refuses a prior above 0.5 ("osd_method=4 (JOBS) requires every error rate to be at most 0.5"; the fit's
# first run stopped on it at alpha 0.3, 15:07 UTC). Every shot's priors are therefore capped at 0.5 after mixing and
# clipping. Shots below the cap are unchanged, so rows decoded before the cap existed (they passed the decoder's
# check) are the same with it. CAPPED counts the shots and columns the cap changed, for the logs.
PRIOR_CAP = 0.5
CAPPED = {"shots": 0, "columns": 0}

# The fit's grid (fixed before any TCBP-JOBS decode with the head's priors). The declared grid is the one
# that made logs/fit_X_p0.004.json for relay-BP: log-odds mixing, alpha 0.1 to 0.7, eps 1e-4. The
# extension rows are decoded in the same run, after the declared grid, and join the candidates only on
# the side where the declared grid's choice sits at its edge (the rule that extended relay-BP's grid).
GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
EXTEND_LOW, EXTEND_HIGH = [0.05], [0.8, 0.9, 1.0]
EPS, MODE = 1e-4, "logodds"
RULE = ("lowest weighted failures (failed shots left wrong + right shots broken, scaled); then fewer weighted shots whose "
        "correction does not reproduce the syndrome; then fewer weighted shots the decoder flags as not converged; then "
        "the smaller alpha. Candidates: the declared grid; if its choice is at an edge, the extension rows on that side join")


# ---------- the decoder ----------

class RelayStandIn:
    """The kit's relay-BP behind the calls this tool makes of a TCBP-JOBS decoder (no GPU: plumbing only)."""

    def __init__(self, H, priors):
        from decoders import relay_decoder

        self.decoder = relay_decoder(H, np.asarray(priors, np.float64))

    def decode(self, syndrome):
        result = self.decoder.decode_detailed(np.ascontiguousarray(syndrome, dtype=np.uint8))
        return SimpleNamespace(result=np.asarray(result.decoding, np.float64), converged=bool(result.success))


class Builder:
    """Builds one decoder per shot from that shot's priors. TCBP-JOBS: the model is fitted once, here."""

    def __init__(self, problem, decoder="tcbp-jobs", preset="tcbp-jobs", overrides=None):
        self.decoder, self.preset, self.overrides = decoder, preset, dict(overrides or {})
        self.H, self.L = problem.H.tocsr(), problem.L
        self.columns = problem.H.shape[1]
        start = time.perf_counter()
        if decoder == "relay":
            self.model_seconds = 0.0
            return
        import cudaq_qec as qec
        from cudaq_qec.plugins import nv_qldpc

        self.qec = qec
        self.dense = np.ascontiguousarray(problem.H.toarray(), dtype=np.uint8)     # row-major, as the plugin's docs ask
        self.model = nv_qldpc.init_tcbp_model(self.dense)                          # once: it does not depend on the priors
        self.options = nv_qldpc.preset(preset, tcbp_model=self.model, **self.overrides)
        self.model_seconds = time.perf_counter() - start

    def build(self, priors):
        if self.decoder == "relay":
            return RelayStandIn(self.H, priors)
        return self.qec.get_decoder("nv-qldpc-decoder", self.dense, error_rate_vec=np.asarray(priors, float), **self.options)


def decode_one(builder, priors, syndrome):
    """(error vector, decoder's converged flag, build seconds, decode seconds) for one shot with its own decoder."""
    from tcbp_jobs_baseline import hard

    start = time.perf_counter()
    decoder = builder.build(priors)
    built = time.perf_counter()
    result = decoder.decode(np.ascontiguousarray(syndrome, dtype=np.uint8))
    error = hard(result.result)
    done = time.perf_counter()
    if error.shape != (builder.columns,):
        raise SystemExit(f"decode returned an error vector of shape {error.shape}; H has {builder.columns} columns")
    flag = getattr(result, "converged", None)
    del decoder, result
    return error, (np.nan if flag is None else float(bool(flag))), built - start, done - built


def capped(priors):
    """The priors with every value above PRIOR_CAP set to PRIOR_CAP; CAPPED counts what changed."""
    over = priors > PRIOR_CAP
    if over.any():
        CAPPED["shots"] += 1
        CAPPED["columns"] += int(over.sum())
        return np.minimum(priors, PRIOR_CAP)
    return priors


def row_hash(logits_row):
    """The first 8 bytes of the sha256 of a float16 logits row, as an unsigned integer (0 for the kit's priors)."""
    if logits_row is None:
        return 0
    return int.from_bytes(hashlib.sha256(np.ascontiguousarray(logits_row, np.float16).tobytes()).digest()[:8], "little")


def decode_shots(builder, problem, syndromes, truth, logits, setting, label, warm=True):
    """Rows of FIELDS for these shots. setting None: the kit's priors. Prints a running mean every 200 shots."""
    from predecoder.priors import shot_priors

    if warm:                                                    # start-up of the GPU stays out of the times
        decode_one(builder, problem.priors, syndromes[0])
    rows = np.zeros((len(syndromes), len(FIELDS)), np.float64)
    hashes = np.zeros(len(syndromes), np.uint64)
    start = time.time()
    for k in range(len(syndromes)):
        if setting is None:
            priors, hashes[k] = problem.priors, 0
        else:
            priors = shot_priors(logits[k], problem.priors, setting["alpha"], setting["eps"], setting["mode"])
            hashes[k] = row_hash(logits[k])
        priors = capped(priors)
        error, flag, build, decode = decode_one(builder, priors, syndromes[k])
        predicted = (problem.L @ error) % 2
        reproduced = np.array_equal((problem.H @ error) % 2, syndromes[k])
        rows[k] = (bool((predicted != truth[k]).any()), reproduced, flag, int(error.sum()), build, decode,
                   int(np.asarray(predicted, np.int64) @ BITS), 0)
        if (k + 1) % 200 == 0:
            gc.collect()
            print(f"    {label}: {k + 1} of {len(syndromes)} shots, build {rows[:k + 1, 4].mean() * 1e3:.1f} ms, decode "
                  f"{rows[:k + 1, 5].mean() * 1e3:.1f} ms mean (last 200: {rows[k - 199:k + 1, 4].mean() * 1e3:.1f} + "
                  f"{rows[k - 199:k + 1, 5].mean() * 1e3:.1f} ms), {time.time() - start:.0f} s; priors capped at {PRIOR_CAP} so far "
                  f"on {CAPPED['shots']} shots, {CAPPED['columns']} columns", flush=True)
    return rows, hashes


# ---------- helpers ----------

def check_environment(decoder):
    from hierarchy.tcbp_member import check_environment as check

    return check(decoder)


def where(decoder):
    from hierarchy.tcbp_member import gpu_name

    return {"node": platform.node(), "processor": platform.processor() or platform.machine(), "cpus": os.cpu_count(),
            "gpu": gpu_name() if decoder == "tcbp-jobs" else "not used", "omp_threads": os.environ.get("OMP_NUM_THREADS")}


def gpu_memory():
    try:
        import subprocess
        return subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, Exception):          # noqa: BLE001  a note in the log only
        return "unknown"


def load(path, final):
    from evaluation.data import load_dataset

    if Path(path).name.startswith(HELD_OUT) and not final:
        raise SystemExit(f"{path} is a held-out file: it is run once, from a frozen commit, with --final")
    return load_dataset(path)


def problem_of(data):
    from predecoder.decode import kit_problem

    return kit_problem(data.basis, data.p)


def tag(what):
    return hashlib.sha256(json.dumps(what, sort_keys=True).encode()).hexdigest()[:12]


def summarise(rows):
    from predecoder.decode import wilson

    failed = rows[:, 0] > 0
    seconds = rows[:, 4] + rows[:, 5]
    low, high = wilson(int(failed.sum()), len(failed))
    flags = rows[:, 2]
    return {"shots": len(rows), "failures": int(failed.sum()), "share": float(failed.mean()), "wilson95": [low, high],
            "not_reproduced": int((rows[:, 1] == 0).sum()),
            "flag_not_converged": None if np.isnan(flags).any() else int((flags == 0).sum()),
            "build_ms_mean": float(rows[:, 4].mean() * 1e3), "decode_ms_mean": float(rows[:, 5].mean() * 1e3),
            "build_decode_ms_mean": float(seconds.mean() * 1e3), "build_decode_ms_median": float(np.median(seconds) * 1e3),
            "build_decode_ms_p99": float(np.percentile(seconds, 99) * 1e3)}


# ---------- probe ----------

def probe(args):
    data = load(args.data, args.final)
    problem = problem_of(data)
    builder = Builder(problem, args.decoder, args.preset, args.opt)
    print(f"model fitted in {builder.model_seconds:.1f} s")
    if args.decoder != "relay":
        print("preset options:", {key: type(value).__name__ for key, value in builder.options.items()})
        print("get_decoder doc:", (builder.qec.get_decoder.__doc__ or "")[:2000])
    decoder = builder.build(problem.priors)
    print("decoder type:", type(decoder).__name__)
    print("decoder attributes:", [name for name in dir(decoder) if not name.startswith("_")])
    for name in dir(decoder):
        if not name.startswith("_") and any(word in name.lower() for word in ("rate", "prior", "weight", "llr", "set", "update")):
            print(f"  {name}: {(getattr(getattr(decoder, name), '__doc__', '') or '')[:600]}")
    print("decode doc:", (getattr(decoder.decode, "__doc__", "") or "")[:1500])
    result = decoder.decode(np.ascontiguousarray(data.basis_input[0], dtype=np.uint8))
    print("result type:", type(result).__name__, "attributes:", [name for name in dir(result) if not name.startswith("_")])
    print("result.converged:", getattr(result, "converged", "absent"), " opt_results:", getattr(result, "opt_results", "absent"))


# ---------- decode ----------

def decode_command(args):
    from predecoder.priors import load_settings

    data = load(args.data, args.final)
    problem = problem_of(data)
    shots = len(data.detectors) if args.limit is None else min(args.limit, len(data.detectors))
    meta, logits_path = read_logits(args.logits, data) if args.logits else (None, None)
    if "head" in args.arms and meta is None:
        raise SystemExit("the head arm needs --logits")
    if meta is not None and meta["shots"] < shots:
        raise SystemExit(f"{args.logits} holds the logits of {meta['shots']} shots only")
    if "head" in args.arms and not args.settings:
        raise SystemExit("the head arm needs --settings (the fitted knobs): the head's numbers as they are make BP worse")
    setting = load_settings(args.settings) if args.settings else None
    logits = np.load(logits_path, mmap_mode="r") if logits_path else None
    version = check_environment(args.decoder)
    out = Path(args.out)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    syndromes, truth = np.ascontiguousarray(data.basis_input[:shots], dtype=np.uint8), data.observables[:shots]
    print(f"{data.path.name}: {shots} shots, p={data.p:g}, arms {args.arms}, decoder {args.decoder} ({version}), preset "
          f"{args.preset}, overrides {args.opt}; head priors {setting}; {where(args.decoder)}", flush=True)
    start = time.time()
    builder = Builder(problem, args.decoder, args.preset, args.opt)
    print(f"  model fitted in {builder.model_seconds:.1f} s; GPU memory {gpu_memory()}", flush=True)
    warm = True
    per_shot, results = {"truth": truth}, {}
    for arm in args.arms:
        what = {"data": data.sha256, "arm": arm, "decoder": args.decoder, "preset": args.preset, "opt": args.opt}
        if arm == "head":
            what |= {"head": meta["model"]["head_sha256"], "trunk": meta["model"]["trunk_sha256"], "priors": setting,
                     "logits_device": meta["model"]["device"]}
        parts = []
        for first in range(0, shots, CHUNK):
            last = min(first + CHUNK, shots)
            path = out / "parts" / f"{arm}_{tag(what)}_{first:06d}_{shots}.npz"
            if not path.exists():
                rows, hashes = decode_shots(builder, problem, syndromes[first:last], truth[first:last],
                                            logits[first:last] if arm == "head" else None,
                                            setting if arm == "head" else None, f"{arm} {first}-{last - 1}", warm)
                warm = False
                np.savez(path, rows=rows, hashes=hashes)
            with np.load(path) as file:
                parts.append((file["rows"], file["hashes"]))
            if (first // CHUNK) % 10 == 9:
                print(f"  {arm}: {last} of {shots} shots, {time.time() - start:.0f} s; GPU memory {gpu_memory()}", flush=True)
        rows = np.concatenate([rows for rows, _ in parts])
        hashes = np.concatenate([hashes for _, hashes in parts])
        results[arm] = summarise(rows)
        for i, field in enumerate(FIELDS[:-1]):
            per_shot[f"{field}_{arm}"] = rows[:, i]
        per_shot[f"predicted_{arm}"] = ((rows[:, 6].astype(np.int64)[:, None] & BITS) > 0).astype(np.uint8)
        per_shot[f"logits_hash_{arm}"] = hashes
        row = results[arm]
        print(f"  {arm}: failures {row['failures']} of {shots}, {row['share']:.4f} [{row['wilson95'][0]:.4f}, "
              f"{row['wilson95'][1]:.4f}]; not reproduced {row['not_reproduced']}; flagged not converged "
              f"{row['flag_not_converged']}; build {row['build_ms_mean']:.1f} ms + decode {row['decode_ms_mean']:.1f} ms mean, "
              f"p99 {row['build_decode_ms_p99']:.0f} ms", flush=True)
    if args.batch_check:
        # The stored TCBP-JOBS files come from one decoder built once and run with decode_batch. The same decoder,
        # built here from this run's model, separates "batch against one shot at a time" from "another model fit".
        from tcbp_jobs_baseline import hard

        decoder = builder.build(problem.priors)
        errors = []
        for first in range(0, shots, 4096):
            batch = np.ascontiguousarray(syndromes[first:first + 4096], dtype=np.uint8)
            if args.decoder == "relay":
                errors.append(np.stack([hard(decoder.decode(s).result) for s in batch]))
            else:
                errors.append(hard(decoder.decode_batch(batch).result))
        errors = np.concatenate(errors).astype(np.uint8)
        per_shot["predicted_batch"] = np.asarray((problem.L @ errors.T) % 2, np.uint8).T
        per_shot["n_faults_batch"] = errors.sum(axis=1)
        per_shot["failed_batch"] = (per_shot["predicted_batch"] != truth).any(axis=1)
        results["batch"] = {"failures": int(per_shot["failed_batch"].sum()), "shots": shots,
                            "what": "one decoder built once with the kit's priors, this run's model, decode_batch"}
        line = f"  batch check (kit's priors, built once, decode_batch, this run's model): failures {results['batch']['failures']}"
        if "fresh" in args.arms:
            differ = (per_shot["predicted_batch"] != per_shot["predicted_fresh"]).any(axis=1)
            results["batch"]["bits_differ_from_fresh"] = int(differ.sum())
            results["batch"]["faults_differ_from_fresh"] = int((per_shot["n_faults_batch"] != per_shot["n_faults_fresh"]).sum())
            line += (f"; its 12 bits differ from the fresh arm's on {int(differ.sum())} shots, its number of faults on "
                     f"{results['batch']['faults_differ_from_fresh']}")
        print(line, flush=True)
    np.savez_compressed(out / "per_shot.npz", **per_shot)
    with open(out / "summary.json", "w") as file:
        json.dump({"data": data.identity(), "shots": shots, "decoder": args.decoder, "version": version, "preset": args.preset,
                   "overrides": args.opt, "head_priors": setting, "prior_cap": PRIOR_CAP, "capped_this_run": dict(CAPPED),
                   "logits": meta, "arms": results,
                   "model_seconds": builder.model_seconds, "machine": where(args.decoder),
                   "timer": "build: get_decoder with the shot's priors; decode: decode() and the cut to 0/1; one process, "
                            "shots one at a time; the network and head are not included (predecoder.evaluate)",
                   "seconds": time.time() - start}, file, indent=1)
    print(f"saved {out / 'per_shot.npz'}; {time.time() - start:.0f} s", flush=True)


# ---------- fit ----------

def settings_grid():
    """The control (the kit's priors), the declared grid, then the extension rows."""
    rows = [{"mode": "linear", "alpha": 0.0, "eps": 1e-6}]
    rows += [{"mode": MODE, "alpha": alpha, "eps": EPS} for alpha in GRID + EXTEND_HIGH + EXTEND_LOW]
    return rows


def choose(rows):
    """The chosen row by RULE, among rows of the grid (not the control). rows carry alpha and the scores."""
    def key(row):
        return (row["score"], row["not_reproduced_weighted"], row["flag_weighted"], row["alpha"])

    declared = [row for row in rows if row["alpha"] in GRID]
    best = min(declared, key=key)
    candidates = declared
    if best["alpha"] == min(GRID):
        candidates = declared + [row for row in rows if row["alpha"] in EXTEND_LOW]
    elif best["alpha"] == max(GRID):
        candidates = declared + [row for row in rows if row["alpha"] in EXTEND_HIGH]
    return min(candidates, key=key), candidates is not declared


def fit_command(args):
    weights = args.weights or [1.0] * len(args.data)
    if not len(args.data) == len(args.logits) == len(weights):
        raise SystemExit("give one --logits folder and one weight per --data file")
    if not all(Path(path).name.startswith("train_") for path in args.data):
        raise SystemExit("the knobs are fitted on train shots only: give train_* files")
    files = []
    for path, folder in zip(args.data, args.logits, strict=True):
        data = load(path, False)
        meta, logits_path = read_logits(folder, data)
        files.append((data, meta, np.load(logits_path, mmap_mode="r")))
    problem = problem_of(files[0][0])
    if any(data.p != problem.p or data.basis != problem.basis for data, _, _ in files):
        raise SystemExit("all files must have the same basis and p")
    version = check_environment(args.decoder)
    grid = settings_grid()[:args.rows]
    cache = Path(args.out).with_suffix(".parts")
    cache.mkdir(parents=True, exist_ok=True)
    start = time.time()
    print(f"fit of the knobs for {args.decoder} ({version}) on {', '.join(f'{d.path.name} (weight {w:g})' for (d, _, _), w in zip(files, weights))}, "
          f"p={problem.p:g}; {len(grid)} rows; {where(args.decoder)}", flush=True)
    builder = Builder(problem, args.decoder, args.preset, args.opt)
    print(f"  model fitted in {builder.model_seconds:.1f} s; GPU memory {gpu_memory()}", flush=True)
    warm = True
    results = {}
    for setting in grid:
        for index, (data, meta, logits) in enumerate(files):
            shots = len(data.detectors) if args.shots is None else min(args.shots, len(data.detectors))
            head = None if setting["alpha"] == 0 else meta["model"]["head_sha256"]
            what = {"data": data.sha256, "head": head, "shots": shots, "setting": setting, "decoder": args.decoder,
                    "preset": args.preset, "opt": args.opt}
            path = cache / f"{tag(what)}.npz"
            if not path.exists():
                rows, hashes = decode_shots(builder, problem, np.ascontiguousarray(data.basis_input[:shots], dtype=np.uint8),
                                            data.observables[:shots], logits[:shots], None if setting["alpha"] == 0 else setting,
                                            f"{data.path.stem} alpha {setting['alpha']:g}", warm)
                warm = False
                np.savez(path, rows=rows, hashes=hashes)
            with np.load(path) as file:
                results[(json.dumps(setting, sort_keys=True), index)] = file["rows"]
            print(f"  capped at {PRIOR_CAP} so far: {CAPPED['shots']} shots, {CAPPED['columns']} columns", flush=True)
            print(f"  {data.path.name}: {shots} shots, {setting}, failures {int(results[(json.dumps(setting, sort_keys=True), index)][:, 0].sum())}, "
                  f"{time.time() - start:.0f} s; GPU memory {gpu_memory()}", flush=True)
    rows = []
    names = [data.path.stem for data, _, _ in files]
    print(f"{'mode':8s} {'alpha':>6s} {'eps':>8s} " + " ".join(f"{'fail ' + n[:18]:>24s}" for n in names)
          + f" {'score':>8s} {'not repr. (w)':>14s} {'flag (w)':>9s} {'build ms':>9s} {'decode ms':>9s}")
    for setting in grid:
        per_file = [results[(json.dumps(setting, sort_keys=True), i)] for i in range(len(files))]
        failures = [int(values[:, 0].sum()) for values in per_file]
        flags = [values[:, 2] for values in per_file]
        row = setting | {
            "failures": failures, "score": float(sum(f * w for f, w in zip(failures, weights))),
            "not_reproduced_weighted": float(sum(int((v[:, 1] == 0).sum()) * w for v, w in zip(per_file, weights))),
            "flag_weighted": float(sum(int((np.nan_to_num(f, nan=1.0) == 0).sum()) * w for f, w in zip(flags, weights))),
            "build_ms_mean": float(np.concatenate([v[:, 4] for v in per_file]).mean() * 1e3),
            "decode_ms_mean": float(np.concatenate([v[:, 5] for v in per_file]).mean() * 1e3)}
        rows.append(row)
        print(f"{row['mode']:8s} {row['alpha']:6g} {row['eps']:8g} " + " ".join(f"{f:24d}" for f in failures)
              + f" {row['score']:8.1f} {row['not_reproduced_weighted']:14.1f} {row['flag_weighted']:9.1f} "
              f"{row['build_ms_mean']:9.1f} {row['decode_ms_mean']:9.1f}")
    control = rows[0]
    print(f"control (the kit's priors, built per shot): failures {control['failures']} (each file's shots: "
          f"{[len(results[(json.dumps(grid[0], sort_keys=True), i)]) for i in range(len(files))]})")
    chosen, extended = None, False
    if all(alpha in [row["alpha"] for row in rows] for alpha in GRID):
        chosen, extended = choose(rows[1:])
        print(f"chosen: mode {chosen['mode']}, alpha {chosen['alpha']:g}, eps {chosen['eps']:g}"
              f"{' (the declared grid chose its edge; the extension rows joined)' if extended else ''}. Rule: {RULE}")
    else:
        print("the declared grid is not complete: nothing chosen")
    with open(args.out, "w") as file:
        json.dump({"data": [data.identity() for data, _, _ in files], "weights": weights, "logits": [m for _, m, _ in files],
                   "decoder": args.decoder, "version": version, "preset": args.preset, "overrides": args.opt,
                   "model_seconds": builder.model_seconds, "machine": where(args.decoder), "rows": rows,
                   "chosen": None if chosen is None else {key: chosen[key] for key in ("mode", "alpha", "eps")},
                   "extended": extended, "rule": RULE, "prior_cap": PRIOR_CAP, "capped_this_run": dict(CAPPED), "grid": GRID, "extension": {"low": EXTEND_LOW, "high": EXTEND_HIGH},
                   "seconds": time.time() - start}, file, indent=1)
    print(f"saved {args.out}; {time.time() - start:.0f} s", flush=True)


# ---------- score (this PC; no decoder) ----------

def score_command(args):
    from predecoder.decode import paired, wilson

    data = load(args.data, args.final)
    with np.load(args.tcbp_head) as file:
        ours = {key: file[key] for key in file.files}
    shots = len(ours["truth"])
    truth = data.observables[:shots]
    if not np.array_equal(ours["truth"], truth):
        raise SystemExit(f"{args.tcbp_head} was not made from {args.data}")
    arms = {}

    def check(name, predicted, stored_failed=None):
        failed = (predicted != truth).any(axis=1)
        if stored_failed is not None and not np.array_equal(failed, np.asarray(stored_failed, bool)):
            raise SystemExit(f"{name}: its stored failures do not match its predictions against the truth")
        return failed

    for arm in ("head", "fresh"):
        if f"predicted_{arm}" in ours:
            arms[f"TCBP-JOBS, {arm} priors per shot"] = check(arm, ours[f"predicted_{arm}"], ours[f"failed_{arm}"] > 0)
    if "predicted_batch" in ours:
        arms["TCBP-JOBS, the kit's priors, built once, decode_batch (same run)"] = check("batch", ours["predicted_batch"],
                                                                                         ours["failed_batch"])
    if args.tcbp:
        with np.load(args.tcbp) as file:
            if str(file["data_sha256"]) != data.sha256:
                raise SystemExit(f"{args.tcbp} was made from another shot file")
            arms["TCBP-JOBS alone (stored)"] = check("T", file["predicted_T"][:shots], file["failed_T"][:shots])
    relay = None
    if args.relay_head:
        with np.load(args.relay_head) as file:
            relay = {key: file[key][:shots] for key in ("predicted_head", "failed_head", "converged_head", "truth")}
        if not np.array_equal(relay["truth"], truth):
            raise SystemExit(f"{args.relay_head} was not made from {args.data}")
        arms["relay-BP, head priors per shot (H)"] = check("H", relay["predicted_head"], relay["failed_head"] > 0)
        if args.tcbp:
            with np.load(args.tcbp) as file:
                stored = file["predicted_T"][:shots]
            frozen = np.where((relay["converged_head"] > 0)[:, None], relay["predicted_head"], stored)
            arms["the frozen fallback rule: H where H converged, else T (stored)"] = check("rule", frozen)
        if "predicted_head" in ours:
            converged = relay["converged_head"] > 0
            chosen = np.where(converged[:, None], relay["predicted_head"], ours["predicted_head"])
            arms["fallback b2: H where H converged, else TCBP-JOBS with head priors"] = check("b2", chosen)
            flag = ours["flag_head"]
            if not np.isnan(flag).any():
                chosen = np.where((flag > 0)[:, None], ours["predicted_head"], relay["predicted_head"])
                arms["fallback b1: TCBP-JOBS with head priors where its flag says converged, else H"] = check("b1", chosen)
    summary = {"data": data.identity(), "shots": shots, "arms": {}, "pairs": {}}
    for name, failed in arms.items():
        low, high = wilson(int(failed.sum()), shots)
        summary["arms"][name] = {"failures": int(failed.sum()), "share": float(failed.mean()), "wilson95": [low, high]}
        print(f"{name:72s} {int(failed.sum()):5d} of {shots}  {failed.mean():.4f} [{low:.4f}, {high:.4f}]")
    names = list(arms)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both, only_a, only_b, p_value = paired(arms[a], arms[b])
            summary["pairs"][f"{a} | {b}"] = {"both": both, "only_first": only_a, "only_second": only_b, "p": p_value}
            print(f"  {a}\n    against {b}: both {both}, only the first {only_a}, only the second {only_b}, exact p = {p_value:.3g}")
    if "logits_hash_head" in ours and args.logits:
        meta, logits_path = read_logits(args.logits, data)
        local = np.load(logits_path, mmap_mode="r")
        same = np.array([row_hash(local[k]) for k in range(shots)], np.uint64) == ours["logits_hash_head"]
        summary["logits_rows_equal_to_local"] = int(same.sum())
        print(f"the head arm's logits rows equal those of {args.logits} on {int(same.sum())} of {shots} shots")
    if args.out:
        with open(args.out, "w") as file:
            json.dump(summary, file, indent=1)


# ---------- pair (the kit's score: shot k of X with shot k of Z) ----------

def pair_command(args):
    """Failed pairs of TH and of the stored TCBP-JOBS files, paired, with the rate per cycle."""
    from decoders import per_cycle_rate
    from predecoder.decode import paired, wilson

    failed = {"TH": [], "T": []}
    for data_path, ours_path, tcbp_path in ((args.data_x, args.ours_x, args.tcbp_x), (args.data_z, args.ours_z, args.tcbp_z)):
        data = load(data_path, args.final)
        with np.load(ours_path) as file:
            ours = {key: file[key] for key in ("truth", "predicted_head")}
        with np.load(tcbp_path) as file:
            stored = {key: file[key] for key in ("truth", "predicted_T", "data_sha256")}
        if not (np.array_equal(ours["truth"], data.observables) and np.array_equal(stored["truth"], data.observables)
                and str(stored["data_sha256"]) == data.sha256):
            raise SystemExit(f"{ours_path} or {tcbp_path} was not made from {data_path}")
        failed["TH"].append((ours["predicted_head"] != data.observables).any(axis=1))
        failed["T"].append((stored["predicted_T"] != data.observables).any(axis=1))
    result = {}
    for name, (fx, fz) in failed.items():
        pairs = fx | fz
        low, high = wilson(int(pairs.sum()), len(pairs))
        rate = [float(value) for value in per_cycle_rate(pairs, args.cycles)]     # (rate, its standard error)
        result[name] = {"failed_pairs": int(pairs.sum()), "failed_X": int(fx.sum()), "failed_Z": int(fz.sum()),
                        "pairs": len(pairs), "wilson95": [low, high], "per_cycle_rate": rate}
        print(f"{name}: {int(pairs.sum())} failed pairs of {len(pairs)} [X {int(fx.sum())} Z {int(fz.sum())}], "
              f"{pairs.mean():.4f} [{low:.4f}, {high:.4f}], rate per cycle {rate}")
    both, only_th, only_t, p_value = paired(failed["TH"][0] | failed["TH"][1], failed["T"][0] | failed["T"][1])
    result["pair"] = {"both": both, "only_TH": only_th, "only_T": only_t, "p": p_value}
    print(f"TH against T, pairs: both {both}, only TH {only_th}, only T {only_t}, exact two-sided p = {p_value:.3g}")
    if args.out:
        with open(args.out, "w") as file:
            json.dump(result, file, indent=1, default=str)


# ---------- logits-diff (no decoder, no torch) ----------

def logits_diff_command(args):
    """How far two logits files of the same shots differ: the largest gap, and the rows and entries that differ."""
    with open(Path(args.a) / "meta.json") as file:
        meta_a = json.load(file)
    with open(Path(args.b) / "meta.json") as file:
        meta_b = json.load(file)
    if meta_a["data"]["sha256"] != meta_b["data"]["sha256"]:
        raise SystemExit("the two folders hold the logits of different shot files")
    a, b = np.load(Path(args.a) / "logits.npy", mmap_mode="r"), np.load(Path(args.b) / "logits.npy", mmap_mode="r")
    shots = min(len(a), len(b), args.limit or len(a))
    gap, rows, entries = 0.0, 0, 0
    for first in range(0, shots, 500):
        x = np.asarray(a[first:min(first + 500, shots)], np.float32)
        y = np.asarray(b[first:min(first + 500, shots)], np.float32)
        differ = x != y
        gap = max(gap, float(np.abs(x - y).max()))
        rows += int(differ.any(axis=1).sum())
        entries += int(differ.sum())
    result = {"a": {"folder": str(args.a), "model": meta_a["model"]}, "b": {"folder": str(args.b), "model": meta_b["model"]},
              "shots": shots, "largest_gap": gap, "rows_differ": rows, "entries_differ": entries,
              "entries": shots * a.shape[1], "per_shot_seconds_b": meta_b.get("per_shot_seconds")}
    print(f"{shots} shots: the float16 logits differ on {rows} rows and {entries} of {shots * a.shape[1]} entries; "
          f"largest gap {gap:.3g} ({meta_a['model'].get('device')} {meta_a['model'].get('precision', '')} against "
          f"{meta_b['model'].get('device')} {meta_b['model'].get('precision', '')})")
    if args.out:
        with open(args.out, "w") as file:
            json.dump(result, file, indent=1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("decode", "fit", "probe"):
        sub = commands.add_parser(name)
        sub.add_argument("--decoder", choices=("tcbp-jobs", "relay"), default="tcbp-jobs")
        sub.add_argument("--preset", default="tcbp-jobs")
        sub.add_argument("--opt", type=json.loads, default={}, help="decoder option overrides, as a JSON object")
        if name == "fit":
            sub.add_argument("--data", nargs="+", required=True)
            sub.add_argument("--weights", type=float, nargs="+")
            sub.add_argument("--logits", nargs="+", required=True)
            sub.add_argument("--shots", type=int, help="only the first this many shots of each file (smoke runs)")
            sub.add_argument("--rows", type=int, help="only the first this many rows of the grid (smoke runs)")
            sub.add_argument("--out", required=True)
        else:
            sub.add_argument("--data", required=True)
            sub.add_argument("--final", action="store_true", help="allow a held-out file (test_*, go_*)")
        if name == "decode":
            sub.add_argument("--logits")
            sub.add_argument("--settings", help="the fit file of this tool (or of priors.py)")
            sub.add_argument("--arms", nargs="+", choices=("fresh", "head"), default=["head"])
            sub.add_argument("--limit", type=int)
            sub.add_argument("--batch-check", action="store_true",
                             help="also decode the shots with one decoder built once (the kit's priors), decode_batch")
            sub.add_argument("--out", required=True)
    sub = commands.add_parser("score")
    sub.add_argument("--data", required=True)
    sub.add_argument("--tcbp-head", required=True, help="per_shot.npz of the decode command")
    sub.add_argument("--tcbp", help="the stored per-shot file of TCBP-JOBS alone (hierarchy.tcbp_member)")
    sub.add_argument("--relay-head", help="per_shot.npz of predecoder.compare with the head arm")
    sub.add_argument("--logits", help="this PC's logits folder: check that the head arm used the same rows")
    sub.add_argument("--final", action="store_true")
    sub.add_argument("--out")
    sub = commands.add_parser("pair")
    for name in ("data-x", "data-z", "ours-x", "ours-z", "tcbp-x", "tcbp-z"):
        sub.add_argument(f"--{name}", required=True)
    sub.add_argument("--cycles", type=int, default=12)
    sub.add_argument("--final", action="store_true")
    sub.add_argument("--out")
    sub = commands.add_parser("logits-diff")
    sub.add_argument("--a", required=True, help="a folder of predecoder.evaluate")
    sub.add_argument("--b", required=True, help="another, for the same shot file")
    sub.add_argument("--limit", type=int)
    sub.add_argument("--out")
    args = parser.parse_args(argv)
    {"decode": decode_command, "fit": fit_command, "probe": probe, "score": score_command,
     "logits-diff": logits_diff_command, "pair": pair_command}[args.command](args)


if __name__ == "__main__":
    main()
