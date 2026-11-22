"""The measurement: relay-BP with the kit's priors against relay-BP with the head's priors, on the same shots.

Run from challenge-1-ml-decoder-kit/ (needs the logits of predecoder.evaluate for the same file):

  python -m predecoder.compare --data data/p0.004/validation_X.npz --logits <folder of predecoder.evaluate> \\
      --settings <fit.json of predecoder.priors> --arms fresh head kit --workers 4 --out <folder>

Three arms, each relay-BP at the kit's settings, on the same Stim shots of the file:

  kit     the kit's relay-BP, built once with the kit's priors, decoding the shots in file order
          (what evaluation.evaluate runs as "relay-bp"; it carries its random state from shot to shot).
          With --kit-chunk N, one such decoder per chunk of N shots, the chunks side by side.
  fresh   relay-BP built per shot with the kit's priors (seed 0 on every shot)
  head    relay-BP built per shot with this shot's priors from the head (seed 0 on every shot)

"head" against "fresh" changes the priors and nothing else: that pair reads the kill line.
"head" against "kit" is what a user of the kit's decoder would see.

For every arm: the failures (a shot fails when any of its 12 bits is wrong) with a 95% Wilson
interval, the shots relay-BP did not converge on, the "hard" shots (relay-BP's first leg alone,
80 iterations with the same priors, does not converge: a separate, untimed decode), the mean
iterations and the time per shot. For every pair: the shots both
get wrong, the shots only one gets wrong, and the exact two-sided binomial test on those.

The time per shot of "head" is the network and the head (from predecoder.evaluate, one shot at a
time, on its device) plus building the decoder plus decoding (here, in the worker processes).
Both depend on the machine and on what else runs on it: the log names the processes.

The "fresh" and "head" arms are decoded in chunks of --chunk shots, each saved to
<out>/parts/ as soon as it is done, so a stopped run continues where it stopped. The "kit" arm
runs one process per chunk of --kit-chunk shots (default: one chunk, the whole file), each in
file order, each saved when it ends.

Files named test_* or go_* are refused without --final.
"""
import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import time
from pathlib import Path

import numpy as np

from evaluation.data import load_dataset
from predecoder.decode import (decode_session, decode_with_priors, first_leg_converges, kit_problem, kit_session, paired,
                               settings, wilson)
from predecoder.files import HELD_OUT, read_logits
from predecoder.priors import load_settings, shot_priors

ARMS = ("kit", "fresh", "head")
FIELDS = ("failed", "converged", "iterations", "seconds_build", "seconds_decode", "first_leg", "predicted_code")
BITS = 1 << np.arange(12)            # the 12 predicted bits as one number, bit i = observable i
_JOB = {}


def _start(H, L, kit_priors, logits_path, prior_settings, syndromes, truth):
    _JOB.update(H=H, L=L, kit=kit_priors, logits=np.load(logits_path, mmap_mode="r") if logits_path else None,
                settings=prior_settings, syndromes=syndromes, truth=truth)


def _row(result, truth, first_leg):
    return (bool((result.predicted != truth).any()), result.converged, result.iterations, result.seconds_build,
            result.seconds_decode, first_leg, int(result.predicted.astype(np.int64) @ BITS))


def _run(task):
    """One task: (arm, first shot, last shot). Returns (arm, first, rows)."""
    arm, first, last = task
    H, L, syndromes, truth = _JOB["H"], _JOB["L"], _JOB["syndromes"], _JOB["truth"]
    rows = []
    if arm == "kit":
        decoder = kit_session(H, _JOB["kit"])
        for k in range(first, last):     # its first leg is the fresh arm's (no randomness in a first leg): not run again
            rows.append(_row(decode_session(decoder, L, syndromes[k]), truth[k], np.nan))
    for k in range(first, last) if arm != "kit" else ():
        if arm == "fresh":
            priors = _JOB["kit"]
        else:
            s = _JOB["settings"]
            priors = shot_priors(_JOB["logits"][k], _JOB["kit"], s["alpha"], s["eps"], s["mode"])
        result = decode_with_priors(H, L, priors, syndromes[k])
        rows.append(_row(result, truth[k], first_leg_converges(H, priors, syndromes[k])))   # untimed
    return arm, first, np.array(rows, dtype=np.float64)


def part_tag(data_sha256, arm, extra=None):
    """The fingerprint in a part file's name: the shot file, the arm, and for the head arm the head and the knobs."""
    what = {"data": data_sha256, "arm": arm} | (extra or {})
    return hashlib.sha256(json.dumps(what, sort_keys=True).encode()).hexdigest()[:12]


def _padded(rows):
    """A part file with the columns of FIELDS; parts written before a column existed get NaN there."""
    if rows.shape[1] == len(FIELDS):
        return rows
    return np.hstack([rows, np.full((len(rows), len(FIELDS) - rows.shape[1]), np.nan)])


def summarise(arm, values, network_seconds=0.0):
    failed, converged, iterations = values["failed"].astype(bool), values["converged"].astype(bool), values["iterations"]
    first_leg = values["first_leg"]
    hard = None if np.isnan(first_leg).any() else int((first_leg == 0).sum())
    seconds = values["seconds_build"] + values["seconds_decode"] + network_seconds
    low, high = wilson(int(failed.sum()), len(failed))
    return {"arm": arm, "shots": len(failed), "failures": int(failed.sum()), "share": float(failed.mean()),
            "wilson95": [low, high], "unconverged": int((~converged).sum()), "hard": hard,
            "mean_iterations": float(iterations.mean()), "network_seconds_mean": network_seconds,
            "build_seconds_mean": float(values["seconds_build"].mean()), "decode_seconds_mean": float(values["seconds_decode"].mean()),
            "seconds_mean": float(seconds.mean()), "seconds_p99": float(np.percentile(seconds, 99))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--logits", help="the folder predecoder.evaluate wrote for this file (needed by the head arm)")
    parser.add_argument("--settings", help="the fit file of predecoder.priors (default: alpha 1, eps 1e-6, linear)")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=100)
    parser.add_argument("--kit-chunk", type=int, help="shots per built-once decoder of the kit arm (default: all)")
    parser.add_argument("--limit", type=int, help="only the first this many shots")
    parser.add_argument("--out", required=True)
    parser.add_argument("--final", action="store_true", help="allow a held-out file (test_*, go_*)")
    args = parser.parse_args(argv)
    if Path(args.data).name.startswith(HELD_OUT) and not args.final:
        raise SystemExit(f"{args.data} is a held-out file: it is run once, from a frozen commit, with --final")

    start = time.time()
    data = load_dataset(args.data)
    problem = kit_problem(data.basis, data.p)
    shots = len(data.detectors) if args.limit is None else min(args.limit, len(data.detectors))
    meta, logits_path = (read_logits(args.logits, data) if args.logits else (None, None))
    if "head" in args.arms and meta is None:
        raise SystemExit("the head arm needs --logits")
    if meta is not None and meta["shots"] < shots:
        raise SystemExit(f"{args.logits} holds the logits of {meta['shots']} shots only")
    prior_settings = load_settings(args.settings)
    out = Path(args.out)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    kit_chunk = args.kit_chunk or shots

    def part(arm, first):
        """A part file's name carries a fingerprint of what it holds: the shot file, and for the head arm the head and the knobs."""
        what = {"kit_chunk": kit_chunk} if arm == "kit" and kit_chunk != shots else {}
        if arm == "head":
            what |= {"head": meta["model"]["head_sha256"], "trunk": meta["model"]["trunk_sha256"], "priors": prior_settings}
        return out / "parts" / f"{arm}_{part_tag(data.sha256, arm, what)}_{first:06d}_{shots}.npy"

    tasks = []
    kit_firsts = list(range(0, shots, kit_chunk))
    if "kit" in args.arms:                                     # first, as they are the longest
        tasks += [("kit", first, min(first + kit_chunk, shots)) for first in kit_firsts if not part("kit", first).exists()]
    for arm in ("fresh", "head"):
        if arm in args.arms:
            tasks += [(arm, first, min(first + args.chunk, shots)) for first in range(0, shots, args.chunk)
                      if not part(arm, first).exists()]
    print(f"{data.path.name}: {shots} shots, p={data.p:g}, arms {args.arms}; head priors {prior_settings}; "
          f"{len(tasks)} tasks to run in {args.workers} processes ({os.cpu_count()} CPUs here, {platform.node()})", flush=True)
    syndromes, truth = np.ascontiguousarray(data.basis_input[:shots]), data.observables[:shots]
    initargs = (problem.H.tocsr(), problem.L, problem.priors, str(logits_path) if logits_path else None, prior_settings,
                syndromes, truth)
    if tasks:
        with multiprocessing.get_context("spawn").Pool(args.workers, _start, initargs) as pool:
            for done, (arm, first, rows) in enumerate(pool.imap_unordered(_run, tasks), 1):
                np.save(part(arm, first), rows)
                if done % 10 == 0 or arm == "kit":
                    print(f"  {done} of {len(tasks)} tasks done ({arm} from shot {first}), {time.time() - start:.0f} s", flush=True)

    results, per_shot = {}, {"truth": truth}
    network = (meta or {}).get("per_shot_seconds", {}).get("mean") or 0.0
    for arm in args.arms:
        firsts = kit_firsts if arm == "kit" else range(0, shots, args.chunk)
        rows = np.concatenate([_padded(np.load(part(arm, first))) for first in firsts])
        values = {field: rows[:, i] if i < rows.shape[1] else np.full(len(rows), np.nan) for i, field in enumerate(FIELDS)}
        if arm == "kit" and "fresh" in args.arms:
            values["first_leg"] = np.concatenate([_padded(np.load(part("fresh", first))) for first in range(0, shots, args.chunk)])[:, 5]
        results[arm] = summarise(arm, values, network if arm == "head" else 0.0)
        for field, column in values.items():
            per_shot[f"{field}_{arm}"] = column
        if not np.isnan(values["predicted_code"]).any():
            per_shot[f"predicted_{arm}"] = ((values["predicted_code"].astype(np.int64)[:, None] & BITS) > 0).astype(np.uint8)
    pairs = {}
    for a, b in (("head", "fresh"), ("head", "kit"), ("fresh", "kit")):
        if a in results and b in results:
            both, only_a, only_b, p_value = paired(per_shot[f"failed_{a}"], per_shot[f"failed_{b}"])
            ia, ib = per_shot[f"iterations_{a}"], per_shot[f"iterations_{b}"]
            pairs[f"{a}_vs_{b}"] = {"both": both, f"only_{a}": only_a, f"only_{b}": only_b, "p": p_value,
                                    f"fewer_iterations_{a}": int((ia < ib).sum()), f"fewer_iterations_{b}": int((ib < ia).sum()),
                                    "same_iterations": int((ia == ib).sum()),
                                    f"iterations_p99_{a}": float(np.percentile(ia, 99)), f"iterations_p99_{b}": float(np.percentile(ib, 99))}

    print(f"relay-BP at the kit's settings; {time.time() - start:.0f} s. Time per shot: build + decode in these worker "
          f"processes" + (f", plus the network and head ({network * 1e3:.1f} ms mean, {meta['model']['device']}, "
                          f"from predecoder.evaluate)" if meta else ""))
    print(f"{'arm':6s} {'failures':>9s} {'share, 95% interval':>28s} {'unconverged':>11s} {'hard':>6s} {'mean iter':>9s} "
          f"{'build ms':>8s} {'decode ms':>9s} {'total ms':>8s} {'p99 ms':>7s}")
    for arm, row in results.items():
        print(f"{arm:6s} {row['failures']:9d} {row['share']:9.4f} [{row['wilson95'][0]:.4f}, {row['wilson95'][1]:.4f}] "
              f"{row['unconverged']:11d} {str(row['hard']):>6s} {row['mean_iterations']:9.0f} {row['build_seconds_mean'] * 1e3:8.1f} "
              f"{row['decode_seconds_mean'] * 1e3:9.1f} {row['seconds_mean'] * 1e3:8.1f} {row['seconds_p99'] * 1e3:7.0f}")
    for name, pair in pairs.items():
        a, b = name.split("_vs_")
        print(f"paired {a} against {b}: both fail {pair['both']}, only {a} {pair[f'only_{a}']}, only {b} {pair[f'only_{b}']}, "
              f"exact two-sided p = {pair['p']:.3g}")
        print(f"  iterations, shot by shot: {a} fewer on {pair[f'fewer_iterations_{a}']}, {b} fewer on "
              f"{pair[f'fewer_iterations_{b}']}, the same on {pair['same_iterations']}; p99 {pair[f'iterations_p99_{a}']:.0f} "
              f"against {pair[f'iterations_p99_{b}']:.0f}")
    if "head_vs_fresh" in pairs:
        pair = pairs["head_vs_fresh"]
        passed = pair["only_head"] < pair["only_fresh"] and pair["p"] < 0.05
        print("kill line (head against fresh: fewer failures, paired, p < 0.05): " + ("PASSED" if passed else "NOT PASSED"))
    np.savez_compressed(out / "per_shot.npz", **per_shot)
    with open(out / "summary.json", "w") as file:
        json.dump({"data": data.identity(), "shots": shots, "relay": settings(), "head_priors": prior_settings,
                   "kit_arm": f"one decoder built once per chunk of {kit_chunk} shots, decoding in file order",
                   "logits": meta, "arms": results, "pairs": pairs, "workers": args.workers, "machine": platform.node(),
                   "cpus": os.cpu_count()}, file, indent=1)


if __name__ == "__main__":
    main()
