"""Evaluate decoders on paired saved X/Z Gross memory shots."""
import argparse
import csv
import importlib.metadata
import json
import os
import platform
import resource
import sys
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

from evaluation.adapters import KIT_NAMES, KitSession, Problem, create_session, validate_shot_result
from evaluation.data import load_dataset, validate_splits

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
               "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS")
SUMMARY_FIELDS = ("decoder", "p", "cycles", "shots_X", "shots_Z", "paired_shots", "failures_X",
                  "failures_Z", "failures_pair", "per_cycle_rate", "standard_error", "mean_pair_ms",
                  "median_pair_ms", "p99_pair_ms", "mean_pair_kit_ms", "setup_seconds", "before_shot_seconds",
                  "decode_loop_seconds", "evaluation_wall_seconds", "batch_throughput_pairs_per_second",
                  "converged_fraction", "first_leg_converged_fraction", "fallback_fraction", "mean_iterations",
                  "threshold", "threshold_X", "threshold_Z", "artifacts")


def run_basis(session, data):
    """Process every shot once, in file order. Truth is never passed to a decoder."""
    n = len(data.detectors)
    predictions = np.empty((n, 12), dtype=np.uint8)
    seconds = np.empty(n, dtype=np.float64)
    kit_seconds = np.full(n, np.nan, dtype=np.float64)
    before_seconds = np.zeros(n, dtype=np.float64)
    metrics = []
    basis_input = data.basis_input
    for k in range(n):
        before = getattr(session, "before_shot", None)
        if before is not None:
            start = perf_counter()
            before(k)
            before_seconds[k] = perf_counter() - start
        start = perf_counter()
        result = session.decode_shot(data.detectors[k], basis_input[k])
        elapsed = perf_counter() - start
        predictions[k] = validate_shot_result(result)
        if isinstance(session, KitSession):
            seconds[k] = result.decode_seconds
            kit_seconds[k] = result.decode_seconds
        elif result.decode_seconds is not None:
            raise ValueError("plugins cannot override the full decode_shot timer")
        else:
            seconds[k] = elapsed
            reported_kit = result.metrics.get("kit_decode_seconds")
            if reported_kit is not None:
                kit_seconds[k] = float(reported_kit)
        metrics.append(result.metrics)
    failed = np.any(predictions != data.observables, axis=1)
    return {"predictions": predictions, "failed": failed, "seconds": seconds, "kit_seconds": kit_seconds,
            "before_seconds": before_seconds, "metrics": metrics}


def score_pair(x, z, cycles):
    from decoders import per_cycle_rate

    if len(x["failed"]) != len(z["failed"]) or not len(x["failed"]):
        raise ValueError("X and Z must have equal nonzero shot counts")
    failed_pair = x["failed"] | z["failed"]
    seconds_pair = x["seconds"] + z["seconds"]
    rate, error = per_cycle_rate(failed_pair, cycles)
    return failed_pair, seconds_pair, {"failures_X": int(x["failed"].sum()),
                                      "failures_Z": int(z["failed"].sum()),
                                      "failures_pair": int(failed_pair.sum()),
                                      "per_cycle_rate": float(rate), "standard_error": float(error),
                                      "mean_pair_ms": float(seconds_pair.mean() * 1e3),
                                      "median_pair_ms": float(np.median(seconds_pair) * 1e3),
                                      "p99_pair_ms": float(np.percentile(seconds_pair, 99) * 1e3)}


def _metric_mean(results, key):
    values = [shot[key] for basis in results.values() for shot in basis["metrics"] if key in shot]
    return float(np.mean(values)) if values else None


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _versions():
    result = {}
    for name in ("numpy", "scipy", "stim", "ldpc", "relay-bp", "matplotlib"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}


def _machine():
    return {"system": platform.system(), "release": platform.release(), "machine": platform.machine(),
            "processor": platform.processor(), "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(), "workers": 1,
            "thread_environment": {key: os.environ.get(key) for key in THREAD_VARS},
            "native_decoder_threads": "not exposed by the installed decoder API",
            "resource_note": "ru_maxrss is peak memory of this parent process only; it excludes other workers"}


def _peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _save_summaries(out_dir, summary, metadata):
    with (out_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    (out_dir / "summary.json").write_text(json.dumps(_jsonable(metadata), indent=2))


def evaluate(datasets, circuits, decoder_specs, out_dir, options=None, cycles=12, circuit_identities=None):
    """The core runner; caller provides validated saved datasets and circuit problems."""
    from circuit import dem_matrices

    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir}: output directory is not empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    test = datasets["test"]
    p = validate_splits(datasets)
    problems = {}
    for basis in "XZ":
        H, L, priors = dem_matrices(circuits[basis], basis)
        if H.shape != (936, 8784) or L.shape != (12, 8784):
            raise ValueError(f"{basis}: expected kit basis-specific H (936, 8784), L (12, 8784)")
        problems[basis] = Problem(basis, H, L, priors, p, circuits[basis])
    metadata = {"cycles": cycles, "p": p, "circuits": circuit_identities,
                "plugin_options": options or {},
                "datasets": {name: {b: d.identity() for b, d in pair.items()}
                                                   for name, pair in datasets.items()},
                "package_versions": _versions(), "machine": _machine(),
                "randomness": {"dataset_seeds": {name: {b: d.seed for b, d in pair.items()}
                                                 for name, pair in datasets.items()},
                               "relay_bp_seed": 0, "relay_bp_seed_source": "installed API default; kit factory leaves it unset",
                               "bposd_seed": None, "bposd_seed_note": "kit factory does not set a seed",
                               "decoder_order": list(decoder_specs)},
                "outcome_policy": "Every returned prediction is scored, including unconverged results; exceptions fail the run.",
                "timing": "Kit adapters time decoder API calls before observable projection, as in the kit. Plugins default to timing their complete decode_shot, including router and fallback work. before_shot and setup are separate; batch throughput uses the complete evaluation wall time.",
                "decoders": []}
    summary = []
    for index, spec in enumerate(decoder_specs, 1):
        start_wall = perf_counter()
        rss_before = _peak_rss()
        sessions = {basis: create_session(spec, problems[basis], options or {}) for basis in "XZ"}
        start_loop = perf_counter()
        results = {basis: run_basis(sessions[basis], test[basis]) for basis in "XZ"}
        loop_seconds = perf_counter() - start_loop
        failed_pair, seconds_pair, scored = score_pair(results["X"], results["Z"], cycles)
        wall_seconds = perf_counter() - start_wall
        stem = f"{index:02d}_{spec.replace(':', '_').replace('.', '_').replace('/', '_')}"
        artifact = stem + "_shots.npz"
        np.savez_compressed(out_dir / artifact,
                            predictions_X=results["X"]["predictions"], predictions_Z=results["Z"]["predictions"],
                            failed_X=results["X"]["failed"], failed_Z=results["Z"]["failed"],
                            failed_pair=failed_pair, seconds_X=results["X"]["seconds"],
                            seconds_Z=results["Z"]["seconds"], seconds_pair=seconds_pair,
                            kit_seconds_X=results["X"]["kit_seconds"], kit_seconds_Z=results["Z"]["kit_seconds"],
                            before_shot_seconds_X=results["X"]["before_seconds"],
                            before_shot_seconds_Z=results["Z"]["before_seconds"])
        metrics_path = stem + "_metrics.json"
        (out_dir / metrics_path).write_text(json.dumps(_jsonable({b: results[b]["metrics"] for b in "XZ"}), indent=2))
        thresholds = {b: getattr(sessions[b], "settings", {}).get("risk_threshold") for b in "XZ"}
        shared_threshold = (options or {}).get("threshold")
        if shared_threshold is None and thresholds["X"] == thresholds["Z"]:
            shared_threshold = thresholds["X"]
        row = {"decoder": spec, "p": p, "cycles": cycles, "shots_X": len(test["X"].detectors),
               "shots_Z": len(test["Z"].detectors), "paired_shots": len(failed_pair), **scored,
               "mean_pair_kit_ms": float(np.mean(results["X"]["kit_seconds"] + results["Z"]["kit_seconds"]) * 1e3)
               if np.isfinite(results["X"]["kit_seconds"]).all() and np.isfinite(results["Z"]["kit_seconds"]).all() else None,
               "setup_seconds": float(sum(s.setup_seconds for s in sessions.values())),
               "before_shot_seconds": float(sum(r["before_seconds"].sum() for r in results.values())),
               "decode_loop_seconds": loop_seconds, "evaluation_wall_seconds": wall_seconds,
               "batch_throughput_pairs_per_second": len(failed_pair) / wall_seconds,
               "converged_fraction": _metric_mean(results, "converged"),
               "first_leg_converged_fraction": _metric_mean(results, "first_leg_converged"),
               "fallback_fraction": _metric_mean(results, "fallback_used"),
               "mean_iterations": _metric_mean(results, "iterations"),
               "threshold": shared_threshold, "threshold_X": thresholds["X"],
               "threshold_Z": thresholds["Z"], "artifacts": artifact}
        summary.append(row)
        metadata["decoders"].append({"name": spec, "configuration": {b: getattr(s, "settings", None) for b, s in sessions.items()},
                                     "setup_seconds_by_basis": {b: s.setup_seconds for b, s in sessions.items()},
                                     "peak_parent_rss_bytes_before": rss_before,
                                     "peak_parent_rss_bytes_after": _peak_rss(),
                                     "shot_metrics": metrics_path, "summary": row})
        _save_summaries(out_dir, summary, metadata)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "validation", "test"):
        for basis in "XZ":
            parser.add_argument(f"--{split}-{basis.lower()}", required=split == "test")
    parser.add_argument("--circuit-x", default=None)
    parser.add_argument("--circuit-z", default=None)
    parser.add_argument("--decoder", action="append", dest="decoders", help=f"kit name {KIT_NAMES} or module:function")
    parser.add_argument("--plugin-options", help="JSON object passed to plugin factories")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from circuit import GROSS_CIRCUITS, detector_ids, load_circuit

    paths = {"X": args.circuit_x or GROSS_CIRCUITS["X"], "Z": args.circuit_z or GROSS_CIRCUITS["Z"]}
    options = json.loads(Path(args.plugin_options).read_text()) if args.plugin_options else {}
    if not isinstance(options, dict):
        parser.error("--plugin-options must contain a JSON object")
    datasets = {}
    for split in ("train", "validation", "test"):
        pair_paths = {b: getattr(args, f"{split}_{b.lower()}") for b in "XZ"}
        if any(pair_paths.values()):
            if not all(pair_paths.values()):
                parser.error(f"{split}: supply both X and Z datasets")
            # Read metadata to learn p before building a potentially large matrix.
            pair = {b: load_dataset(pair_paths[b], b) for b in "XZ"}
            datasets[split] = pair
    p = validate_splits(datasets)
    circuits = {b: load_circuit(paths[b], p) for b in "XZ"}
    for pair in datasets.values():
        for b in "XZ":
            expected = detector_ids(circuits[b], b)
            if not np.array_equal(pair[b].basis_detectors, expected):
                raise ValueError(f"{pair[b].path}: basis_detectors do not match {b} circuit")
    output = Path(args.out)
    if output.resolve() in {d.path for pair in datasets.values() for d in pair.values()}:
        parser.error("output directory must differ from input datasets")
    rows = evaluate(datasets, circuits, args.decoders or ["bposd", "relay-bp"], output, options,
                    circuit_identities={b: _identity(paths[b]) for b in "XZ"})
    print(f"saved {len(rows)} decoder summaries in {output}")


if __name__ == "__main__":
    main()
