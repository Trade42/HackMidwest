"""Audit a paired saved-shot decoder comparison and report percent reductions.

The printed baselines.log is deliberately not accepted as a reference: it has
different sampled shots and no per-shot failure masks or timings.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ARRAYS = ("predictions_X", "predictions_Z", "failed_X", "failed_Z", "failed_pair",
          "seconds_X", "seconds_Z", "seconds_pair", "seconds_total_X", "seconds_total_Z",
          "seconds_total_pair", "kit_seconds_X", "kit_seconds_Z")
ENV_PACKAGES = ("numpy", "scipy", "stim", "ldpc", "relay-bp")
MACHINE_KEYS = ("system", "machine", "processor", "workers", "thread_environment")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary_path(path):
    path = Path(path)
    if path.is_dir():
        path = path / "summary.json"
    if path.name != "summary.json":
        raise ValueError("reference and candidate must be saved-shot summary.json files, not baselines.log")
    return path.resolve()


def _load_run(path, decoder):
    path = _summary_path(path)
    metadata = json.loads(path.read_text())
    entries = [entry for entry in metadata.get("decoders", []) if entry.get("name") == decoder]
    if len(entries) != 1:
        raise ValueError(f"{path}: expected exactly one completed {decoder!r} entry; found {len(entries)}")
    row = entries[0]["summary"]
    artifact = (path.parent / row["artifacts"]).resolve()
    if artifact.parent != path.parent:
        raise ValueError("shot artifact must be inside its result directory")
    with np.load(artifact, allow_pickle=False) as file:
        missing = set(ARRAYS) - set(file.files)
        if missing:
            raise ValueError(f"{artifact}: missing {sorted(missing)}; rerun with the updated evaluator")
        arrays = {key: file[key] for key in ARRAYS}
    return {"metadata": metadata, "row": row, "configuration": entries[0].get("configuration"),
            "arrays": arrays, "summary_path": path,
            "summary_sha256": _sha256(path), "artifact_path": artifact,
            "artifact_sha256": _sha256(artifact), "decoder": decoder}


def _check_same_experiment(reference, candidate):
    left, right = reference["metadata"], candidate["metadata"]
    if left.get("cycles") != right.get("cycles") or left.get("cycles") != 12 or left.get("p") != right.get("p"):
        raise ValueError("comparisons require the same p and 12 syndrome cycles")
    for basis in "XZ":
        a = left["datasets"]["test"][basis]
        b = right["datasets"]["test"][basis]
        for key in ("sha256", "basis", "p", "seed", "shots"):
            if a.get(key) != b.get(key):
                raise ValueError(f"{basis}: test datasets differ in {key}; paired gain is unavailable")
        circuit_a = (left.get("circuits") or {}).get(basis)
        circuit_b = (right.get("circuits") or {}).get(basis)
        if not circuit_a or not circuit_b or circuit_a.get("sha256") != circuit_b.get("sha256"):
            raise ValueError(f"{basis}: circuit identities are missing or differ")
    for key in MACHINE_KEYS:
        if left.get("machine", {}).get(key) != right.get("machine", {}).get(key):
            raise ValueError(f"machine setting {key} differs; rerun latency comparisons in one environment")
    for package in ENV_PACKAGES:
        if left.get("package_versions", {}).get(package) != right.get("package_versions", {}).get(package):
            raise ValueError(f"package {package} differs; rerun comparisons with matching versions")


def _truth(metadata, basis):
    identity = metadata["datasets"]["test"][basis]
    path = Path(identity["path"])
    if _sha256(path) != identity["sha256"]:
        raise ValueError(f"{basis}: test NPZ differs from the recorded dataset identity")
    with np.load(path, allow_pickle=False) as file:
        truth = file["observables"]
    if truth.shape != (identity["shots"], 12) or not np.isin(truth, [0, 1]).all():
        raise ValueError(f"{basis}: test observables have an invalid shape or values")
    return truth


def _binary(array, shape, name):
    if array.shape != shape or not np.isin(array, [0, 1]).all():
        raise ValueError(f"{name}: expected binary shape {shape}, got {array.shape}")
    return array.astype(bool, copy=False)


def _times(array, shape, name, allow_nan=False):
    valid = (np.isfinite(array) | (np.isnan(array) if allow_nan else False))
    if array.shape != shape or not valid.all() or (array[np.isfinite(array)] < 0).any():
        raise ValueError(f"{name}: expected finite nonnegative times of shape {shape}")
    return array


def _audit(run, truths):
    from decoders import per_cycle_rate

    arrays = run["arrays"]
    row = run["row"]
    n = row["paired_shots"]
    for basis in "XZ":
        predictions = _binary(arrays[f"predictions_{basis}"], (n, 12), f"predictions_{basis}")
        failed = _binary(arrays[f"failed_{basis}"], (n,), f"failed_{basis}")
        if not np.array_equal(failed, np.any(predictions != truths[basis], axis=1)):
            raise ValueError(f"{run['decoder']}: {basis} failure mask disagrees with saved truth")
        for prefix in ("seconds", "seconds_total", "kit_seconds"):
            _times(arrays[f"{prefix}_{basis}"], (n,), f"{prefix}_{basis}", allow_nan=prefix == "kit_seconds")
        if int(failed.sum()) != row[f"failures_{basis}"]:
            raise ValueError(f"{run['decoder']}: {basis} failure count disagrees with summary")
    pair = _binary(arrays["failed_pair"], (n,), "failed_pair")
    if not np.array_equal(pair, arrays["failed_X"] | arrays["failed_Z"]):
        raise ValueError(f"{run['decoder']}: paired failure mask is not X OR Z")
    for key in ("seconds", "seconds_total"):
        pair_times = _times(arrays[f"{key}_pair"], (n,), f"{key}_pair")
        if not np.allclose(pair_times, arrays[f"{key}_X"] + arrays[f"{key}_Z"], rtol=1e-12, atol=1e-12):
            raise ValueError(f"{run['decoder']}: paired {key} is not X + Z")
    if int(pair.sum()) != row["failures_pair"]:
        raise ValueError(f"{run['decoder']}: pair failure count disagrees with summary")
    rate, error = per_cycle_rate(pair, 12)
    if not np.isclose(rate, row["per_cycle_rate"], rtol=1e-12, atol=1e-12):
        raise ValueError(f"{run['decoder']}: kit per-cycle rate disagrees with summary")
    if not np.isclose(error, row["standard_error"], rtol=1e-12, atol=1e-12):
        raise ValueError(f"{run['decoder']}: kit standard error disagrees with summary")
    if not np.isclose(arrays["seconds_total_pair"].mean() * 1e3,
                      row["mean_pair_total_ms"], rtol=1e-12, atol=1e-9):
        raise ValueError(f"{run['decoder']}: total latency disagrees with summary")
    return float(rate), float(error)


def _percent_reduction(reference, candidate):
    return None if reference <= 0 else float(100 * (1 - candidate / reference))


def _paired_failure_report(reference, candidate, cycles, repetitions, seed):
    ref, new = reference.astype(bool), candidate.astype(bool)
    n = len(ref)
    both = int((ref & new).sum())
    reference_only = int((ref & ~new).sum())
    candidate_only = int((~ref & new).sum())
    neither = n - both - reference_only - candidate_only
    discordant = reference_only + candidate_only
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, np.array([both, reference_only, candidate_only, neither]) / n,
                             size=repetitions)
    ref_fraction = (counts[:, 0] + counts[:, 1]) / n
    new_fraction = (counts[:, 0] + counts[:, 2]) / n
    ref_rate = 1 - (1 - ref_fraction) ** (1 / cycles)
    new_rate = 1 - (1 - new_fraction) ** (1 / cycles)
    finite = ref_rate > 0
    interval = (np.percentile(100 * (1 - new_rate[finite] / ref_rate[finite]), [2.5, 97.5]).tolist()
                if finite.mean() >= .95 else None)
    return {"both_fail": both, "reference_only_fail": reference_only,
            "candidate_only_fail": candidate_only, "neither_fail": neither,
            "exact_discordant_sign_test_p": float(binomtest(candidate_only, discordant).pvalue)
            if discordant else 1.0,
            "per_cycle_rate_reduction_bootstrap_95pct": interval,
            "bootstrap_defined_fraction": float(finite.mean())}


def _paired_latency_interval(reference, candidate, repetitions, seed):
    n = len(reference)
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    for k in range(repetitions):
        ids = rng.integers(0, n, size=n)
        samples[k] = 100 * (1 - candidate[ids].mean() / reference[ids].mean())
    return np.percentile(samples, [2.5, 97.5]).tolist()


def compare(reference_path, candidate_path, candidate_decoder, reference_decoder="relay-bp",
            repetitions=2000, seed=0):
    if not 100 <= repetitions <= 10000 or seed < 0:
        raise ValueError("bootstrap repetitions must be 100..10000 and seed nonnegative")
    reference = _load_run(reference_path, reference_decoder)
    candidate = _load_run(candidate_path, candidate_decoder)
    _check_same_experiment(reference, candidate)
    truths = {basis: _truth(reference["metadata"], basis) for basis in "XZ"}
    ref_rate, ref_error = _audit(reference, truths)
    new_rate, new_error = _audit(candidate, truths)
    a, b = reference["arrays"], candidate["arrays"]
    ref_total, new_total = a["seconds_total_pair"], b["seconds_total_pair"]
    ref_mean, new_mean = float(ref_total.mean() * 1e3), float(new_total.mean() * 1e3)
    ref_p99, new_p99 = float(np.percentile(ref_total, 99) * 1e3), float(np.percentile(new_total, 99) * 1e3)
    paired = _paired_failure_report(a["failed_pair"], b["failed_pair"], 12, repetitions, seed)
    return {"classification": "held_out_paired_saved_shots",
            "note": "Percent reductions use the same saved test shots. Negative values mean the candidate is worse. Separate runs may still experience system-load drift.",
            "p": reference["metadata"]["p"], "cycles": 12,
            "paired_shots": len(a["failed_pair"]),
            "test_datasets": reference["metadata"]["datasets"]["test"],
            "machine": reference["metadata"].get("machine"),
            "package_versions": reference["metadata"].get("package_versions"),
            "reference": {"decoder": reference_decoder, "failures_pair": int(a["failed_pair"].sum()),
                          "per_cycle_rate": ref_rate, "standard_error": ref_error,
                          "mean_total_pair_ms": ref_mean, "p99_total_pair_ms": ref_p99,
                          "configuration": reference["configuration"],
                          "summary_path": str(reference["summary_path"]),
                          "summary_sha256": reference["summary_sha256"],
                          "artifact_sha256": reference["artifact_sha256"]},
            "candidate": {"decoder": candidate_decoder, "failures_pair": int(b["failed_pair"].sum()),
                          "per_cycle_rate": new_rate, "standard_error": new_error,
                          "mean_total_pair_ms": new_mean, "p99_total_pair_ms": new_p99,
                          "configuration": candidate["configuration"],
                          "summary_path": str(candidate["summary_path"]),
                          "summary_sha256": candidate["summary_sha256"],
                          "artifact_sha256": candidate["artifact_sha256"]},
            "per_cycle_rate_reduction_percent": _percent_reduction(ref_rate, new_rate),
            "mean_total_latency_reduction_percent": _percent_reduction(ref_mean, new_mean),
            "p99_total_latency_reduction_percent": _percent_reduction(ref_p99, new_p99),
            "paired_failures": paired,
            "mean_total_latency_reduction_bootstrap_95pct":
                _paired_latency_interval(ref_total, new_total, repetitions, seed) if ref_mean > 0 else None,
            "bootstrap": {"repetitions": repetitions, "seed": seed,
                          "method": "resample paired shots; failure counts use equivalent multinomial outcomes"}}


def export(report, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"{output}: comparison output already exists")
    output.mkdir(parents=True)
    (output / "comparison.json").write_text(json.dumps(report, indent=2))
    row = {"p": report["p"], "paired_shots": report["paired_shots"],
           "reference_decoder": report["reference"]["decoder"],
           "candidate_decoder": report["candidate"]["decoder"],
           "reference_failures_pair": report["reference"]["failures_pair"],
           "candidate_failures_pair": report["candidate"]["failures_pair"],
           "reference_per_cycle_rate": report["reference"]["per_cycle_rate"],
           "candidate_per_cycle_rate": report["candidate"]["per_cycle_rate"],
           "per_cycle_rate_reduction_percent": report["per_cycle_rate_reduction_percent"],
           "per_cycle_rate_reduction_ci_low_percent":
               report["paired_failures"]["per_cycle_rate_reduction_bootstrap_95pct"][0]
               if report["paired_failures"]["per_cycle_rate_reduction_bootstrap_95pct"] else None,
           "per_cycle_rate_reduction_ci_high_percent":
               report["paired_failures"]["per_cycle_rate_reduction_bootstrap_95pct"][1]
               if report["paired_failures"]["per_cycle_rate_reduction_bootstrap_95pct"] else None,
           "reference_mean_total_pair_ms": report["reference"]["mean_total_pair_ms"],
           "candidate_mean_total_pair_ms": report["candidate"]["mean_total_pair_ms"],
           "mean_total_latency_reduction_percent": report["mean_total_latency_reduction_percent"],
           "mean_total_latency_reduction_ci_low_percent":
               report["mean_total_latency_reduction_bootstrap_95pct"][0]
               if report["mean_total_latency_reduction_bootstrap_95pct"] else None,
           "mean_total_latency_reduction_ci_high_percent":
               report["mean_total_latency_reduction_bootstrap_95pct"][1]
               if report["mean_total_latency_reduction_bootstrap_95pct"] else None,
           "p99_total_latency_reduction_percent": report["p99_total_latency_reduction_percent"],
           "reference_only_fail": report["paired_failures"]["reference_only_fail"],
           "candidate_only_fail": report["paired_failures"]["candidate_only_fail"]}
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(row))
        writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-results", required=True, help="saved-shot result directory or summary.json; never baselines.log")
    parser.add_argument("--candidate-results", required=True, help="saved-shot result directory or summary.json")
    parser.add_argument("--reference-decoder", default="relay-bp")
    parser.add_argument("--candidate-decoder", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = compare(args.reference_results, args.candidate_results, args.candidate_decoder,
                     args.reference_decoder, args.bootstrap_repetitions, args.bootstrap_seed)
    export(report, args.out)
    print(json.dumps({"output": args.out, "p": report["p"],
                      "rate_reduction_percent": report["per_cycle_rate_reduction_percent"],
                      "total_latency_reduction_percent": report["mean_total_latency_reduction_percent"]}))


if __name__ == "__main__":
    main()
