"""Validation-only commit threshold selection and X p=.004 held-out go/no-go screen."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from evaluation.adapters import Problem
from evaluation.data import load_dataset
from evaluation.evaluate import _machine, _peak_rss, _versions, run_basis
from evaluation.predecoder import ComposedSession, FrozenCommitter, model_factory_sha256
from evaluation.predecoder_layout import matrix_identity
from evaluation.router import TieredSession


def _problem(basis, p):
    from circuit import GROSS_CIRCUITS, dem_matrices, detector_ids, load_circuit

    circuit = load_circuit(GROSS_CIRCUITS[basis], p)
    H, L, priors = dem_matrices(circuit, basis)
    return Problem(basis, H, L, priors, p, circuit), detector_ids(circuit, basis)


def _summary(result):
    metrics = result["metrics"]
    n = len(metrics)
    failed = result["failed"]
    hard = sum(not x["first_leg_converged"] for x in metrics)
    iterations = [x["iterations"] for x in metrics if x.get("iterations") is not None]
    bp_iterations = [x["bp_context_iterations"] for x in metrics if x.get("bp_context_iterations") is not None]
    ci = binomtest(int(failed.sum()), n).proportion_ci(confidence_level=.95, method="exact")
    return {"shots": n, "failures": int(failed.sum()), "failure_fraction": float(failed.mean()),
            "failure_95pct_interval": [float(ci.low), float(ci.high)],
            "first_leg_hard_shots": hard, "first_leg_hard_fraction": hard / n,
            "mean_relay_iterations": float(np.mean(iterations)) if iterations else None,
            "mean_bp_context_iterations": float(np.mean(bp_iterations)) if bp_iterations else None,
            "mean_total_ms": float(result["seconds"].mean() * 1e3),
            "median_total_ms": float(np.median(result["seconds"]) * 1e3),
            "p99_total_ms": float(np.percentile(result["seconds"], 99) * 1e3),
            "mean_kit_decode_ms": float(np.mean(result["kit_seconds"]) * 1e3)
            if np.isfinite(result["kit_seconds"]).all() else None}


def _paired(candidate, control):
    a, b = candidate["failed"], control["failed"]
    n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum())
    diff = a.astype(np.int8) - b.astype(np.int8)
    half_width = 1.96 * float(np.std(diff, ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
    return {"candidate_only_failures": n10, "control_only_failures": n01,
            "paired_failure_fraction_difference": float(diff.mean()),
            "approximate_95pct_difference_interval": [float(diff.mean() - half_width), float(diff.mean() + half_width)],
            "exact_discordant_sign_test_p": float(binomtest(n10, n10 + n01).pvalue) if n10 + n01 else 1.0}


def promotion_gate(base, bp, cnn):
    """The approved screening requirements on identical 5,000 X shots."""
    return {"hard_fraction_reduction_at_least_one_third": base["first_leg_hard_fraction"] > 0 and
            cnn["first_leg_hard_fraction"] <= (2 / 3) * base["first_leg_hard_fraction"],
            "no_more_observed_failures_than_no_predecoder": cnn["failures"] <= base["failures"],
            "improves_bp_control_hardness_without_more_observed_failures":
            cnn["first_leg_hard_fraction"] < bp["first_leg_hard_fraction"] and cnn["failures"] <= bp["failures"],
            "measured_total_latency_improves": cnn["mean_total_ms"] < base["mean_total_ms"]}


def _load_saved(path, basis, ids):
    return load_dataset(path, basis, ids)


def select(args):
    # This command only opens validation shots. No final test path is accepted.
    problem, ids = _problem(args.basis, args.p)
    data = _load_saved(args.validation, args.basis, ids)
    if data.p != args.p:
        raise ValueError("validation p differs from requested p")
    base = run_basis(TieredSession(problem, args.router_model), data)
    router_metadata = json.loads(Path(args.router_model).read_text())
    if router_metadata.get("train_dataset", {}).get("sha256") == data.sha256:
        raise ValueError("commit validation data overlaps router training data")
    baseline_summary = _summary(base)
    model_hash = hashlib.sha256(Path(args.model_path).read_bytes()).hexdigest() if args.mode == "cnn" else None
    model_config = json.loads(Path(args.model_config).read_text()) if args.model_config else {}
    if not isinstance(model_config, dict):
        raise ValueError("model config must be a JSON object")
    training_sources = []
    if args.mode == "cnn":
        training_record = json.loads(Path(args.model_training_manifest).read_text())
        if training_record.get("basis") != args.basis or training_record.get("p") != args.p:
            raise ValueError("CNN training manifest basis or p mismatch")
        if training_record.get("model_sha256") != model_hash:
            raise ValueError("CNN training manifest model hash mismatch")
        training_sources = training_record.get("training_datasets", [])
        if not training_sources or any(source.get("sha256") == data.sha256 or source.get("seed") == data.seed
                                       for source in training_sources):
            raise ValueError("CNN training sources are missing or overlap validation")
    template = {"selection_stage": "validation_candidate", "mode": args.mode, "basis": args.basis,
                "p": args.p, "matrix_sha256": matrix_identity(problem.H, problem.L, problem.priors),
                "model_factory": args.model_factory,
                "model_factory_sha256": model_factory_sha256(args.model_factory) if args.mode == "cnn" else None,
                "model_path": args.model_path,
                "model_sha256": model_hash, "model_config": model_config,
                "training_datasets": training_sources, "output_kind": args.output_kind,
                "use_bp_context": args.use_bp_context}
    reports = []
    for threshold in args.thresholds:
        config = template | {"threshold": threshold}
        committer = FrozenCommitter(problem, validation_candidate=config)
        session = ComposedSession(problem, {"router_models": {args.basis: args.router_model}},
                                  validation_committer=committer)
        result = run_basis(session, data)
        reports.append({"threshold": threshold, "summary": _summary(result),
                        "paired_vs_no_predecoder": _paired(result, base)})
    eligible = [r for r in reports if r["summary"]["failures"] <= baseline_summary["failures"] + args.max_added_failures]
    if not eligible:
        raise ValueError("no validation threshold satisfies the supplied added-failure budget")
    chosen = min(eligible, key=lambda r: (r["summary"]["mean_total_ms"], r["summary"]["failures"]))
    manifest = (template | {"selection_stage": "validation", "threshold": chosen["threshold"],
                            "validation_dataset": data.identity(), "router_model_sha256": hashlib.sha256(Path(args.router_model).read_bytes()).hexdigest(),
                            "selection_rule": "lowest measured validation mean total latency among thresholds within max_added_failures of no-predecoder",
                            "max_added_failures": args.max_added_failures, "validation_no_predecoder": baseline_summary,
                            "validation_candidates": reports})
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"{output}: frozen manifest already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2))
    return {"output": str(output), "selected_threshold": chosen["threshold"],
            "validation_failures": chosen["summary"]["failures"], "candidate_thresholds": len(reports)}


def go(args):
    # Construct and hash every model/decoder before the held-out file is opened.
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"{output}: output already exists")
    problem, ids = _problem("X", .004)
    no_pre_session = TieredSession(problem, args.router_model)
    options = {"router_models": {"X": args.router_model}}
    bp_session = ComposedSession(problem, options | {"commit_manifests": {"X": args.bp_manifest}})
    cnn_session = ComposedSession(problem, options | {"commit_manifests": {"X": args.cnn_manifest}})
    if bp_session.committer.mode != "bp" or cnn_session.committer.mode != "cnn":
        raise ValueError("go/no-go needs one frozen BP control and one frozen CNN committer")
    router_hash = hashlib.sha256(Path(args.router_model).read_bytes()).hexdigest()
    for session in (bp_session, cnn_session):
        expected = session.committer.manifest.get("router_model_sha256")
        if expected and expected != router_hash:
            raise ValueError("downstream router differs from the validation-frozen router")
    data = _load_saved(args.test, "X", ids)
    if data.p != .004 or len(data.detectors) != 5000:
        raise ValueError("go/no-go requires exactly 5,000 X-memory test shots at p=0.004")
    for manifest in (bp_session.committer.manifest, cnn_session.committer.manifest, no_pre_session.model):
        for key in ("train_dataset", "validation_dataset"):
            previous = manifest.get(key)
            if previous and (previous["sha256"] == data.sha256 or previous["seed"] == data.seed):
                raise ValueError("held-out test duplicates a training or validation source")
        for previous in manifest.get("training_datasets", []):
            if previous["sha256"] == data.sha256 or previous["seed"] == data.seed:
                raise ValueError("held-out test duplicates a CNN training source")
    sessions = {"no_predecoder": no_pre_session, "bp_confidence": bp_session, "learned_predecoder": cnn_session}
    rss_before = _peak_rss()
    results = {name: run_basis(session, data) for name, session in sessions.items()}
    rss_after = _peak_rss()
    summaries = {name: _summary(result) for name, result in results.items()}
    base, bp, cnn = (summaries[name] for name in sessions)
    gate = promotion_gate(base, bp, cnn)
    report = {"experiment": "X memory p=0.004, 5000 held-out circuit shots", "test_dataset": data.identity(),
              "package_versions": _versions(), "machine": _machine(),
              "peak_parent_rss_bytes_before": rss_before, "peak_parent_rss_bytes_after": rss_after,
              "setup_seconds": {name: session.setup_seconds for name, session in sessions.items()},
              "devices": {"no_predecoder": "cpu", "bp_confidence": "cpu",
                          "learned_predecoder": cnn_session.committer.settings.get("device", "unknown")},
              "router_sha256": router_hash,
              "bp_manifest_sha256": hashlib.sha256(Path(args.bp_manifest).read_bytes()).hexdigest(),
              "cnn_manifest_sha256": hashlib.sha256(Path(args.cnn_manifest).read_bytes()).hexdigest(),
              "screens_only": "No more observed failures does not prove equal underlying error rates.",
              "pipelines": {"no_predecoder": {"preprocessor_ml": False, "router_ml": True},
                            "bp_confidence": {"preprocessor_ml": False, "router_ml": True},
                            "learned_predecoder": {"preprocessor_ml": True, "router_ml": True}},
              "summaries": summaries,
              "paired_vs_no_predecoder": _paired(results["learned_predecoder"], results["no_predecoder"]),
              "paired_vs_bp_confidence": _paired(results["learned_predecoder"], results["bp_confidence"]),
              "gate": gate, "promote_tier0": all(gate.values()),
              "timing": "total per-shot includes tier0, memory-BP context, host-synchronized model output, residual transform, router, decoding and reruns; construction is separate"}
    output.mkdir(parents=True)
    for name, result in results.items():
        np.savez_compressed(output / f"{name}_shots.npz", predictions=result["predictions"],
                            failed=result["failed"], seconds_total=result["seconds"],
                            seconds_kit_decode=result["kit_seconds"])
        (output / f"{name}_metrics.json").write_text(json.dumps(result["metrics"], indent=2))
    (output / "go_no_go.json").write_text(json.dumps(report, indent=2))
    return {"output": str(output), "promote_tier0": report["promote_tier0"], "gate": gate}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    select_parser = sub.add_parser("select", help="select commit threshold on validation only")
    select_parser.add_argument("--mode", choices=("bp", "cnn"), required=True)
    select_parser.add_argument("--basis", choices="XZ", required=True)
    select_parser.add_argument("--p", type=float, required=True)
    select_parser.add_argument("--validation", required=True)
    select_parser.add_argument("--router-model", required=True)
    select_parser.add_argument("--thresholds", type=float, nargs="+", required=True)
    select_parser.add_argument("--max-added-failures", type=int, required=True)
    select_parser.add_argument("--model-factory")
    select_parser.add_argument("--model-path")
    select_parser.add_argument("--model-config", help="optional JSON object passed to the CNN factory")
    select_parser.add_argument("--model-training-manifest", help="CNN training provenance JSON")
    select_parser.add_argument("--output-kind", choices=("probability", "logit"), default="probability")
    select_parser.add_argument("--use-bp-context", action="store_true")
    select_parser.add_argument("--out", required=True)
    go_parser = sub.add_parser("go", help="open the held-out 5000-shot X test only after freezing configurations")
    go_parser.add_argument("--test", required=True)
    go_parser.add_argument("--router-model", required=True)
    go_parser.add_argument("--bp-manifest", required=True)
    go_parser.add_argument("--cnn-manifest", required=True)
    go_parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.stage == "select":
        if args.max_added_failures < 0:
            parser.error("--max-added-failures must be nonnegative")
        if args.mode == "cnn" and (not args.model_factory or not args.model_path or not args.model_training_manifest):
            parser.error("CNN selection needs --model-factory, --model-path, and --model-training-manifest")
        result = select(args)
    else:
        result = go(args)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
