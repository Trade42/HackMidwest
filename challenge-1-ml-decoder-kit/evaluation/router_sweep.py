"""Measure a validation-only risk-threshold curve with actual per-shot reruns."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluation.adapters import Problem
from evaluation.data import load_dataset, validate_splits
from evaluation.evaluate import _identity, _machine, _versions, run_basis, score_pair
from evaluation.router import TieredSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-x", required=True)
    parser.add_argument("--validation-z", required=True)
    parser.add_argument("--model-x", required=True)
    parser.add_argument("--model-z", required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from circuit import GROSS_CIRCUITS, dem_matrices, detector_ids, load_circuit

    datasets = {"validation": {"X": load_dataset(args.validation_x, "X"),
                               "Z": load_dataset(args.validation_z, "Z")}}
    p = validate_splits(datasets)
    problems = {}
    for basis in "XZ":
        circuit = load_circuit(GROSS_CIRCUITS[basis], p)
        if not np.array_equal(datasets["validation"][basis].basis_detectors, detector_ids(circuit, basis)):
            raise ValueError(f"{basis}: validation detector indices differ from kit circuit")
        H, L, priors = dem_matrices(circuit, basis)
        problems[basis] = Problem(basis, H, L, priors, p, circuit)
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"{output}: output already exists")
    output.mkdir(parents=True)
    rows = []
    for index, threshold in enumerate(args.thresholds):
        if not 0 <= threshold <= 1:
            raise ValueError("router risk thresholds must be in [0,1]")
        sessions = {"X": TieredSession(problems["X"], args.model_x),
                    "Z": TieredSession(problems["Z"], args.model_z)}
        for session in sessions.values():
            session.model["threshold"] = threshold  # validation-only sweep; frozen files remain unchanged
            session.settings["risk_threshold"] = threshold
        results = {basis: run_basis(sessions[basis], datasets["validation"][basis]) for basis in "XZ"}
        failed, seconds, scored = score_pair(results["X"], results["Z"], 12)
        row = {"decoder": "tiered-router-validation", "validation_only": True, "p": p,
               "threshold": threshold, "paired_shots": len(failed), **scored}
        rows.append(row)
        np.savez_compressed(output / f"threshold_{index:02d}_shots.npz",
                            failed_X=results["X"]["failed"], failed_Z=results["Z"]["failed"],
                            failed_pair=failed, predictions_X=results["X"]["predictions"],
                            predictions_Z=results["Z"]["predictions"], seconds_X=results["X"]["seconds"],
                            seconds_Z=results["Z"]["seconds"], seconds_pair=seconds)
        with (output / "summary.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / "manifest.json").write_text(json.dumps({"classification": "validation_threshold_sweep_not_final_test",
                                                       "datasets": {b: d.identity() for b, d in datasets["validation"].items()},
                                                       "models": {"X": _identity(args.model_x), "Z": _identity(args.model_z)},
                                                       "package_versions": _versions(), "machine": _machine(),
                                                       "thresholds": args.thresholds}, indent=2))
    print(json.dumps({"output": str(output), "rows": len(rows), "validation_only": True}))


if __name__ == "__main__":
    main()
