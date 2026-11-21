"""Inspect synthetic mechanism-truth commits; never report competitive LER."""
import argparse
import json

import numpy as np

from evaluation.predecoder import oracle_commits_for_diagnostic, residual_syndrome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-shard", required=True)
    args = parser.parse_args()
    with np.load(args.synthetic_shard, allow_pickle=False) as file:
        metadata = json.loads(str(file["metadata"].item()))
        if metadata.get("source_type") != "independent_dem_mechanisms":
            raise ValueError("oracle diagnostic accepts only synthetic DEM shards")
        errors = file["mechanism_truth"]
        syndromes = file["basis_syndromes"]
        observables = file["observables"]
    from circuit import GROSS_CIRCUITS, dem_matrices, load_circuit

    H, L, priors = dem_matrices(load_circuit(GROSS_CIRCUITS[metadata["basis"]], metadata["p"]), metadata["basis"])
    from evaluation.predecoder_layout import matrix_identity

    if matrix_identity(H, L, priors) != metadata["matrix_sha256"]:
        raise ValueError("synthetic shard matrix identity mismatch")
    for e, syndrome, truth in zip(errors, syndromes, observables, strict=True):
        commits = oracle_commits_for_diagnostic(e)
        if residual_syndrome(H, syndrome, commits).any() or np.any((L @ commits) % 2 != truth):
            raise ValueError("synthetic shard fails its own algebraic oracle identity")
    print(json.dumps({"classification": "oracle_diagnostic_only_not_competitive",
                      "source_type": metadata["source_type"], "shots_checked": len(errors),
                      "matrix_sha256": metadata["matrix_sha256"]}))


if __name__ == "__main__":
    main()
