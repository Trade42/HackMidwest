"""Generate first-solution Relay-BP router features from saved train/validation shots."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from evaluation.adapters import _prediction
from evaluation.data import load_dataset
from evaluation.predecoder_layout import matrix_identity, matrix_semantic_identity
from evaluation.router import FEATURE_NAMES, candidate_features, first_leg_converged


def generate(data, H, L, priors, committer=None):
    from decoders import relay_decoder

    start = perf_counter()
    decoder = relay_decoder(H, priors, stop_nconv=1)
    setup_seconds = perf_counter() - start
    n = len(data.detectors)
    features = np.empty((n, len(FEATURE_NAMES)), dtype=np.float64)
    candidate = np.empty((n, 12), dtype=np.uint8)
    wrong = np.empty(n, dtype=bool)
    easy = np.empty(n, dtype=bool)
    seconds = np.empty(n, dtype=np.float64)
    for k, original_syndrome in enumerate(data.basis_input):
        if committer is not None:
            from evaluation.predecoder import residual_syndrome

            commits, _ = committer.commit(original_syndrome, data.detectors[k])
            syndrome = residual_syndrome(H, original_syndrome, commits)
        else:
            syndrome = original_syndrome
        start = perf_counter()
        result = decoder.decode_detailed(syndrome)
        seconds[k] = perf_counter() - start
        features[k] = candidate_features(result, syndrome)
        residual_prediction = _prediction(L, result.decoding)
        if committer is not None:
            from evaluation.predecoder import combine_observables

            candidate[k] = combine_observables(L, commits, residual_prediction)
        else:
            candidate[k] = residual_prediction
        wrong[k] = bool(np.any(candidate[k] != data.observables[k]))
        easy[k] = first_leg_converged(result)
    return {"features": features, "candidate_predictions": candidate, "wrong": wrong,
            "first_leg_converged": easy, "decode_seconds": seconds, "setup_seconds": setup_seconds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--basis", choices="XZ", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--commit-manifest", help="optional frozen tier-0 commits; labels include their logical frame")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from circuit import GROSS_CIRCUITS, dem_matrices, detector_ids, load_circuit

    data = load_dataset(args.data, args.basis)
    circuit = load_circuit(GROSS_CIRCUITS[args.basis], data.p)
    if not np.array_equal(data.basis_detectors, detector_ids(circuit, args.basis)):
        raise ValueError("dataset basis detector indices do not match the kit circuit")
    H, L, priors = dem_matrices(circuit, args.basis)
    committer = None
    commit_hash = None
    if args.commit_manifest:
        from evaluation.adapters import Problem
        from evaluation.predecoder import FrozenCommitter

        committer = FrozenCommitter(Problem(args.basis, H, L, priors, data.p, circuit), args.commit_manifest)
        commit_hash = hashlib.sha256(Path(args.commit_manifest).read_bytes()).hexdigest()
    output = Path(args.out)
    if output.exists() or output.resolve() == data.path:
        raise FileExistsError(f"{output}: output already exists")
    result = generate(data, H, L, priors, committer)
    metadata = {"split": args.split, "basis": args.basis, "p": data.p, "dataset": data.identity(),
                "matrix_sha256": matrix_identity(H, L, priors),
                "matrix_semantic_sha256": matrix_semantic_identity(H, L, priors),
                "feature_names": list(FEATURE_NAMES),
                "commit_manifest_sha256": commit_hash,
                "setup_seconds": result.pop("setup_seconds"),
                "label": "wrong = any(first_candidate_observables != truth); truth used only after decoding"}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result, metadata=json.dumps(metadata))
    print(json.dumps({"output": str(output), "shots": len(data.detectors), "hard_shots": int((~result["first_leg_converged"]).sum()),
                      "wrong_candidates": int(result["wrong"].sum())}))


if __name__ == "__main__":
    main()
