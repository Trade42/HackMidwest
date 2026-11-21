"""Generate one deterministic synthetic mechanism shard for training or diagnostics.

These Bernoulli DEM samples are not Stim circuit shots and are not final test data.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.predecoder_layout import matrix_identity


def sample_mechanisms(H, L, priors, shots, seed):
    if shots <= 0 or seed < 0:
        raise ValueError("shots must be positive and seed nonnegative")
    rng = np.random.default_rng(seed)
    errors = (rng.random((shots, H.shape[1])) < priors).astype(np.uint8)
    syndrome = np.asarray((H @ errors.T).T % 2, dtype=np.uint8)
    observables = np.asarray((L @ errors.T).T % 2, dtype=np.uint8)
    return errors, syndrome, observables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", choices="XZ", required=True)
    parser.add_argument("--p", type=float, required=True)
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from circuit import GROSS_CIRCUITS, dem_matrices, load_circuit

    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"{output}: shard already exists")
    H, L, priors = dem_matrices(load_circuit(GROSS_CIRCUITS[args.basis], args.p), args.basis)
    errors, syndrome, observables = sample_mechanisms(H, L, priors, args.shots, args.seed)
    metadata = {"source_type": "independent_dem_mechanisms", "basis": args.basis, "p": args.p,
                "seed": args.seed, "shard": args.shard, "shots": args.shots,
                "matrix_sha256": matrix_identity(H, L, priors),
                "warning": "mechanism_truth is training/diagnostic only; never pass it to deployable decoders"}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, mechanism_truth=errors, basis_syndromes=syndrome,
                        observables=observables, metadata=json.dumps(metadata))
    print(json.dumps({"output": str(output), **{k: metadata[k] for k in ("basis", "p", "seed", "shard", "shots")}}))


if __name__ == "__main__":
    main()
