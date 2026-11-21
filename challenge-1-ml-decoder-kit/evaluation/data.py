"""Load and validate the kit's saved memory-experiment datasets."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

FIELDS = {"detectors", "observables", "basis_detectors", "p", "seed", "basis"}


@dataclass(frozen=True)
class Dataset:
    path: Path
    sha256: str
    basis: str
    p: float
    seed: int
    detectors: np.ndarray
    observables: np.ndarray
    basis_detectors: np.ndarray

    @property
    def basis_input(self):
        return self.detectors[:, self.basis_detectors]

    def identity(self):
        return {"path": str(self.path), "sha256": self.sha256, "basis": self.basis,
                "p": self.p, "seed": self.seed, "shots": len(self.detectors),
                "source_format": "kit_make_dataset_npz_stim_circuit_samples"}


def _scalar(value, name, path):
    if value.shape != ():
        raise ValueError(f"{path}: {name} must be a scalar")
    return value.item()


def _binary(value, shape, name, path):
    if value.ndim != 2 or value.shape[1] != shape or value.shape[0] == 0:
        raise ValueError(f"{path}: {name} must have nonempty shape (shots, {shape}); got {value.shape}")
    if not np.issubdtype(value.dtype, np.integer) and value.dtype != np.bool_:
        raise ValueError(f"{path}: {name} must contain binary integers")
    if not np.isin(value, [0, 1]).all():
        raise ValueError(f"{path}: {name} contains values other than 0 and 1")
    return value.astype(np.uint8, copy=False)


def load_dataset(path, expected_basis=None, expected_ids=None):
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as file:
        missing = FIELDS - set(file.files)
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        basis = _scalar(file["basis"], "basis", path)
        p = _scalar(file["p"], "p", path)
        seed = _scalar(file["seed"], "seed", path)
        det = _binary(file["detectors"], 1728, "detectors", path)
        obs = _binary(file["observables"], 12, "observables", path)
        ids = file["basis_detectors"]
    if basis not in ("X", "Z") or (expected_basis and basis != expected_basis):
        raise ValueError(f"{path}: basis must be {expected_basis or 'X or Z'}; got {basis!r}")
    if not isinstance(p, (float, int)) or not np.isfinite(p) or not 0 < p < 1:
        raise ValueError(f"{path}: p must be a finite probability in (0, 1)")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"{path}: seed must be a nonnegative integer")
    if len(det) != len(obs):
        raise ValueError(f"{path}: detector and observable shot counts differ")
    if ids.shape != (936,) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(f"{path}: basis_detectors must contain 936 integer indices")
    if (ids < 0).any() or (ids >= 1728).any() or len(np.unique(ids)) != 936:
        raise ValueError(f"{path}: basis_detectors must be unique indices in [0, 1728)")
    if expected_ids is not None and not np.array_equal(ids, expected_ids):
        raise ValueError(f"{path}: basis_detectors do not match the circuit's {basis} checks")
    digest = sha256(path.read_bytes()).hexdigest()
    return Dataset(path, digest, basis, float(p), int(seed), det, obs, ids.astype(np.int64))


def validate_splits(splits):
    """splits maps split name to an X/Z pair; returns the shared test p."""
    seen_hashes, seen_sources = set(), set()
    test_p = None
    for name, pair in splits.items():
        if set(pair) != {"X", "Z"}:
            raise ValueError(f"{name}: both X and Z datasets are required")
        x, z = pair["X"], pair["Z"]
        if len(x.detectors) != len(z.detectors):
            raise ValueError(f"{name}: X and Z shot counts differ")
        if x.p != z.p:
            raise ValueError(f"{name}: X and Z noise probabilities differ")
        for data in pair.values():
            if data.sha256 in seen_hashes or (data.basis, data.p, data.seed) in seen_sources:
                raise ValueError(f"{name}: dataset is reused across splits or has the same basis/p/seed")
            seen_hashes.add(data.sha256)
            seen_sources.add((data.basis, data.p, data.seed))
        if name == "test":
            test_p = x.p
    return test_p
