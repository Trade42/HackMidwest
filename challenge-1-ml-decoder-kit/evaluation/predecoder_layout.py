"""Translation-verified Gross-code detector and error-mechanism grids.

The kit orders the 936 basis detectors as 13 blocks of 72 checks, and each
check block uses the 12 x 6 torus index from make_circuit.py. Fault columns
are located by translating their H-column detector support, never by reshape.
"""
import argparse
import hashlib
import json

import numpy as np


def matrix_identity(H, L, priors):
    digest = hashlib.sha256()
    for matrix in (H.tocsc(), L.tocsc()):
        for value in (matrix.shape, matrix.indptr, matrix.indices, matrix.data):
            digest.update(np.asarray(value).tobytes())
    digest.update(np.asarray(priors, dtype=np.float64).tobytes())
    return digest.hexdigest()


class GrossLayout:
    syndrome_shape = (13, 12, 6)
    fault_shape = (122, 12, 6)

    def __init__(self, H, L, priors):
        if H.shape != (936, 8784) or L.shape != (12, 8784) or np.shape(priors) != (8784,):
            raise ValueError("Gross layout needs H (936,8784), L (12,8784), priors (8784,)")
        self.H, self.L = H.tocsc(), L.tocsr()
        self.priors = np.asarray(priors, dtype=np.float64)
        self.identity = matrix_identity(H, L, priors)
        supports = [tuple(map(int, self.H.indices[self.H.indptr[j]:self.H.indptr[j + 1]]))
                    for j in range(8784)]
        lookup = {support: j for j, support in enumerate(supports)}
        if len(lookup) != 8784:
            raise ValueError("H columns do not have unique detector supports")
        self.detector_permutations = {}
        self.fault_permutations = {}
        for dx in range(12):
            for dy in range(6):
                row_perm = np.array([t * 72 + ((i // 6 + dx) % 12) * 6 + (i % 6 + dy) % 6
                                     for t in range(13) for i in range(72)], dtype=np.int32)
                col_perm = np.empty(8784, dtype=np.int32)
                for j, support in enumerate(supports):
                    translated = tuple(sorted(int(row_perm[i]) for i in support))
                    try:
                        col_perm[j] = lookup[translated]
                    except KeyError as exc:
                        raise ValueError(f"H is not spatially translation invariant at ({dx},{dy})") from exc
                if len(np.unique(row_perm)) != 936 or len(np.unique(col_perm)) != 8784:
                    raise ValueError(f"translation ({dx},{dy}) is not a permutation")
                if not np.allclose(self.priors, self.priors[col_perm], rtol=0, atol=1e-12):
                    raise ValueError(f"priors vary under translation ({dx},{dy})")
                # This checks H[row_perm[i], col_perm[j]] = H[i, j] for every nonzero.
                for j, support in enumerate(supports):
                    target = supports[col_perm[j]]
                    if target != tuple(sorted(int(row_perm[i]) for i in support)):
                        raise ValueError(f"translation ({dx},{dy}) does not preserve H")
                self.detector_permutations[(dx, dy)] = row_perm
                self.fault_permutations[(dx, dy)] = col_perm
        grid = np.empty(self.fault_shape, dtype=np.int32)
        assigned = set()
        group = 0
        for j in range(8784):
            if j in assigned:
                continue
            orbit = [int(self.fault_permutations[(dx, dy)][j]) for dx in range(12) for dy in range(6)]
            if len(set(orbit)) != 72 or set(orbit) & assigned:
                raise ValueError("fault translation orbit is incomplete or overlaps")
            if group >= 122:
                raise ValueError("more than 122 fault translation orbits")
            grid[group] = np.asarray(orbit, dtype=np.int32).reshape(12, 6)
            assigned.update(orbit)
            group += 1
        if group != 122 or len(assigned) != 8784 or not np.array_equal(np.sort(grid.ravel()), np.arange(8784)):
            raise ValueError("fault grid does not map each column exactly once")
        self.fault_indices = grid
        self.prior_grid = self.vector_to_fault_grid(self.priors)

    def vector_to_fault_grid(self, vector):
        value = np.asarray(vector)
        if value.shape != (8784,):
            raise ValueError("fault vector must have 8784 entries")
        return value[self.fault_indices]

    def fault_grid_to_vector(self, grid):
        value = np.asarray(grid)
        if value.shape != self.fault_shape:
            raise ValueError("fault grid must have shape (122,12,6)")
        result = np.empty(8784, dtype=value.dtype)
        result[self.fault_indices] = value
        return result

    def vector_to_syndrome_grid(self, vector):
        value = np.asarray(vector)
        if value.shape != (936,):
            raise ValueError("syndrome vector must have 936 entries")
        return value.reshape(self.syndrome_shape)

    def syndrome_grid_to_vector(self, grid):
        value = np.asarray(grid)
        if value.shape != self.syndrome_shape:
            raise ValueError("syndrome grid must have shape (13,12,6)")
        return value.reshape(936)

    def verify_roundtrips(self):
        faults = np.arange(8784)
        syndrome = np.arange(936)
        if not np.array_equal(self.fault_grid_to_vector(self.vector_to_fault_grid(faults)), faults):
            raise ValueError("fault grid round trip failed")
        if not np.array_equal(self.syndrome_grid_to_vector(self.vector_to_syndrome_grid(syndrome)), syndrome):
            raise ValueError("syndrome grid round trip failed")
        return True


def circular_pad_spatial(grid, padding):
    """Wrap only the last two (12 x 6) spatial axes; never wrap syndrome time."""
    value = np.asarray(grid)
    if value.ndim < 3 or padding < 0:
        raise ValueError("expected a grid with time and two spatial axes, and nonnegative padding")
    widths = [(0, 0)] * value.ndim
    widths[-2:] = [(padding, padding), (padding, padding)]
    return np.pad(value, widths, mode="wrap")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", choices="XZ", required=True)
    parser.add_argument("--p", type=float, default=.004)
    args = parser.parse_args()
    from circuit import GROSS_CIRCUITS, dem_matrices, load_circuit

    H, L, priors = dem_matrices(load_circuit(GROSS_CIRCUITS[args.basis], args.p), args.basis)
    layout = GrossLayout(H, L, priors)
    layout.verify_roundtrips()
    print(json.dumps({"basis": args.basis, "p": args.p, "matrix_sha256": layout.identity,
                      "syndrome_shape": layout.syndrome_shape, "fault_shape": layout.fault_shape,
                      "translations_verified": 72, "fault_columns_mapped": 8784}))


if __name__ == "__main__":
    main()
