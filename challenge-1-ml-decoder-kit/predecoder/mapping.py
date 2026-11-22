"""From the network's fault mechanisms to the columns of the decoders' H.

The network is trained on shots drawn from the FULL detector error model of the memory
circuit: every detector (1,728), about 68,000 mechanisms. The decoders (relay-BP, BP-OSD) solve
the smaller BASIS problem: only the 936 detectors on checks of the memory basis, 8,784
mechanisms. The kit builds that problem (circuit.dem_matrices with a basis) by keeping, for
every mechanism of the circuit, only its effect on the basis detectors and the observables,
and merging the mechanisms that have the same effect.

So every full mechanism projects onto one basis column, or onto none (when it flips no basis
detector and no observable, as a pure X error does in X memory). A basis column "fired" in a
shot when an ODD number of its full mechanisms fired: two of them together cancel, which is
exactly how the kit merged their probabilities ("exactly one fires").

`ColumnMap` holds that projection and checks it three ways:
- the merged probability of every column's mechanisms equals the kit's prior of that column;
- on sampled shots, H x (fired columns) equals the shot's basis detectors;
- and L x (fired columns) equals the shot's observables.
"""
import numpy as np
import scipy.sparse as sp
import torch

from circuit import dem_matrices, detector_ids
from make_circuit import memory_experiment


class ColumnMap:
    """The projection from the full error model's mechanisms onto the basis problem's columns."""

    def __init__(self, full_H, full_L, full_priors, basis_H, basis_L, basis_priors, basis_rows):
        """full_*: the full model (all detectors). basis_*: the kit's basis problem. basis_rows: the detector ids of H's rows."""
        full_H, basis_H = sp.csc_matrix(full_H), sp.csc_matrix(basis_H)
        full_L, basis_L = sp.csc_matrix(full_L), sp.csc_matrix(basis_L)
        row_of = {int(d): row for row, d in enumerate(basis_rows)}
        column_of = {self._effect(basis_H, basis_L, k): k for k in range(basis_H.shape[1])}
        if len(column_of) != basis_H.shape[1]:
            raise ValueError("two columns of the basis problem have the same effect")
        target = np.full(full_H.shape[1], -1, dtype=np.int64)
        for j in range(full_H.shape[1]):
            detectors = full_H.indices[full_H.indptr[j]:full_H.indptr[j + 1]]
            kept = tuple(sorted(row_of[int(d)] for d in detectors if int(d) in row_of))
            observables = tuple(sorted(full_L.indices[full_L.indptr[j]:full_L.indptr[j + 1]].tolist()))
            if kept or observables:
                if (kept, observables) not in column_of:
                    raise ValueError(f"mechanism {j} projects onto an effect that is not a column of the basis problem")
                target[j] = column_of[(kept, observables)]
        self.target = target                                   # [full mechanisms]: basis column, or -1
        self.num_columns = basis_H.shape[1]
        self.basis_H, self.basis_L = basis_H, basis_L
        self.basis_rows = np.asarray(basis_rows)
        merged = self.merged_priors(np.asarray(full_priors, np.float64))
        self.prior_error = float(np.abs(merged - np.asarray(basis_priors, np.float64)).max())
        if self.prior_error > 1e-9:
            raise ValueError(f"the merged priors differ from the kit's by up to {self.prior_error:.2e}")
        if (np.bincount(target[target >= 0], minlength=self.num_columns) == 0).any():
            raise ValueError("a basis column has no mechanism of the full model")

    @staticmethod
    def _effect(H, L, k):
        return (tuple(sorted(H.indices[H.indptr[k]:H.indptr[k + 1]].tolist())),
                tuple(sorted(L.indices[L.indptr[k]:L.indptr[k + 1]].tolist())))

    def merged_priors(self, full_priors):
        """The probability that an odd number of each column's mechanisms fires: 1 - 2q = product of (1 - 2p)."""
        used = self.target >= 0
        log_factor = np.zeros(self.num_columns)
        np.add.at(log_factor, self.target[used], np.log(np.abs(1 - 2 * full_priors[used])))
        sign = np.ones(self.num_columns)
        np.multiply.at(sign, self.target[used], np.sign(1 - 2 * full_priors[used]))
        return (1 - sign * np.exp(log_factor)) / 2

    def columns_fired(self, shot, mechanism, shots):
        """uint8 [shots, columns]: the parity of the fired mechanisms of every column, from DemSampler's (shot, mechanism)."""
        target = torch.as_tensor(self.target, device=mechanism.device)[mechanism]
        keep = target >= 0
        flat = shot[keep] * self.num_columns + target[keep]
        counts = torch.bincount(flat, minlength=shots * self.num_columns).reshape(shots, self.num_columns)
        return (counts & 1).to(torch.uint8)

    def check(self, columns, detectors, observables):
        """Raise unless H x columns = the basis detectors and L x columns = the observables, on every shot (numpy uint8)."""
        columns = sp.csr_matrix(np.asarray(columns, np.int64))
        syndrome = (columns @ self.basis_H.T.astype(np.int64)).toarray() % 2
        logical = (columns @ self.basis_L.T.astype(np.int64)).toarray() % 2
        if not np.array_equal(syndrome, np.asarray(detectors)[:, self.basis_rows]):
            raise AssertionError("H x fired columns differs from the basis detectors")
        if not np.array_equal(logical, np.asarray(observables)):
            raise AssertionError("L x fired columns differs from the observables")


def column_map(code="gross", basis="X", rounds=None, p=0.004, sampler=None):
    """The ColumnMap of one memory circuit at noise strength p.

    With a neural.sampler.DemSampler (detectors="all") the full model is the sampler's own
    matrices, so the mechanism indices of its sampled faults are the ones mapped here.
    """
    circuit = memory_experiment(code, basis, rounds, p)
    basis_H, basis_L, basis_priors = dem_matrices(circuit, basis)
    if sampler is None:
        full_H, full_L, full_priors = dem_matrices(circuit, None)
    else:
        if sampler.num_detectors != circuit.num_detectors:
            raise ValueError("the sampler must use the full error model (detectors='all')")
        full_H, full_L, full_priors = sampler.H, sampler.L, sampler.priors[p].cpu().numpy().astype(np.float64)
        # The sampler stores float32 priors; take the float64 ones of the same model for the prior check.
        _, _, exact = dem_matrices(circuit, None)
        if len(exact) != len(full_priors) or np.abs(exact - full_priors).max() > 1e-6:
            raise ValueError("the sampler's priors are not this circuit's at this p")
        full_priors = exact
    return ColumnMap(full_H, full_L, full_priors, basis_H, basis_L, basis_priors, detector_ids(circuit, basis))
