"""Training shots for the head from the neutral-atom-shaped noise model (neutral_atom_noise/) instead of the kit's.

`predecoder.train --noise-factor F` uses `ModelProblem` in place of train.py's `Problem`. Everything else of the
training is unchanged: the same trunk (frozen), the same head, the same loss, the same kit columns.

What changes, and why:
- **The shots.** Every step draws fresh shots from the FULL detector error model of
  `neutral_atom_noise.model.model_circuit(basis, neutral_atom(F))`: the kit's circuit with the model's noise in
  place of the kit's (35,064 mechanisms at F = 0.8, against the kit's 67,824). It is the same circuit without
  noise as the kit's (checked here), so the trunk reads the same 1,728 detectors in the same order.
- **The labels.** The model's mechanisms are mapped onto the kit's 8,784 columns by `mapping.ColumnMap`, built
  from the kit's own H and L (`neutral_atom_noise.priors.kit_problem`), never from the model's basis DEM, whose
  column order is its own. The map raises if a mechanism lands on an effect that is not a kit column.
- **The head's base prior.** On the kit's noise the head reads the kit's priors at the training p: the true merged
  prior of the sampler's model. The same here is the model's own merged prior on the kit's columns,
  `neutral_atom_noise.priors.matched_priors` (the "matched" vector). The map checks that the mechanisms' merged
  probabilities equal it (to 1e-9). So the untrained head gives exactly the matched priors, and the same vector
  must be the head's input and the mixing base wherever the head is used (`predecoder.evaluate --base-priors`).

The levels of a ModelProblem are the factors: `problem.prior_logit[0.8]`, `problem.batch(shots, 0.8, generator)`.
"""
import numpy as np
import stim
import torch

from circuit import dem_matrices, detector_ids
from make_circuit import memory_experiment
from neural.sampler import DemSampler
from predecoder.files import vector_sha256
from predecoder.head import prior_logit
from predecoder.mapping import ColumnMap


class CircuitSampler(DemSampler):
    """DemSampler (detectors="all", the kit's observables) of given circuits, one per level, instead of the kit's."""

    def __init__(self, circuits, device="cpu"):
        self.device = torch.device(device)
        self.priors, self.exact_priors = {}, {}
        for level, circuit in circuits.items():
            H, L, priors = dem_matrices(circuit, None)
            if not self.priors:
                self.H, self.L = H.tocsc(), L.tocsc()
            elif H.shape != self.H.shape or (H != self.H).nnz or (L != self.L).nnz:
                raise ValueError("the error model's mechanisms changed with the level")
            self.exact_priors[level] = np.asarray(priors, np.float64)
            self.priors[level] = torch.tensor(priors, dtype=torch.float32, device=self.device)
        self.num_detectors, self.num_mechanisms = self.H.shape
        self.num_observables = self.L.shape[0]
        self._detector_rows = self._padded_rows(self.H)
        self._observable_rows = self._padded_rows(self.L)


def same_circuit_without_noise(a, b):
    return stim.Circuit(str(a.without_noise())) == stim.Circuit(str(b.without_noise()))


class ModelProblem:
    """train.Problem for the neutral-atom-shaped model at the given factors: the same attributes and batch()."""

    def __init__(self, trunk_config, factors, device):
        from neutral_atom_noise.model import model_circuit, neutral_atom
        from neutral_atom_noise.priors import kit_problem, matched_priors

        code, basis, rounds = trunk_config["code"], trunk_config["basis"], trunk_config["rounds"]
        kit_circuit = memory_experiment(code, basis, rounds, 0.004)
        self.levels = sorted(set(float(f) for f in factors))
        circuits = {f: model_circuit(basis, neutral_atom(f)) for f in self.levels}
        for f, circuit in circuits.items():
            if not same_circuit_without_noise(circuit, kit_circuit):
                raise ValueError(f"the model's circuit at factor {f:g} is not the trunk's circuit without noise")
        self.sampler = CircuitSampler(circuits, device)
        H, L, _ = kit_problem(basis)
        rows = detector_ids(kit_circuit, basis)
        self.base, self.base_sha256, self.prior_logit = {}, {}, {}
        for f in self.levels:
            base, fired = matched_priors(basis, neutral_atom(f))
            if not fired.all():
                # With a column the model never fires, the base would hold a floor the mechanisms cannot reproduce.
                raise ValueError(f"the model at factor {f:g} never fires {int((~fired).sum())} kit columns")
            mapping = ColumnMap(self.sampler.H, self.sampler.L, self.sampler.exact_priors[f], H, L, base, rows)
            if f == self.levels[0]:
                self.map = mapping
            elif not np.array_equal(mapping.target, self.map.target):
                raise ValueError("the column map changed with the factor")
            self.base[f], self.base_sha256[f] = base, vector_sha256(base)
            self.prior_logit[f] = prior_logit(base).to(device)
        self.basis_rows = torch.from_numpy(self.map.basis_rows).to(device)
        self.trunk_reads_all = trunk_config["detectors"] == "all"

    def batch(self, shots, level, generator):
        """(trunk input, basis bits, column labels, detectors, observables) of fresh shots, as train.Problem.batch."""
        detectors, observables, (shot, mechanism) = self.sampler.sample(shots, level, generator, return_faults=True)
        labels = self.map.columns_fired(shot, mechanism, shots)
        basis_bits = detectors[:, self.basis_rows]
        trunk_input = detectors if self.trunk_reads_all else basis_bits
        return trunk_input, basis_bits, labels, detectors, observables
