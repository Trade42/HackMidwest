"""The decoding problem and the two pictures of the noise a decoder can be given.

Every row of our tables decodes with the SAME matrices: the kit's H (936 x 8,784 for one memory basis) and L.
Only the prior vector differs:

  generic   the kit's uniform model at one strength p (`uniform_priors`): what a decoder assumes when it is
            told nothing about the hardware.
  matched   the detector error model of the very circuit that made the shots (`matched_priors`), written onto
            the kit's columns. A column the model never fires gets FLOOR instead of 0, because belief
            propagation needs a nonzero prior.
"""
import numpy as np

from circuit import GROSS_CIRCUITS, dem_matrices, load_circuit
from neutral_atom_noise.model import model_circuit

FLOOR = 1e-9    # prior of a kit column that the model's circuit never fires


def effects(H, L):
    """For each column: (the detector rows it flips, the observables it flips)."""
    H, L = H.tocsc(), L.tocsc()
    return [(tuple(H.indices[H.indptr[k]:H.indptr[k + 1]].tolist()), tuple(L.indices[L.indptr[k]:L.indptr[k + 1]].tolist()))
            for k in range(H.shape[1])]


def kit_problem(basis, p=0.001):
    """(H, L, priors) of the kit's circuit at uniform strength p. H and L do not depend on p."""
    return dem_matrices(load_circuit(GROSS_CIRCUITS[basis], p), basis)


def uniform_priors(basis, p):
    """Generic priors: the kit's uniform model at strength p, on the kit's columns."""
    return kit_problem(basis, p)[2]


def matched_priors(basis, noise, floor=FLOOR):
    """Matched priors: the error model of `model_circuit(basis, noise)`, on the kit's columns.

    Returns (priors, fired) where fired marks the columns the model's circuit can fire. Raises if the model has
    a mechanism that is not a column of the kit's H: then the two problems would differ in more than the priors.
    """
    H, L, _ = kit_problem(basis)
    column = {effect: k for k, effect in enumerate(effects(H, L))}
    if len(column) != H.shape[1]:
        raise ValueError("two columns of the kit's H and L share an effect")
    H_model, L_model, priors_model = dem_matrices(model_circuit(basis, noise), basis)
    priors, fired = np.full(H.shape[1], floor), np.zeros(H.shape[1], bool)
    for effect, probability in zip(effects(H_model, L_model), priors_model, strict=True):
        if effect not in column:
            raise ValueError(f"the model fires a mechanism that the kit's matrix does not have: {effect}")
        priors[column[effect]], fired[column[effect]] = probability, True
    return priors, fired


def parse_priors(spec, basis, noise):
    """Priors from a command-line word: 'matched', 'generic-2q' or 'uniform:P'.

    'generic-2q' is the kit's uniform model at the p whose two-qubit depolarizing error equals the model's
    total two-qubit Pauli error. 'uniform:P' is the kit's uniform model at p = P.
    Returns (priors, the uniform p or None).
    """
    if spec == "matched":
        return matched_priors(basis, noise)[0], None
    p = noise.two_qubit_error() if spec == "generic-2q" else float(spec.removeprefix("uniform:"))
    return uniform_priors(basis, p), p
