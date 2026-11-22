"""Tests of tcbp_check.py that need no GPU: the prior cap, the column order of H, and the scoring arithmetic.

From challenge-1-ml-decoder-kit/:  python -m pytest -p no:cacheprovider neutral_atom_noise/test_tcbp_check.py
"""
import numpy as np

from neutral_atom_noise.priors import kit_problem, uniform_priors
from neutral_atom_noise.tcbp_check import PRIOR_CAP, cap, matrix_hash, sign_test, wilson

# The hash of the kit's X-memory H and L that convert recorded in transfer.json (2026-11-22). The server's decode
# refuses to run unless the H and L it rebuilds there hash to the same value.
H_L_SHA256_X = "a4a80003a0b234e0e10b5d65d38ce5181c39670732e0f473df0aee1796c2c8b0"


def test_cap_sets_values_above_one_half_to_one_half_and_counts_them():
    capped, changed = cap(np.array([0.1, 0.5, 0.7, 0.99, 1e-9]))
    assert np.array_equal(capped, [0.1, 0.5, PRIOR_CAP, PRIOR_CAP, 1e-9])
    assert changed == 2
    assert PRIOR_CAP == 0.5


def test_column_order_of_H_does_not_depend_on_p_and_matches_the_shipped_hash():
    H, L, _ = kit_problem("X")
    H_p, L_p, priors = kit_problem("X", 0.0326371)
    assert H.shape == (936, 8784) and L.shape == (12, 8784)
    assert matrix_hash(H, L) == matrix_hash(H_p, L_p) == H_L_SHA256_X
    assert np.array_equal(priors, uniform_priors("X", 0.0326371))
    assert priors.max() < PRIOR_CAP


def test_matrix_hash_sees_a_column_swap():
    H, L, _ = kit_problem("X")
    order = np.arange(H.shape[1])
    order[[0, 1]] = order[[1, 0]]
    assert matrix_hash(H[:, order], L[:, order]) != matrix_hash(H, L)


def test_sign_test_and_wilson():
    assert sign_test(0, 0) == 1.0
    assert sign_test(3, 3) == 1.0
    assert abs(sign_test(0, 5) - 2 / 32) < 1e-12
    assert abs(sign_test(1, 9) - 2 * 11 / 1024) < 1e-12
    low, high = wilson(0, 10)
    assert low == 0.0 and abs(high - 0.2775) < 1e-4
    low, high = wilson(91, 2000)
    assert low < 91 / 2000 < high


def test_phase2_models_have_the_declared_rates():
    import stim

    from circuit import GROSS_CIRCUITS
    from neutral_atom_noise.tcbp_models import biased_channel, biased_circuit, biased_two_qubit_error, si1000_noise

    px, py, pz = biased_channel(0.003, 100)
    assert abs(px + py + pz - 0.003) < 1e-15 and px == py and abs(pz / (px + py) - 100) < 1e-9
    assert abs(biased_two_qubit_error(0.003) - 0.005991) < 1e-12
    noise = si1000_noise(0.004)
    assert (noise.cx_depolarize2, noise.measure_flip, noise.reset_flip) == (0.004, 0.02, 0.008)
    assert abs(noise.idle_depolarize - 0.0004) < 1e-15
    circuit = biased_circuit(stim.Circuit.from_file(GROSS_CIRCUITS["X"]), 0.003, 100)
    names = {instruction.name for instruction in circuit.flattened()}
    assert "DEPOLARIZE1" not in names and "DEPOLARIZE2" not in names and "PAULI_CHANNEL_1" in names
