"""Tests of the neutral-atom-shaped noise model.

  python -m pytest -p no:cacheprovider test_kit.py neutral_atom_noise      (from challenge-1-ml-decoder-kit/)
"""
import math

import numpy as np
import pytest
import stim

from circuit import GROSS_CIRCUITS, dem_matrices, detector_ids, load_circuit, sample
from decoders import bposd_decoder, score
from make_circuit import memory_experiment
from neutral_atom_noise.model import (
    CHANNELS,
    CZ_TRANSITIONS,
    MODEL,
    PUBLISHED,
    Noise,
    kit_uniform,
    model_circuit,
    neutral_atom,
    with_noise,
)
from neutral_atom_noise.priors import FLOOR, effects, kit_problem, matched_priors, parse_priors, uniform_priors

BASES = ["X", "Z"]


def _noise_names(circuit):
    return {i.name for i in circuit.flattened() if stim.gate_data(i.name).is_noisy_gate and i.gate_args_copy()}


def _annotations(circuit):
    return [str(i) for i in circuit.flattened() if i.name in ("DETECTOR", "OBSERVABLE_INCLUDE")]


# ---------- the rewrite ----------

@pytest.mark.parametrize("basis", BASES)
def test_uniform_rates_give_the_kits_detector_error_model(basis):
    """THE test of the rewrite: with every rate at the kit's uniform p, our circuit is the kit's.

    The detector error models are equal term for term (same detectors, observables and probabilities, in the
    same order), and so are the circuits themselves.
    """
    p = 0.004
    ours = model_circuit(basis, kit_uniform(p))
    kit = load_circuit(GROSS_CIRCUITS[basis], p)
    ours_dem, kit_dem = ours.detector_error_model().flattened(), kit.detector_error_model().flattened()
    assert len(ours_dem) == len(kit_dem) > 60_000
    assert ours_dem == kit_dem
    assert ours == kit == memory_experiment("gross", basis, p=p)


@pytest.mark.parametrize("basis", BASES)
def test_only_the_noise_differs_from_the_kit(basis):
    """At the model's own rates the gates, detectors and observables are still exactly the kit's."""
    ours, kit = model_circuit(basis, neutral_atom(0.5)), load_circuit(GROSS_CIRCUITS[basis])
    assert ours.without_noise() == kit.without_noise()
    assert _annotations(ours) == _annotations(kit)
    assert (ours.num_detectors, ours.num_observables, ours.num_qubits) == (1728, 12, 288)
    assert ours.get_detector_coordinates() == kit.get_detector_coordinates()
    assert _noise_names(ours) <= set(CHANNELS)
    det, obs = sample(ours.without_noise(), shots=5, seed=0)
    assert not det.any() and not obs.any()


def test_a_cz_phase_error_is_z_on_the_control_and_x_on_the_target():
    kit_style = stim.Circuit("R 0 1\nX_ERROR(0.1) 0 1\nCX 0 1\nDEPOLARIZE2(0.1) 0 1\nDEPOLARIZE1(0.1) 2\n"
                             "X_ERROR(0.1) 1\nM 1")
    ours = with_noise(kit_style, Noise(cx_phase=0.01, cx_scatter=0.02, idle_phase=0.03, measure_flip=0.04, reset_flip=0.05))
    assert ours == stim.Circuit("R 0 1\nX_ERROR(0.05) 0 1\nCX 0 1\nZ_ERROR(0.01) 0\nX_ERROR(0.01) 1\n"
                                "DEPOLARIZE1(0.02) 0 1\nZ_ERROR(0.03) 2\nX_ERROR(0.04) 1\nM 1")


def test_zero_rates_leave_no_channel_and_unknown_noise_is_refused():
    kit_style = stim.Circuit("R 0\nX_ERROR(0.1) 0\nM 0")
    assert with_noise(kit_style, Noise()) == stim.Circuit("R 0\nM 0")
    with pytest.raises(ValueError):
        with_noise(stim.Circuit("R 0\nY_ERROR(0.1) 0\nM 0"), Noise())
    with pytest.raises(ValueError):                       # a flip that belongs to no reset and no readout
        with_noise(stim.Circuit("R 0\nH 0\nX_ERROR(0.1) 0\nH 0\nM 0"), Noise())


def test_repeat_blocks_stay_repeat_blocks():
    ours = model_circuit("X", neutral_atom(1.0))
    kit = stim.Circuit.from_file(GROSS_CIRCUITS["X"])
    assert [type(i) for i in ours if isinstance(i, stim.CircuitRepeatBlock)] == [stim.CircuitRepeatBlock]
    assert sum(isinstance(i, stim.CircuitRepeatBlock) for i in kit) == 1


# ---------- the table of rates ----------

def test_every_figure_has_a_source_and_the_rates_follow_from_them():
    for figure in PUBLISHED.values():
        assert figure.source.startswith("arXiv:") and figure.kind in ("M", "F") and figure.what
    assert all(rate.status in ("published", "derived", "our choice") and rate.how for rate in MODEL.values())
    assert MODEL["cx_phase"].value == PUBLISHED["cz_phase"].value == 0.0073
    assert MODEL["measure_flip"].value == pytest.approx(0.0125)
    assert MODEL["reset_flip"].value == pytest.approx(0.021)
    assert MODEL["loss"].value == pytest.approx(0.0025) and CZ_TRANSITIONS[4][1] == 5000
    assert MODEL["cx_scatter"].value == pytest.approx(0.75 * (17.4 + 18.5 + 185.0 + 197.5) / 2 * 1e-6)
    assert MODEL["idle_phase"].value == pytest.approx((1 - math.exp(-0.006 / 2.8)) / 2)
    assert set(MODEL) <= set(Noise().as_dict())


def test_one_factor_scales_every_rate():
    full, half = neutral_atom(1.0).as_dict(), neutral_atom(0.5).as_dict()
    assert all(half[name] == pytest.approx(0.5 * full[name]) for name in full)
    assert all(full[name] > 0 for name in MODEL)
    assert full["cx_depolarize2"] == full["idle_depolarize"] == 0
    # two phase errors and two scattering errors per CX
    assert neutral_atom(1.0).two_qubit_error() == pytest.approx(1 - (1 - 0.0073) ** 2 * (1 - MODEL["cx_scatter"].value) ** 2)
    assert kit_uniform(0.004).two_qubit_error() == pytest.approx(0.004)


# ---------- the priors ----------

@pytest.mark.parametrize("basis", BASES)
def test_matched_priors_are_the_models_error_model_on_the_kits_columns(basis):
    noise = neutral_atom(0.5)
    H, L, _ = kit_problem(basis)
    priors, fired = matched_priors(basis, noise)
    H_model, L_model, priors_model = dem_matrices(model_circuit(basis, noise), basis)
    assert priors.shape == (8784,) and priors.min() > 0 and priors.max() < 0.5
    assert fired.sum() == len(priors_model)
    # the same effect has the same probability in both
    by_effect = dict(zip(effects(H_model, L_model), priors_model, strict=True))
    kit_effects = effects(H, L)
    assert all(priors[k] == by_effect[kit_effects[k]] for k in np.nonzero(fired)[0])
    assert (priors[~fired] == FLOOR).all()


def test_columns_the_model_never_fires_get_the_floor():
    """Pure phase noise on the CX: no hook errors, so many of the kit's columns cannot fire."""
    priors, fired = matched_priors("X", Noise(cx_phase=0.005, measure_flip=0.01, reset_flip=0.01))
    assert 0 < fired.sum() < len(fired)
    assert (priors[~fired] == FLOOR).all() and priors[fired].min() > 1e-4


def test_generic_priors_come_from_the_kits_uniform_circuit():
    noise = neutral_atom(0.5)
    priors, p = parse_priors("generic-2q", "X", noise)
    assert p == pytest.approx(noise.two_qubit_error())
    assert (priors == dem_matrices(load_circuit(GROSS_CIRCUITS["X"], p), "X")[2]).all()
    priors, p = parse_priors("uniform:0.004", "X", noise)
    assert p == 0.004 and (priors == uniform_priors("X", 0.004)).all()
    assert parse_priors("matched", "X", noise)[1] is None
    # H and L do not depend on p, so every row of the tables decodes the same problem
    H3, L3, _ = kit_problem("X", 0.003)
    H5, L5, _ = kit_problem("X", 0.005)
    assert (H3 != H5).nnz == 0 and (L3 != L5).nnz == 0


def test_bposd_with_matched_priors_decodes_low_noise_shots():
    noise = neutral_atom(0.1)
    circuit = model_circuit("X", noise)
    H, L, _ = kit_problem("X")
    det, obs = sample(circuit, shots=20, seed=0)
    decoder = bposd_decoder(H, matched_priors("X", noise)[0])
    failed, _ = score(decoder.decode, L, det[:, detector_ids(circuit, "X")], obs)
    assert det.any() and not failed.any()
