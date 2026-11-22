"""Tests of the head trained on the neutral-atom-shaped noise model (predecoder/noise.py, train.py --noise-factor).
CPU only. Run from challenge-1-ml-decoder-kit/:

  python -m pytest -p no:cacheprovider predecoder/test_noise.py
"""
import json

import numpy as np
import pytest
import torch

from circuit import detector_ids, sample
from neural.sampler import rate_z_scores
from neutral_atom_noise.model import model_circuit, neutral_atom
from neutral_atom_noise.priors import matched_priors, uniform_priors
from predecoder.files import vector_sha256
from predecoder.head import build_head
from predecoder.noise import ModelProblem
from predecoder.test_predecoder import tiny_trunk

FACTOR = 0.8
TRUNK_CONFIG = {"code": "gross", "basis": "X", "rounds": 12, "detectors": "all"}


@pytest.fixture(scope="module")
def problem():
    return ModelProblem(TRUNK_CONFIG, [FACTOR], "cpu")


def test_labels_reproduce_the_syndrome_and_the_observables(problem):
    generator = torch.Generator().manual_seed(5)
    _, basis_bits, labels, detectors, observables = problem.batch(100, FACTOR, generator)
    problem.map.check(labels.numpy(), detectors.numpy(), observables.numpy())     # raises otherwise
    assert np.array_equal(basis_bits.numpy(), detectors.numpy()[:, problem.map.basis_rows])
    assert labels.sum(dim=1).float().mean() > 40                                    # about 87 columns fire per shot


def test_the_base_is_the_matched_vector_and_the_merged_mechanisms_equal_it(problem):
    matched, fired = matched_priors("X", neutral_atom(FACTOR))
    assert fired.all()
    assert np.array_equal(problem.base[FACTOR], matched)
    assert problem.map.prior_error < 1e-9
    assert problem.base_sha256[FACTOR] == vector_sha256(matched)
    assert problem.sampler.num_mechanisms < 67824                                   # another model than the kit's
    assert np.array_equal(problem.map.basis_rows, detector_ids(model_circuit("X", neutral_atom(FACTOR)), "X"))


def test_error_model_shots_match_stim_shots_of_the_model_circuit(problem):
    """The head trains on error-model shots and is scored on Stim shots: their detector and observable rates agree."""
    shots, chunk = 20000, 2000
    generator = torch.Generator().manual_seed(9)
    ours_det, ours_obs = np.zeros(1728), np.zeros(12)
    for _ in range(shots // chunk):
        detectors, observables = problem.sampler.sample(chunk, FACTOR, generator)
        ours_det += detectors.numpy().sum(axis=0)
        ours_obs += observables.numpy().sum(axis=0)
    det, obs = sample(model_circuit("X", neutral_atom(FACTOR)), shots, 11)
    z_det = rate_z_scores(ours_det, det.sum(axis=0).astype(float), shots)
    z_obs = rate_z_scores(ours_obs, obs.sum(axis=0).astype(float), shots)
    assert np.abs(z_det).max() < 5.0 and np.abs(z_obs).max() < 5.0
    assert abs(ours_det.sum() / shots - det.sum() / shots) < 0.01 * det.sum() / shots + 1.0


def test_an_untrained_head_gives_the_matched_priors(problem, tmp_path):
    from neural.train import load_decoder

    trunk, saved = load_decoder(tiny_trunk(tmp_path / "trunk.pt"))
    head = build_head(problem.map.basis_H, problem.map.basis_rows, saved["config"], project=8, hidden=8, shape_width=4)
    trunk_input, basis_bits, _, _, _ = problem.batch(3, FACTOR, torch.Generator().manual_seed(1))
    with torch.no_grad():
        logits = head(trunk.features(trunk_input), basis_bits, problem.prior_logit[FACTOR])
    assert torch.equal(logits, problem.prior_logit[FACTOR][None].expand(3, -1))


def model_shot_file(path, shots, p=0.0326371, seed=12):
    """A small file of Stim shots of the model circuit in the kit's format (what predecoder.evaluate reads)."""
    circuit = model_circuit("X", neutral_atom(FACTOR))
    detectors, observables = sample(circuit, shots, seed)
    np.savez_compressed(path, p=p, seed=seed, basis="X", detectors=detectors.astype(np.uint8),
                        observables=observables.astype(np.uint8), basis_detectors=detector_ids(circuit, "X"))
    return path


@pytest.fixture(scope="module")
def heads(tmp_path_factory):
    """(trunk, retrained head, kit head, folder): a few steps of predecoder.train each, on a tiny trunk."""
    from predecoder import train

    folder = tmp_path_factory.mktemp("noise_heads")
    trunk = tiny_trunk(folder / "trunk.pt")
    common = ["--trunk", trunk, "--steps", "4", "--batch", "4", "--report-every", "2", "--report-shots", "8",
              "--project", "8", "--hidden", "8", "--shape-width", "4", "--device", "cpu", "--threads", "2",
              "--lr", "1e-2", "--warmup-steps", "1"]
    train.main(common + ["--out", str(folder / "na"), "--noise-factor", str(FACTOR)])
    train.main(common + ["--out", str(folder / "kit")])
    return trunk, str(folder / "na" / "head.pt"), str(folder / "kit" / "head.pt"), folder


def test_the_retrained_checkpoint_records_its_noise_and_starts_at_the_base(heads):
    _, head_path, _, _ = heads
    saved = torch.load(head_path, map_location="cpu", weights_only=True)
    matched, _ = matched_priors("X", neutral_atom(FACTOR))
    assert saved["config"]["noise_factors"] == [FACTOR]
    assert saved["config"]["base_sha256"] == {f"{FACTOR:g}": vector_sha256(matched)}
    first = saved["history"][0]
    assert first["step"] == 0 and first["rows"][0]["log_loss_head"] == pytest.approx(first["rows"][0]["log_loss_kit"], rel=1e-6)


def test_evaluate_needs_the_base_the_head_was_trained_with(heads, tmp_path):
    from predecoder import evaluate

    trunk, head_path, _, _ = heads
    shots = model_shot_file(tmp_path / "na_X.npz", 3)
    common = ["--head", head_path, "--trunk", trunk, "--data", str(shots), "--device", "cpu", "--time-shots", "1"]
    with pytest.raises(SystemExit, match="--base-priors"):
        evaluate.main(common + ["--out", str(tmp_path / "none")])
    np.save(tmp_path / "generic.npy", uniform_priors("X", 0.0326371))
    with pytest.raises(SystemExit, match="not the base"):
        evaluate.main(common + ["--out", str(tmp_path / "wrong"), "--base-priors", str(tmp_path / "generic.npy")])
    matched, _ = matched_priors("X", neutral_atom(FACTOR))
    np.save(tmp_path / "matched.npy", matched)
    evaluate.main(common + ["--out", str(tmp_path / "ok"), "--base-priors", str(tmp_path / "matched.npy")])
    meta = json.loads((tmp_path / "ok" / "meta.json").read_text())
    assert meta["base_priors"]["trained_with_this_base"] is True
    assert meta["base_priors"]["sha256"] == vector_sha256(matched)
    assert np.load(tmp_path / "ok" / "logits.npy").shape == (3, 8784)


def test_a_kit_head_takes_another_base_and_says_so(heads, tmp_path):
    from predecoder import evaluate

    trunk, _, kit_head, _ = heads
    shots = model_shot_file(tmp_path / "na_X.npz", 2)
    matched, _ = matched_priors("X", neutral_atom(FACTOR))
    np.save(tmp_path / "matched.npy", matched)
    common = ["--head", kit_head, "--trunk", trunk, "--data", str(shots), "--device", "cpu", "--time-shots", "0"]
    evaluate.main(common + ["--out", str(tmp_path / "generic")])
    evaluate.main(common + ["--out", str(tmp_path / "matched"), "--base-priors", str(tmp_path / "matched.npy")])
    on_generic = json.loads((tmp_path / "generic" / "meta.json").read_text())["base_priors"]
    on_matched = json.loads((tmp_path / "matched" / "meta.json").read_text())["base_priors"]
    assert on_generic["trained_with_this_base"] is True and on_generic["sha256"] == vector_sha256(uniform_priors("X", 0.0326371))
    assert on_matched["trained_with_this_base"] is False and on_matched["sha256"] == vector_sha256(matched)
    a, b = np.load(tmp_path / "generic" / "logits.npy"), np.load(tmp_path / "matched" / "logits.npy")
    assert not np.array_equal(a, b)                # the base reaches the head
