"""Tests of the predecoder. Run from challenge-1-ml-decoder-kit/:

  python -m pytest -p no:cacheprovider predecoder

A tiny untrained network stands in for the trained one: the tests check the plumbing and the
physics of the mapping, not how well a trained head predicts.
"""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from circuit import dem_matrices, detector_ids
from evaluation.data import load_dataset
from evaluation.evaluate import run_basis
from make_circuit import memory_experiment
from neural.graph import build_graph
from neural.sampler import DemSampler
from predecoder import decode as decode_module
from predecoder.decode import decode_with_priors, first_leg_converges, kit_problem
from predecoder.head import PriorHead, column_shapes, prior_logit
from predecoder.mapping import column_map
from predecoder.priors import head_probabilities, shot_priors

KIT = Path(__file__).resolve().parents[1]
VALIDATION = KIT / "data" / "p0.004" / "validation_X.npz"
FAST_RELAY = {"num_sets": 3}          # a short relay-BP, for tests that only check the plumbing


# ---------- helpers ----------

@pytest.fixture(scope="module")
def sampled():
    """DEM shots of the full model at p = 0.004 with their fired mechanisms, and the column map."""
    sampler = DemSampler("gross", "X", None, "all", [0.004])
    mapping = column_map("gross", "X", None, 0.004, sampler)
    detectors, observables, (shot, mechanism) = sampler.sample(200, 0.004, torch.Generator().manual_seed(3), return_faults=True)
    columns = mapping.columns_fired(shot, mechanism, 200)
    return sampler, mapping, detectors.numpy(), observables.numpy(), columns.numpy()


def tiny_trunk(path):
    """An untrained network in the format neural.train.load_decoder reads."""
    from neural import train
    from neural.model import CascadeDecoder

    torch.manual_seed(7)
    model = CascadeDecoder(code="gross", basis="X", detectors="all", width=8, depth=1)
    torch.save({"config": train.DEFAULTS | model.config | {"seed": 7}, "step": 0, "ema_model": model.state_dict(),
                "commit": "made by test_predecoder"}, path)
    return str(path)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """(trunk path, head path) after a few steps of predecoder.train on the tiny trunk."""
    from predecoder import train

    folder = tmp_path_factory.mktemp("head")
    trunk = tiny_trunk(folder / "trunk.pt")
    train.main(["--trunk", trunk, "--out", str(folder / "run"), "--steps", "4", "--batch", "4", "--report-every", "2",
                "--report-shots", "8", "--project", "8", "--hidden", "8", "--shape-width", "4", "--device", "cpu",
                "--threads", "2", "--lr", "1e-2", "--warmup-steps", "1"])
    return trunk, str(folder / "run" / "head.pt")


def small_file(path, count, start=0):
    with np.load(VALIDATION) as file:
        np.savez_compressed(path, p=file["p"], seed=file["seed"], basis=file["basis"],
                            detectors=file["detectors"][start:start + count],
                            observables=file["observables"][start:start + count], basis_detectors=file["basis_detectors"])
    return path


# ---------- the mapping ----------

def test_mapping_reproduces_syndrome_and_observables(sampled):
    """The labels are the sampled mechanisms of the same error model the decoders' H comes from."""
    _, mapping, detectors, observables, columns = sampled
    H, L, priors = dem_matrices(memory_experiment("gross", "X", None, 0.004), "X")
    rows = detector_ids(memory_experiment("gross", "X", None, 0.004), "X")
    assert np.array_equal((columns.astype(np.int64) @ H.T.toarray().astype(np.int64)) % 2, detectors[:, rows])
    assert np.array_equal((columns.astype(np.int64) @ L.T.toarray().astype(np.int64)) % 2, observables)
    mapping.check(columns, detectors, observables)
    assert mapping.prior_error < 1e-12 and np.allclose(mapping.merged_priors(np.asarray(mapping_priors(0.004))), priors)
    # The fired columns are not trivial: about the kit's sum of priors per shot, and a wrong map would break the check.
    assert 30 < columns.sum(axis=1).mean() < 65
    broken = columns.copy()
    broken[0, np.nonzero(columns[0])[0][0]] ^= 1
    with pytest.raises(AssertionError):
        mapping.check(broken, detectors, observables)


def mapping_priors(p):
    _, _, priors = dem_matrices(memory_experiment("gross", "X", None, p), None)
    return priors


def test_mapping_is_the_same_at_other_noise_strengths():
    sampler = DemSampler("gross", "X", None, "all", [0.003, 0.005])
    first = column_map("gross", "X", None, 0.003, sampler)
    second = column_map("gross", "X", None, 0.005, sampler)
    assert np.array_equal(first.target, second.target) and (first.target >= 0).sum() < len(first.target)


# ---------- the head ----------

def test_column_shapes_are_122_shapes_of_72():
    problem = kit_problem("X", 0.004)
    graph = build_graph("gross", "X")
    rows = detector_ids(problem.circuit, "X")
    shapes = column_shapes(problem.H, graph.detector_round[rows], graph.detector_check[rows] % 72, graph.ell, graph.m)
    assert shapes.max() + 1 == 122 and (np.bincount(shapes) == 72).all()


def test_untrained_head_gives_the_kit_priors():
    problem = kit_problem("X", 0.004)
    head = PriorHead(problem.H, detector_ids(problem.circuit, "X"), trunk_width=8, project=4, hidden=4, shape_width=2)
    features = torch.randn(3, 13, 144, 8)
    syndrome = (torch.rand(3, 936) < 0.05).to(torch.uint8)
    logits = head(features, syndrome, prior_logit(problem.priors))
    assert torch.allclose(logits, prior_logit(problem.priors)[None].expand(3, -1))


def test_shifting_the_syndrome_shifts_the_head_output():
    """The head has the network's symmetry: a torus shift of the shot moves every column's logit with its column."""
    from neural.model import CascadeDecoder
    from neural.test_neural import detector_shift

    torch.manual_seed(1)
    problem = kit_problem("X", 0.004)
    graph = build_graph("gross", "X")
    rows = detector_ids(problem.circuit, "X")
    trunk = CascadeDecoder("gross", "X", width=8, depth=1).double().eval()
    head = PriorHead(problem.H, rows, trunk_width=8, project=4, hidden=4, shape_width=2).double()
    torch.nn.init.normal_(head.mlp[-1].weight)
    detectors = (torch.rand(2, graph.num_detectors) < 0.03).long()
    shift = (2, 1)
    moved_detector = detector_shift(graph, shift)                       # detector d goes to moved_detector[d]
    shifted = torch.zeros_like(detectors)
    shifted[:, torch.from_numpy(moved_detector)] = detectors
    row_of = {int(d): i for i, d in enumerate(rows)}
    H = problem.H.tocsc()
    column_of = {tuple(sorted(H.indices[H.indptr[k]:H.indptr[k + 1]])): k for k in range(H.shape[1])}
    moved_column = np.array([column_of[tuple(sorted(row_of[int(moved_detector[rows[i]])]
                                                    for i in H.indices[H.indptr[k]:H.indptr[k + 1]]))]
                             for k in range(H.shape[1])])
    priors = prior_logit(problem.priors).double()
    assert torch.equal(priors[torch.from_numpy(moved_column)], priors)        # the priors have the symmetry too
    with torch.no_grad():
        before = head(trunk.features(detectors), detectors[:, torch.from_numpy(rows)], priors)
        after = head(trunk.features(shifted), shifted[:, torch.from_numpy(rows)], priors)
    assert torch.allclose(after[:, torch.from_numpy(moved_column)], before, atol=1e-9)
    assert (before - priors).abs().max() > 1e-3


# ---------- priors ----------

def test_shot_priors_mix_and_clip():
    kit = np.array([0.01, 0.01, 0.01])
    logits = np.array([-50.0, 0.0, 50.0])
    q = head_probabilities(logits)
    assert np.allclose(shot_priors(logits, kit, 1.0, 1e-3), [1e-3, 0.5, 1 - 1e-3])
    assert np.allclose(shot_priors(logits, kit, 0.5, 1e-6), 0.5 * q + 0.005)
    assert np.allclose(shot_priors(logits, kit, 0.0, 1e-6, "logodds"), kit)
    assert np.allclose(shot_priors(np.array([0.0]), np.array([0.5]), 0.3, 1e-6, "logodds"), [0.5])
    # float16 rounding, as in the saved files
    assert head_probabilities(np.float32(1.00001)) == head_probabilities(np.float16(1.00001))


def test_priors_change_what_relay_bp_does(sampled):
    """relay-BP really reads the per-shot priors: told the fired columns (an oracle), it converges at once."""
    _, mapping, detectors, observables, columns = sampled
    problem = kit_problem("X", 0.004)
    syndrome = detectors[0, mapping.basis_rows]
    plain = decode_with_priors(problem.H, problem.L, problem.priors, syndrome)
    told = problem.priors.copy()
    told[columns[0] == 1] = 0.5
    oracle = decode_with_priors(problem.H, problem.L, told, syndrome)
    assert oracle.converged and np.array_equal(oracle.predicted, observables[0])
    assert oracle.iterations < plain.iterations
    assert first_leg_converges(problem.H, told, syndrome)
    again = decode_with_priors(problem.H, problem.L, told, syndrome)       # a decoder built per shot repeats itself
    assert again.iterations == oracle.iterations and np.array_equal(again.predicted, oracle.predicted)


# ---------- training, the plugin, the scripts ----------

def test_training_writes_a_head_that_loads(trained):
    from predecoder.train import load_head

    trunk, head_path = trained
    saved = torch.load(head_path, map_location="cpu", weights_only=True)
    assert saved["step"] == 4 and len(saved["history"]) == 3
    row = saved["history"][-1]["rows"][0]
    assert row["log_loss_head"] > 0 and row["log_loss_kit"] > 0
    assert saved["history"][0]["rows"][0]["log_loss_head"] == pytest.approx(saved["history"][0]["rows"][0]["log_loss_kit"])
    _, head, _ = load_head(head_path, "cpu", trunk)
    assert head.num_columns == 8784


def test_head_refuses_another_trunk(trained, tmp_path):
    from predecoder.train import load_head

    _, head_path = trained
    other = tiny_trunk(tmp_path / "other.pt")
    state = torch.load(other, weights_only=True)
    state["commit"] = "another"
    torch.save(state, other)
    with pytest.raises(SystemExit):
        load_head(head_path, "cpu", other)


def plugin_session(trained, monkeypatch=None):
    from predecoder.plugin import make_decoder

    trunk, head_path = trained
    return make_decoder(kit_problem("X", 0.004), {"heads": {"X": head_path}, "trunks": {"X": trunk},
                                                  "settings": {"mode": "linear", "alpha": 0.7, "eps": 1e-4},
                                                  "relay_changes": FAST_RELAY, "torch_threads": 2})


def test_plugin_gets_detector_bits_only(trained, tmp_path):
    """Truth shuffle: the observables of the file are permuted between shots; no prediction changes."""
    data = load_dataset(small_file(tmp_path / "small_X.npz", 6))
    shuffled = dataclasses.replace(data, observables=data.observables[np.random.default_rng(0).permutation(6)])
    assert not np.array_equal(shuffled.observables, data.observables)
    first = run_basis(plugin_session(trained), data)
    second = run_basis(plugin_session(trained), shuffled)
    assert np.array_equal(first["predictions"], second["predictions"])
    assert all(m["stage"] == "head priors" for m in first["metrics"])


def test_plugin_hands_its_priors_to_relay_bp(trained, tmp_path, monkeypatch):
    """The per-shot priors reach the decoder: the priors relay-BP is built with are the head's, mixed and clipped."""
    session = plugin_session(trained)
    data = load_dataset(small_file(tmp_path / "small_X.npz", 2))
    seen = []
    real = decode_module.relay_decoder

    def spy(H, priors, **changes):
        seen.append(np.array(priors))
        return real(H, priors, **changes)

    monkeypatch.setattr(decode_module, "relay_decoder", spy)
    for k in range(2):
        expected, _ = session.shot_priors(data.detectors[k], data.basis_input[k])
        session.decode_shot(data.detectors[k], data.basis_input[k])
        assert np.array_equal(seen[-1], expected)
    assert not np.allclose(seen[0], session.priors) and not np.array_equal(seen[0], seen[1])
    assert seen[0].min() >= 1e-4 and seen[0].max() <= 1 - 1e-4


def test_plugin_without_a_head_runs_the_kit_decoder():
    from predecoder.plugin import make_decoder

    session = make_decoder(kit_problem("Z", 0.004), {"heads": {"X": "unused"}})
    assert session.predecoder is None and "kit's relay-BP" in session.settings["decoder"]


def test_evaluate_then_compare(trained, tmp_path):
    """The scripts end to end on 4 shots: logits saved, the three arms decoded, the file numbers add up."""
    from predecoder import compare, evaluate

    trunk, head_path = trained
    shots = small_file(tmp_path / "small_X.npz", 4)
    evaluate.main(["--head", head_path, "--trunk", trunk, "--data", str(shots), "--out", str(tmp_path / "logits"),
                   "--device", "cpu", "--threads", "2", "--time-shots", "2"])
    logits = np.load(tmp_path / "logits" / "logits.npy")
    assert logits.shape == (4, 8784) and logits.dtype == np.float16
    compare.main(["--data", str(shots), "--logits", str(tmp_path / "logits"), "--workers", "2", "--chunk", "2",
                  "--out", str(tmp_path / "cmp")])
    summary = json.loads((tmp_path / "cmp" / "summary.json").read_text())
    per_shot = np.load(tmp_path / "cmp" / "per_shot.npz")
    for arm in ("kit", "fresh", "head"):
        assert summary["arms"][arm]["shots"] == 4
        assert summary["arms"][arm]["failures"] == int(per_shot[f"failed_{arm}"].sum())
        assert np.array_equal((per_shot[f"predicted_{arm}"] != per_shot["truth"]).any(axis=1), per_shot[f"failed_{arm}"] > 0)
    assert summary["arms"]["head"]["network_seconds_mean"] > 0


def test_held_out_files_are_refused(tmp_path):
    from predecoder import compare, evaluate

    held_out = small_file(tmp_path / "test_X.npz", 2)
    with pytest.raises(SystemExit):
        evaluate.main(["--head", "x", "--data", str(held_out), "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit):
        compare.main(["--data", str(held_out), "--out", str(tmp_path / "o")])


def test_fit_choice_rule():
    """Lowest weighted failures; ties go to fewer weighted hard shots, then fewer iterations."""
    from predecoder.priors import choose

    rows = [{"name": "a", "score": 30.0, "hard_weighted": 90, "mean_iterations": 200},
            {"name": "b", "score": 25.0, "hard_weighted": 400, "mean_iterations": 500},
            {"name": "c", "score": 25.0, "hard_weighted": 300, "mean_iterations": 600}]
    assert choose(rows)["name"] == "c"


def test_subsets_and_the_fit_run_end_to_end(trained, tmp_path):
    """subset.py splits a train file by a per-shot file; evaluate.py and the weighted fit of priors.py run on the parts."""
    from predecoder import evaluate, priors, subset

    trunk, head_path = trained
    source = small_file(tmp_path / "train_small_X.npz", 4)
    data = load_dataset(source)
    np.savez(tmp_path / "per_shot.npz", truth=data.observables, failed_fresh=np.array([False, True, False, True]))
    subset.main(["--data", str(source), "--per-shot", str(tmp_path / "per_shot.npz"), "--right", "1", "--seed", "0",
                 "--out", str(tmp_path / "sub")])
    failed = load_dataset(tmp_path / "sub" / "train_small_X_fresh_failed.npz")
    right = load_dataset(tmp_path / "sub" / "train_small_X_fresh_right.npz")
    assert np.array_equal(failed.detectors, data.detectors[[1, 3]]) and len(right.detectors) == 1
    with np.load(tmp_path / "sub" / "train_small_X_fresh_right.npz") as file:
        assert np.array_equal(right.detectors[0], data.detectors[int(file["shot"][0])]) and int(file["shot"][0]) in (0, 2)
    folders = []
    for name in ("failed", "right"):
        folders.append(str(tmp_path / f"logits_{name}"))
        evaluate.main(["--head", head_path, "--trunk", trunk, "--data", str(tmp_path / "sub" / f"train_small_X_fresh_{name}.npz"),
                       "--out", folders[-1], "--device", "cpu", "--threads", "2", "--time-shots", "0"])
    priors.main(["--data", str(tmp_path / "sub" / "train_small_X_fresh_failed.npz"), str(tmp_path / "sub" / "train_small_X_fresh_right.npz"),
                 "--weights", "1", "2", "--logits", *folders, "--modes", "logodds", "--alphas", "1.0", "--workers", "2",
                 "--out", str(tmp_path / "fit.json")])
    fit = json.loads((tmp_path / "fit.json").read_text())
    assert len(fit["rows"]) == 2 and fit["chosen"]["mode"] == "logodds"
    control = fit["rows"][0]
    assert control["score"] == control["failures"][0] + 2 * control["failures"][1]
    with pytest.raises(SystemExit):        # validation shots are never used to fit
        priors.main(["--data", str(VALIDATION), "--logits", folders[0], "--out", str(tmp_path / "no.json")])


def test_calibration_runs(trained, tmp_path):
    from predecoder import calibration

    trunk, head_path = trained
    calibration.main(["--head", head_path, "--trunk", trunk, "--shots", "4", "--chunk", "2", "--device", "cpu",
                      "--out", str(tmp_path / "cal.json")])
    rows = json.loads((tmp_path / "cal.json").read_text())["rows"]
    assert rows["all shots"]["shots"] == 4 and rows["all shots"]["head"]["log_loss"] > 0


def test_candidate_counts(tmp_path):
    """The head's answer beside the members: the counts of a hand-made case."""
    from predecoder import candidate

    truth = np.zeros((4, 12), np.uint8)
    wrong = np.zeros(12, np.uint8)
    wrong[0] = 1
    def bits(*bad):
        out = truth.copy()
        out[list(bad)] = wrong
        return out
    np.savez(tmp_path / "ours.npz", truth=truth, predicted_head=bits(3))
    np.savez(tmp_path / "members.npz", truth=truth, predicted_R=bits(0, 1), predicted_P1=bits(0, 1), predicted_P2=bits(0, 2))
    np.savez(tmp_path / "tcbp.npz", truth=truth, predicted_T=bits(0, 1))
    candidate.main(["--ours", str(tmp_path / "ours.npz"), "--members", str(tmp_path / "members.npz"),
                    "--tcbp", str(tmp_path / "tcbp.npz"), "--out", str(tmp_path / "c.json")])
    c = json.loads((tmp_path / "c.json").read_text())
    assert c["failures"] == {"R": 2, "P1": 2, "P2": 2, "T": 2, "H": 1}
    assert c["disagree"] == 2 and c["all_members_fail"] == 1 and c["head_right_where_all_members_fail"] == 1
    assert c["oracle_with_head"] == 0 and c["members_agree_but_wrong"] == 1 and c["head_right_there"] == 1
