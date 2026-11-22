"""Tests of the learned selector, on hand-made member files whose answers are known.

Run from challenge-1-ml-decoder-kit/:  python -m pytest -p no:cacheprovider selector
"""
import json

import numpy as np
import pytest

from selector import evaluate as E
from selector import features as F
from selector import fit as FIT
from selector import model as M

SHA = {"X": "a" * 64, "Z": "b" * 64}


def flip(bits, k):
    out = bits.copy()
    out[k % 12] ^= 1
    return out


def make_members(folder, basis="X", stem="train", shots=80, seed=1, truth=None, with_joint=True):
    """Member files of one made-up data file: P1, P2, R (candidates), T (tcbp) and J (joint, on the choice shots only).

    Shots 0 to 3 are fixed cases; the rest are drawn at random, half of them choice shots.
    Returns the paths and the truth.
    """
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 2, (shots, 12)).astype(np.uint8) if truth is None else truth
    wrong = {name: np.zeros(shots, int) for name in ("P1", "P2", "T", "R", "J")}   # 0: right, k > 0: bit k-1 flipped
    # shot 0: unanimous and right; shot 1: T wrong, P1 = P2 right; shot 2: P1 wrong, P2 and T right; shot 3: all differ, R right
    wrong["T"][1] = 1
    wrong["P1"][2] = 2
    wrong["P1"][3], wrong["P2"][3], wrong["T"][3] = 3, 4, 5
    for shot in range(4, shots):
        if rng.random() < 0.5:
            for name in ("P1", "P2", "T", "R", "J"):
                wrong[name][shot] = 0 if rng.random() < 0.55 else int(rng.integers(1, 4))
    predicted = {name: np.array([truth[s] if wrong[name][s] == 0 else flip(truth[s], wrong[name][s] - 1) for s in range(shots)], np.uint8)
                 for name in wrong}
    common = {"data_sha256": SHA[basis], "data_name": f"{stem}_{basis}.npz", "basis": basis, "p": 0.004}

    def fields(name, rows):
        right = wrong[name][rows] == 0
        return {f"predicted_{name}": predicted[name][rows], f"converged_{name}": right | (rng.random(len(rows)) < 0.5),
                f"weight_{name}": 200 + 30 * (~right) + rng.normal(0, 5, len(rows)), f"n_faults_{name}": rng.integers(40, 80, len(rows)),
                f"iterations_{name}": rng.integers(1, 18060, len(rows)), f"seconds_{name}": np.full(len(rows), 0.05),
                f"failed_{name}": (predicted[name][rows] != truth[rows]).any(axis=1)}

    everything = np.arange(shots)
    folder.mkdir(parents=True, exist_ok=True)
    cand = folder / f"{stem}_{basis}.chunks.npz"
    np.savez(cand, truth=truth, **common, **fields("R", everything), **fields("P1", everything), **fields("P2", everything))
    tcbp = folder / f"{stem}_{basis}.tcbp.npz"
    t = fields("T", everything)
    t["converged_T"][:] = True
    t["seconds_T"][:] = np.nan
    np.savez(tcbp, truth=truth, **common, **t)
    paths = [cand, tcbp]
    if with_joint:
        split = np.flatnonzero(~((predicted["P1"] == predicted["P2"]).all(1) & (predicted["P1"] == predicted["T"]).all(1)))
        joint = folder / f"{stem}_{basis}.joint.npz"
        np.savez(joint, shot=split, truth=truth[split], shots_in_file=shots, **common, **fields("J", split))
        paths.append(joint)
    return paths, truth


def fixed_model(pool=F.POOL, prefer="has_T"):
    names = list(F.BASE_FEATURES)
    weights = np.zeros(len(names))
    weights[names.index(prefer)] = 1.0
    return {"features": names, "pool": list(pool), "mean": np.zeros(len(names)), "scale": np.ones(len(names)), "weights": weights,
            "lambda": 0.0, "trained_on": [{"data_name": "made up", "p": 0.004, "data_sha256": "c" * 64}]}


# ---------- the choice table ----------

def test_table_on_fixed_cases(tmp_path):
    paths, truth = make_members(tmp_path, with_joint=False)
    view, info = F.load(paths)
    choices = F.table(view, pool=("T", "P1", "P2", "R"))
    choice = F.choice_shots(view)
    assert not choice[0] and choice[1] and choice[2] and choice[3]
    assert list(choices.shots) == list(np.flatnonzero(choice))
    names = list(choices.names)
    for shot in (1, 2, 3):
        rows = np.flatnonzero(choices.shot == shot)
        bits = choices.bits[rows]
        assert len({b.tobytes() for b in bits}) == len(rows), "the answers of a shot are distinct"
        assert choices.gives[rows].sum() == sum(view.members[n].covers[shot] for n in ("T", "P1", "P2", "R"))
        first = choices.features[rows[0]]
        assert first[names.index("has_T")] == 1.0, "TCBP-JOBS's answer comes first"
        assert first[names.index("pt_weight_minus_T")] == 0.0 and first[names.index("R_weight_minus_T")] == 0.0
    rows1 = np.flatnonzero(choices.shot == 1)          # T alone against P1 = P2 (R right too)
    other = choices.features[rows1[1]]
    assert other[names.index("n_priortilt")] == 2.0 and other[names.index("has_T")] == 0.0
    right = F.labels(choices, truth)
    assert right[rows1].tolist() == [False, True]


def test_joint_enters_only_its_listed_shots(tmp_path):
    paths, _ = make_members(tmp_path)
    view, _ = F.load(paths)
    choices = F.table(view)
    j = F.POOL.index("J")
    assert set(choices.shot[choices.gives[:, j]]) <= set(np.flatnonzero(view.members["J"].covers))
    no_j = F.table(view, pool=("T", "P1", "P2", "R"))
    assert not no_j.gives[:, j].any()


# ---------- no truth reaches a choice ----------

def shuffled_copy(folder, paths, truth, permutation):
    """The same member files with the truth moved between shots, and every truth-derived array with it."""
    new_truth = truth[permutation]
    out = []
    for path in paths:
        with np.load(path, allow_pickle=False) as file:
            arrays = {key: file[key] for key in file.files}
        rows = arrays["shot"] if "shot" in arrays else np.arange(len(truth))
        arrays["truth"] = new_truth[rows]
        for key in list(arrays):
            if key.startswith("failed_"):
                name = key[len("failed_"):]
                arrays[key] = (arrays[f"predicted_{name}"] != new_truth[rows]).any(axis=1)
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / path.name
        np.savez(target, **arrays)
        out.append(target)
    return out


@pytest.mark.parametrize("prefer", ["has_T", "n_priortilt", "pt_weight_minus_T", "R_log_iterations"])
def test_shuffling_the_truth_changes_no_choice(tmp_path, prefer):
    paths, truth = make_members(tmp_path / "a")
    permutation = np.random.default_rng(7).permutation(len(truth))
    shuffled = shuffled_copy(tmp_path / "b", paths, truth, permutation)
    model = fixed_model(prefer=prefer)
    model["weights"] = model["weights"] + np.linspace(-0.3, 0.3, len(model["weights"]))   # every feature counts a little
    picks = []
    for files in (paths, shuffled):
        view, info = F.load(files)
        answer, choice, _, choices, picked = E.selector_row(view, model)
        picks.append((answer, choice, choices.features, picked, info.truth))
    (a1, c1, f1, p1, t1), (a2, c2, f2, p2, t2) = picks
    assert not np.array_equal(t1, t2), "the truth did move"
    assert np.array_equal(f1, f2) and np.array_equal(p1, p2) and np.array_equal(c1, c2) and np.array_equal(a1, a2)


def test_a_fitted_model_chooses_alike_on_shuffled_truth(tmp_path):
    paths, truth = make_members(tmp_path / "train", shots=200, seed=3)
    out = tmp_path / "model.json"
    FIT.main(["--train", *map(str, paths), "--out", str(out)])
    model = M.load(out)
    val, vtruth = make_members(tmp_path / "val", stem="validation", shots=120, seed=4)
    shuffled = shuffled_copy(tmp_path / "val_shuffled", val, vtruth, np.random.default_rng(5).permutation(len(vtruth)))
    first = E.selector_row(F.load(val)[0], model)
    second = E.selector_row(F.load(shuffled)[0], model)
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[4], second[4])


# ---------- fitting ----------

def test_fit_learns_an_obvious_rule(tmp_path):
    paths, _ = make_members(tmp_path, shots=300, seed=2)
    out = tmp_path / "m.json"
    model = FIT.main(["--train", *map(str, paths), "--pool", "T", "P1", "P2", "R", "--out", str(out)])
    assert json.loads(out.read_text())["pool"] == ["T", "P1", "P2", "R"]
    # the made-up wrong answers are 30 heavier and converge half as often: the weight and convergence features must count
    names = model["features"]
    assert model["weights"][names.index("pt_converged")] > 0
    view, info = F.load(paths)
    answer, *_ = E.selector_row(view, M.load(out))
    t_fail = (view.members["T"].predicted != info.truth).any(1).sum()
    assert (answer != info.truth).any(1).sum() < t_fail


@pytest.mark.parametrize("stem,data_name", [("test", None), ("go", None), ("selector_test", None), ("train", "test_X.npz"),
                                            ("train", "go_X.npz"), ("validation", None)])
def test_fit_refuses_held_out_and_validation_files(tmp_path, stem, data_name):
    paths, _ = make_members(tmp_path, stem=stem)
    if data_name:
        for path in paths:
            with np.load(path) as file:
                arrays = {key: file[key] for key in file.files}
            arrays["data_name"] = np.array(data_name)
            np.savez(path, **arrays)
    with pytest.raises(SystemExit, match="refused"):
        FIT.main(["--train", *map(str, paths), "--out", str(tmp_path / "m.json")])
    assert not (tmp_path / "m.json").exists()


def test_fit_refuses_the_folder_of_the_test_run(tmp_path):
    paths, _ = make_members(tmp_path / "files" / "clean" / "cand_p0.004")
    with pytest.raises(SystemExit, match="refused"):
        FIT.main(["--train", *map(str, paths), "--out", str(tmp_path / "m.json")])


def test_cross_validation_counts(tmp_path):
    paths, truth = make_members(tmp_path, shots=200, seed=9)
    view, info = F.load(paths)
    choices = F.table(view)
    right = F.labels(choices, info.truth)
    cv = M.cross_validate(choices.features, choices.starts, right, lambdas=(0.1, 1.0), baseline=choices.gives[:, F.POOL.index("T")])
    with_right = sum(right[choices.starts[i]:choices.starts[i + 1]].any() for i in range(len(choices.shots)))
    for counts in cv.values():
        assert counts["shots"] == with_right
        assert counts["right"] + counts["only_selector_wrong"] + counts["both_wrong"] == counts["shots"]
    assert M.choose_lambda({0.1: {"right": 5}, 1.0: {"right": 5}}) == 1.0


# ---------- applying and scoring ----------

def test_evaluate_scores_both_bases_and_pairs(tmp_path):
    x, xtruth = make_members(tmp_path / "x", basis="X", stem="validation", shots=60, seed=11, with_joint=False)
    z, ztruth = make_members(tmp_path / "z", basis="Z", stem="validation", shots=60, seed=12, with_joint=False)
    model_path = tmp_path / "model.json"
    M.save(fixed_model(pool=("T", "P1", "P2", "R")), model_path)
    results = E.main(["--model", str(model_path), "--x", *map(str, x), "--z", *map(str, z), "--save", str(tmp_path / "out" / "v")])
    saved = np.load(tmp_path / "out" / "v_X.npz")
    # a model that always prefers TCBP-JOBS's answer is TCBP-JOBS alone, and (without J) so is rule f
    assert np.array_equal(saved["failed"], saved["failed_T"]) and np.array_equal(saved["failed_T"], saved["failed_f"])
    assert results["pairs"]["selector"]["failed_pairs"] == results["pairs"]["T"]["failed_pairs"]
    assert results["X"]["failures"]["selector"] == int(saved["failed"].sum())
    with pytest.raises(SystemExit):
        E.main(["--model", str(model_path), "--expect-sha256", "0" * 64, "--x", *map(str, x)])


def test_evaluate_needs_every_always_on_member(tmp_path):
    x, _ = make_members(tmp_path, with_joint=False)
    model_path = tmp_path / "model.json"
    M.save(fixed_model(), model_path)
    with pytest.raises(SystemExit):
        E.main(["--model", str(model_path), "--x", str(x[0])])


def test_evaluate_on_held_out_files_needs_the_frozen_sha(tmp_path):
    x, _ = make_members(tmp_path, stem="test", with_joint=False)
    model_path = tmp_path / "model.json"
    sha = M.save(fixed_model(pool=("T", "P1", "P2", "R")), model_path)
    with pytest.raises(SystemExit, match="expect-sha256"):
        E.main(["--model", str(model_path), "--x", *map(str, x)])
    results = E.main(["--model", str(model_path), "--expect-sha256", sha, "--x", *map(str, x)])
    assert results["X"]["shots"] == 80


@pytest.mark.parametrize("drop", ["converged_T", "weight_T", "weight_P1", "converged_R", "iterations_P2"])
def test_a_member_file_without_a_field_the_features_read_is_refused(tmp_path, drop):
    paths, _ = make_members(tmp_path, with_joint=False)
    for path in paths:
        with np.load(path) as file:
            arrays = {key: file[key] for key in file.files if key != drop}
        np.savez(path, **arrays)
    with pytest.raises(SystemExit, match="holds no"):
        F.load(paths)


def test_a_nan_weight_of_T_is_refused(tmp_path):
    paths, _ = make_members(tmp_path, with_joint=False)
    with np.load(paths[1]) as file:
        arrays = {key: file[key] for key in file.files}
    arrays["weight_T"][3] = np.nan
    np.savez(paths[1], **arrays)
    with pytest.raises(SystemExit, match="no weight for T"):
        F.load(paths)
