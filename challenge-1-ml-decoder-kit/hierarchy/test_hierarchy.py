"""Tests of hierarchy/. Run from challenge-1-ml-decoder-kit/:

  python -m pytest -p no:cacheprovider hierarchy

The tests that decode need PriorTilt (the module priortilt_rs, and PRIORTILT_REPO naming a clone
of its repository). Without it they are skipped, and so are the network tests without torch.
The decoding tests use the first shots of the committed training files and small iteration
budgets, so their answers are the same on every machine and they finish in seconds.
"""
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from evaluation.adapters import Problem, create_session, validate_shot_result
from hierarchy import candidates, compare, joint_member, rules, select, tcbp_member
from hierarchy.plugin import HierarchySession, Network
from hierarchy.rules import ORDER
from hierarchy.stages import Candidate, kit_problem, load_shots, sha256_of

KIT = Path(__file__).resolve().parent.parent
QUICK = ["--priortilt-max-rounds", "20", "--relay-sets", "3"]          # small budgets: many shots stay unconverged
QUICK_OPTIONS = {"priortilt_max_rounds": 20, "relay_sets": 3}
SHOTS = 24
needs_priortilt = pytest.mark.skipif(
    importlib.util.find_spec("priortilt_rs") is None or not os.environ.get("PRIORTILT_REPO"),
    reason="PriorTilt is not installed, or PRIORTILT_REPO does not name its clone (see hierarchy/README.md)")

RIGHT = np.zeros(12, np.uint8)
WRONG_A = np.eye(12, dtype=np.uint8)[0]
WRONG_B = np.eye(12, dtype=np.uint8)[1]


# ---------- helpers ----------

@pytest.fixture(scope="module")
def shot_files(tmp_path_factory):
    """The first shots of the committed training files of both bases, as two small shot files."""
    folder = tmp_path_factory.mktemp("shots")
    paths = {}
    for basis in "XZ":
        with np.load(KIT / "data" / "p0.004" / f"train_{basis}.npz") as file:
            paths[basis] = folder / f"small_{basis}.npz"
            np.savez_compressed(paths[basis], p=file["p"], seed=file["seed"], basis=file["basis"],
                                detectors=file["detectors"][:SHOTS], observables=file["observables"][:SHOTS],
                                basis_detectors=file["basis_detectors"])
    return paths


def tiny_checkpoint(path, basis, detectors="all", logicals="kit"):
    """An untrained network in the format neural.train.load_decoder reads, made with neural's own model class."""
    import torch

    from neural import train
    from neural.model import CascadeDecoder

    torch.manual_seed(7)
    model = CascadeDecoder(code="gross", basis=basis, detectors=detectors, width=8, depth=1, logicals=logicals)
    torch.save({"config": train.DEFAULTS | model.config | {"seed": 7}, "step": 0,
                "ema_model": model.state_dict(), "commit": "made by test_hierarchy"}, path)
    return str(path)


def fake_problem(basis="X"):
    """The sizes of the kit's problem without its content, for sessions whose decoders are stubs."""
    return Problem(basis, sp.csc_matrix((936, 5), dtype=np.uint8), sp.csr_matrix((12, 5), dtype=np.uint8),
                   np.full(5, 0.01), 0.004, SimpleNamespace(num_detectors=1728))


def candidate(name, predicted, weight, converged=True):
    return Candidate(name, np.array(predicted, np.uint8), converged, iterations=10, n_faults=3, weight=weight, seconds=0.001)


class StubHierarchy:
    """Stage decoders that return fixed answers, and count how often relay-BP was asked."""

    def __init__(self, first, second, third):
        self.answers, self.third_calls = (first, second, third), 0

    def first_two(self, syndrome):
        return self.answers[0], self.answers[1]

    def third(self, syndrome):
        self.third_calls += 1
        return self.answers[2]


class StubNetwork:
    """A network on the kit's observables that is sure of one 12-bit answer, and remembers what it was shown."""
    model_from_kit = None

    def __init__(self, favourite, strength=8.0):
        self.logits, self.seen = np.where(np.array(favourite) == 1, strength, -strength), []

    def __call__(self, full, basis):
        self.seen.append(np.array(full))
        return self.logits, (self.logits > 0).astype(np.uint8)


def write_candidates(path, shots, data_sha256="a" * 64, session=False):
    """A candidates file from rows of (predicted R, P1, P2), (weight R, P1, P2), (converged R, P1, P2)."""
    arrays = {"truth": np.zeros((len(shots), 12), np.uint8), "data_sha256": data_sha256}
    for index, name in enumerate(ORDER):
        arrays[f"predicted_{name}"] = np.array([row[0][index] for row in shots], np.uint8)
        arrays[f"weight_{name}"] = np.array([row[1][index] for row in shots], float)
        arrays[f"converged_{name}"] = np.array([row[2][index] for row in shots], bool)
        arrays[f"iterations_{name}"] = np.full(len(shots), 100)
        arrays[f"seconds_{name}"] = np.full(len(shots), 0.3 if name == "R" else 0.1)
    if session:                                    # relay-BP ran only where P1 and P2 differ
        arrays["ran_R"] = (arrays["predicted_P1"] != arrays["predicted_P2"]).any(axis=1)
        arrays["weight_R"] = np.where(arrays["ran_R"], arrays["weight_R"], np.nan)
    np.savez(path, **arrays)
    return str(path)


def write_logits(path, logits, observables=None):
    logits = np.array(logits, np.float32)
    np.savez(path, logits=logits, predictions=(logits > 0).astype(np.uint8),
             observables=np.zeros((len(logits), 12), np.uint8) if observables is None else observables)
    return str(path)


# Five shots whose outcome under every rule is known by hand. The truth is "no bit flipped".
ALL = (True, True, True)
HAND_MADE = [
    ((RIGHT, RIGHT, RIGHT), (10, 10, 10), ALL),                         # 0: all right
    ((RIGHT, WRONG_A, WRONG_A), (10, 20, 20), ALL),                     # 1: P1 = P2, both wrong; relay-BP right
    ((RIGHT, WRONG_A, RIGHT), (30, 10, 20), ALL),                       # 2: P1 wrong and lightest; the others right
    ((WRONG_B, RIGHT, WRONG_A), (5, 9, 7), (False, True, True)),        # 3: only P1 right; relay-BP lightest, unconverged
    ((WRONG_A, WRONG_A, WRONG_B), (10, 10, 10), ALL),                   # 4: all wrong
]
# The network is sure of "no flip" on shot 2 and of the wrong answer A on shots 3 and 4.
HAND_MADE_LOGITS = [-8 * np.ones(12), -8 * np.ones(12), -8 * np.ones(12), 16 * WRONG_A - 8.0, 16 * WRONG_A - 8.0]


# ---------- the rules ----------

def test_network_score_is_the_log_probability_of_the_twelve_bits():
    logits = np.linspace(-3, 3, 12)
    predicted = np.stack([RIGHT, WRONG_A, np.ones(12, np.uint8)])[:, None, :]
    expected = [sum(math.log(1 / (1 + math.exp(-x))) if bit else math.log(1 / (1 + math.exp(x)))
                    for x, bit in zip(logits, bits, strict=True)) for bits in predicted[:, 0]]
    assert rules.network_score(predicted, logits[None])[:, 0] == pytest.approx(expected)
    assert rules.log_sigmoid(np.array([-800.0, 0.0, 800.0])) == pytest.approx([-800.0, -math.log(2), 0.0])


def test_rules_choose_as_documented():
    predicted = np.stack([np.stack([RIGHT, RIGHT]), np.stack([WRONG_A, RIGHT]), np.stack([WRONG_B, RIGHT])])   # [3, 2, 12]
    weight = np.array([[5.0, 7.0], [4.0, 7.0], [4.0, 9.0]])
    assert rules.lowest_weight(weight).tolist() == [1, 0]                       # equal weights: the first in ORDER
    allowed = np.array([[True, False], [False, False], [True, False]])
    assert rules.lowest_weight(weight, allowed).tolist() == [2, 0]              # no allowed candidate: all of them
    favours_b = (16 * WRONG_B - 8.0)[None].repeat(2, axis=0)
    assert rules.most_likely(predicted, favours_b).tolist() == [2, 0]           # equal answers score the same: the first wins
    assert rules.most_likely(predicted, favours_b, allowed=np.array([[True] * 2, [True] * 2, [False] * 2])).tolist() == [0, 0]
    # The network slightly prefers answer B; with a small constant the lighter answer A wins.
    mild = (0.2 * WRONG_B - 0.1 * WRONG_A)[None].repeat(2, axis=0)
    tilted = np.array([[9.0, 1.0], [4.0, 1.0], [8.0, 1.0]])
    assert rules.most_likely(predicted, mild).tolist() == [2, 0]
    assert rules.most_likely(predicted, mild, tilted, constant=1.0).tolist() == [1, 0]
    assert rules.agree(predicted).tolist() == [False, True]
    assert rules.tie_break(predicted, np.array([2, 2])).tolist() == [2, ORDER.index("P1")]


def test_wilson_interval_and_exact_test():
    low, high = rules.wilson_interval(31, 5000)
    assert (low, high) == pytest.approx((0.00437, 0.00879), abs=1e-5)           # the laptop's printed interval for 31 of 5,000
    assert rules.wilson_interval(0, 100)[0] == 0.0 and rules.wilson_interval(100, 100)[1] == 1.0
    a, b = np.array([1, 1, 0, 0, 0, 0, 0], bool), np.array([1, 0, 1, 1, 1, 1, 1], bool)
    assert rules.paired_test(a, b) == (1, 1, 5, pytest.approx(2 * 7 / 64))      # 1 against 5 discordant: 2 * (1 + 6) / 2^6
    assert rules.paired_test(a, a) == (2, 0, 0, 1.0)


def test_wilson_interval_equals_the_neural_folder():
    pytest.importorskip("torch")
    from neural.evaluate_stim import wilson_interval

    for failures, shots in [(0, 50), (3, 50), (66, 25000), (50, 50)]:
        assert rules.wilson_interval(failures, shots) == pytest.approx(wilson_interval(failures, shots))


# ---------- select.py ----------

def failures(rows):
    return {row.key: None if row.failed is None else int(row.failed.sum()) for row in rows}


def test_select_scores_every_rule_on_hand_made_candidates(tmp_path):
    pool, sources = select.load_candidates([write_candidates(tmp_path / "c.npz", HAND_MADE)])
    network = select.load_logits([write_logits(tmp_path / "l.npz", HAND_MADE_LOGITS)], pool, sources)
    assert network.ms is None                                                   # no .json beside the logits
    network.ms = 10.0
    rows = select.score(pool, network, constants=[(1.0, "given")])
    assert failures(rows) == {"R": 2, "P1": 3, "P2": 3, "weight": 4, "converged": 3, "network": 3,
                              "network_C1": 4, "perfect": 1, "perfect_tie_break": 2}
    by_key = {row.key: row for row in rows}
    assert by_key["network"].failed.tolist() == [False, True, False, True, True]
    count = select.facts(pool)
    assert (count["p1_p2_differ"], count["not_all_equal"], int(count["pickable"].sum())) == (3, 4, 3)
    assert count["unconverged"] == {"R": 1, "P1": 0, "P2": 0} and int(count["none_converged"].sum()) == 0
    # Time: P1 and P2 on every shot (200 ms), relay-BP (300 ms) and the network (10 ms) on the 3 of 5 shots where they differ.
    assert by_key["weight"].ms.mean() == pytest.approx(200 + 300 * 3 / 5)
    assert by_key["network"].ms.mean() == pytest.approx(200 + 310 * 3 / 5)
    assert by_key["converged"].ms.mean() == pytest.approx(500)
    numbers = select.numbers(pool, sources, rows)
    assert numbers["rules"][5]["right_picks"] == 1 and numbers["right_pick_possible"] == 3


def test_select_marks_what_a_session_file_cannot_measure(tmp_path):
    pool, sources = select.load_candidates([write_candidates(tmp_path / "s.npz", HAND_MADE, session=True)])
    rows = select.score(pool, select.load_logits([write_logits(tmp_path / "l.npz", HAND_MADE_LOGITS)], pool, sources))
    assert failures(rows) == {"R": None, "P1": 3, "P2": 3, "weight": 4, "converged": None, "network": 3,
                              "perfect": None, "perfect_tie_break": 2}
    assert "relay-BP ran on 3 of 5 shots" in rows[0].note
    # Two files are scored as one set, in the order given.
    both, _ = select.load_candidates([write_candidates(tmp_path / "c.npz", HAND_MADE), str(tmp_path / "s.npz")])
    assert both["truth"].shape == (10, 12) and failures(select.score(both))["weight"] == 8


def test_select_says_when_a_constant_is_fitted_on_the_scored_shots(tmp_path, capsys):
    scored = write_candidates(tmp_path / "scored.npz", HAND_MADE, data_sha256="a" * 64)
    other = write_candidates(tmp_path / "other.npz", HAND_MADE, data_sha256="b" * 64)
    logits = write_logits(tmp_path / "l.npz", HAND_MADE_LOGITS)
    fit = ["--fit-logits", logits, "--constants", "1", "1000"]
    select.main(["--candidates", scored, "--logits", logits, "--fit-candidates", other, *fit])
    printed = capsys.readouterr().out
    assert "chosen C = 1000" in printed and "fitted on other shots" in printed and "WARNING" not in printed
    select.main(["--candidates", scored, "--logits", logits, "--fit-candidates", scored, *fit, "--save", str(tmp_path / "n.json")])
    printed = capsys.readouterr().out
    assert "FITTED ON THESE SAME SHOTS" in printed and "WARNING: C was fitted on the shots it is scored on" in printed
    assert json.loads((tmp_path / "n.json").read_text())["shots"] == 5
    select.main(["--candidates", scored, "--logits", logits, "--constant", "30"])
    assert "counts as a result only if it was chosen on other shots" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="together"):
        select.main(["--candidates", scored, "--logits", logits, "--constants", "10"])


def test_select_refuses_logits_of_other_shots(tmp_path):
    pool, sources = select.load_candidates([write_candidates(tmp_path / "c.npz", HAND_MADE)])
    other = write_logits(tmp_path / "l.npz", HAND_MADE_LOGITS, observables=np.ones((5, 12), np.uint8))
    with pytest.raises(SystemExit, match="observables differ"):
        select.load_logits([other], pool, sources)


# ---------- held-out files ----------

@pytest.mark.parametrize("name", ["test_X.npz", "go_X.npz"])
def test_held_out_files_are_refused_by_name(tmp_path, name):
    with pytest.raises(SystemExit, match="held-out"):
        load_shots(tmp_path / name)                    # refused before it is opened: the file does not even exist
    with pytest.raises(SystemExit, match="held-out"):
        candidates.main(["--data", str(tmp_path / name), "--out", str(tmp_path / "out")])


def test_final_allows_a_held_out_file_and_folders_leave_them_alone(tmp_path, shot_files, capsys):
    held_out = tmp_path / "test_X.npz"
    held_out.write_bytes(shot_files["X"].read_bytes())
    (tmp_path / "validation_X.npz").write_bytes(shot_files["X"].read_bytes())
    assert len(load_shots(held_out, final=True).detectors) == SHOTS
    assert [path.name for path in candidates.expand([tmp_path])] == ["validation_X.npz"]
    assert "left alone, held out: test_X.npz" in capsys.readouterr().out


# ---------- candidates.py ----------

def arrays_of(path):
    with np.load(path) as file:
        return {key: file[key] for key in file.files}


@needs_priortilt
def test_candidates_do_not_depend_on_the_number_of_processes(tmp_path, shot_files):
    command = ["--data", str(shot_files["X"]), "--chunk-size", "6", *QUICK]
    candidates.main([*command, "--out", str(tmp_path / "one"), "--processes", "1"])
    candidates.main([*command, "--out", str(tmp_path / "four"), "--processes", "4"])
    one, four = arrays_of(tmp_path / "one" / "small_X.chunks.npz"), arrays_of(tmp_path / "four" / "small_X.chunks.npz")
    changing = {key for key in one if key.startswith("seconds_")} | {"machine", "processes"}   # the times, and where they were taken
    assert one.keys() == four.keys() and changing < one.keys()
    for key in one.keys() - changing:
        assert np.array_equal(one[key], four[key], equal_nan=one[key].dtype.kind == "f"), key
    assert one["processes"].tolist() == [1] * 4 and four["processes"].tolist() == [4] * 4
    assert compare.same(compare.compare(compare.load(tmp_path / "one" / "small_X.chunks.npz"),
                                        compare.load(tmp_path / "four" / "small_X.chunks.npz")))
    assert one["ran_R"].all() and len(one["truth"]) == SHOTS
    assert json.loads(str(one["settings_P1"]))["seed"] == 1 and json.loads(str(one["settings_P2"]))["seed"] == 1001
    assert json.loads(str(one["settings_P1"]))["max_rounds"] == 20
    for name in ORDER:                                  # an unconverged candidate is scored on what it returned
        assert np.array_equal(one[f"failed_{name}"], (one[f"predicted_{name}"] != one["truth"]).any(axis=1))
    assert not one["converged_P1"].all() and one["converged_P1"].any()

    # A run stopped after two chunks continues to the same file.
    candidates.main([*command, "--out", str(tmp_path / "resumed"), "--processes", "1", "--chunks", "0:2"])
    assert not (tmp_path / "resumed" / "small_X.chunks.npz").exists()
    candidates.main([*command, "--out", str(tmp_path / "resumed"), "--processes", "1"])
    resumed = arrays_of(tmp_path / "resumed" / "small_X.chunks.npz")
    assert all(np.array_equal(one[key], resumed[key], equal_nan=one[key].dtype.kind == "f") for key in one.keys() - changing)
    # Other settings in the same folder are refused, not mixed in.
    with pytest.raises(SystemExit, match="another run"):
        candidates.main([*command, "--out", str(tmp_path / "resumed"), "--seeds", "5", "6"])


# ---------- the plugin ----------

def test_plugin_returns_the_agreed_answer_without_relay_or_network():
    stages = StubHierarchy(candidate("P1", WRONG_A, 5.0), candidate("P2", WRONG_A, 7.0), candidate("R", RIGHT, 1.0))
    network = StubNetwork(RIGHT)
    session = HierarchySession(fake_problem(), {"rule": "network"}, hierarchy=stages, network=network)
    result = session.decode_shot(np.zeros(1728, np.uint8), np.zeros(936, np.uint8))
    assert validate_shot_result(result).tolist() == WRONG_A.tolist() and result.decode_seconds is None
    assert stages.third_calls == 0 and network.seen == []
    assert result.metrics["stage"] == "agree" and result.metrics["agreed"] and not result.metrics["fallback_used"]
    assert result.metrics["winner"] == "P1" and result.metrics["iterations"] == 20 and "converged_R" not in result.metrics


@pytest.mark.parametrize("favourite", ORDER)
def test_stub_network_picks_the_candidate_it_prefers(favourite):
    answers = {"R": WRONG_B, "P1": RIGHT, "P2": WRONG_A}
    stages = StubHierarchy(candidate("P1", answers["P1"], 1.0), candidate("P2", answers["P2"], 2.0), candidate("R", answers["R"], 3.0))
    network = StubNetwork(answers[favourite])
    session = HierarchySession(fake_problem(), {"rule": "network"}, hierarchy=stages, network=network)
    full = (np.arange(1728) % 3 == 0).astype(np.uint8)
    result = session.decode_shot(full, np.zeros(936, np.uint8))
    assert validate_shot_result(result).tolist() == answers[favourite].tolist()
    assert result.metrics["winner"] == favourite and result.metrics["stage"] == "network" and result.metrics["fallback_used"]
    assert stages.third_calls == 1 and len(network.seen) == 1 and np.array_equal(network.seen[0], full)   # the shot's bits, once
    assert result.metrics["logits"] == network.logits.tolist() and result.metrics["iterations"] == 30
    assert result.metrics["network_answer"] == answers[favourite].tolist()
    assert result.metrics["predicted_R"] == answers["R"].tolist() and result.metrics["converged_R"] is True
    json.dumps(result.metrics)                          # the evaluator writes the metrics as JSON


def test_weight_rule_and_constant():
    stages = StubHierarchy(candidate("P1", RIGHT, 9.0), candidate("P2", WRONG_A, 4.0, converged=False), candidate("R", WRONG_B, 6.0))
    shot = (np.zeros(1728, np.uint8), np.zeros(936, np.uint8))
    by_weight = HierarchySession(fake_problem(), {"rule": "weight"}, hierarchy=stages)
    result = by_weight.decode_shot(*shot)
    assert result.metrics["winner"] == "P2" and result.metrics["stage"] == "weight" and "logits" not in result.metrics
    assert result.metrics["converged"] is False and result.metrics["any_converged"] is True
    # The network prefers P1's answer by a little; with a small constant the lightest candidate wins.
    mild = StubNetwork(RIGHT, strength=0.05)
    assert HierarchySession(fake_problem(), {"rule": "network"}, hierarchy=stages, network=mild).decode_shot(*shot).metrics["winner"] == "P1"
    tilted = HierarchySession(fake_problem(), {"rule": "network", "constant": 1.0}, hierarchy=stages, network=mild)
    assert tilted.decode_shot(*shot).metrics["winner"] == "P2" and tilted.settings["constant"] == 1.0


def test_a_basis_without_a_checkpoint_uses_the_weight_rule_and_says_so():
    stages = StubHierarchy(candidate("P1", RIGHT, 9.0), candidate("P2", WRONG_A, 4.0), candidate("R", WRONG_B, 6.0))
    session = HierarchySession(fake_problem("Z"), {"rule": "network", "checkpoints": {"X": "some.pt"}, "constant": 30}, hierarchy=stages)
    assert session.rule == "weight" and session.settings["rule"] == "weight" and session.settings["rule_asked"] == "network"
    assert "no checkpoint was given for Z memory" in session.settings["rule_note"] and session.settings["constant"] is None
    assert session.decode_shot(np.zeros(1728, np.uint8), np.zeros(936, np.uint8)).metrics["winner"] == "P2"
    with pytest.raises(ValueError, match="unknown options"):
        HierarchySession(fake_problem(), {"checkpoint": "typo.pt"}, hierarchy=stages)
    with pytest.raises(ValueError, match="rule must be"):
        HierarchySession(fake_problem(), {"rule": "majority"}, hierarchy=stages)


@pytest.mark.parametrize("basis, detectors, logicals", [("X", "all", "kit"), ("X", "basis", "kit"), ("Z", "all", "light")])
def test_network_reads_the_shot_and_nothing_else(tmp_path, basis, detectors, logicals):
    torch = pytest.importorskip("torch")
    from neural.train import load_decoder

    torch.set_num_threads(1)
    checkpoint = tiny_checkpoint(tmp_path / "net.pt", basis, detectors, logicals)
    network = Network(checkpoint, fake_problem(basis))
    random = np.random.default_rng(3)
    full, part = random.integers(0, 2, 1728, dtype=np.uint8), random.integers(0, 2, 936, dtype=np.uint8)
    model, _ = load_decoder(checkpoint)
    shot = torch.from_numpy((full if detectors == "all" else part)[None])
    with torch.no_grad():
        direct = model(shot)
    logits, answer = network(full, part)
    assert logits == pytest.approx(direct.numpy()[0], abs=1e-6)
    assert np.array_equal(answer, model.kit_bits(shot, direct > 0).numpy()[0])
    assert network.record["reads_detectors"] == detectors and network.record["logical_operators"] == logicals
    assert network.record["step"] == 0 and len(network.record["sha256"]) == 64
    with pytest.raises(ValueError, match=f"decodes {basis} memory"):
        Network(checkpoint, fake_problem("Z" if basis == "X" else "X"))

    # A candidate's bits in the network's own operators give back the candidate's kit bits
    # through neural/'s own conversion, for any 12 bits.
    kit_candidates = random.integers(0, 2, (5, 1, 12), dtype=np.uint8)
    own = rules.bits_for_network(kit_candidates, (logits > 0)[None], answer[None], network.model_from_kit)
    assert (network.model_from_kit is None) == (logicals == "kit")
    for candidate, bits in zip(kit_candidates[:, 0], own[:, 0], strict=True):
        assert np.array_equal(model.kit_bits(shot, torch.from_numpy(bits[None])).numpy()[0], candidate)
    if logicals == "light":
        assert not np.array_equal(own, kit_candidates)      # these operators are not the kit's


@pytest.fixture(scope="module")
def session_candidates(tmp_path_factory, shot_files):
    """The small shot files decoded the way a session decodes them, with the quick budgets."""
    out = tmp_path_factory.mktemp("session")
    candidates.main(["--data", str(shot_files["X"]), str(shot_files["Z"]), "--out", str(out), "--mode", "session",
                     "--processes", "1", *QUICK])
    return {basis: out / f"small_{basis}.session.npz" for basis in "XZ"}


@needs_priortilt
def test_plugin_through_the_evaluator_equals_select_on_session_candidates(tmp_path, shot_files, session_candidates, monkeypatch):
    """The whole chain on 24 shots per basis: the network rule for X, the weight rule for Z, which has no checkpoint."""
    torch = pytest.importorskip("torch")
    from evaluation import evaluate
    from neural import evaluate_stim

    torch.set_num_threads(1)
    checkpoint = tiny_checkpoint(tmp_path / "x.pt", "X")
    options = tmp_path / "options.json"
    options.write_text(json.dumps({"rule": "network", "checkpoints": {"X": checkpoint}, "torch_threads": 1, **QUICK_OPTIONS}))
    monkeypatch.setattr(sys, "argv", ["evaluate", "--test-x", str(shot_files["X"]), "--test-z", str(shot_files["Z"]),
                                      "--decoder", "hierarchy.plugin:make_decoder", "--plugin-options", str(options),
                                      "--out", str(tmp_path / "eval")])
    evaluate.main()
    summary = json.loads((tmp_path / "eval" / "summary.json").read_text())
    settings = summary["decoders"][0]["configuration"]
    assert settings["X"]["rule"] == "network" and settings["X"]["network"]["checkpoint"] == checkpoint
    assert settings["Z"]["rule"] == "weight" and "no checkpoint" in settings["Z"]["rule_note"]
    assert settings["X"]["decoders"]["P1"]["seed"] == 1 and settings["X"]["decoders"]["P2"]["seed"] == 1001
    assert 0 < summary["decoders"][0]["summary"]["fallback_fraction"] < 1
    shots_file = tmp_path / "eval" / "01_hierarchy_plugin_make_decoder_shots.npz"
    metrics = json.loads((tmp_path / "eval" / "01_hierarchy_plugin_make_decoder_metrics.json").read_text())

    evaluate_stim.main(["--checkpoint", checkpoint, "--data", str(shot_files["X"]), "--latency-shots", "3", "--threads", "1",
                        "--save", str(tmp_path / "logits_X")])
    for basis, rule in (("X", "network"), ("Z", "weight")):
        pool, sources = select.load_candidates([session_candidates[basis]])
        network = select.load_logits([str(tmp_path / "logits_X.npz")], pool, sources) if basis == "X" else None
        rows = select.score(pool, network)
        assert select.check_evaluator(shots_file, rows, rule, sources) == 0
        ran = pool["ran"][ORDER.index("R")]
        assert 0 < ran.sum() < SHOTS                    # the quick budgets leave both kinds of shot in the file
        assert [not shot["agreed"] for shot in metrics[basis]] == ran.tolist()
        assert {shot["stage"] for shot in metrics[basis]} == {"agree", rule}
        if basis == "X":                                # the plugin's batch-one logits equal the offline ones
            online = np.array([shot["logits"] for shot in metrics[basis] if not shot["agreed"]])
            assert online == pytest.approx(network.logits[ran], abs=1e-4)
            assert network.ms > 0 and "from the .json beside the logits" in network.where


@needs_priortilt
def test_plugin_with_a_network_on_other_logical_operators_equals_select(tmp_path, shot_files, session_candidates):
    """Z memory with a `--logicals light` network: the session made by the evaluator's create_session against select.py."""
    torch = pytest.importorskip("torch")
    from neural import evaluate_stim

    torch.set_num_threads(1)
    checkpoint = tiny_checkpoint(tmp_path / "z.pt", "Z", logicals="light")
    session = create_session("hierarchy.plugin:make_decoder", kit_problem("Z", 0.004),
                             {"rule": "network", "checkpoints": {"Z": checkpoint}, "torch_threads": 1, **QUICK_OPTIONS})
    assert session.setup_seconds > 0 and session.settings["network"]["logical_operators"] == "light"
    json.dumps(session.settings)
    data = load_shots(shot_files["Z"])
    results = [session.decode_shot(full, part) for full, part in zip(data.detectors, data.basis_input, strict=True)]
    answers = np.stack([validate_shot_result(result) for result in results])

    evaluate_stim.main(["--checkpoint", checkpoint, "--data", str(shot_files["Z"]), "--latency-shots", "3", "--threads", "1",
                        "--save", str(tmp_path / "logits_Z")])
    pool, sources = select.load_candidates([session_candidates["Z"]])
    network = select.load_logits([str(tmp_path / "logits_Z.npz")], pool, sources)
    by_key = {row.key: row for row in select.score(pool, network)}
    assert np.array_equal(answers, by_key["network"].predicted)
    asked = [not result.metrics["agreed"] for result in results]
    assert asked == pool["ran"][ORDER.index("R")].tolist() and any(asked)
    assert not np.array_equal(network.candidates, pool["predicted"])       # the candidates were turned into the light operators


# ---------- tcbp_member.py, without a GPU ----------

def test_tcbp_member_turns_error_vectors_into_answers():
    H = sp.csc_matrix(np.array([[1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1]], np.uint8))
    L = sp.csr_matrix(np.vstack([[1, 1, 0, 0], np.zeros((11, 4), int)]).astype(np.uint8))
    problem = Problem("X", H, L, np.array([0.1, 0.2, 0.1, 0.5]), 0.004, None)
    llr = np.log((1 - problem.priors) / problem.priors)
    errors = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]], np.uint8)
    seen = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 0]], np.uint8)            # the last correction also flips detector 2
    rows = tcbp_member.answers(errors, seen, problem, llr)
    assert rows["predicted"][:, 0].tolist() == [1, 0, 0] and not rows["predicted"][:, 1:].any()
    assert rows["converged"].tolist() == [True, True, False] and rows["n_faults"].tolist() == [1, 2, 2]
    assert rows["weight"] == pytest.approx([math.log(9), math.log(9) + math.log(4), math.log(9)])

    result = SimpleNamespace(result=[[0.9, 0.1, 0.0, 0.6]])
    assert tcbp_member.error_vectors(result, (1, 4), "decode_batch").tolist() == [[1, 0, 0, 1]]
    assert tcbp_member.error_vectors([result, result], (2, 1, 4), "decode_batch").shape == (2, 1, 4)   # a list of results
    with pytest.raises(SystemExit, match="shape"):
        tcbp_member.error_vectors(result, (1, 5), "decode_batch")
    with pytest.raises(SystemExit, match="without a `result`"):
        tcbp_member.error_vectors(object(), (1, 4), "decode")


def test_tcbp_member_plumbing_with_relay_bp_in_place_of_the_gpu_decoder(tmp_path, shot_files, capsys):
    """Everything of tcbp_member.py but the decoder itself, on a small file, serial and batched."""
    command = ["--data", str(shot_files["X"]), "--decoder", "relay", "--opt", "num_sets=3"]
    tcbp_member.main([*command, "--out", str(tmp_path / "serial")])
    tcbp_member.main([*command, "--out", str(tmp_path / "batch"), "--batch"])
    printed = capsys.readouterr().out
    assert "plumbing stand-in, not TCBP-JOBS" in printed and "serial decode, the first decode of the basis untimed" in printed
    serial = arrays_of(tmp_path / "serial" / "small_X.relay-plumbing.npz")
    batch = arrays_of(tmp_path / "batch" / "small_X.relay-plumbing.npz")
    for rows, mode in ((serial, "serial"), (batch, "batch")):
        assert rows["predicted_T"].shape == (SHOTS, 12) and str(rows["mode"]) == mode and str(rows["decoder"]) == "relay"
        assert np.array_equal(rows["failed_T"], (rows["predicted_T"] != rows["truth"]).any(axis=1))
        assert str(rows["data_sha256"]) == sha256_of(shot_files["X"]) and json.loads(str(rows["overrides"])) == {"num_sets": 3}
        assert rows["converged_T"].any() and (rows["n_faults_T"] > 0).all() and (rows["weight_T"] > 0).all()
        assert 0 <= int(rows["repeat_differ"]) <= int(rows["repeat_shots"]) == SHOTS
    assert (serial["seconds_T"] > 0).all() and np.isnan(batch["seconds_T"]).all()
    assert np.array_equal(serial["truth"], load_shots(shot_files["X"]).observables)

    tcbp_member.main([*command, "--out", str(tmp_path / "serial")])              # a second run keeps the finished file
    assert "kept" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="other data or settings"):
        tcbp_member.main([*command[:-1], "num_sets=4", "--out", str(tmp_path / "serial")])
    with pytest.raises(SystemExit, match="held-out"):
        tcbp_member.main(["--data", str(tmp_path / "test_X.npz"), "--decoder", "relay", "--out", str(tmp_path / "x")])
    with pytest.raises(SystemExit, match="NAME=VALUE"):
        tcbp_member.main([*command[:-1], "num_sets", "--out", str(tmp_path / "x")])
    if importlib.util.find_spec("cudaq_qec") is None:                           # without the GPU package it says so at once
        with pytest.raises(SystemExit, match="cudaq-qec"):
            tcbp_member.main(["--data", str(shot_files["X"]), "--out", str(tmp_path / "x")])


# ---------- joint_member.py ----------

def test_joint_member_decodes_the_listed_shots_and_refuses_a_foreign_selection(tmp_path, shot_files, capsys):
    """Two listed shots on the joint problem with a small budget, by selection file and resumed; the refusals."""
    data = load_shots(shot_files["X"])
    selection = tmp_path / "small_X.selection.npz"
    np.savez(selection, shot=np.array([5, 2]), data_sha256=data.sha256)
    command = ["--data", str(shot_files["X"]), "--out", str(tmp_path / "joint"), "--processes", "1", "--relay-sets", "1"]
    joint_member.main([*command, "--shots", "2"])
    joint_member.main([*command, "--selection", str(selection)])               # shot 2 is kept, shot 5 is decoded
    assert "1 kept from an earlier run" in capsys.readouterr().out
    rows = arrays_of(tmp_path / "joint" / "small_X.joint.npz")
    assert rows["shot"].tolist() == [2, 5] and int(rows["shots_in_file"]) == SHOTS and str(rows["data_sha256"]) == data.sha256
    assert np.array_equal(rows["truth"], data.observables[[2, 5]]) and rows["predicted_J"].shape == (2, 12)
    assert np.array_equal(rows["failed_J"], (rows["predicted_J"] != rows["truth"]).any(axis=1))
    assert (rows["iterations_J"] > 0).all() and (rows["seconds_J"] > 0).all() and (rows["weight_J"] > 0).all()
    settings = json.loads(str(rows["settings_J"]))
    assert settings["settings"]["num_sets"] == 1 and settings["settings"]["stop_nconv"] == 5       # the kit's settings but the test budget
    assert joint_member.joint_problem("X", 0.004).H.shape[0] == 1728                               # every detector of the shot

    np.savez(selection, shot=np.array([1]), data_sha256="0" * 64)
    with pytest.raises(SystemExit, match="was made from a data file"):
        joint_member.main([*command, "--selection", str(selection)])
    with pytest.raises(SystemExit, match="held-out"):
        joint_member.main(["--data", str(tmp_path / "go_X.npz"), "--shots", "0", "--out", str(tmp_path / "x")])
    with pytest.raises(SystemExit, match="has shots 0 to"):
        joint_member.main([*command, "--shots", str(SHOTS)])
    with pytest.raises(SystemExit, match="--selection .* or --shots"):
        joint_member.main(command)
