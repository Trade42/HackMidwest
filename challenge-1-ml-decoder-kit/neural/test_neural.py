"""Tests of the neural decoder. They run on the CPU in under two minutes.

  python -m pytest -p no:cacheprovider neural       (from challenge-1-ml-decoder-kit/)
"""
import inspect

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from torch.nn import functional as F

from circuit import dem_matrices, detector_ids, sample
from make_circuit import code_layout, memory_experiment
from neural import evaluate_stim, train
from neural.graph import build_graph
from neural.model import EVENT_CODES, CascadeDecoder
from neural.sampler import DemSampler, compare_with_stim

torch.set_num_threads(1)   # small tensors; one thread is fastest and repeatable
BASES = ["X", "Z"]
GENERATORS = [(1, 0), (0, 1)]   # the two shifts that generate every translation of the torus
# A very small experiment for the training tests: the [[72,12,6]] code with 2 cycles.
SMALL = train.DEFAULTS | {"code": "bb72", "rounds": 2, "width": 16, "depth": 2, "batch": 8, "steps": 6,
                          "p_start": 0.001, "p_train": 0.003, "steps_start": 2, "steps_anneal": 2, "anneal_levels": 2,
                          "warmup_steps": 2, "lr_decay_steps": 6, "report_every": 3, "report_shots": 40,
                          "report_p": [0.001, 0.003], "seed": 5}


@pytest.fixture(scope="module", params=BASES)
def gross(request):
    return build_graph("gross", request.param)


# ---------- the structure tables against the kit ----------

@pytest.mark.parametrize("code", ["gross", "bb72"])
def test_every_check_has_its_six_data_qubits(code):
    graph, layout = build_graph(code, "X"), code_layout(code)
    supports = np.vstack([layout.hx, layout.hz])
    for check, neighbours in enumerate(graph.check_neighbours):
        assert len(set(neighbours)) == 6 and set(neighbours) == set(np.nonzero(supports[check])[0])


def test_every_data_qubit_has_three_checks_of_each_type(gross):
    n2 = gross.n2
    for data, checks in enumerate(gross.data_neighbours):
        assert (checks[:3] < n2).all() and (checks[3:] >= n2).all() and len(set(checks)) == 6
        assert all(data in gross.check_neighbours[check] for check in checks)


@pytest.mark.parametrize("shift", GENERATORS)
def test_neighbour_tables_move_with_the_torus(gross, shift):
    """Neighbour s of a shifted node is the shifted neighbour s. So one weight per slot is one weight per torus offset."""
    moved = gross.torus_shift(*shift)
    for table in (gross.check_neighbours, gross.data_neighbours):
        assert np.array_equal(table[moved], moved[table])


def test_two_hops_reach_22_checks(gross):
    """Check to data to check reaches 22 checks, itself included: the count the paper gives for its convolution."""
    reached = {int(check) for data in gross.check_neighbours[0] for check in gross.data_neighbours[data]}
    assert len(reached) == 22 and sum(check < gross.n2 for check in reached) == 13


def test_detector_layout(gross):
    n2, rounds = gross.n2, gross.rounds
    basis_type = "XZ".index(gross.basis)
    basis, other = gross.basis_detectors, np.setdiff1d(np.arange(gross.num_detectors), gross.basis_detectors)
    assert (len(basis), len(other)) == (936, 792)
    # The layout the team's reviewer confirmed for the 936: row = 72 * round + 6 * x + y.
    assert np.array_equal(gross.detector_round[basis] * n2 + gross.detector_check[basis] - basis_type * n2, np.arange(936))
    assert (gross.detector_check[other] // n2 == 1 - basis_type).all()
    assert set(gross.detector_round[other]) == set(range(1, rounds))   # no detector in round 0 or at the readout
    places = gross.detector_round * 2 * n2 + gross.detector_check
    assert len(set(places)) == gross.num_detectors


def detector_shift(graph, shift):
    """Where every detector goes when its check moves with the torus and its round stays."""
    places = graph.detector_round * 2 * graph.n2 + graph.detector_check
    at_place = np.full(places.max() + 1, -1)
    at_place[places] = np.arange(graph.num_detectors)
    target = at_place[graph.detector_round * 2 * graph.n2 + graph.torus_shift(*shift)[graph.detector_check]]
    assert (target >= 0).all()
    return target


def test_error_model_is_the_same_after_a_torus_shift(gross):
    """Proof of the layout of all 1,728 detectors: a shift maps the full error model onto itself.

    Every mechanism's set of detectors, moved by the shift, is again a mechanism's set of
    detectors, with the same probability. A set is compared through the sum of one random
    64-bit number per detector, which no two different sets share in practice.
    """
    H, _, priors = dem_matrices(memory_experiment("gross", gross.basis, p=0.001))
    H = H.tocsc()
    tag = np.random.default_rng(0).integers(0, 2 ** 63, size=H.shape[0], dtype=np.uint64)

    def mechanisms(relabel):
        """(tag of the detector set, probability) of every mechanism, sorted."""
        set_tags = np.add.reduceat(tag[relabel[H.indices]], H.indptr[:-1])
        order = np.lexsort((priors, set_tags))
        return set_tags[order], priors[order]

    assert np.diff(H.indptr).min() > 0                     # reduceat needs every mechanism to flip a detector
    original_tags, original_priors = mechanisms(np.arange(H.shape[0]))
    assert len(np.unique(original_tags)) > 0.9 * H.shape[1]
    for shift in GENERATORS:
        tags, shifted_priors = mechanisms(detector_shift(gross, shift))
        assert np.array_equal(tags, original_tags) and np.allclose(shifted_priors, original_priors, rtol=1e-9)
    scrambled = np.random.default_rng(1).permutation(H.shape[0])
    assert not np.array_equal(mechanisms(scrambled)[0], original_tags)    # the test can fail


def test_observables_read_the_logical_operators_of_the_memory_basis(gross):
    """X memory is labelled by the X logical operators: they commute with every Z check and are not X stabilizers."""
    layout = code_layout("gross")
    same, other = (layout.hx, layout.hz) if gross.basis == "X" else (layout.hz, layout.hx)
    support = gross.observable_support
    assert support.shape == (12, 144)
    assert not ((other @ support.T) % 2).any()
    from make_circuit import gf2_rank
    assert gf2_rank(np.vstack([same, support])) == gf2_rank(same) + 12


# ---------- the model ----------

def test_shifting_the_syndrome_shifts_the_activations(gross):
    torch.manual_seed(0)
    model = CascadeDecoder("gross", gross.basis, width=16, depth=2).double()
    detectors = (torch.rand(2, gross.num_detectors) < 0.05).long()
    for shift in [(1, 0), (0, 1), (5, 2)]:
        nodes = torch.from_numpy(gross.torus_shift(*shift))
        shifted = torch.zeros_like(detectors)
        shifted[:, torch.from_numpy(detector_shift(gross, shift))] = detectors
        for mode in (model.eval, model.train):
            mode()
            with torch.no_grad():
                before, after = model.features(detectors), model.features(shifted)
            moved = torch.zeros_like(before)
            moved[:, :, nodes] = before
            assert torch.allclose(moved, after, atol=1e-10) and before.abs().mean() > 0.1


def test_readout_equals_scatter_then_average():
    """The readout averages over time before it scatters to the data qubits. The paper's order gives the same logits."""
    torch.manual_seed(0)
    model = CascadeDecoder("bb72", "X", width=16, depth=1).double().eval()
    features = model.features((torch.rand(3, model.num_inputs) < 0.1).long())
    on_data = model.to_data.checks_to_data(F.silu(model.norm(features)))      # [shots, cycles, data, width]
    pooled = torch.matmul(model.pool, on_data.mean(dim=1))
    assert torch.allclose(model.readout(features), model.head(pooled).squeeze(-1), atol=1e-10)


def test_basis_input_leaves_the_other_checks_without_a_detector():
    graph = build_graph("bb72", "Z")
    full = CascadeDecoder("bb72", "Z", detectors="all", width=8, depth=1)
    basis = CascadeDecoder("bb72", "Z", detectors="basis", width=8, depth=1)
    basis.load_state_dict(full.state_dict())
    assert (full.num_inputs, basis.num_inputs) == (graph.num_detectors, len(graph.basis_detectors))
    detectors = (torch.rand(2, graph.num_detectors) < 0.2).long()
    all_in, basis_in = full.embed(detectors), basis.embed(detectors[:, torch.from_numpy(graph.basis_detectors)])
    z_checks, x_checks = slice(graph.n2, 2 * graph.n2), slice(0, graph.n2)
    assert torch.equal(all_in[:, :, z_checks], basis_in[:, :, z_checks])            # the memory basis: same input
    no_detector = basis.embedding.weight[EVENT_CODES * 0 + 2]
    assert (basis_in[:, :, x_checks] == no_detector).all()                          # the other checks: nothing
    assert (all_in[:, 0, x_checks] == no_detector).all() and not (all_in[:, 1, x_checks] == no_detector).all()


def test_the_model_sees_detectors_only():
    assert list(inspect.signature(CascadeDecoder.forward).parameters) == ["self", "detectors"]
    model = CascadeDecoder("bb72", "X", rounds=2, width=8, depth=1)
    assert model(torch.zeros(4, model.num_inputs, dtype=torch.uint8)).shape == (4, 12)
    with pytest.raises(ValueError):
        model(torch.zeros(4, model.num_inputs + 12, dtype=torch.uint8))             # no room for labels


def test_mixed_precision_runs():
    model = CascadeDecoder("bb72", "X", rounds=2, width=8, depth=1)
    detectors = torch.zeros(4, model.num_inputs, dtype=torch.uint8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        features, logits = model.features(detectors), model(detectors)
    F.binary_cross_entropy_with_logits(logits.float(), torch.zeros(4, 12)).backward()
    assert features.dtype == torch.bfloat16 and logits.dtype == torch.bfloat16
    assert train.predict(model.eval(), detectors, torch.bfloat16).dtype == torch.uint8


def test_tiny_model_overfits_a_small_fixed_batch():
    torch.manual_seed(0)
    sampler = DemSampler("bb72", "X", rounds=2, noise_levels=[0.003])
    detectors, observables = sampler.sample(16, 0.003, torch.Generator().manual_seed(1))
    assert 0.05 < observables.float().mean() < 0.95
    model = CascadeDecoder("bb72", "X", rounds=2, width=16, depth=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(60):
        logits = model(detectors)
        loss = F.binary_cross_entropy_with_logits(logits, observables.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.05 and torch.equal((logits > 0).to(torch.uint8), observables)


# ---------- the sampler ----------

@pytest.mark.parametrize("detectors", ["all", "basis"])
def test_sampled_syndromes_are_H_times_the_faults(detectors):
    sampler = DemSampler("bb72", "Z", detectors=detectors, noise_levels=[0.004])
    H, L, priors = dem_matrices(memory_experiment("bb72", "Z", p=0.004), "Z" if detectors == "basis" else None)
    syndromes, labels, (shot, mechanism) = sampler.sample(300, 0.004, torch.Generator().manual_seed(3), return_faults=True)
    faults = sp.csr_matrix((np.ones(len(shot), dtype=np.int64), (shot.numpy(), mechanism.numpy())), shape=(300, H.shape[1]))
    assert np.array_equal(syndromes.numpy(), (faults @ H.T.astype(np.int64)).toarray() % 2)
    assert np.array_equal(labels.numpy(), (faults @ L.T.astype(np.int64)).toarray() % 2)
    expected = 300 * priors.sum()
    assert abs(len(shot) - expected) < 5 * np.sqrt(expected)        # the faults fire at the kit's priors
    assert syndromes.shape == (300, 252 if detectors == "basis" else 432)


@pytest.mark.parametrize("detectors", ["all", "basis"])
def test_sampler_agrees_with_stim(detectors):
    result = compare_with_stim("bb72", "X", None, detectors, p=0.004, shots=20000, seed=11)
    assert result["passed"], result


def test_same_seed_gives_the_same_shots():
    sampler = DemSampler("bb72", "X", rounds=2, noise_levels=[0.002, 0.004])
    first = sampler.sample(50, 0.004, torch.Generator().manual_seed(9))
    again = sampler.sample(50, 0.004, torch.Generator().manual_seed(9))
    other = sampler.sample(50, 0.004, torch.Generator().manual_seed(10))
    assert torch.equal(first[0], again[0]) and torch.equal(first[1], again[1]) and not torch.equal(first[0], other[0])
    low, high = (sampler.sample(2000, p, torch.Generator().manual_seed(9))[0].float().mean() for p in (0.002, 0.004))
    assert low < high


# ---------- the trainer ----------

def test_curriculum_and_learning_rate_schedules():
    config = train.DEFAULTS | {"p_start": 0.001, "p_train": 0.005, "anneal_levels": 4, "steps_start": 10,
                               "steps_anneal": 40, "warmup_steps": 5, "lr_decay_steps": 100, "lr": 1.0}
    noise = [train.noise_at(step, config) for step in range(80)]
    assert noise[:10] == [0.001] * 10 and noise[50:] == [0.005] * 30
    assert noise == sorted(noise) and sorted(set(noise)) == [0.001, 0.002, 0.003, 0.004, 0.005]
    assert [noise.count(p) for p in (0.002, 0.003, 0.004)] == [10, 10, 10]
    rate = [train.learning_rate_at(step, config) for step in range(200)]
    assert rate[0] == pytest.approx(0.2) and max(rate) == pytest.approx(rate[4], rel=0.01)
    assert rate[100:] == [pytest.approx(0.1)] * 100 and rate[5:101] == sorted(rate[5:101], reverse=True)


def test_score_counts_shots_and_bits():
    observables = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=torch.uint8)
    predictions = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.uint8)
    assert train.score(predictions, observables) == {"shots": 4, "bits": 12, "shots_wrong": 2, "bits_right": 10,
                                                     "shots_wrong_zero": 2, "bits_right_zero": 9}


def state_of(run):
    return [run.model.state_dict(), run.ema_model.state_dict(), run.optimizer.state_dict()["state"]]


def same(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[key], b[key]) for key in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def test_checkpoint_round_trips_and_resumes_to_the_same_steps(tmp_path):
    straight = train.new_run(SMALL, "cpu")
    for _ in range(6):
        train.train_step(straight)

    stopped = train.new_run(SMALL, "cpu")
    for _ in range(3):
        train.train_step(stopped)
    train.save_checkpoint(stopped, tmp_path / "checkpoint.pt")
    resumed = train.load_checkpoint(tmp_path / "checkpoint.pt", "cpu")
    assert resumed.step == 3 and resumed.config == stopped.config and same(state_of(resumed), state_of(stopped))
    for _ in range(3):
        train.train_step(resumed)
    assert resumed.step == 6 and same(state_of(resumed), state_of(straight))

    decoder, saved = train.load_decoder(tmp_path / "checkpoint.pt")
    assert same(decoder.state_dict(), stopped.ema_model.state_dict()) and not decoder.training
    assert saved["config"]["seed"] == 5 and saved["config"]["basis"] == "X" and saved["step"] == 3
    assert train.noise_levels(saved["config"]) == [0.001, 0.002, 0.003] and len(saved["commit"]) > 6
    assert {key: saved["config"][key] for key in train.MODEL_KEYS} == decoder.config


def test_a_different_seed_gives_a_different_run():
    runs = [train.new_run(SMALL | {"seed": seed}, "cpu") for seed in (5, 6)]
    for run in runs:
        train.train_step(run)
    assert not same(runs[0].model.state_dict(), runs[1].model.state_dict())


def log_lines(folder):
    """The lines of a run's log that do not depend on the machine's speed."""
    return [line for line in (folder / "train.log").read_text().splitlines() if not line.startswith("speed")]


def test_command_runs_twice_the_same_and_resumes(tmp_path):
    options = ["--tiny", "--code", "bb72", "--rounds", "2", "--width", "16", "--depth", "1", "--batch", "8",
               "--p-start", "0.001", "--p-train", "0.003", "--steps-start", "2", "--steps-anneal", "2",
               "--anneal-levels", "2", "--lr-decay-steps", "6", "--report-every", "3", "--report-shots", "40",
               "--report-p", "0.001", "0.003", "--seed", "4", "--threads", "1"]
    train.main(["--out", str(tmp_path / "a"), "--steps", "6", *options])
    train.main(["--out", str(tmp_path / "b"), "--steps", "6", *options])
    assert log_lines(tmp_path / "a")[:-1] == log_lines(tmp_path / "b")[:-1]       # the last line names the folder

    train.main(["--out", str(tmp_path / "c"), "--steps", "3", *options])
    train.main(["--out", str(tmp_path / "c"), "--resume", str(tmp_path / "c" / "checkpoint.pt"), "--steps", "6"])
    assert log_lines(tmp_path / "a")[-5:-1] == log_lines(tmp_path / "c")[-5:-1]   # the report of step 6
    assert log_lines(tmp_path / "a")[-5].startswith("step 6 of 6")
    with pytest.raises(SystemExit):
        train.main(["--out", str(tmp_path / "c"), "--resume", str(tmp_path / "c" / "checkpoint.pt"), "--width", "64"])


# ---------- the evaluation on Stim shots ----------

def test_wilson_interval_and_rate_per_cycle():
    assert evaluate_stim.wilson_interval(5, 100) == pytest.approx((0.0215, 0.1118), abs=5e-4)
    assert evaluate_stim.wilson_interval(0, 100) == pytest.approx((0.0, 0.0370), abs=5e-4)
    assert evaluate_stim.wilson_interval(100, 100)[1] == 1.0
    assert evaluate_stim.per_cycle(0.5, 1) == 0.5 and evaluate_stim.per_cycle(1 - 0.5 ** 12, 12) == pytest.approx(0.5)


@pytest.mark.parametrize("detectors", ["all", "basis"])
def test_evaluation_on_a_saved_file_of_stim_shots(tmp_path, detectors):
    run = train.new_run(SMALL | {"detectors": detectors}, "cpu")
    train.train_step(run)
    checkpoint = str(tmp_path / "checkpoint.pt")
    train.save_checkpoint(run, checkpoint)
    circuit = memory_experiment("bb72", "X", 2, 0.003)
    syndromes, labels = sample(circuit, 300, seed=21)
    saved = {"p": 0.003, "seed": 21, "basis": "X", "detectors": syndromes, "observables": labels,
             "basis_detectors": detector_ids(circuit, "X")}                     # the arrays of make_dataset.py
    np.savez_compressed(tmp_path / "validation_X.npz", **saved)
    np.savez_compressed(tmp_path / "test_X.npz", **saved)

    numbers = evaluate_stim.main(["--checkpoint", checkpoint, "--data", str(tmp_path / "validation_X.npz"),
                                  "--latency-shots", "5", "--threads", "1", "--save", str(tmp_path / "result")])
    model, _ = train.load_decoder(checkpoint)
    inputs = syndromes if detectors == "all" else syndromes[:, detector_ids(circuit, "X")]
    wrong = (train.predict(model, torch.from_numpy(inputs)).numpy() != labels).any(axis=1)
    assert (numbers["failures"], numbers["shots"]) == (int(wrong.sum()), 300)
    assert numbers["always_zero_failures"] == int(labels.any(axis=1).sum()) and numbers["latency_shots"] == 5
    assert numbers["failure_interval"][0] <= numbers["failure_fraction"] <= numbers["failure_interval"][1]
    assert np.array_equal(np.load(tmp_path / "result.npz")["failed"], wrong)

    with pytest.raises(SystemExit):                                              # held-out shots need --final
        evaluate_stim.main(["--checkpoint", checkpoint, "--data", str(tmp_path / "test_X.npz")])
    sampled = evaluate_stim.main(["--checkpoint", checkpoint, "--p", "0.003", "--shots", "50", "--seed", "3",
                                  "--latency-shots", "2", "--threads", "1"])
    assert sampled["shots"] == 50
