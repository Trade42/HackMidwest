"""Execution backends: ideal, local noisy (frame sampler and Aer), and their helpers."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli

import iceberg
from qsel import QaoaSimulator, qaoa_circuit, random_qubo, tune_angles
from qsel.backends import (Backend, IdealBackend, LocalNoisyBackend, circuit_fingerprint,
                           circuit_stats, compact_circuit, make_backend)
from qsel.frame_sim import GenericFrameSampler, UnsupportedCircuit, _conj


def tvd(counts: dict[str, int], reference: np.ndarray) -> float:
    total = sum(counts.values())
    emp = np.zeros_like(reference)
    for key, c in counts.items():
        emp[int(key, 2)] = c / total
    return 0.5 * float(np.abs(emp - reference).sum())


@pytest.fixture(scope="module")
def problem():
    qubo = random_qubo(4, edge_fraction=0.5, seed=2)
    angles = tune_angles(qubo, p=2).angles
    bare = qaoa_circuit(qubo, angles)
    # "direct" is the 8-qubit textbook encoding; it needs routing on a grid, which is
    # what these tests exercise. The bus encoding (9 qubits here) has its own tests below.
    return qubo, angles, bare, iceberg.encode(bare, style="direct")


@pytest.fixture(scope="module")
def bus_circuit(problem):
    return iceberg.encode(problem[2])


# ------------------------------------------------------------------- helpers
def test_circuit_stats_ignores_barriers_and_idle_qubits():
    qc = QuantumCircuit(5, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.barrier()
    qc.cz(1, 2)
    qc.measure([0, 1], [0, 1])
    st = circuit_stats(qc)
    assert st["qubits"] == 3 and st["two_qubit_gates"] == 2 and st["two_qubit_depth"] == 2
    assert st["depth"] == 4 and "barrier" not in st["count_ops"]
    small = compact_circuit(qc)
    assert small.num_qubits == 3 and small.num_clbits == 2
    assert circuit_stats(small)["two_qubit_gates"] == 2


def test_fingerprint_tracks_structure_not_metadata(problem):
    qubo, angles, bare, _ = problem
    again = qaoa_circuit(qubo, angles)
    again.metadata = {"other": 1}
    assert circuit_fingerprint(bare) == circuit_fingerprint(again)
    other = qaoa_circuit(qubo, type(angles)(tuple(g + 0.01 for g in angles.gammas), angles.betas))
    assert circuit_fingerprint(bare) != circuit_fingerprint(other)


def test_make_backend():
    from qsel.pauli_backend import PauliFrameBackend

    assert isinstance(make_backend({"kind": "ideal", "seed": 3}), IdealBackend)
    for kind in ("local-aer", "local_aer", "local-noisy"):
        nb = make_backend({"kind": kind, "p2": 0.01, "grid_shape": [3, 3], "coupling": "grid"})
        assert isinstance(nb, LocalNoisyBackend) and nb.grid_shape == (3, 3) and nb.p2 == 0.01
        assert isinstance(nb, Backend)
    pf = make_backend({"kind": "local-pauli", "noise": {"p2": 0.02, "p1": 0.0, "p_meas": 0.0},
                       "grid_shape": [4, 6], "seed": 4})
    assert isinstance(pf, PauliFrameBackend) and pf.noise.p2 == 0.02 and pf.grid_shape == (4, 6)
    assert isinstance(make_backend({"kind": "local-pauli"}), PauliFrameBackend)
    with pytest.raises(ValueError):
        make_backend({"kind": "nope"})


# --------------------------------------------------------------------- ideal
def test_ideal_backend_reproduces_exact_probabilities(problem):
    qubo, angles, bare, enc = problem
    backend = IdealBackend(seed=7)
    exact = QaoaSimulator(qubo).probabilities(angles)
    assert np.abs(backend.probabilities(bare) - exact).max() < 1e-10
    counts = backend.run([bare], 40000)[0]
    assert sum(counts.values()) == 40000
    assert tvd(counts, exact) < 0.02
    logical, stats = iceberg.postselect(backend.run([enc], 40000)[0], enc)
    assert stats.kept_share == 1.0
    assert tvd(logical, exact) < 0.02


def test_ideal_backend_is_deterministic_and_order_independent(problem):
    _, _, bare, enc = problem
    a = IdealBackend(seed=1).run([bare, enc], 500)
    b = IdealBackend(seed=1).run([enc, bare], 500)
    assert a[0] == b[1] and a[1] == b[0]
    assert IdealBackend(seed=2).run([bare], 500)[0] != a[0]


# --------------------------------------------------------------- local noisy
@pytest.mark.parametrize("method", ["frame", "aer"])
def test_noiseless_noisy_backend_equals_ideal_through_grid_routing(problem, method):
    """Zero noise on a routed, basis-translated circuit: the measurement map survived."""
    qubo, angles, bare, enc = problem
    exact = QaoaSimulator(qubo).probabilities(angles)
    backend = LocalNoisyBackend(p1=0, p2=0, p_meas=0, coupling="grid", grid_shape=(3, 3),
                                layout_trials=2, seed=0, method=method)
    c_bare, c_enc = backend.run([bare, enc], 20000)
    assert backend.last_methods == [method, method]
    assert set(backend.last_compiled[1]["count_ops"]) <= {"cz", "rz", "sx", "x", "measure"}
    assert tvd(c_bare, exact) < 0.03
    logical, stats = iceberg.postselect(c_enc, enc)
    assert stats.kept_share == 1.0
    assert tvd(logical, exact) < 0.03


def test_noisy_iceberg_discards_shots_and_noise_hurts(problem):
    qubo, angles, bare, enc = problem
    backend = LocalNoisyBackend(p1=0.001, p2=0.01, p_meas=0.01, coupling="grid", grid_shape=(3, 3),
                                layout_trials=2, seed=3)
    c_bare, c_enc = backend.run([bare, enc], 20000)
    _, stats = iceberg.postselect(c_enc, enc)
    assert 0.2 < stats.kept_share < 0.95
    assert stats.fail_sz > 0 and stats.fail_sx > 0
    exact = QaoaSimulator(qubo).probabilities(angles)
    assert tvd(c_bare, exact) > 0.01
    assert backend.run([bare], 2000) == backend.run([bare], 2000)          # seeded


def test_bus_circuit_is_compiled_in_place_without_routing(bus_circuit):
    from qsel.pauli_backend import PauliFrameBackend

    backend = LocalNoisyBackend(coupling="grid", grid_shape=(3, 3))
    stats = backend.compiled_stats(bus_circuit)
    assert stats["qubits"] == bus_circuit.num_qubits == 9
    assert set(stats["count_ops"]) <= {"cz", "rz", "sx", "x", "measure"}
    reference = PauliFrameBackend(grid_shape=(3, 3)).stats(bus_circuit)["two_qubit_gates"]
    assert reference - 4 <= stats["two_qubit_gates"] <= reference        # no SWAP was added
    with pytest.raises(ValueError):
        LocalNoisyBackend(coupling="grid", grid_shape=(2, 4)).compile(bus_circuit)
    logical, post = iceberg.postselect(
        LocalNoisyBackend(p1=0, p2=0, p_meas=0, grid_shape=(3, 3)).run([bus_circuit], 3000)[0], bus_circuit)
    assert post.kept_share == 1.0 and sum(logical.values()) == 3000


def test_aer_and_the_pauli_frame_backend_agree_on_a_bus_circuit(problem, bus_circuit):
    """Two independent simulators and two compilers, same noise: same statistics."""
    from qsel.pauli_backend import PauliFrameBackend
    from qsel.sampling import sample_stats

    qubo = problem[0]
    shots = 40000
    for p2 in (0.01, 0.03):
        aer = LocalNoisyBackend(p1=0.0, p2=p2, p_meas=0.01, grid_shape=(3, 3), seed=1)
        fast = PauliFrameBackend({"p1": 0.0, "p2": p2, "p_meas": 0.01}, grid_shape=(3, 3), seed=1)
        la, sa = iceberg.postselect(aer.run([bus_circuit], shots)[0], bus_circuit)
        lf, sf = iceberg.postselect(fast.run([bus_circuit], shots)[0], bus_circuit)
        assert aer.last_methods == ["aer"]
        assert aer.last_compiled[0]["two_qubit_gates"] == fast.last_stats[0]["two_qubit_gates"]
        assert sa.kept_share == pytest.approx(sf.kept_share, abs=0.015)
        assert sample_stats(qubo, la).approx_ratio == pytest.approx(sample_stats(qubo, lf).approx_ratio, abs=0.015)
        assert sa.kept_share < 0.7


def test_compile_keeps_the_best_trial_and_caches(problem):
    _, _, _, enc = problem
    one = LocalNoisyBackend(coupling="grid", grid_shape=(3, 3), layout_trials=1, seed=0)
    many = LocalNoisyBackend(coupling="grid", grid_shape=(3, 3), layout_trials=6, seed=0)
    assert many.compiled_stats(enc)["two_qubit_gates"] <= one.compiled_stats(enc)["two_qubit_gates"]
    assert many.compile(enc) is many.compile(enc)
    full = LocalNoisyBackend(coupling="full")
    assert full.compiled_stats(enc)["two_qubit_gates"] <= many.compiled_stats(enc)["two_qubit_gates"]
    with pytest.raises(ValueError):
        LocalNoisyBackend(coupling="grid", grid_shape=(2, 2)).compile(enc)


# -------------------------------------------------------------- frame sampler
def test_pauli_conjugation_matches_qiskit():
    rng = np.random.default_rng(0)
    gates_1q = ["h", "s", "sdg", "x", "y", "z", "sx", "sxdg"]
    gates_2q = ["cx", "cz", "swap"]
    for _ in range(60):
        n = 4
        qc = QuantumCircuit(n)
        prims = []
        for _ in range(12):
            if rng.random() < 0.4:
                g = str(rng.choice(gates_2q))
                a, b = (int(v) for v in rng.choice(n, 2, replace=False))
                getattr(qc, g)(a, b)
                prims.append((g, (a, b)))
            else:
                g = str(rng.choice(gates_1q))
                a = int(rng.integers(n))
                getattr(qc, g)(a)
                prims.append((g, (a,)))
        pauli = Pauli("".join(rng.choice(list("IXYZ"), n)))
        X = pauli.x.astype(np.uint8)[None, :].copy()
        Z = pauli.z.astype(np.uint8)[None, :].copy()
        r = np.zeros(1, dtype=np.uint8)
        for g, qs in prims:
            _conj(g, qs, X, Z, r)
        want = pauli.evolve(qc, frame="s")
        assert (want.x.astype(np.uint8) == X[0]).all() and (want.z.astype(np.uint8) == Z[0]).all()
        assert want.phase == (2 * int(r[0])) % 4


def test_frame_sampler_ideal_distribution_is_exact(problem):
    _, _, bare, enc = problem
    backend = LocalNoisyBackend(coupling="grid", grid_shape=(3, 3), layout_trials=2, seed=0)
    ideal = IdealBackend()
    for circuit in (bare, enc):
        frame = GenericFrameSampler(compact_circuit(backend.compile(circuit)))
        assert np.abs(frame.ideal_probabilities() - ideal.probabilities(circuit)).max() < 1e-6
    # logical-level circuits (rzz, rxx, cx, h) are supported too
    frame = GenericFrameSampler(enc)
    assert np.abs(frame.ideal_probabilities() - ideal.probabilities(enc)).max() < 1e-6
    assert len(frame.support) == 2 ** (4 + 1)          # k + 1 qubits' worth, not 2**8


def test_frame_sampler_matches_aer_density_matrix(problem):
    """Exact reference: Aer's density-matrix probabilities under the same noise model."""
    from qiskit_aer import AerSimulator

    _, _, _, enc = problem
    backend = LocalNoisyBackend(p1=0.004, p2=0.03, p_meas=0.0, coupling="grid", grid_shape=(3, 3),
                                layout_trials=2, seed=1)
    small = compact_circuit(backend.compile(enc))
    dm = small.copy()
    dm.remove_final_measurements(inplace=True)
    dm.save_probabilities()
    by_qubit = AerSimulator(method="density_matrix", noise_model=backend.noise_model()).run(dm) \
        .result().data(0)["probabilities"]
    idx = np.arange(len(by_qubit))
    clbits = np.zeros_like(idx)
    for inst in small.data:
        if inst.operation.name == "measure":
            qb, cb = small.find_bit(inst.qubits[0]).index, small.find_bit(inst.clbits[0]).index
            clbits |= ((idx >> qb) & 1) << cb
    exact = np.bincount(clbits, weights=by_qubit, minlength=1 << small.num_clbits)
    shots = 120000
    counts = GenericFrameSampler(small).sample(shots, p1=0.004, p2=0.03, p_meas=0.0, seed=5)
    noise_scale = 0.4 * float(np.sqrt(exact * (1 - exact) / shots).sum())
    assert tvd(counts, exact) < 2.5 * noise_scale + 0.002
    # and the noise is not negligible, so the comparison means something
    assert tvd(counts, IdealBackend().probabilities(enc)) > 0.2


def test_frame_sampler_readout_flips():
    qc = QuantumCircuit(2, 2)
    qc.x(0)
    qc.measure([0, 1], [0, 1])
    counts = GenericFrameSampler(qc).sample(40000, p_meas=0.1, seed=2)
    assert counts["01"] / 40000 == pytest.approx(0.81, abs=0.01)
    assert counts["00"] / 40000 == pytest.approx(0.09, abs=0.01)


def test_frame_sampler_rejects_unsupported_gates_and_backend_falls_back():
    qc = QuantumCircuit(2, 2)
    qc.u(0.3, 0.2, 0.1, 0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    with pytest.raises(UnsupportedCircuit):
        GenericFrameSampler(qc)
    backend = LocalNoisyBackend(coupling="full", basis=("cx", "u"), p1=0.0, p2=0.01, p_meas=0.0)
    counts = backend.run([qc], 500)[0]
    assert backend.last_methods == ["aer"] and sum(counts.values()) == 500
    with pytest.raises(UnsupportedCircuit):
        LocalNoisyBackend(coupling="full", basis=("cx", "u"), method="frame").run([qc], 10)


@pytest.mark.slow
def test_frame_sampler_agrees_with_aer_trajectories_beyond_density_matrix_size():
    """8 paragraphs, direct encoding, 12 qubits, routed on a grid: compare statistics."""
    qubo = random_qubo(8, edge_fraction=0.25, seed=3)
    enc = iceberg.encode(qaoa_circuit(qubo, tune_angles(qubo, p=1).angles), style="direct")
    kept = {}
    for method in ("frame", "statevector"):
        backend = LocalNoisyBackend(coupling="grid", grid_shape=(3, 4), layout_trials=2, seed=0, method=method)
        _, ps = iceberg.postselect(backend.run([enc], 1500)[0], enc)
        kept[method] = ps.kept_share
    assert kept["frame"] == pytest.approx(kept["statevector"], abs=0.06)
