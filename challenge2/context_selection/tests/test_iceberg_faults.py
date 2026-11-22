"""Which single faults do the checks catch? Verified exhaustively, two independent ways."""

from __future__ import annotations

import numpy as np
import pytest

from iceberg import encode_bus, encode_direct, layout_of, postselect
from iceberg.faults import enumerate_faults, format_report
from iceberg.frames import FrameSimulator, PauliNoise
from qsel import QaoaSimulator, qaoa_circuit, random_qubo, tune_angles

PROTECTED = ("init", "syndrome", "measure")


def instance(n: int = 4, p: int = 1, seed: int = 1, frac: float = 0.5):
    qubo = random_qubo(n, edge_fraction=frac, seed=seed)
    angles = tune_angles(qubo, p=p).angles
    return qubo, qaoa_circuit(qubo, angles), angles


def undetected(report, blocks=None):
    return [f for f in report.faults if f.outcome == "undetected error" and (blocks is None or f.block in blocks)]


# ------------------------------------------------------------ fault tolerance
@pytest.mark.parametrize("make", [encode_direct, encode_bus, lambda c: encode_bus(c, edge_mode="bus")])
@pytest.mark.parametrize("n,p", [(4, 1), (4, 2), (6, 1)])
def test_no_single_fault_in_preparation_or_readout_goes_undetected(make, n, p):
    """State preparation, the S_X check and the readout are fault tolerant to one fault."""
    qubo, bare, _ = instance(n, p, seed=n + p)
    report = enumerate_faults(make(bare), energies=qubo.energies_all())
    assert undetected(report, PROTECTED) == []
    assert all(not f.outcome.startswith("partly") for f in report.faults)
    assert report.totals()["detected"] > 0.8 * report.totals()["faults"]


def test_bare_circuit_detects_nothing():
    qubo, bare, _ = instance()
    report = enumerate_faults(bare, energies=qubo.energies_all())
    assert report.totals()["detected"] == 0
    assert report.coefficients()["discard"] == pytest.approx(0.0, abs=1e-9)


def test_each_flag_closes_a_real_hole():
    """Removing a flag lets specific single faults through; this is why they are there."""
    qubo, bare, _ = instance()
    e = qubo.energies_all()
    assert undetected(enumerate_faults(encode_direct(bare, init_flag=False), energies=e), ("init",))
    assert undetected(enumerate_faults(encode_direct(bare, sx_flag=False), energies=e), ("syndrome",))
    assert undetected(enumerate_faults(encode_bus(bare, init_flag=False), energies=e), ("init",))
    full = len(undetected(enumerate_faults(encode_bus(bare), energies=e)))
    no_copy = len(undetected(enumerate_faults(encode_bus(bare, precopy=False), energies=e)))
    assert no_copy > full          # a fault that flips S_X and its own readout qubit cancels itself


def test_single_readout_errors_are_always_caught():
    qubo, bare, _ = instance(6, 1, 3, 0.3)
    for enc in (encode_direct(bare), encode_bus(bare)):
        report = enumerate_faults(enc)
        readout = [f for f in report.faults if f.kind == "readout"]
        assert len(readout) == enc.num_qubits
        assert all(f.outcome == "detected" for f in readout)


def test_bus_removes_the_many_bit_flip_errors():
    """Direct hub gates leave undetected errors that flip most logical bits; the bus does not.

    An undetected ``X_i X_b`` flips every logical bit except ``i``. In the bus encoding
    the worst harmful undetected single fault flips at most two logical bits.
    """
    qubo, bare, _ = instance(6, 1, 3, 0.3)
    worst = {}
    for name, enc in (("direct", encode_direct(bare)), ("bus", encode_bus(bare))):
        fs = FrameSimulator(enc, level="native")
        locs = fs._locations()
        rows = len(locs)
        fx = np.zeros(rows, dtype=np.uint64)
        fz = np.zeros(rows, dtype=np.uint64)
        at: dict[int, list[int]] = {}
        for r, (pos, op, _, bits) in enumerate(locs):
            for qb, (bx, bz) in zip(op.qubits, bits):
                fx[r] |= np.uint64(bx) << np.uint64(qb)
                fz[r] |= np.uint64(bz) << np.uint64(qb)
            at.setdefault(pos, []).append(r)

        def inject(pos, op, X, Z, at=at, fx=fx, fz=fz):
            rs = at.get(pos)
            if rs:
                X[rs] ^= fx[rs]
                Z[rs] ^= fz[rs]

        _, mask = fs._propagate(rows, inject)
        flips = fs._logical_flips(mask)
        report = fs.single_faults(qubo.energies_all(), include_readout=False)
        harmful = [r for r in range(rows) if report.faults[r].outcome == "undetected error"]
        worst[name] = max(bin(int(flips[r])).count("1") for r in harmful)
    assert worst["direct"] == 5          # k - 1 logical bits flipped at once
    assert worst["bus"] <= 2


def test_bus_has_lower_first_order_error_than_bare_and_direct():
    """The headline numbers: surviving error per unit CZ error rate, at the CZ level."""
    for seed in range(3):
        qubo = random_qubo(8, edge_fraction=0.25, seed=seed)
        angles = tune_angles(qubo, p=1).angles
        bare = qaoa_circuit(qubo, angles)
        e = qubo.energies_all()
        loss = {}
        for name, circ in (("bare", bare), ("direct", encode_direct(bare)), ("bus", encode_bus(bare, seed=seed))):
            co = FrameSimulator(circ, level="cz").single_faults(e).coefficients()
            loss[name] = co["d_energy"]
        assert loss["bus"] < loss["bare"] < loss["direct"]


# ------------------------------------------------- deeper circuits (p >= 2)
def _single_fault_table(enc, energies):
    """Per single fault: bus bits flipped at the end of its block, logical bits flipped, verdict."""
    fs = FrameSimulator(enc, level="native")
    lay = layout_of(enc)
    lane = np.uint64(sum(1 << q for q in (lay.sx_ancilla, *lay.flags)))
    locs = fs._locations()
    rows = len(locs)
    fx = np.zeros(rows, dtype=np.uint64)
    fz = np.zeros(rows, dtype=np.uint64)
    at: dict[int, list[int]] = {}
    block_end: dict[int, list[int]] = {}
    ends = {}
    for pos, op in enumerate(fs.ops):
        ends[op.block] = pos                      # blocks are contiguous except repeats per layer
    last_of_run = [pos for pos, op in enumerate(fs.ops)
                   if pos + 1 == len(fs.ops) or fs.ops[pos + 1].block != op.block]
    for r, (pos, op, _, bits) in enumerate(locs):
        for qb, (bx, bz) in zip(op.qubits, bits):
            fx[r] |= np.uint64(bx) << np.uint64(qb)
            fz[r] |= np.uint64(bz) << np.uint64(qb)
        at.setdefault(pos, []).append(r)
        block_end.setdefault(next(e for e in last_of_run if e >= pos), []).append(r)
    dirty = np.zeros(rows, dtype=np.uint64)

    def inject(pos, op, X, Z):
        rs = at.get(pos)
        if rs:
            X[rs] ^= fx[rs]
            Z[rs] ^= fz[rs]
        done = block_end.get(pos)
        if done:
            dirty[done] = X[done] & lane

    _, mask = fs._propagate(rows, inject)
    report = fs.single_faults(energies, include_readout=False)
    flips = fs._logical_flips(mask)
    return locs, dirty, flips, report


@pytest.mark.parametrize("n,p,seed,frac", [(2, 2, 1, 1.0), (3, 2, 1, 1.0), (4, 1, 1, 0.5), (4, 2, 4, 1.0),
                                           (4, 3, 2, 0.7), (6, 2, 3, 0.5), (8, 2, 5, 0.4)])
def test_a_flipped_bus_qubit_always_discards_the_shot(n, p, seed, frac):
    """The bus is trustworthy: no single fault leaves it flipped and still passes the checks."""
    qubo, bare, _ = instance(n, p, seed, frac)
    for mode in ("direct", "bus"):
        enc = encode_bus(bare, edge_mode=mode)
        locs, dirty, flips, report = _single_fault_table(enc, qubo.energies_all())
        leaked = [(locs[r][1].block, locs[r][1].name, locs[r][1].qubits, locs[r][2])
                  for r in range(len(locs)) if dirty[r] and report.faults[r].p_detect < 0.5
                  and locs[r][1].block not in ("syndrome",)]
        assert leaked == []
        harmful = [r for r in range(len(locs)) if report.faults[r].outcome == "undetected error"]
        assert harmful and max(bin(int(flips[r])).count("1") for r in harmful) <= 2
        assert not [r for r in harmful if locs[r][1].block in PROTECTED]


def test_two_layer_gap_found_by_independent_review_is_closed():
    """Regression: a Y on the S_X bus qubit during a fan-out used to pass every check at p = 2.

    With only one witness copy (``precopy=1``) the fault flips the S_X qubit, its neighbour
    and S_X itself, and all three cancel in the readout. The second copy exposes it.
    """
    import math

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    bare = QuantumCircuit(4, 4)
    bare.h(range(4))
    for (i, j), th in zip([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], [0.9, 0.5, 0.7, 1.1, 0.4, 0.8]):
        bare.rzz(th, i, j)
    for i, th in enumerate([0.6, 1.0, 0.35, 0.75]):
        bare.rz(th, i)
    for i, th in enumerate([0.45, 0.95, 0.65, 0.85]):
        bare.rx(th, i)
    bare.rzz(math.pi / 2, 0, 3)
    bare.rzz(math.pi / 2, 1, 2)
    for i, th in enumerate([0.5, 0.9, 0.3, 0.7]):
        bare.rz(th, i)
    for i in range(4):
        bare.rx(math.pi / 2, i)
    bare.measure(range(4), range(4))
    ideal = Statevector(bare.remove_final_measurements(inplace=False)).probabilities()

    def kept_after_fault(enc):
        lay = layout_of(enc)
        root = lay.sx_ancilla
        labels = [name for name, count in enc.metadata["iceberg_blocks"] for _ in range(count)]
        idx = next(i for i, g in enumerate(enc.data) if labels[i] == "fields" and g.operation.name == "cx"
                   and [enc.find_bit(q).index for q in g.qubits] == [root, root - 1])
        bad = enc.copy_empty_like()
        for i, g in enumerate(enc.data):
            bad.append(g.operation, g.qubits, g.clbits)
            if i == idx:
                bad.y(root)
        probs = Statevector(bad.remove_final_measurements(inplace=False)).probabilities()
        counts = {format(i, f"0{enc.num_qubits}b"): int(round(x * 1e9)) for i, x in enumerate(probs) if round(x * 1e9)}
        logical, stats = postselect(counts, enc)
        dist = np.zeros(16)
        for key, c in logical.items():
            dist[int(key, 2)] = c
        tvd = 0.5 * np.abs(dist / dist.sum() - ideal).sum() if dist.sum() else 0.0
        return stats.kept_share, tvd

    kept_old, tvd_old = kept_after_fault(encode_bus(bare, precopy=1))
    assert kept_old == pytest.approx(1.0) and tvd_old > 0.3          # the gap: kept and wrong
    kept_new, _ = kept_after_fault(encode_bus(bare))
    assert kept_new == pytest.approx(0.0, abs=1e-9)                  # now discarded


def test_small_problems_get_a_three_qubit_lane():
    """Regression: with a two-qubit lane the second witness copy cannot exist.

    One or two paragraphs are padded to four code-logical qubits. The two faults an
    independent review found on the two-qubit lane are now discarded.
    """
    import math

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    def kept_after_y_on_root(bare, pick):
        enc = encode_bus(bare)
        lay = layout_of(enc)
        assert lay.k_code == 4 and len(lay.flags) == 2
        labels = [name for name, count in enc.metadata["iceberg_blocks"] for _ in range(count)]
        idx = pick(enc, labels, lay.sx_ancilla)
        bad = enc.copy_empty_like()
        for i, g in enumerate(enc.data):
            bad.append(g.operation, g.qubits, g.clbits)
            if i == idx:
                bad.y(lay.sx_ancilla)
        probs = Statevector(bad.remove_final_measurements(inplace=False)).probabilities()
        counts = {format(i, f"0{enc.num_qubits}b"): int(round(x * 1e9)) for i, x in enumerate(probs) if round(x * 1e9)}
        return postselect(counts, enc)[1].kept_share

    def first_cx_from_root(block):
        def pick(enc, labels, root):
            return next(i for i, g in enumerate(enc.data) if labels[i] == block and g.operation.name == "cx"
                        and root in [enc.find_bit(q).index for q in g.qubits])
        return pick

    two_layers = QuantumCircuit(2, 2)
    two_layers.h([0, 1])
    two_layers.rzz(0.9, 0, 1); two_layers.rz(0.6, 0); two_layers.rz(1.0, 1)
    two_layers.rx(0.45, 0); two_layers.rx(0.95, 1)
    two_layers.rzz(math.pi / 2, 0, 1); two_layers.rz(0.5, 0); two_layers.rz(0.9, 1)
    two_layers.rx(math.pi / 2, 0); two_layers.rx(math.pi / 2, 1)
    two_layers.measure([0, 1], [0, 1])
    assert kept_after_y_on_root(two_layers, first_cx_from_root("fields")) == pytest.approx(0.0, abs=1e-9)

    no_h = QuantumCircuit(1, 1)
    no_h.rx(0.8, 0)
    no_h.measure(0, 0)
    assert kept_after_y_on_root(no_h, first_cx_from_root("init")) == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError):
        encode_bus(no_h, precopy=3)


# ------------------------------------------------------ frames vs statevector
@pytest.mark.parametrize("name", ["bare", "direct", "direct-z", "direct-x", "bus", "bus-all"])
def test_frame_analysis_matches_brute_force_fault_by_fault(name):
    qubo, bare, _ = instance(5, 2, 2, 0.4)
    circ = {
        "bare": bare,
        "direct": encode_direct(bare),
        "direct-z": encode_direct(bare, checks="z"),
        "direct-x": encode_direct(bare, checks="x"),
        "bus": encode_bus(bare),
        "bus-all": encode_bus(bare, edge_mode="bus"),
    }[name]
    e = qubo.energies_all()
    brute = enumerate_faults(circ, energies=e)
    frames = FrameSimulator(circ, level="native").single_faults(e)
    a = {(f.kind, f.op_index, f.qubits, f.pauli): f for f in brute.faults}
    b = {(f.kind, f.op_index, f.qubits, f.pauli): f for f in frames.faults}
    assert a.keys() == b.keys()
    for key in a:
        assert a[key].p_detect == pytest.approx(b[key].p_detect, abs=1e-9)
        assert a[key].tvd == pytest.approx(b[key].tvd, abs=1e-9)
        assert a[key].d_energy == pytest.approx(b[key].d_energy, abs=1e-9)
    assert brute.locations == frames.locations


def test_cz_level_model_matches_a_compiled_circuit():
    """Our CZ-level fault model equals brute force on the transpiled {cz, rz, sx, x} circuit."""
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap

    from iceberg.sim import compact

    qubo, bare, _ = instance()
    e = qubo.energies_all()
    enc = encode_bus(bare)
    model = FrameSimulator(enc, level="cz").single_faults(e).coefficients()
    out = transpile(enc, basis_gates=["cz", "rz", "sx", "x"], coupling_map=CouplingMap.from_grid(3, 3),
                    optimization_level=1, seed_transpiler=1)
    small = compact(out)
    small.metadata = {"iceberg": layout_of(enc).to_dict()}
    brute = enumerate_faults(small, energies=e, virtual_gates=("rz",)).coefficients()
    assert brute["two_qubit_gates"] == model["two_qubit_gates"]
    assert brute["discard"] == pytest.approx(model["discard"], rel=1e-6)
    assert brute["error"] == pytest.approx(model["error"], rel=1e-6)
    assert brute["d_energy"] == pytest.approx(model["d_energy"], rel=1e-6)


def test_one_fault_model_is_consistent():
    qubo, bare, _ = instance()
    report = FrameSimulator(encode_bus(bare), level="cz").single_faults(qubo.energies_all())
    quiet = report.predict(0.0, 0.0, 0.0)
    assert quiet["kept_share"] == pytest.approx(1.0)
    assert quiet["logical_tvd"] == pytest.approx(0.0)
    noisy = report.predict(1e-4, 1e-3, 1e-3)
    assert 0.8 < noisy["kept_share"] < 1.0
    assert noisy["p_multi"] < 0.02
    assert "TOTAL" in format_report(report)


# ----------------------------------------------------------------- sampling
def test_noiseless_sampling_keeps_every_shot_and_matches_ideal():
    qubo, bare, angles = instance(6, 1, 3, 0.3)
    ideal = QaoaSimulator(qubo).probabilities(angles)
    quiet = PauliNoise(0.0, 0.0, 0.0)
    for enc in (encode_direct(bare), encode_bus(bare)):
        counts = FrameSimulator(enc, level="cz").sample(50_000, quiet, seed=3)
        logical, stats = postselect(counts, enc)
        assert stats.kept == stats.shots == 50_000
        dist = np.zeros(1 << qubo.n)
        for key, c in logical.items():
            dist[int(key, 2)] += c
        assert 0.5 * np.abs(dist / dist.sum() - ideal).sum() < 0.03


def test_sampling_is_reproducible_and_noise_discards_shots():
    qubo, bare, _ = instance(6, 1, 3, 0.3)
    enc = encode_bus(bare)
    fs = FrameSimulator(enc, level="cz")
    noise = PauliNoise(p1=5e-4, p2=5e-3, p_meas=5e-3)
    a = fs.sample(4000, noise, seed=7)
    assert a == fs.sample(4000, noise, seed=7)
    assert a != fs.sample(4000, noise, seed=8)
    _, stats = postselect(a, enc)
    predicted = fs.single_faults().predict(noise.p1, noise.p2, noise.p_meas)["kept_share"]
    assert 0.2 < stats.kept_share < 0.95
    assert abs(stats.kept_share - predicted) < 0.08      # one-fault model, loose at this noise


def test_dephasing_noise_is_caught_more_often_than_depolarizing():
    """Phase errors on CZ gates are what the checks are best at catching."""
    qubo, bare, _ = instance(6, 1, 3, 0.3)
    report = FrameSimulator(encode_bus(bare), level="cz").single_faults(qubo.energies_all())
    two = [f for f in report.faults if f.kind == "gate2"]
    phase = [f for f in two if set(f.pauli) <= {"I", "Z"}]
    assert np.mean([f.accepted_error for f in phase]) <= np.mean([f.accepted_error for f in two]) + 1e-12
    w = PauliNoise(two_qubit="dephasing").weights()
    assert w.sum() == pytest.approx(1.0) and np.count_nonzero(w) == 3


@pytest.mark.slow
def test_frame_sampler_matches_exact_density_matrix_simulation():
    """Multi-fault behaviour: kept share and mean energy agree with Aer's exact probabilities."""
    from itertools import product

    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, pauli_error

    from iceberg import postselect_probs

    p1, p2 = 0.004, 0.02
    model = NoiseModel()
    model.add_all_qubit_quantum_error(
        pauli_error([("X", p1 / 3), ("Y", p1 / 3), ("Z", p1 / 3), ("I", 1 - p1)]), ["h", "rx"])
    two = [("".join(p), p2 / 15) for p in product("IXYZ", repeat=2) if p != ("I", "I")] + [("II", 1 - p2)]
    model.add_all_qubit_quantum_error(pauli_error(two), ["cx", "rzz", "rxx"])
    aer = AerSimulator(method="density_matrix", noise_model=model,
                       basis_gates=["h", "rx", "rz", "cx", "rzz", "rxx"])
    noise = PauliNoise(p1=p1, p2=p2, p_meas=0.0, noisy_rz=False)
    qubo, bare, _ = instance(4, 2, 7)
    e = qubo.energies_all()
    for enc in (encode_direct(bare), encode_bus(bare)):
        probe = enc.remove_final_measurements(inplace=False)
        probe.save_probabilities()
        probs = np.asarray(aer.run(probe).result().data()["probabilities"])
        mass, kept = postselect_probs(probs, layout_of(enc))
        exact_energy = float(mass @ e) / kept
        fs = FrameSimulator(enc, level="native")
        shots = 2_000_000
        logical, stats = postselect(fs.sample(shots, noise, seed=11), enc)
        energy = sum(c * e[int(k, 2)] for k, c in logical.items()) / stats.kept
        assert abs(stats.kept_share - kept) < 5 * np.sqrt(kept * (1 - kept) / shots)
        assert abs(energy - exact_energy) < 5 * float(e.std()) / np.sqrt(stats.kept)
