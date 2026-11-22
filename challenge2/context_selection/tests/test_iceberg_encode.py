"""The encoder must reproduce the bare circuit's logical distribution exactly."""

from __future__ import annotations

import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from iceberg import (IcebergLayout, UnsupportedGateError, encode, encode_bus, encode_direct,
                     grid_coupling, layout_of, place, postselect, postselect_probs)
from iceberg.bus import edge_cost
from iceberg.sim import outcome_probabilities, prefix_states, run, to_program
from qsel import Angles, QaoaSimulator, qaoa_circuit, random_qubo, tune_angles

ENCODERS = {
    "direct": lambda c: encode_direct(c),
    "direct-no-flags": lambda c: encode_direct(c, init_flag=False, sx_flag=False),
    "direct-z": lambda c: encode_direct(c, checks="z"),
    "direct-x": lambda c: encode_direct(c, checks="x"),
    "bus": lambda c: encode_bus(c),
    "bus-all": lambda c: encode_bus(c, edge_mode="bus"),
    "bus-no-precopy": lambda c: encode_bus(c, precopy=False, init_flag=False),
}


def exact_logical(enc: QuantumCircuit) -> tuple[np.ndarray, float]:
    prog = to_program(enc)
    probs = outcome_probabilities(prog, run(prog))
    mass, kept = postselect_probs(probs, layout_of(enc))
    return mass / kept, kept


def bare_probs(circuit: QuantumCircuit) -> np.ndarray:
    return Statevector(circuit.remove_final_measurements(inplace=False)).probabilities()


def instance(n: int, p: int, seed: int, frac: float = 0.4):
    qubo = random_qubo(n, edge_fraction=frac, seed=seed)
    angles = tune_angles(qubo, p=p).angles
    return qubo, qaoa_circuit(qubo, angles), angles


@pytest.mark.parametrize("name", sorted(ENCODERS))
@pytest.mark.parametrize("n,p,seed", [(2, 1, 0), (4, 1, 1), (5, 2, 2), (6, 1, 3), (7, 1, 4)])
def test_encoded_equals_bare_on_ideal_simulator(name, n, p, seed):
    """Plan checkpoint H7: same logical distribution, and no shot is discarded."""
    _, bare, _ = instance(n, p, seed)
    dist, kept = exact_logical(ENCODERS[name](bare))
    assert kept == pytest.approx(1.0, abs=1e-10)
    assert np.abs(dist - bare_probs(bare)).max() < 1e-10


@pytest.mark.slow
@pytest.mark.parametrize("name", ["direct", "bus"])
def test_encoded_equals_bare_at_ten_paragraphs(name):
    qubo, bare, angles = instance(10, 1, 3, frac=0.2)
    dist, kept = exact_logical(ENCODERS[name](bare))
    assert kept == pytest.approx(1.0, abs=1e-10)
    assert np.abs(dist - QaoaSimulator(qubo).probabilities(angles)).max() < 1e-10


def test_qubit_budget_matches_the_plan():
    _, bare, _ = instance(10, 1, 3, frac=0.2)
    assert encode_direct(bare).num_qubits == 14          # 12 code qubits + ancilla + flag
    assert encode_direct(bare, checks="z").num_qubits == 13
    bus = encode_bus(bare)
    assert bus.num_qubits == 18                          # 12 code qubits + 6 bus ancillas
    rows = {r for r, _ in layout_of(bus).positions}
    cols = {c for _, c in layout_of(bus).positions}
    assert rows == {0, 1, 2} and cols == set(range(6))   # fits Sqale's 6 x 4 grid


def test_bus_circuit_is_nearest_neighbour_and_never_couples_data_to_a_hub():
    _, bare, _ = instance(10, 2, 5, frac=0.5)
    for mode in ("direct", "bus"):
        enc = encode_bus(bare, edge_mode=mode)
        lay = layout_of(enc)
        allowed = set(grid_coupling(lay))
        hubs = {lay.top, lay.bottom}
        data = set(lay.data)
        for inst in enc.data:
            if inst.operation.name == "barrier" or len(inst.qubits) != 2:
                continue
            pair = tuple(sorted(enc.find_bit(q).index for q in inst.qubits))
            assert pair in allowed, f"{inst.operation.name} on {pair} is not nearest neighbour"
            if inst.operation.name in ("rzz", "rxx"):
                assert not (set(pair) & hubs)
        # hub gates only appear in state preparation and the checks
        blocks = dict(zip(range(len(enc.data)), _labels(enc)))
        for pos, inst in enumerate(enc.data):
            if len(inst.qubits) == 2 and inst.operation.name != "barrier":
                pair = {enc.find_bit(q).index for q in inst.qubits}
                if pair & hubs and pair & data:
                    assert blocks[pos] in ("init", "syndrome")


def _labels(circuit: QuantumCircuit) -> list[str]:
    out: list[str] = []
    for name, count in circuit.metadata["iceberg_blocks"]:
        out.extend([name] * count)
    return out


def test_stabilizers_hold_after_every_block():
    """S_X = S_Z = +1 on the code qubits throughout the encoded circuit."""
    _, bare, _ = instance(4, 2, 7)
    for enc in (encode_direct(bare, readout=False), encode_bus(bare, readout=False)):
        lay = layout_of(enc)
        prog = to_program(enc)
        psi = prefix_states(prog)[-1]
        n = prog.num_qubits
        idx = np.arange(1 << n)
        code_mask = sum(1 << j for j in lay.code_qubits)
        parity = np.array([bin(int(v) & code_mask).count("1") & 1 for v in idx])
        assert float(np.sum(np.abs(psi) ** 2 * (1 - 2 * parity))) == pytest.approx(1.0, abs=1e-10)  # S_Z
        flipped = psi[idx ^ code_mask]                                                       # S_X |psi>
        assert float(np.real(np.vdot(psi, flipped))) == pytest.approx(1.0, abs=1e-10)


def test_logical_operators_act_as_expected():
    """Encoded X, Y, Z gates flip or phase the right logical qubit."""
    bare = QuantumCircuit(4, 4)
    bare.h(range(4))
    bare.rz(0.7, 0)
    bare.rzz(0.3, 1, 2)
    bare.rx(1.1, 0)
    bare.x(1)
    bare.y(2)
    bare.z(3)
    bare.cz(0, 3)
    bare.rx(0.4, 3)
    bare.measure(range(4), range(4))
    dist, kept = exact_logical(encode_direct(bare))
    assert kept == pytest.approx(1.0, abs=1e-10)
    assert np.abs(dist - bare_probs(bare)).max() < 1e-10


def test_zero_start_and_phase_gates():
    bare = QuantumCircuit(2, 2)
    bare.rx(0.9, 0)
    bare.rxx(0.5, 0, 1)
    bare.s(0)
    bare.t(1)
    bare.rx(0.3, 1)
    bare.measure([0, 1], [0, 1])
    enc = encode_direct(bare)
    assert layout_of(enc).init == "zero"
    dist, kept = exact_logical(enc)
    assert kept == pytest.approx(1.0, abs=1e-10)
    assert np.abs(dist - bare_probs(bare)).max() < 1e-10


def test_measurement_map_is_preserved():
    """Logical counts come back in the bare circuit's classical-bit order."""
    bare = QuantumCircuit(4, 4)
    bare.h(range(4))
    bare.rz(1.3, 0)
    bare.rx(0.8, 0)
    bare.rx(2.1, 2)
    bare.measure([0, 1, 2, 3], [2, 0, 3, 1])           # scrambled on purpose
    target = np.zeros(16)
    probs = bare_probs(bare)
    for k, pk in enumerate(probs):
        bits = [(k >> q) & 1 for q in range(4)]
        out = (bits[0] << 2) | (bits[1] << 0) | (bits[2] << 3) | (bits[3] << 1)
        target[out] += pk
    for enc in (encode_direct(bare), encode_bus(bare)):
        dist, _ = exact_logical(enc)
        assert np.abs(dist - target).max() < 1e-10


def test_odd_number_of_paragraphs_is_padded():
    _, bare, _ = instance(5, 1, 2)
    for enc in (encode_direct(bare), encode_bus(bare)):
        lay = layout_of(enc)
        assert (lay.k, lay.k_code) == (5, 6)
        counts = {"0" * enc.num_clbits: 10}
        logical, stats = postselect(counts, enc)
        assert stats.kept == 10 and list(logical) == ["00000"]


@pytest.mark.parametrize("gate", ["h", "ry", "cx"])
def test_unsupported_gates_are_rejected(gate):
    bare = QuantumCircuit(2, 2)
    bare.h([0, 1])
    bare.rz(0.2, 0)
    if gate == "h":
        bare.h(0)
    elif gate == "ry":
        bare.ry(0.3, 0)
    else:
        bare.cx(0, 1)
    bare.measure([0, 1], [0, 1])
    with pytest.raises(UnsupportedGateError):
        encode_direct(bare)
    with pytest.raises(UnsupportedGateError):
        encode_bus(bare)


def test_bad_measurement_maps_and_options_are_rejected():
    bare = QuantumCircuit(2, 2)
    bare.h([0, 1])
    bare.rx(0.3, 0)
    bare.measure([0, 1], [0, 0])                       # two qubits into one classical bit
    for enc in (encode_direct, encode_bus):
        with pytest.raises(UnsupportedGateError):
            enc(bare)
        with pytest.raises(ValueError):
            enc(QuantumCircuit(0, 0))
    ok = QuantumCircuit(2, 2)
    ok.h([0, 1])
    ok.measure([0, 1], [0, 1])
    with pytest.raises(TypeError):
        encode(ok, style="bus", sx_flag=False)
    with pytest.raises(TypeError):
        encode(ok, style="direct", precopy=False)


def test_partial_leading_h_layer_is_rejected():
    bare = QuantumCircuit(2, 2)
    bare.h(0)
    bare.rz(0.2, 1)
    bare.measure([0, 1], [0, 1])
    with pytest.raises(UnsupportedGateError):
        encode_direct(bare)


def test_layout_round_trips_through_metadata():
    _, bare, _ = instance(6, 1, 3)
    for enc in (encode_direct(bare), encode_bus(bare)):
        lay = layout_of(enc)
        assert IcebergLayout.from_dict(lay.to_dict()) == lay
        assert layout_of(enc.metadata) == lay
        assert lay.num_qubits == enc.num_qubits == enc.num_clbits


def test_encode_dispatches_on_style():
    _, bare, _ = instance(4, 1, 1)
    assert layout_of(encode(bare, style="bus")).style == "bus"
    assert layout_of(encode(bare, style="direct")).style == "direct"
    assert layout_of(encode(bare)).style == "bus"
    with pytest.raises(ValueError):
        encode(bare, style="bus", checks="z")
    with pytest.raises(ValueError):
        encode(bare, style="nope")


def test_placement_puts_redundant_pairs_next_to_each_other():
    edges = [(0, 5), (5, 3), (3, 8), (1, 2)]
    sites = place(10, edges, seed=0)
    half = 5
    cost = sum(edge_cost((sites[i] // half, sites[i] % half), (sites[j] // half, sites[j] % half))
               for i, j in edges)
    naive = sum(edge_cost((i // half, i % half), (j // half, j % half)) for i, j in edges)
    assert sorted(sites) == list(range(10))
    assert cost <= naive
    assert cost == 2 * len(edges)        # a path and a separate pair fit as row neighbours


def test_encoded_rotations_carry_the_bare_angles():
    qubo, bare, angles = instance(6, 1, 3)
    for enc in (encode_direct(bare), encode_bus(bare)):
        terms = enc.metadata["iceberg_terms"]
        bare_rot = [i for i in bare.data if i.operation.name in ("rz", "rzz", "rx")]
        assert len(terms) == len(bare_rot)
        got = sorted(round(float(enc.data[idx].operation.params[0]), 9) for _, _, idx in terms)
        want = sorted(round(float(i.operation.params[0]), 9) for i in bare_rot)
        assert got == want


def test_compiler_cannot_remove_the_checks():
    """An optimising transpile keeps the flag couplings and the exact distribution."""
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap

    from iceberg.sim import compact

    _, bare, _ = instance(4, 1, 1, frac=0.5)
    want = bare_probs(bare)
    cases = [(encode_direct(bare), None), (encode_bus(bare), CouplingMap.from_grid(3, 3))]
    for enc, cmap in cases:
        lay = layout_of(enc)
        out = transpile(enc, basis_gates=["cz", "rz", "sx", "x"], coupling_map=cmap,
                        optimization_level=3, seed_transpiler=1)
        small = compact(out)
        small.metadata = {"iceberg": lay.to_dict()}
        dist, kept = exact_logical(small)
        assert kept == pytest.approx(1.0, abs=1e-9)
        assert np.abs(dist - want).max() < 1e-9
        if lay.style == "bus":
            ours = sum(1 for i in enc.data if len(i.qubits) == 2 and i.operation.name != "barrier")
            got = sum(1 for i in out.data if len(i.qubits) == 2 and i.operation.name != "barrier")
            n_rzz = enc.count_ops().get("rzz", 0)
            assert got <= ours + n_rzz           # no SWAPs: each cx is one cz, each rzz is two


def test_angles_sign_convention():
    """Negating every angle gives the mirrored distribution, encoded or not."""
    qubo = random_qubo(4, edge_fraction=0.5, seed=1)
    a = Angles((0.37,), (0.52,))
    b = Angles((-0.37,), (-0.52,))
    pa, _ = exact_logical(encode_bus(qaoa_circuit(qubo, a)))
    pb, _ = exact_logical(encode_bus(qaoa_circuit(qubo, b)))
    assert np.abs(pa - pb).max() < 1e-10       # complex conjugate circuit, same probabilities
    assert not math.isclose(float(pa.max()), 1 / 16)
