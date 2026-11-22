"""Post-selection: which shots are kept and how logical bits are decoded."""

from __future__ import annotations

import numpy as np
import pytest

from iceberg import IcebergLayout, PostselectStats, encode_bus, encode_direct, layout_of, postselect, postselect_probs
from qsel import qaoa_circuit, random_qubo, tune_angles


def key(bits: dict[int, int], width: int) -> str:
    """Qiskit-style key with the given bits set (bit 0 rightmost)."""
    value = sum(v << j for j, v in bits.items())
    return format(value, f"0{width}b")


@pytest.fixture()
def layout() -> IcebergLayout:
    return IcebergLayout.default(4)          # data 0..3, t=4, b=5, a=6, f=7


def test_valid_shot_is_decoded_relative_to_b(layout):
    # data bits 1,0,1,1 with b = 1 and t chosen for even parity -> logical 0,1,0,0
    shot = key({0: 1, 1: 0, 2: 1, 3: 1, 4: 0, 5: 1}, 8)
    logical, stats = postselect({shot: 7}, layout)
    assert logical == {"0010": 7}
    assert stats == PostselectStats(shots=7, kept=7, fail_sz=0, fail_sx=0, fail_flag=0)
    assert stats.kept_share == 1.0


def test_both_gauge_copies_decode_to_the_same_logical_string(layout):
    a = key({0: 1, 4: 1}, 8)                                  # b = 0
    b = key({1: 1, 2: 1, 3: 1, 5: 1}, 8)                      # every code bit flipped
    logical, stats = postselect({a: 3, b: 5}, layout)
    assert logical == {"0001": 8} and stats.kept == 8


def test_each_check_discards(layout):
    good = key({}, 8)
    odd = key({2: 1}, 8)                 # odd code parity
    sx = key({6: 1}, 8)                  # ancilla fired
    flag = key({7: 1}, 8)                # flag fired
    logical, stats = postselect({good: 10, odd: 4, sx: 3, flag: 2}, layout)
    assert logical == {"0000": 10}
    assert (stats.shots, stats.kept, stats.fail_sz, stats.fail_sx, stats.fail_flag) == (19, 10, 4, 3, 2)
    assert stats.discarded == 9
    assert stats.to_dict()["kept_share"] == pytest.approx(10 / 19)


def test_layout_is_required_and_keys_are_validated(layout):
    shot = key({0: 1, 4: 1}, 8)
    with pytest.raises(ValueError, match="needs the encoded circuit"):
        postselect({shot: 1})                      # bus and direct cannot be told apart from counts
    with pytest.raises(ValueError, match="8-bit"):
        postselect({"0" * 7: 1}, layout)
    with pytest.raises(ValueError):
        postselect({"0x1f": 1}, layout)


def test_keys_with_spaces_and_empty_counts(layout):
    shot = key({0: 1, 4: 1}, 8)
    spaced = shot[:2] + " " + shot[2:]
    assert postselect({spaced: 2}, layout)[0] == {"0001": 2}
    logical, stats = postselect({}, layout)
    assert logical == {} and stats.shots == 0 and stats.kept_share == 0.0


def test_half_codes_only_apply_their_own_check():
    z_only = IcebergLayout.default(4, checks="z")       # no ancilla; flag at 6
    x_only = IcebergLayout.default(4, checks="x")
    odd = key({2: 1}, z_only.num_qubits)
    assert postselect({odd: 1}, z_only)[1].kept == 0
    odd_x = key({2: 1}, x_only.num_qubits)
    assert postselect({odd_x: 1}, x_only)[1].kept == 1   # parity is not checked
    fired = key({x_only.sx_ancilla: 1}, x_only.num_qubits)
    assert postselect({fired: 1}, x_only)[1].fail_sx == 1


def test_bus_layout_requires_every_bus_qubit_to_read_zero():
    qubo = random_qubo(4, edge_fraction=0.5, seed=1)
    enc = encode_bus(qaoa_circuit(qubo, tune_angles(qubo).angles))
    lay = layout_of(enc)
    width = enc.num_clbits
    assert postselect({key({}, width): 1}, enc)[1].kept == 1
    for q in lay.flags:
        assert postselect({key({q: 1}, width): 1}, enc)[1].fail_flag == 1
    assert postselect({key({lay.sx_ancilla: 1}, width): 1}, enc)[1].fail_sx == 1
    assert len(lay.flags) + 1 == width - lay.n_code


def test_probability_version_agrees_with_counts_version():
    qubo = random_qubo(4, edge_fraction=0.5, seed=2)
    enc = encode_direct(qaoa_circuit(qubo, tune_angles(qubo).angles))
    lay = layout_of(enc)
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(1 << enc.num_qubits))
    counts = {format(i, f"0{enc.num_qubits}b"): int(round(p * 10**9)) for i, p in enumerate(probs)}
    logical, stats = postselect(counts, lay)
    mass, kept = postselect_probs(probs, lay)
    assert stats.kept_share == pytest.approx(kept, abs=1e-6)
    from_counts = np.zeros(16)
    for k_, c in logical.items():
        from_counts[int(k_, 2)] += c
    assert np.abs(from_counts / from_counts.sum() - mass / kept).max() < 1e-6
