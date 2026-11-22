"""Tests of the loss-time tracker. Run from challenge-1-ml-decoder-kit/:

    python -m pytest -p no:cacheprovider loss_tracker

They sample their own tiny Clifft datasets (fixed seeds) and use the committed tables.
"""
import os

import numpy as np
import pytest

from decoders import relay_decoder
from loss.check_map import forced_text
from loss.lossmap import LossMap, PriorBuilder, parse_condition
from loss.lossmodel import LEFT, RIGHT, Layout, clifft_text, kit_circuit, loss_times, sample_chunk
from loss_tracker.fit import count_tables
from loss_tracker.priors import parse_variant, priors_from_slots, slot_dists, true_round_dists
from loss_tracker.tracker import CLS, SLOTS, Geometry, Tables, Tracker, features, softmax, to_rounds

TABLES = os.path.join(os.path.dirname(__file__), "tables", "X_p0.004_s1e-3.npz")
P, S = 0.004, 1e-3


@pytest.fixture(scope="module")
def tracker():
    return Tracker(Tables.load(TABLES))


@pytest.fixture(scope="module")
def shots():
    """40 shots of the loss circuit (X memory, p = 0.004, s = 1e-3, seed 9001), with the probes' truth."""
    circuit = kit_circuit("X", P)
    layout = Layout(circuit)
    text, probe_layout, new_index = clifft_text(circuit, S, probes=True)
    smp = sample_chunk(text, 40, 9001)
    data_slot, check_k = loss_times(layout, probe_layout, new_index, smp)
    return {"detectors": np.asarray(smp.detectors, np.uint8), "observables": np.asarray(smp.observables, np.uint8),
            "heralds": np.asarray(smp.heralds, np.uint8)[:, new_index], "data_slot": data_slot, "check_k": check_k}


@pytest.fixture(scope="module")
def builder():
    return PriorBuilder(LossMap("X", P), S)


def test_class_map():
    assert CLS.shape == (13, SLOTS)
    assert (CLS[3, :6] == 19).all() and (CLS[5, 36:] == 0).all()           # lost 3 cycles ago; still present
    assert CLS[3, 6:12].tolist() == [13, 14, 15, 16, 17, 18]               # lost 2 cycles ago, after CX 1..6
    assert CLS[5, 30:36].tolist() == [1, 2, 3, 4, 5, 6] and CLS[6, 30:36].tolist() == [7, 8, 9, 10, 11, 12]


def test_tables_are_distributions(tracker):
    t = tracker.tables
    assert np.allclose(t.p1.sum(axis=-1), 1) and np.allclose(t.cond.sum(axis=-1), 1)
    assert np.allclose(t.sub1.sum(axis=-1), 1)                              # still a distribution with dirty bits
    assert np.allclose(t.sub2.sum(axis=-1), 1)
    assert np.allclose(t.sub1[..., 7, :], 1 / 8) and np.allclose(t.sub2[..., 7, 7, :, :], 1 / 8)


def test_posterior_sums_to_one(tracker, shots):
    feat = features(tracker.geo, shots["detectors"], shots["heralds"])
    for order in (0, 1):
        shot, atom, ll = tracker.loglik(feat, order=order)
        post = tracker.posterior(ll)
        assert len(shot) == int(shots["heralds"][:, tracker.geo.data_rec].sum()) > 0
        assert np.allclose(post.sum(axis=1), 1) and (post >= 0).all()
        assert np.allclose(to_rounds(post).sum(axis=1), 1)


def test_empty_history_gives_the_prior(tracker, shots):
    """With every detector bit marked unreliable, the history says nothing: the posterior is the prior."""
    feat = features(tracker.geo, shots["detectors"], shots["heralds"])
    for T in ("X", "Z"):
        feat["herald_dirt"][T][:] = True
    for order in (0, 1):
        _, _, ll = tracker.loglik(feat, order=order)
        assert np.allclose(tracker.posterior(ll), np.exp(tracker.log_prior)[None, :])
    w = (1 - S) ** np.arange(SLOTS)
    assert np.allclose(np.exp(tracker.log_prior), w / w.sum())


def test_explain_mode(tracker, shots):
    """Explaining shared bits: still a distribution, the prior when nothing is usable, and unchanged for an atom
    that shares no check with another flagged atom and has no flagged check atom nearby."""
    geo = tracker.geo
    feat = features(geo, shots["detectors"][:12], shots["heralds"][:12])
    shot, atom, ll = tracker.loglik(feat, explain=True)
    _, _, base = tracker.loglik(feat)
    post = tracker.posterior(ll)
    assert np.allclose(post.sum(axis=1), 1) and np.isfinite(ll).all()
    flagged = feat["flagged"]
    alone = np.ones(len(shot), bool)
    for T in ("X", "Z"):
        alone &= ~flagged[shot[:, None, None], geo.co_atom[T][atom]].any(axis=(1, 2))
    assert alone.any() and not alone.all()
    assert np.allclose(ll[alone], base[alone], atol=1e-8)                   # nothing to explain: the same answer
    assert not np.allclose(ll[~alone], base[~alone], atol=1e-3)
    for T in ("X", "Z"):
        feat["herald_dirt"][T][:] = True
    _, _, ll = tracker.loglik(feat, explain=True)
    assert np.allclose(tracker.posterior(ll), np.exp(tracker.log_prior)[None, :])


def test_a_shot_does_not_depend_on_its_batch(tracker, shots):
    feat = features(tracker.geo, shots["detectors"], shots["heralds"])
    shot, atom, ll = tracker.loglik(feat, chunk=7)
    one = features(tracker.geo, shots["detectors"][11:12], shots["heralds"][11:12])
    _, atom1, ll1 = tracker.loglik(one)
    assert (atom[shot == 11] == atom1).all() and np.allclose(ll[shot == 11], ll1, atol=1e-9)


def test_no_flag_gives_the_lossless_priors(tracker, builder, shots):
    heralds = np.zeros_like(shots["heralds"][0])
    feat = features(tracker.geo, shots["detectors"][:1], heralds[None, :])
    shot, atom, ll = tracker.loglik(feat)
    assert len(shot) == 0 and ll.shape == (0, SLOTS)                        # nothing to track
    pri = priors_from_slots(builder, heralds, {})
    assert np.allclose(pri, np.minimum(builder.m.pri, 0.5), rtol=0, atol=1e-15)     # the kit's priors, to rounding
    assert np.array_equal(pri, builder.priors(parse_condition("exact"), heralds))
    # and in a shot with flags, every column that no flagged atom can flip keeps the kit's prior
    m, k = builder.m, 0
    her = shots["heralds"][k]
    feat = features(tracker.geo, shots["detectors"][k:k + 1], her[None, :])
    _, atoms, ll = tracker.loglik(feat)
    pri = priors_from_slots(builder, her, slot_dists(builder, parse_variant("tracker"), atoms, ll, tracker.log_prior))
    touched = set(m.own_flip[her[builder.own_rec].astype(bool)].tolist()) | set(m.hook[her[builder.other_rec].astype(bool)].ravel().tolist())
    for i in atoms:
        d = m.data[(*LEFT, *RIGHT)[int(i)]]
        touched |= set(d["col"].tolist()) | {int(d["final"])}
    rest = np.setdiff1d(np.arange(m.ncols), sorted(touched))
    assert len(atoms) > 0 and len(rest) > 7000
    assert np.allclose(pri[rest], m.pri[rest], rtol=0, atol=1e-15)
    assert not np.allclose(pri, m.pri, atol=1e-6)


def test_tracker_reads_no_truth(tracker):
    """The tracker's inputs are the detector bits and the heralds: features() takes nothing else."""
    import inspect
    from loss_tracker import track, tracker as module
    assert list(inspect.signature(features).parameters) == ["geo", "detectors", "heralds"]
    for mod in (module, track):
        source = inspect.getsource(mod)
        code = "\n".join(line for line in source.split('"""')[2::2])        # the code without the docstrings
        assert "data_slot" not in code and "check_k" not in code and "observables" not in code


@pytest.mark.parametrize("atom,cycle", [(LEFT[3], 2), (RIGHT[10], 6), (LEFT[40], 9)])
def test_forced_loss_is_found(tracker, atom, cycle):
    """One data atom lost for certain after a chosen CX of a noiseless circuit: the tracker finds the round."""
    j = 6 * cycle + 3
    text, probe_layout, new_index = forced_text(kit_circuit("X", 0), atom, j)
    smp = sample_chunk(text, 60, 4000 + atom)
    det, her = np.asarray(smp.detectors, np.uint8), np.asarray(smp.heralds, np.uint8)[:, new_index]
    feat = features(tracker.geo, det, her)
    shot, idx, ll = tracker.loglik(feat)
    assert (idx == atom - LEFT[0]).all() and len(shot) == 60                # the only flagged data atom, in every shot
    rounds = to_rounds(tracker.posterior(ll))
    est = rounds.argmax(axis=1)
    assert np.bincount(est, minlength=12).argmax() == cycle                 # the true round is the most frequent answer
    assert np.mean(np.abs(est - cycle) <= 1) >= 0.75
    assert rounds[:, max(cycle - 1, 0):cycle + 2].sum(axis=1).mean() >= 0.6
    assert rounds.mean(axis=0).argmax() == cycle


def test_fit_counts_only_clean_triples(tracker, shots):
    c1, c2 = count_tables(tracker.geo, shots["detectors"], shots["heralds"], shots["data_slot"])
    assert c1.sum() > 0 and c2.sum() > 0 and c2.sum() <= c1.sum()
    assert c1[:, 1, 0].sum() == 0 and c1[:, 1, 12].sum() == 0               # no Z-check detectors in rounds 0 and 12
    assert c1[:, 0, 12, 0].sum() > 0 and c1[:, 0, 0, 7:].sum() == 0         # round 0 cannot see an earlier loss


def test_plumbing_true_round_equals_oracle(builder, shots):
    """With the TRUE round put in by hand, this folder's priors are loss-gates' ORACLE `oracle-round`, bit for bit,
    and so is the decoding; the uniform guess gives loss-gates' `exact`."""
    oracle, exact = parse_condition("oracle-round"), parse_condition("exact")
    for k in range(len(shots["heralds"])):
        her, slot = shots["heralds"][k], shots["data_slot"][k]
        mine = priors_from_slots(builder, her, true_round_dists(builder, slot))
        assert np.array_equal(mine, builder.priors(oracle, her, slot, shots["check_k"][k]))
        uniform = {int(i): builder.w72 for i in np.nonzero(slot)[0]}
        assert np.array_equal(priors_from_slots(builder, her, uniform), builder.priors(exact, her))
    lmap = builder.m
    for k in (0, 1):
        her, slot, syndrome = shots["heralds"][k], shots["data_slot"][k], shots["detectors"][k][lmap.rows]
        a = relay_decoder(lmap.H, priors_from_slots(builder, her, true_round_dists(builder, slot)), seed=1).decode_detailed(syndrome)
        b = relay_decoder(lmap.H, builder.priors(oracle, her, slot, shots["check_k"][k]), seed=1).decode_detailed(syndrome)
        assert a.iterations == b.iterations and a.success == b.success
        assert np.array_equal(np.asarray(a.decoding), np.asarray(b.decoding))


def test_variants(tracker, builder, shots):
    feat = features(tracker.geo, shots["detectors"][:1], shots["heralds"][:1])
    _, atoms, ll = tracker.loglik(feat)
    soft = slot_dists(builder, parse_variant("tracker"), atoms, ll, tracker.log_prior)
    slot = slot_dists(builder, parse_variant("tracker-slot"), atoms, ll, tracker.log_prior)
    hard = slot_dists(builder, parse_variant("tracker-hard0"), atoms, ll, tracker.log_prior)
    flat = slot_dists(builder, parse_variant("tracker-T1e9"), atoms, ll, tracker.log_prior)
    post = softmax(tracker.log_prior[None, :] + ll)
    for n, i in enumerate(atoms):
        for d in (soft, slot, hard, flat):
            assert np.isclose(d[int(i)].sum(), 1)
        assert np.allclose(slot[int(i)], post[n])
        assert np.allclose(soft[int(i)].reshape(12, 6).sum(axis=1), post[n].reshape(12, 6).sum(axis=1))
        assert np.count_nonzero(hard[int(i)]) == 6                          # one round, spread over its 6 slots
        assert np.allclose(flat[int(i)].reshape(12, 6).sum(axis=1), np.exp(tracker.log_prior).reshape(12, 6).sum(axis=1), atol=1e-6)
    pri = priors_from_slots(builder, shots["heralds"][0], soft)
    assert pri.shape == builder.m.pri.shape and (pri <= 0.5).all() and (pri > 0).all()
