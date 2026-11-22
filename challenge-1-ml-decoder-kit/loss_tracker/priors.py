"""Per-shot priors for relay-BP from the tracker's posterior over the loss slot of each flagged data atom.

The map from a loss to the decoder's columns is loss-gates' (`loss.lossmap`), unchanged. `priors_from_slots` is
its condition `exact`, with one difference: the distribution over the 72 loss slots of each flagged data atom
is given by the caller instead of being the uniform guess. With the true round put in by hand it must equal
loss-gates' ORACLE `oracle-round`, bit for bit (test_loss_tracker.py and the plumbing check of RESULTS.md).
"""
import numpy as np

from loss.lossmap import _combine
from loss.lossmodel import CYCLES, LEFT, RIGHT
from loss_tracker.tracker import softmax

DATA = (*LEFT, *RIGHT)


def priors_from_slots(builder, heralds, slot_dist, cap=0.5):
    """builder: loss.lossmap.PriorBuilder. slot_dist: {data atom index 0..143: (72,) distribution over its slot}.

    Same order of operations as PriorBuilder.priors for `exact`: flagged own check atoms, then the hook columns
    of flagged other-type check atoms (geometric weights), then every flagged data atom.
    """
    m = builder.m
    cols, vals = [], []
    own_flag = heralds[builder.own_rec].astype(bool)
    other_flag = heralds[builder.other_rec].astype(bool)
    data_flag = heralds[builder.data_rec].astype(bool)
    cols.append(m.own_flip[own_flag]); vals.append(np.full(own_flag.sum(), 0.5))
    for a, c in zip(*np.nonzero(other_flag), strict=True):
        cols.append(m.hook[a, c]); vals.append(0.5 * builder.w6[:5])
    for i in np.nonzero(data_flag)[0]:
        d = m.data[DATA[i]]
        c_, a_ = m.data_activation(DATA[i], slot_dist[int(i)])
        cols.append(c_); vals.append(0.5 * a_)
        cols.append([d["final"]]); vals.append([0.5])
    if not cols:
        return np.minimum(m.pri, cap)
    return _combine(m.pri, np.concatenate(cols).astype(np.int64), np.concatenate(vals), cap)


def spread_round(builder, round_dist):
    """(12,) over the loss cycles -> (72,) over the slots: within a cycle as the ORACLE `oracle-round` does."""
    return (np.asarray(round_dist, np.float64)[:, None] * builder.w6[None, :]).ravel()


def parse_variant(name):
    """'tracker', 'tracker-T2', 'tracker-hard0.8', 'tracker-slot', 'tracker-T1.5-hard0.9' -> a dict.

    T<x>     temperature on the tracker's evidence (the prior over the slot is not tempered);
    hard<x>  where the most probable round has posterior >= x, use that round alone;
    slot     use the posterior over the 72 slots as it is, not summed to rounds and spread again.
    """
    tokens = name.split("-")
    assert tokens[0] == "tracker", name
    v = {"temperature": 1.0, "hard": None, "slot": False}
    for tok in tokens[1:]:
        if tok.startswith("T"):
            v["temperature"] = float(tok[1:])
        elif tok.startswith("hard"):
            v["hard"] = float(tok[4:])
        elif tok == "slot":
            v["slot"] = True
        else:
            raise ValueError(f"unknown tracker variant {name}")
    return v


def slot_dists(builder, variant, atoms, loglik, log_prior):
    """The slot distributions of one shot's flagged atoms, for priors_from_slots. loglik: (atoms, 72)."""
    out = {}
    if len(atoms) == 0:
        return out
    post = softmax(log_prior[None, :] + np.asarray(loglik, np.float64) / variant["temperature"])
    for i, p in zip(atoms, post, strict=True):
        if variant["slot"]:
            assert variant["hard"] is None
            out[int(i)] = p
            continue
        rounds = p.reshape(CYCLES, 6).sum(axis=1)
        if variant["hard"] is not None and rounds.max() >= variant["hard"]:
            best = int(rounds.argmax())
            rounds = np.zeros(CYCLES)
            rounds[best] = 1.0
        out[int(i)] = spread_round(builder, rounds)
    return out


def true_round_dists(builder, data_slot):
    """ORACLE, for the plumbing check only: a one-hot posterior on the TRUE loss cycle of each lost data atom."""
    out = {}
    for i in np.nonzero(data_slot)[0]:
        rounds = np.zeros(CYCLES)
        rounds[(int(data_slot[i]) - 1) // 6] = 1.0
        out[int(i)] = spread_round(builder, rounds)
    return out
