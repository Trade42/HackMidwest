"""A loss-time tracker: for every data atom flagged at the final readout, a posterior over WHEN it was lost.

The tracker reads detector bits and heralds only. It never reads the status probes (`data_slot`, `check_k`):
this module does not take them, and `track.py`, which runs it on a dataset, does not load them.

The model (a change-point model, i.e. a left-to-right hidden Markov model in closed form):

- Hidden variable of a flagged data atom: its loss slot j = 1..72 (lost after its j-th CX). The herald says
  that a loss happened, so j has no "never" value. Prior: geometric in s, P(j) ~ (1 - s)^(j - 1).
- Observations: in every detector round r, the 3 detector bits of the atom's 3 X checks (one pattern of 8) and
  the 3 bits of its 3 Z checks (another pattern of 8). The bits of a triple are ordered by the atom's CX order.
- Emission tables, fitted on training shots (fit.py): P(pattern | class), where the class of (round r, slot j)
  says how far the loss is from the round: 0 = the atom is still present in round r; 1..6 = lost in cycle r
  after its k-th CX of that cycle; 7..12 = lost in cycle r - 1 after its k-th CX; 13..18 = lost in cycle r - 2
  after its k-th CX; 19 = lost three or more cycles before. One set of tables per side (left / right data
  atoms: their CX orders differ) and per check type.
- Two emission models (`order`):
    order 0: the rounds are independent given the slot,
             log P(history | j) = sum over the rounds and the two check types of log P(pattern_r | class(r, j));
    order 1: every round is conditioned on the round before (a first-order chain),
             ... log P(pattern_r | pattern_{r-1}, class(r, j)).
  Order 1 exists because the rounds of a lost atom are NOT independent: in this CX schedule one detector of a
  lost atom repeats, one round later, what its two same-type partners showed (signal.py measures it), so a lost
  atom makes one fresh random bit per check type and round, not two.

Checks that carry no information ("dirty" bits). A detector bit is a fair coin, whatever this atom did, when
  (a) its check atom was flagged in this cycle or the one before (its recorded bit is random);
  (b) another lost data atom on the same check was lost before its CX with that check in this cycle.
And all 3 bits of a triple are unreliable when
  (c) a check atom of the OTHER type next to this atom was flagged in this cycle or the one before: a lost check
      atom leaves a hook error on the data atoms it already met, which flips their other-type checks together,
      like a loss does.
(a) and (c) are known from the heralds. (b) depends on the loss times of the other flagged atoms, which are
unknown: the tracker uses their current posteriors, and repeats (`sweeps`): every atom's posterior is recomputed
from the others' posteriors of the sweep before, all at once (synchronous message passing; the result does not
depend on any atom order). A dirty bit with probability rho is marginalised out of the pattern with weight rho.
"""
import json

import numpy as np

from loss.lossmodel import CYCLES, LEFT, N, RIGHT, X_ANC, Z_ANC, Layout, kit_circuit

DATA = (*LEFT, *RIGHT)
TYPES = ("X", "Z")
ROUNDS = CYCLES + 1          # detector rounds 0..12
SLOTS = 6 * CYCLES           # loss slots 1..72
N_CLASS = 20
ALPHA = 0.5                  # pseudo-count per pattern in every table
KAPPA = 4.0                  # weight of the single-round table as the prior of a conditional table


def class_map():
    """cls[r, j - 1]: the class of loss slot j seen from detector round r (see the module docstring)."""
    j = np.arange(1, SLOTS + 1)
    c, k = (j - 1) // 6, (j - 1) % 6 + 1
    delta = np.arange(ROUNDS)[:, None] - c[None, :]
    return np.where(delta < 0, 0, np.where(delta <= 2, 6 * np.clip(delta, 0, 2) + k[None, :], 19))


CLS = class_map()


def round_groups(basis):
    """group[T][r]: rounds whose lost-atom tables are pooled. The first two rounds and the last stand alone.

    T is the check type. The decoded basis has detectors in rounds 0..12, the other type in rounds 1..11.
    """
    own = np.array([0, 1] + [2] * (CYCLES - 2) + [3])
    other = np.array([-1, 0, 1] + [2] * (CYCLES - 3) + [-1])
    return {"X": own, "Z": other} if basis == "X" else {"X": other, "Z": own}


class Geometry:
    """What the tracker needs from the kit's circuit: the checks next to each data atom, and the detector ids."""

    def __init__(self, basis, p):
        lay = Layout(kit_circuit(basis, p))
        self.basis = basis
        anc = {"X": list(X_ANC), "Z": list(Z_ANC)}
        index = {T: {a: n for n, a in enumerate(anc[T])} for T in TYPES}
        # chk[T][i, m]: the m-th check of type T that data atom i meets in a cycle; pos[T][i, m]: at which of its 6 CX
        self.chk = {T: np.zeros((2 * N, 3), np.int64) for T in TYPES}
        self.pos = {T: np.zeros((2 * N, 3), np.int64) for T in TYPES}
        for i, q in enumerate(DATA):
            partners = [partner for _, _, partner in lay.cx[q]]
            assert all(partners[6 * c:6 * c + 6] == partners[:6] for c in range(CYCLES))
            for T in TYPES:
                mine = [(k + 1, index[T][a]) for k, a in enumerate(partners[:6]) if a in index[T]]
                assert len(mine) == 3
                self.pos[T][i] = [k for k, _ in mine]
                self.chk[T][i] = [a for _, a in mine]
        self.side = np.array([0] * N + [1] * N)                  # 0 = left data atom, 1 = right
        for T in TYPES:
            for side in (0, 1):
                assert (self.pos[T][self.side == side] == self.pos[T][self.side == side][0]).all()
        # co_atom[T][i, m, :], co_slot[T][i, m, :]: the 5 other data atoms on that check, and which of THEIR 3 it is
        self.co_atom = {T: np.zeros((2 * N, 3, 5), np.int64) for T in TYPES}
        self.co_slot = {T: np.zeros((2 * N, 3, 5), np.int64) for T in TYPES}
        for T in TYPES:
            members = {}
            for i in range(2 * N):
                for m in range(3):
                    members.setdefault(int(self.chk[T][i, m]), []).append((i, m))
            for i in range(2 * N):
                for m in range(3):
                    others = [x for x in members[int(self.chk[T][i, m])] if x[0] != i]
                    assert len(others) == 5
                    self.co_atom[T][i, m] = [x[0] for x in others]
                    self.co_slot[T][i, m] = [x[1] for x in others]
        # det[T][a, r]: detector id of check a (0..71 within its type) in round r, or -1
        self.det = {T: np.full((N, ROUNDS), -1, np.int64) for T in TYPES}
        for d, (a, r) in enumerate(lay.det_check):
            T = "X" if a in index["X"] else "Z"
            self.det[T][index[T][a], r] = d
        self.rounds = {T: np.nonzero((self.det[T] >= 0).all(axis=0))[0] for T in TYPES}
        for T in TYPES:
            assert ((self.det[T] >= 0).all(axis=0) == (self.det[T] >= 0).any(axis=0)).all()
        self.groups = round_groups(basis)
        assert all((self.groups[T] >= 0).nonzero()[0].tolist() == self.rounds[T].tolist() for T in TYPES)
        # measurement records (the heralds are indexed by them)
        self.check_rec = {T: np.array([[lay.record_of(a, c) for c in range(CYCLES)] for a in anc[T]]) for T in TYPES}
        self.data_rec = np.array([lay.record_of(q) for q in DATA])
        # lim[T][i, m, r]: atom i has left its m-th T check out of the detector of round r iff its slot j <= lim
        r = np.arange(ROUNDS)
        self.lim = {T: np.minimum(6 * r[None, None, :] + self.pos[T][:, :, None] - 1, SLOTS) for T in TYPES}


def features(geo, detectors, heralds):
    """What the tracker sees of a batch of shots. detectors (shots, 1728), heralds (shots, 1872).

    Returns a dict:
      flagged        (shots, 144) bool   the data atom was heralded at the final readout
      pattern[T]     (shots, 144, 13)    the 3 detector bits of the atom's T checks in each round, as 0..7
      herald_dirt[T] (shots, 144, 3, 13) bool: that bit is unreliable because of a flagged check atom (a) or (c)
    """
    detectors, heralds = np.asarray(detectors), np.asarray(heralds)
    bits, hd = {}, {}
    for T in TYPES:
        bits[T] = np.where(geo.det[T] >= 0, detectors[:, np.maximum(geo.det[T], 0)], 0).astype(np.uint8)
        h = heralds[:, geo.check_rec[T]].astype(bool)             # (shots, 72, 12): flagged in cycle c
        hd[T] = np.zeros((len(h), N, ROUNDS), bool)                # the detector of round r reads cycles r and r - 1
        hd[T][:, :, :CYCLES] |= h
        hd[T][:, :, 1:] |= h
    out = {"flagged": heralds[:, geo.data_rec].astype(bool), "pattern": {}, "herald_dirt": {}, "own_herald": {}, "hook": {}}
    for T, other in (("X", "Z"), ("Z", "X")):
        b = bits[T][:, geo.chk[T], :]                              # (shots, 144, 3, 13)
        out["pattern"][T] = b[:, :, 0] | (b[:, :, 1] << 1) | (b[:, :, 2] << 2)
        own = hd[T][:, geo.chk[T], :]                              # (a) the check's own atom was flagged
        hook = hd[other][:, geo.chk[other], :].any(axis=2)         # (c) an other-type check atom next to it was
        out["herald_dirt"][T] = own | hook[:, :, None, :]
        out["own_herald"][T], out["hook"][T] = own, hook          # the two causes apart (signal.py reports them)
    return out


def others_dirt(geo, T, shot_idx, atom_idx, cdf, shots, to_shot=None, to_atom=None):
    """P(bit m of an atom's T triple in round r is randomised by ANOTHER lost data atom), shape (targets, 3, 13).

    The lost atoms are the entries (shot_idx, atom_idx), with cdf[e, n] = P(slot <= n), n = 0..72. Atoms that are
    not among the entries count as never lost. Distinct atoms are treated as independent. The targets are
    (to_shot, to_atom), by default the entries themselves. The result never falls from one round to the next.
    """
    own = np.take_along_axis(cdf[:, None, :], geo.lim[T][atom_idx].reshape(len(atom_idx), 1, 3 * ROUNDS), axis=2)
    own = own.reshape(len(atom_idx), 3, ROUNDS)                   # P(this atom has randomised its m-th check by r)
    keep = np.ones((shots, 2 * N, 3, ROUNDS))
    keep[shot_idx, atom_idx] = 1 - own
    if to_shot is None:
        to_shot, to_atom = shot_idx, atom_idx
    nb = keep[to_shot[:, None, None], geo.co_atom[T][to_atom], geo.co_slot[T][to_atom]]       # (targets, 3, 5, 13)
    return 1 - nb.prod(axis=2)


def _subsets(S):
    """The flip masks inside the 3-bit mask S."""
    return [f for f in range(8) if f & ~S == 0]


class Tables:
    """Emission tables P(pattern | side, check type, round, class), and their versions with dirty bits removed.

    counts1[side, type, round, class, pattern]                 clean triples
    counts2[side, type, round, class, pattern before, pattern]  clean triples whose round before was clean too
    """

    def __init__(self, counts1, counts2, basis, p, s, meta=None):
        self.counts1, self.counts2 = np.asarray(counts1, np.float64), np.asarray(counts2, np.float64)
        self.basis, self.p, self.s, self.meta = basis, p, s, meta or {}
        groups = round_groups(basis)
        pooled1, pooled2 = np.zeros_like(self.counts1), np.zeros_like(self.counts2)
        for t, T in enumerate(TYPES):
            for g in range(groups[T].max() + 1):                  # the lost classes are pooled over a round group
                rr = np.nonzero(groups[T] == g)[0]
                pooled1[:, t, rr, 1:] = self.counts1[:, t, rr, 1:].sum(axis=1, keepdims=True)
                pooled2[:, t, rr, 1:] = self.counts2[:, t, rr, 1:].sum(axis=1, keepdims=True)
            pooled1[:, t, :, 0] = self.counts1[:, t, :, 0]         # the present class has a table per round
            pooled2[:, t, :, 0] = self.counts2[:, t, :, 0]
        self.pooled1, self.pooled2 = pooled1, pooled2
        self.p1 = (pooled1 + ALPHA) / (pooled1.sum(axis=-1, keepdims=True) + 8 * ALPHA)
        before = pooled2.sum(axis=-1)
        self.p_before = (before + ALPHA) / (before.sum(axis=-1, keepdims=True) + 8 * ALPHA)
        self.cond = (pooled2 + KAPPA * self.p1[..., None, :]) / (before[..., None] + KAPPA)   # P(pattern | before)
        joint = self.p_before[..., None] * self.cond
        y = np.arange(8)
        # sub1[..., S, y]: the single-round table with the bits of S (a 3-bit mask) replaced by fair coins
        self.sub1 = np.zeros(self.p1.shape[:-1] + (8, 8))
        for S in range(8):
            self.sub1[..., S, :] = np.mean([self.p1[..., y ^ f] for f in _subsets(S)], axis=0)
        # sub2[..., Sb, S, yb, y] = P(pattern y | pattern before yb), when the bits Sb of the round before are
        # unknown and the bits S of this round are fair coins
        self.sub2 = np.zeros(joint.shape[:-2] + (8, 8, 8, 8))
        for Sb in range(8):
            den = sum(self.p_before[..., y ^ fb] for fb in _subsets(Sb))                       # (..., yb)
            for S in range(8):
                num = sum(joint[..., y[:, None] ^ fb, y[None, :] ^ f] for fb in _subsets(Sb) for f in _subsets(S))
                self.sub2[..., Sb, S, :, :] = num / len(_subsets(S)) / den[..., None]

    def save(self, path):
        np.savez_compressed(path, counts1=self.counts1.astype(np.int64), counts2=self.counts2.astype(np.int64),
                            meta=np.array(json.dumps({"basis": self.basis, "p": self.p, "s": self.s, **self.meta})))

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            meta = json.loads(str(f["meta"]))
            return cls(f["counts1"], f["counts2"], meta.pop("basis"), meta.pop("p"), meta.pop("s"), meta)


MODES = ("naive", "drop", "coupled")


def subset_weights(rho):
    """rho (..., 3): P(bit m is dirty), independent bits -> (..., 8): P(the dirty set is S)."""
    w = np.ones(rho.shape[:-1] + (8,))
    for S in range(8):
        for m in range(3):
            w[..., S] *= rho[..., m] if S >> m & 1 else 1 - rho[..., m]
    return w


def pair_weights(h_before, h, o_before, o):
    """P(dirty set before = Sb, dirty set now = S), shape (n, R, 8, 8), for the 3 bits of a triple.

    h_*: (n, 3, R) bool, dirty by a herald (known). o_*: (n, 3, R), P(dirty by another lost data atom); such a bit
    stays dirty, so o >= o_before and "dirty before, clean now" has probability 0 from this cause.
    """
    both = np.where(h_before & h, 1.0, np.where(h_before, o, o_before))
    pb, pn = np.where(h_before, 1.0, o_before), np.where(h, 1.0, o)
    joint = np.stack([np.stack([1 - pb - pn + both, pn - both], axis=-1),
                      np.stack([pb - both, both], axis=-1)], axis=-2).clip(0, 1)              # (n, 3, R, 2, 2)
    w = np.ones(both.shape[:1] + both.shape[2:] + (8, 8))
    for Sb in range(8):
        for S in range(8):
            for m in range(3):
                w[:, :, Sb, S] *= joint[:, m, :, Sb >> m & 1, S >> m & 1]
    return w


class Tracker:
    def __init__(self, tables, geo=None):
        self.tables = tables
        self.geo = geo or Geometry(tables.basis, tables.p)
        w = (1 - tables.s) ** np.arange(SLOTS)
        self.log_prior = np.log(w / w.sum())

    def loglik(self, feat, mode="coupled", sweeps=2, order=1, chunk=100):
        """log P(detector history | loss slot j) for every flagged data atom of a batch of shots.

        feat is tracker.features(...): detector patterns and heralds, nothing else.
        Returns (shot_idx, atom_idx, loglik) with loglik of shape (entries, 72). The posterior over the slot is
        softmax(log_prior + loglik); `posterior` does that.
        mode: 'naive'   every bit is trusted unless a herald says otherwise (ignores the other lost atoms);
              'drop'    a check shared with another flagged data atom is ignored in every round;
              'coupled' the other atoms' posteriors say from which round a shared check is ignored; `sweeps`
                        synchronous updates after a first pass that uses their priors.
        order: 0 = independent rounds, 1 = every round conditioned on the round before.
        The result of a shot depends on that shot only (not on the batch or the chunk).
        """
        assert mode in MODES and order in (0, 1)
        shots = len(feat["flagged"])
        parts = [self._loglik_chunk({"flagged": feat["flagged"][a:a + chunk],
                                     "pattern": {T: feat["pattern"][T][a:a + chunk] for T in TYPES},
                                     "herald_dirt": {T: feat["herald_dirt"][T][a:a + chunk] for T in TYPES}},
                                    mode, sweeps, order, a) for a in range(0, shots, chunk)]
        return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))

    def _loglik_chunk(self, feat, mode, sweeps, order, offset):
        geo, tab = self.geo, self.tables
        shots = len(feat["flagged"])
        shot_idx, atom_idx = np.nonzero(feat["flagged"])
        n = len(shot_idx)
        if n == 0:                                    # no flagged data atom: nothing to track
            return shot_idx + offset, atom_idx, np.zeros((0, SLOTS))
        side = geo.side[atom_idx]
        post = np.tile(np.exp(self.log_prior), (n, 1))
        ll = np.zeros((n, SLOTS))
        for _ in range((sweeps if mode == "coupled" else 0) + 1):
            if mode == "naive":
                cdf = np.zeros((n, SLOTS + 1))
            elif mode == "drop":
                cdf = np.ones((n, SLOTS + 1))
            else:
                cdf = np.concatenate([np.zeros((n, 1)), np.cumsum(post, axis=1)], axis=1).clip(0, 1)
            ll = np.zeros((n, SLOTS))
            for t, T in enumerate(TYPES):
                rr = geo.rounds[T]
                h = feat["herald_dirt"][T][shot_idx, atom_idx][:, :, rr]                     # (n, 3, R) bool
                o = others_dirt(geo, T, shot_idx, atom_idx, cdf, shots)[:, :, rr]            # (n, 3, R)
                y = feat["pattern"][T][shot_idx, atom_idx][:, rr]                            # (n, R)
                single = slice(None) if order == 0 else slice(0, 1)                          # rounds without a "before"
                rho = (1 - (1 - h) * (1 - o))[:, :, single]
                w = subset_weights(np.moveaxis(rho, 1, 2))                                   # (n, R1, 8)
                r1 = rr[single]
                sel = tab.sub1[side[:, None], t, r1[None, :], :, :, y[:, single]]           # (n, R1, class, S)
                loge = np.log(np.einsum("nrcs,nrs->nrc", sel, w))
                ll += loge[:, np.arange(len(r1))[:, None], CLS[r1]].sum(axis=1)
                if order == 1:
                    w2 = pair_weights(h[:, :, :-1], h[:, :, 1:], o[:, :, :-1], o[:, :, 1:])  # (n, R - 1, 8, 8)
                    r2 = rr[1:]
                    sel = tab.sub2[side[:, None], t, r2[None, :], :, :, :, y[:, :-1], y[:, 1:]]   # (n, R-1, class, Sb, S)
                    loge = np.log(np.einsum("nrcbs,nrbs->nrc", sel, w2))
                    ll += loge[:, np.arange(len(r2))[:, None], CLS[r2]].sum(axis=1)
            post = softmax(self.log_prior + ll)
        return shot_idx + offset, atom_idx, ll

    def posterior(self, loglik, temperature=1.0):
        """Posterior over the 72 slots. temperature > 1 flattens the evidence (not the prior)."""
        return softmax(self.log_prior + loglik / temperature)


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def to_rounds(post):
    """(entries, 72) over the slots -> (entries, 12) over the loss cycles."""
    return post.reshape(len(post), CYCLES, 6).sum(axis=2)
