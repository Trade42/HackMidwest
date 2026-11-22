"""Which columns of the decoding matrix H a loss can flip, and the per-shot priors built from the heralds.

Everything is read from the kit's circuit. For a memory basis B the decoder sees the B-check detectors, so on a
data atom it sees the Pauli R (Z for X memory, X for Z memory). A lost atom takes no part in its later gates:
an absent CX control acts like a control in |0>, an absent target like a target in |+> (the "Pauli envelope" of
Liu et al. 2603.04156). From this:

- own check atom (a B check), lost in cycle c: its recorded bit is random. Column: its measurement flip in c.
- other check atom, lost after its k-th CX of cycle c: the data it already met are dephased by the partial check.
  Column: R on the check atom right after its k-th CX (k = 1..5; after the 6th CX there is no effect on B).
- data atom, lost after its j-th CX (j = 1..72): R on the atom right before each later CX in which R does not
  commute through (the CX where the data atom is the target for X memory, the control for Z memory). Of these,
  the first after the loss, and every one that starts a run of such CXs ("transition"), fire with probability 1/2;
  so does the final readout flip.

Each listed column fires independently with probability 1/2. check_map.py tests all of this on Clifft shots.
"""
import numpy as np
import stim

from circuit import dem_matrices, detector_ids
from loss.lossmodel import CYCLES, LEFT, N, RIGHT, X_ANC, Z_ANC, Layout, kit_circuit

TAG = 1e-4    # probe errors get probabilities TAG * (k + 1) so that they can be told apart in the DEM


class LossMap:
    def __init__(self, basis, p):
        self.basis, self.p = basis, p
        circuit = kit_circuit(basis, p)
        self.layout = Layout(circuit)
        assert self.layout.memory_basis == basis
        self.H, self.L, self.pri = dem_matrices(circuit, basis)
        self.H, self.L = self.H.tocsc(), self.L.tocsc()
        self.rows = detector_ids(circuit, basis)
        self.row_of = {int(d): r for r, d in enumerate(self.rows)}
        self.ncols = self.H.shape[1]
        self.col_of = {}
        for j in range(self.ncols):
            key = (tuple(self.H.indices[self.H.indptr[j]:self.H.indptr[j + 1]].tolist()),
                   tuple(self.L.indices[self.L.indptr[j]:self.L.indptr[j + 1]].tolist()))
            assert key not in self.col_of, "two columns of H and L share an effect"
            self.col_of[key] = j
        self.R = "Z" if basis == "X" else "X"
        self.sensitive_role = "T" if basis == "X" else "C"
        self.own, self.other = (X_ANC, Z_ANC) if basis == "X" else (Z_ANC, X_ANC)
        self.noiseless = circuit.without_noise().flattened()
        self._build()

    # ---- effects and columns

    def _column(self, detectors, observables):
        key = (tuple(sorted(self.row_of[d] for d in detectors if d in self.row_of)), tuple(sorted(observables)))
        assert key in self.col_of, f"a loss effect {key} is not a column of H"
        return self.col_of[key]

    def _record_column(self, record):
        """Column whose effect equals flipping one measurement record."""
        lay = self.layout
        dets = [d for d, recs in enumerate(lay.det_records) if recs.count(record) % 2]
        obs = [k for k, recs in lay.obs_records.items() if recs.count(record) % 2]
        return self._column(dets, obs)

    def _probe(self, insertions):
        """insertions: list of (tick, qubit). R on the qubit at the start of the layer after `tick` TICKs.

        Returns the column of each insertion. One DEM per tick, with tagged probabilities.
        """
        by_tick = {}
        for i, (tick, q) in enumerate(insertions):
            by_tick.setdefault(tick, []).append((i, q))
        cols = [None] * len(insertions)
        for tick, items in by_tick.items():
            if not self._probe_group(tick, items, cols):
                for item in items:       # two probes had one effect (merged), or one had none: one by one
                    assert self._probe_group(tick, [item], cols), f"probe {item} at tick {tick} has no effect"
        assert all(c is not None for c in cols)
        return cols

    def _probe_group(self, tick, items, cols):
        """Insert the tagged errors before the first instruction of the layer that follows `tick` TICKs."""
        out, t, inserted = stim.Circuit(), 0, False
        for inst in self.noiseless:
            if t == tick and not inserted and inst.name != "TICK":
                for k, (_, q) in enumerate(items):
                    out.append(f"{self.R}_ERROR", [q], TAG * (k + 1))
                inserted = True
            out.append(inst)
            if inst.name == "TICK":
                t += 1
        assert inserted
        errors = [e for e in out.detector_error_model().flattened() if e.type == "error"]
        if len(errors) != len(items):
            return False
        for e in errors:
            k = int(round(e.args_copy()[0] / TAG)) - 1
            tg = e.targets_copy()
            dets = [x.val for x in tg if x.is_relative_detector_id()]
            obs = [x.val for x in tg if x.is_logical_observable_id()]
            cols[items[k][0]] = self._column(dets, obs)
        return True

    def _build(self):
        lay = self.layout
        # own check atoms: measurement flip of check a in cycle c
        self.own_flip = np.array([[self._record_column(lay.record_of(a, c)) for c in range(CYCLES)] for a in self.own])
        # other check atoms: R on the atom right after its k-th CX of cycle c, k = 1..5
        ins = []
        for a in self.other:
            for c in range(CYCLES):
                for k in range(1, 6):
                    tick = lay.cx[a][6 * c + k - 1][0]
                    ins.append((tick + 1, a))
        self.hook = np.array(self._probe(ins)).reshape(N, CYCLES, 5)
        # data atoms: R right before each sensitive CX, and the final readout flip
        self.data = {}
        ins, owners = [], []
        for q in (*LEFT, *RIGHT):
            for j, (tick, role, _) in enumerate(lay.cx[q]):
                if role == self.sensitive_role:
                    ins.append((tick, q))
                    owners.append((q, j + 1))
        cols = self._probe(ins)
        for q in (*LEFT, *RIGHT):
            roles = [role for _, role, _ in lay.cx[q]]
            pos = [j for (qq, j) in owners if qq == q]
            col = [c for (qq, _), c in zip(owners, cols, strict=True) if qq == q]
            transition = [j == 1 or roles[j - 2] != self.sensitive_role for j in pos]
            final = self._record_column(lay.record_of(q))
            self.data[q] = {"pos": np.array(pos), "col": np.array(col), "transition": np.array(transition), "final": final}
            # the starter's map has one transition per cycle; the schedule must give exactly that
            per_cycle = np.bincount((np.array(pos)[np.array(transition)] - 1) // 6, minlength=CYCLES)
            assert (per_cycle[1:] == 1).all() and per_cycle[0] in (1, 2), f"data atom {q}: transitions {per_cycle}"

    # ---- generators of one loss event (the columns that fire with probability 1/2)

    def data_generators(self, q, j):
        """Columns that fire with probability 1/2 when data atom q is lost after its j-th CX (1..72)."""
        d = self.data[q]
        later = d["pos"] > j
        gens = set(d["col"][later & d["transition"]].tolist())
        if later.any():
            gens.add(int(d["col"][np.argmax(later)]))        # the first sensitive CX after the loss
        gens.add(int(d["final"]))
        return gens

    def data_activation(self, q, post):
        """P(column fires | loss slot ~ post) / 1, for each sensitive column of q; post[j - 1] = P(lost after CX j)."""
        d = self.data[q]
        cdf_before = np.concatenate([[0.0], np.cumsum(post)])          # cdf_before[m] = P(j < m + 1) = P(j <= m)
        act = np.where(d["transition"], cdf_before[d["pos"] - 1], post[np.maximum(d["pos"] - 2, 0)] * (d["pos"] > 1))
        return d["col"], act

    def hook_generators(self, a_index, c, k):
        """Column that fires with probability 1/2 when other-check atom a_index (0..71) is lost after its k-th CX."""
        return set() if k == 6 else {int(self.hook[a_index, c, k - 1])}


def geometric(n, s):
    """P(lost after the m-th of n CX | lost at one of them), m = 1..n, for loss probability s per CX."""
    w = (1 - s) ** np.arange(n)
    return w / w.sum()


# ---- per-shot priors

CONDITIONS = {
    # name: own / hook / data / final / unflagged, see priors()
    "blind":           dict(flags=False),
    "blind-rate":      dict(flags=False, unflagged="rate"),
    "starter":         dict(own=True, hook="uniform", data="starter", final=True),
    "starter-nofinal": dict(own=True, hook="uniform", data="starter", final=False),
    "starter-nohook":  dict(own=True, hook=None, data="starter", final=True),
    "starter-noown":   dict(own=False, hook="uniform", data="starter", final=True),
    "starter-ratebase": dict(own=True, hook="uniform", data="starter", final=True, unflagged="rate"),
    "exact":           dict(own=True, hook="geometric", data="uniform", final=True),
    "oracle-round":    dict(own=True, hook="geometric", data="oracle-round", final=True),
    "oracle-slot":     dict(own=True, hook="geometric", data="oracle-slot", final=True),
    "oracle-all":      dict(own=True, hook="oracle", data="oracle-slot", final=True),
    # the starter's map (one column per cycle, uniform hooks) with the true loss round in place of the uniform guess
    "starter-oracle-round": dict(own=True, hook="uniform", data="starter-oracle-round", final=True),
}


def parse_condition(name):
    """'starter', 'starter@0.4' (cap 0.4) or 'starter@0.5/stop1' (relay-BP stopping at its first solution)."""
    base, _, rest = name.partition("@")
    cap, _, relay = rest.partition("/")
    spec = dict(CONDITIONS[base])
    spec["cap"] = float(cap) if cap else 0.5
    spec["relay"] = {"stop_nconv": 1} if relay == "stop1" else {}
    assert relay in ("", "stop1"), name
    return spec


class PriorBuilder:
    """priors(spec, heralds, data_slot, check_k) for one shot. Truth (data_slot, check_k) is read only by oracles."""

    def __init__(self, lmap, s):
        self.m, self.s = lmap, s
        lay = lmap.layout
        self.own_rec = np.array([[lay.record_of(a, c) for c in range(CYCLES)] for a in lmap.own])
        self.other_rec = np.array([[lay.record_of(a, c) for c in range(CYCLES)] for a in lmap.other])
        self.data_rec = np.array([lay.record_of(q) for q in (*LEFT, *RIGHT)])
        self.w6 = geometric(6, s)
        self.w72 = geometric(72, s)
        # the starter's flicker column per cycle and its prior (c + 0.5) / 12
        self.starter_cols = {}
        for q, d in lmap.data.items():
            tr = d["transition"] & (d["pos"] > 1)
            cols = d["col"][tr]
            assert len(cols) == CYCLES and ((d["pos"][tr] - 1) // 6 == np.arange(CYCLES)).all()
            self.starter_cols[q] = cols
        self._rate = self._rate_bumps()

    def _rate_bumps(self):
        """(columns, activation) for every atom without flags: a decoder that knows s but not where."""
        m, s = self.m, self.s
        cols, act = [], []
        p_cycle = 1 - (1 - s) ** 6
        cols.append(m.own_flip.ravel()); act.append(np.full(m.own_flip.size, p_cycle))
        k_act = s * (1 - s) ** np.arange(5)
        cols.append(m.hook.ravel()); act.append(np.tile(k_act, N * CYCLES))
        unconditional = s * (1 - s) ** np.arange(72)            # P(lost after CX j), not conditioned on a flag
        for q, d in m.data.items():
            c, a = m.data_activation(q, unconditional)
            cols.append(c); act.append(a)
            cols.append([d["final"]]); act.append([1 - (1 - s) ** 72])
        return np.concatenate(cols).astype(np.int64), 0.5 * np.concatenate(act)

    def priors(self, spec, heralds, data_slot=None, check_k=None):
        m = self.m
        if not spec.get("flags", True):
            if spec.get("unflagged") == "rate":
                return _combine(m.pri, *self._rate, spec["cap"])
            return np.minimum(m.pri, spec["cap"])
        cols, vals = [], []
        own_flag = heralds[self.own_rec].astype(bool)            # (72, 12)
        other_flag = heralds[self.other_rec].astype(bool)
        data_flag = heralds[self.data_rec].astype(bool)          # (144,)
        if spec.get("unflagged") == "rate":
            rc, rv = self._rate
            keep = np.ones(len(rc), bool)
            # drop the rate bumps of flagged atoms; they get their flag bumps below
            flagged_cols = set(m.own_flip[own_flag].tolist()) | set(m.hook[other_flag].ravel().tolist())
            for i in np.nonzero(data_flag)[0]:
                d = m.data[(*LEFT, *RIGHT)[i]]
                flagged_cols |= set(d["col"].tolist()) | {d["final"]}
            keep = ~np.isin(rc, list(flagged_cols))
            cols.append(rc[keep]); vals.append(rv[keep])
        if spec["own"]:
            cols.append(m.own_flip[own_flag]); vals.append(np.full(own_flag.sum(), 0.5))
        if spec["hook"]:
            for a, c in zip(*np.nonzero(other_flag), strict=True):
                if spec["hook"] == "uniform":
                    cols.append(m.hook[a, c]); vals.append(np.full(5, 0.5 / 6))
                elif spec["hook"] == "geometric":
                    cols.append(m.hook[a, c]); vals.append(0.5 * self.w6[:5])
                elif spec["hook"] == "oracle":
                    k = int(check_k[(N + a) if m.basis == "X" else a, c])
                    assert k > 0, "a flagged check atom with no recorded loss"
                    if k < 6:
                        cols.append([m.hook[a, c, k - 1]]); vals.append([0.5])
        for i in np.nonzero(data_flag)[0]:
            q = (*LEFT, *RIGHT)[i]
            d = m.data[q]
            if spec["data"] == "starter":
                cols.append(self.starter_cols[q]); vals.append(0.5 * (np.arange(CYCLES) + 0.5) / CYCLES)
            elif spec["data"] == "starter-oracle-round":
                c = (data_slot[i] - 1) // 6
                post = np.zeros(72); post[6 * c:6 * c + 6] = self.w6
                c_, a_ = m.data_activation(q, post)
                tr = d["transition"] & (d["pos"] > 1)          # exactly the starter's 12 columns
                assert (c_[tr] == self.starter_cols[q]).all()
                cols.append(c_[tr]); vals.append(0.5 * a_[tr])
            else:
                if spec["data"] == "uniform":
                    post = self.w72
                elif spec["data"] == "oracle-slot":
                    post = np.zeros(72); post[data_slot[i] - 1] = 1.0
                elif spec["data"] == "oracle-round":
                    c = (data_slot[i] - 1) // 6
                    post = np.zeros(72); post[6 * c:6 * c + 6] = self.w6
                c_, a_ = m.data_activation(q, post)
                cols.append(c_); vals.append(0.5 * a_)
            if spec["final"]:
                cols.append([d["final"]]); vals.append([0.5])
        if not cols:
            return np.minimum(m.pri, spec["cap"])
        return _combine(m.pri, np.concatenate(cols).astype(np.int64), np.concatenate(vals), spec["cap"])


def _combine(base, cols, vals, cap):
    """XOR-combine independent flips: P = (1 - prod(1 - 2 v)) / 2 over the base prior and every bump; then cap."""
    prod = 1 - 2 * base.astype(np.float64)
    np.multiply.at(prod, cols, 1 - 2 * np.asarray(vals, np.float64))
    return np.minimum((1 - prod) / 2, cap)
