"""Step 1: the signal a loss-time tracker can use, measured on training shots with the probes as the truth.

usage (from challenge-1-ml-decoder-kit/):
    python -m loss_tracker.signal --data TRAIN.npz --tables loss_tracker/tables/X_p0.004_s1e-3.npz \
        --out loss_tracker/results/signal_train

What it reports:
  sharing    how often a lost data atom is NOT alone: another lost data atom on one of its 6 checks, a flagged
             check atom next to it, a check that was already randomised when it was lost;
  by_delta   for clean triples, the firing rate of each of the 3 detectors and the joint pattern, by the round
             relative to the loss round;
  repeat     for an atom lost three or more cycles ago: how often detector m of round r equals detector m' of
             round r - 1 (0.5 = unrelated); this is why the tracker conditions a round on the round before;
  hook       what a flagged check atom does to the OTHER-type triple of the data atoms next to it;
  own        the flagged check atom's own two detectors;
  shared     a bit that another lost atom has randomised: is it a fair coin, unrelated to the other two bits?
"""
import argparse
import json

import numpy as np

from loss.lossmodel import CYCLES, load_dataset
from loss_tracker.fit import true_cdf
from loss_tracker.tracker import CLS, ROUNDS, TYPES, Geometry, Tables, features, others_dirt

DELTA = ["<= -2", "-1", "0 (the loss round)", "+1", "+2", ">= +3"]
WEIGHT = np.array([bin(y).count("1") for y in range(8)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--shots", type=int)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = load_dataset(a.data)
    meta = data["meta"]
    n = a.shots or meta["shots"]
    geo = Geometry(meta["basis"], meta["p"])
    atoms = len(geo.side)
    share = np.zeros(8)                                      # counters over lost atoms, see below
    dirty_bits = np.zeros((2, 4))                            # [type, number of dirty bits] over triple-rounds from the loss on
    by_delta = np.zeros((2, 2, len(DELTA), 8))               # [side, type, delta bin, pattern], clean triples
    hook = np.zeros((2, 2, 2, 8))                            # [side, type, hook possible, pattern], atoms never lost
    shared = np.zeros((2, 2, 3, 2, 4))                       # [side, type, dirty bit m, its value, pattern of the other two]
    own = np.zeros((2, 2, 2))                                # [type, round = c or c + 1, (fired, reads)]
    valid = {T: np.isin(np.arange(ROUNDS), geo.rounds[T]) for T in TYPES}
    for lo in range(0, n, 500):
        det, her = data["detectors"][lo:lo + 500], data["heralds"][lo:lo + 500]
        slot = data["data_slot"][lo:lo + 500].astype(np.int64)
        shots = len(slot)
        feat = features(geo, det, her)
        s_idx, a_idx = np.nonzero(slot)
        cdf = true_cdf(slot[s_idx, a_idx])
        cyc = (slot - 1) // 6
        all_s, all_a = (x.ravel() for x in np.indices((shots, atoms)))
        od = {T: others_dirt(geo, T, s_idx, a_idx, cdf, shots, all_s, all_a).reshape(shots, atoms, 3, ROUNDS) > 0.5 for T in TYPES}
        lost = slot > 0
        # ---- sharing, per lost atom
        sh = {T: od[T][:, :, :, ROUNDS - 1].any(axis=2) for T in TYPES}                  # another lost atom on a T check
        at_loss = {T: np.take_along_axis(od[T], np.clip(cyc, 0, None)[:, :, None, None], axis=3)[..., 0].any(axis=2) for T in TYPES}
        nb = np.zeros((shots, atoms), bool)
        for T in TYPES:
            nb |= her[:, geo.check_rec[T]].astype(bool).any(axis=2)[:, geo.chk[T]].any(axis=2)
        share += [lost.sum(), (lost & sh["X"]).sum(), (lost & sh["Z"]).sum(), (lost & (sh["X"] | sh["Z"])).sum(),
                  (lost & nb).sum(), (lost & ~sh["X"] & ~sh["Z"] & ~nb).sum(),
                  (lost & (at_loss["X"] | at_loss["Z"])).sum(), (lost & sh["X"] & sh["Z"]).sum()]
        rr = np.arange(ROUNDS)[None, None, :]
        for t, T in enumerate(TYPES):
            dirt = od[T] | feat["herald_dirt"][T]
            y = feat["pattern"][T].astype(np.int64)
            side = np.broadcast_to(geo.side[None, :, None], y.shape)
            ok_round = np.broadcast_to(valid[T][None, None, :], y.shape)
            after = lost[:, :, None] & (rr >= cyc[:, :, None]) & ok_round
            dirty_bits[t] += np.bincount(dirt.sum(axis=2)[after], minlength=4)
            # ---- firing by the round relative to the loss, clean triples, rounds 1..11 (round 12 holds the final readout)
            clean = ~dirt.any(axis=2) & ok_round & (rr >= 1) & (rr <= CYCLES - 1) & lost[:, :, None]
            delta = np.clip(rr - cyc[:, :, None], -2, 3) + 2
            idx = (side[clean] * len(DELTA) + np.broadcast_to(delta, y.shape)[clean]) * 8 + y[clean]
            by_delta[:, t] += np.bincount(idx, minlength=2 * len(DELTA) * 8).reshape(2, len(DELTA), 8)
            # ---- a flagged other-type check atom next to an atom that was never lost
            base = ~lost[:, :, None] & ok_round & ~od[T].any(axis=2) & ~feat["own_herald"][T].any(axis=2)
            idx = (side[base] * 2 + feat["hook"][T][base]) * 8 + y[base]
            hook[:, t] += np.bincount(idx, minlength=2 * 2 * 8).reshape(2, 2, 8)
            # ---- a bit randomised by another lost atom, on an atom that was never lost: exactly that bit dirty
            for m in range(3):
                only = ~lost[:, :, None] & ok_round & ~feat["herald_dirt"][T].any(axis=2) & od[T][:, :, m] & (od[T].sum(axis=2) == 1)
                bit = (y >> m) & 1
                rest = ((y >> ((m + 1) % 3)) & 1) | (((y >> ((m + 2) % 3)) & 1) << 1)
                idx = (side[only] * 2 + bit[only]) * 4 + rest[only]
                shared[:, t, m] += np.bincount(idx, minlength=16).reshape(2, 2, 4)
            # ---- the flagged check atom's own detectors: rounds c and c + 1
            h = her[:, geo.check_rec[T]].astype(bool)                                   # (shots, 72, 12)
            bits = np.where(geo.det[T] >= 0, det[:, np.maximum(geo.det[T], 0)], 0)      # (shots, 72, 13)
            for k in (0, 1):
                rounds_ok = valid[T][k:k + CYCLES]
                sel = h & rounds_ok[None, None, :]
                own[t, k] += [bits[:, :, k:k + CYCLES][sel].sum(), sel.sum()]
    tables = Tables.load(a.tables)
    out = {"dataset": {**meta, "file": a.data.split("/")[-1], "shots_used": n},
           "cx_order": {("left" if s == 0 else "right"): {T: geo.pos[T][s * 72].tolist() for T in TYPES} for s in (0, 1)},
           "other_data_atoms_on_its_checks": int(len({int(x) for T in TYPES for x in geo.co_atom[T][0].ravel()}))}
    names = ["lost_data_atoms", "another_lost_atom_on_an_X_check", "another_lost_atom_on_a_Z_check", "another_lost_atom_on_any_check",
             "a_flagged_check_atom_next_to_it", "alone_in_every_sense", "a_check_already_randomised_when_it_was_lost",
             "another_lost_atom_on_X_and_on_Z_checks"]
    out["sharing"] = {k: int(v) for k, v in zip(names, share, strict=True)}
    out["sharing_share"] = {k: float(v / share[0]) for k, v in zip(names[1:], share[1:], strict=True)}
    out["triple_rounds_from_the_loss_on"] = {T: {f"{k}_dirty_bits": float(dirty_bits[t, k] / dirty_bits[t].sum()) for k in range(4)}
                                             for t, T in enumerate(TYPES)}
    md = [f"Dataset {a.data.split('/')[-1]} (seed {meta['seed']}), first {n} shots; {int(share[0])} lost data atoms "
          f"({share[0] / n:.2f} per shot). Each data atom shares its 6 checks with {out['other_data_atoms_on_its_checks']} other data atoms.", "",
          "| Of the lost data atoms | share |", "|---|---|"]
    md += [f"| {k.replace('_', ' ')} | {v:.3f} |" for k, v in out["sharing_share"].items()]
    md += ["", "Triple-rounds from the loss round on, by the number of the 3 detector bits that are dirty (flagged check atom, hook, or another lost atom):", "",
           "| check type | 0 dirty (clean) | 1 | 2 | 3 |", "|---|---|---|---|---|"]
    md += [f"| {T} | " + " | ".join(f"{dirty_bits[t, k] / dirty_bits[t].sum():.3f}" for k in range(4)) + " |" for t, T in enumerate(TYPES)]
    out["by_delta"] = []
    md += ["", "Clean triples of lost atoms, rounds 2..12 of 13 (detector rounds 1..11): firing rate of each detector, in the atom's CX order, and the joint pattern.", "",
           "| side | check type (CX positions) | round − loss round | triples | rate of detector 1 / 2 / 3 | none fires | exactly one | exactly two | all three |",
           "|---|---|---|---|---|---|---|---|---|"]
    for s, sname in enumerate(("left", "right")):
        for t, T in enumerate(TYPES):
            for d, dname in enumerate(DELTA):
                c = by_delta[s, t, d]
                tot = c.sum()
                rates = [float(c[[y for y in range(8) if y >> m & 1]].sum() / tot) for m in range(3)]
                w = [float(c[WEIGHT == k].sum() / tot) for k in range(4)]
                out["by_delta"].append({"side": sname, "type": T, "cx_positions": geo.pos[T][s * 72].tolist(), "delta": dname, "triples": int(tot),
                                        "rates": rates, "weight_distribution": w, "patterns": (c / tot).tolist()})
                md.append(f"| {sname} | {T} {tuple(geo.pos[T][s * 72].tolist())} | {dname} | {int(tot)} | " + " / ".join(f"{x:.3f}" for x in rates)
                          + " | " + " | ".join(f"{x:.3f}" for x in w) + " |")
    # the steady patterns and the repeat across rounds, from the fitted tables (class 19 = lost three or more cycles ago)
    out["steady"] = []
    md += ["", "An atom lost three or more cycles ago, clean triples (the fitted tables; patterns are written detector 1, 2, 3):", "",
           "| side | check type (CX positions) | pattern probabilities 000, 100, 010, 110, 001, 101, 011, 111 | the same for a present atom (round 6) |", "|---|---|---|---|"]
    for s, sname in enumerate(("left", "right")):
        for t, T in enumerate(TYPES):
            r = int(geo.rounds[T][len(geo.rounds[T]) // 2])
            lostp, pres = tables.p1[s, t, r, 19], tables.p1[s, t, r, 0]
            out["steady"].append({"side": sname, "type": T, "lost": lostp.tolist(), "present": pres.tolist()})
            md.append(f"| {sname} | {T} {tuple(geo.pos[T][s * 72].tolist())} | " + " ".join(f"{x:.3f}" for x in lostp) + " | " + " ".join(f"{x:.3f}" for x in pres) + " |")
    out["repeat"] = []
    md += ["", "The same atoms: P(detector m of round r = detector m' of round r − 1). 0.5 means unrelated.", "",
           "| side | check type | m = 1: m' = 1, 2, 3 | m = 2: m' = 1, 2, 3 | m = 3: m' = 1, 2, 3 | present atom, largest departure from its no-relation value |", "|---|---|---|---|---|---|"]
    for s, sname in enumerate(("left", "right")):
        for t, T in enumerate(TYPES):
            r = int(geo.rounds[T][len(geo.rounds[T]) // 2])
            agree = np.zeros((3, 3))
            c2 = tables.pooled2[s, t, r, 19]
            for m in range(3):
                for mb in range(3):
                    agree[m, mb] = sum(c2[yb, y] for yb in range(8) for y in range(8) if (y >> m & 1) == (yb >> mb & 1)) / c2.sum()
            c2p = tables.pooled2[s, t, r, 0]
            pb, pn = c2p.sum(axis=1) / c2p.sum(), c2p.sum(axis=0) / c2p.sum()
            dep = 0.0
            for m in range(3):
                for mb in range(3):
                    got = sum(c2p[yb, y] for yb in range(8) for y in range(8) if (y >> m & 1) == (yb >> mb & 1)) / c2p.sum()
                    qb, qn = sum(pb[yb] for yb in range(8) if yb >> mb & 1), sum(pn[y] for y in range(8) if y >> m & 1)
                    dep = max(dep, abs(got - (qb * qn + (1 - qb) * (1 - qn))))
            out["repeat"].append({"side": sname, "type": T, "agree": agree.tolist(), "present_largest_departure": float(dep)})
            md.append(f"| {sname} | {T} | " + " | ".join(", ".join(f"{x:.3f}" for x in agree[m]) for m in range(3)) + f" | {dep:.3f} |")
    out["hook"] = []
    md += ["", "A data atom that was never lost, its triple otherwise clean: the pattern when a check atom of the OTHER type next to it was flagged in this cycle or the one before.", "",
           "| side | check type of the triple | flagged other-type neighbour | triple-rounds | none fires | exactly one | exactly two | all three |", "|---|---|---|---|---|---|---|---|"]
    for s, sname in enumerate(("left", "right")):
        for t, T in enumerate(TYPES):
            for hk in (0, 1):
                c = hook[s, t, hk]
                w = [float(c[WEIGHT == k].sum() / c.sum()) for k in range(4)]
                out["hook"].append({"side": sname, "type": T, "hook_possible": bool(hk), "triple_rounds": int(c.sum()), "weight_distribution": w,
                                    "patterns": (c / c.sum()).tolist()})
                md.append(f"| {sname} | {T} | {'yes' if hk else 'no'} | {int(c.sum())} | " + " | ".join(f"{x:.4f}" for x in w) + " |")
    out["own"] = {T: {"round_c": float(own[t, 0, 0] / own[t, 0, 1]), "round_c_plus_1": float(own[t, 1, 0] / own[t, 1, 1]),
                      "reads": [int(own[t, 0, 1]), int(own[t, 1, 1])]} for t, T in enumerate(TYPES)}
    md += ["", "A check atom flagged in cycle c: the firing rate of its own detectors in rounds c and c + 1: "
           + "; ".join(f"{T} checks {out['own'][T]['round_c']:.3f} and {out['own'][T]['round_c_plus_1']:.3f} ({out['own'][T]['reads'][0]} and {out['own'][T]['reads'][1]} reads)"
                       for T in TYPES) + "."]
    out["shared"] = []
    md += ["", "A data atom that was never lost, with exactly one of its 3 bits on a check that another lost atom has already left: that bit, and the other two bits.", "",
           "| side | check type | bit | triple-rounds | rate of that bit | other two bits 00/10/01/11 when it is 0 | when it is 1 | largest difference |", "|---|---|---|---|---|---|---|---|"]
    for s, sname in enumerate(("left", "right")):
        for t, T in enumerate(TYPES):
            for m in range(3):
                c = shared[s, t, m]
                d0, d1 = c[0] / c[0].sum(), c[1] / c[1].sum()
                out["shared"].append({"side": sname, "type": T, "bit": m + 1, "triple_rounds": int(c.sum()), "rate": float(c[1].sum() / c.sum()),
                                      "rest_given_0": d0.tolist(), "rest_given_1": d1.tolist(), "largest_difference": float(np.abs(d0 - d1).max())})
                md.append(f"| {sname} | {T} | {m + 1} | {int(c.sum())} | {c[1].sum() / c.sum():.3f} | " + " ".join(f"{x:.3f}" for x in d0) + " | "
                          + " ".join(f"{x:.3f}" for x in d1) + f" | {np.abs(d0 - d1).max():.3f} |")
    with open(a.out + ".json", "w") as f:
        json.dump(out, f, indent=1)
    with open(a.out + ".md", "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
