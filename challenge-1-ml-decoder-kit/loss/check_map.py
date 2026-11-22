"""Test the loss-to-column map on Clifft shots with loss and no Pauli noise.

usage (from challenge-1-ml-decoder-kit/):
    python -m loss.check_map --basis X --s 1e-3 --shots 1000 --seed 101 --out loss/results/check_map_X_s1e-3.json
    python -m loss.check_map --basis X --forced --shots 400 --seed 201 --out loss/results/check_forced_X.json

Random mode: every shot's stacked (basis detectors, observables) vector must lie
  (a) in the span of the columns its flags allow, over all loss times (what a flag-aware decoder may use), and
  (b) in the span of the generators of the TRUE loss times (from the probes): the envelope claim itself.
When the true generators are linearly independent, the shot's vector fixes which of them fired; each should fire
with probability 1/2 and independently. The tally is reported by kind of generator.

Forced mode: one atom is lost with certainty after one chosen CX (LOSS(1)), nothing else is noisy. One run per
location: own check atom and other check atom after each of their 6 CX in cycle 5, and data atoms after several
CX. Same tests, per location.
"""
import argparse
import json
import time
from collections import defaultdict

import numpy as np

from loss.lossmap import LossMap
from loss.lossmodel import CYCLES, LEFT, N, RIGHT, X_ANC, Layout, clifft_text, kit_circuit, loss_times, sample_chunk

DATA = (*LEFT, *RIGHT)


def column_ints(lmap):
    """Each column of [H; L] as a Python int bitmask (detector rows, then observables)."""
    H, L = lmap.H, lmap.L
    nr = H.shape[0]
    out = []
    for j in range(H.shape[1]):
        v = 0
        for r in H.indices[H.indptr[j]:H.indptr[j + 1]]:
            v |= 1 << int(r)
        for o in L.indices[L.indptr[j]:L.indptr[j + 1]]:
            v |= 1 << (nr + int(o))
        out.append(v)
    return out


def shot_int(det_rows, obs):
    v = 0
    for r in np.nonzero(det_rows)[0]:
        v |= 1 << int(r)
    nr = len(det_rows)
    for o in np.nonzero(obs)[0]:
        v |= 1 << (nr + int(o))
    return v


def solve(gens, target):
    """GF(2): (in span, independent, coefficients or None). gens: list of ints."""
    basis = {}                       # pivot bit -> (vector, combination bitmask over gens)
    independent = True
    for i, g in enumerate(gens):
        v, comb = g, 1 << i
        while v:
            top = v.bit_length() - 1
            if top not in basis:
                basis[top] = (v, comb)
                break
            bv, bc = basis[top]
            v ^= bv
            comb ^= bc
        else:
            independent = False
    v, comb = target, 0
    while v:
        top = v.bit_length() - 1
        if top not in basis:
            return False, independent, None
        bv, bc = basis[top]
        v ^= bv
        comb ^= bc
    return True, independent, (comb if independent else None)


def truth_generators(lmap, heralds_rec, data_slot, check_k, builder):
    """[(column, kind)] of the true loss events of one shot."""
    gens = []
    own_flag = heralds_rec[builder["own_rec"]].astype(bool)
    for a, c in zip(*np.nonzero(own_flag), strict=True):
        gens.append((int(lmap.own_flip[a, c]), "own check: measurement flip"))
    other_flag = heralds_rec[builder["other_rec"]].astype(bool)
    off = N if lmap.basis == "X" else 0
    for a, c in zip(*np.nonzero(other_flag), strict=True):
        k = int(check_k[off + a, c])
        assert k > 0
        for col in lmap.hook_generators(a, c, k):
            gens.append((col, f"other check: hook k={k}"))
    for i in np.nonzero(data_slot)[0]:
        q = DATA[i]
        j = int(data_slot[i])
        d = lmap.data[q]
        later = d["pos"] > j
        if later.any():
            m = int(np.argmax(later))
            gens.append((int(d["col"][m]), "data: first after loss" + (" (transition)" if d["transition"][m] else " (same run)")))
            for mm in np.nonzero(later & d["transition"])[0]:
                if mm != m:
                    gens.append((int(d["col"][mm]), "data: later transition"))
        gens.append((int(d["final"]), "data: final readout"))
    return gens


def flag_generators(lmap, heralds_rec, builder):
    cols = []
    own_flag = heralds_rec[builder["own_rec"]].astype(bool)
    cols += lmap.own_flip[own_flag].tolist()
    other_flag = heralds_rec[builder["other_rec"]].astype(bool)
    cols += lmap.hook[other_flag].ravel().tolist()
    for i in np.nonzero(heralds_rec[builder["data_rec"]])[0]:
        d = lmap.data[DATA[i]]
        cols += d["col"].tolist() + [d["final"]]
    return sorted(set(cols))


def analyse(lmap, colint, smp, layout, probe_layout, new_index):
    data_slot, check_k = loss_times(layout, probe_layout, new_index, smp)
    det = np.asarray(smp.detectors, np.uint8)[:, lmap.rows]
    obs = np.asarray(smp.observables, np.uint8)
    her = np.asarray(smp.heralds, np.uint8)[:, new_index]
    lay = lmap.layout
    builder = {"own_rec": np.array([[lay.record_of(a, c) for c in range(CYCLES)] for a in lmap.own]),
               "other_rec": np.array([[lay.record_of(a, c) for c in range(CYCLES)] for a in lmap.other]),
               "data_rec": np.array([lay.record_of(q) for q in DATA])}
    res = {"shots": len(det), "in_flag_span": 0, "in_truth_span": 0, "independent": 0}
    fired, total = defaultdict(int), defaultdict(int)
    pair_both, pair_n = 0, 0
    for k in range(len(det)):
        target = shot_int(det[k], obs[k])
        fg = flag_generators(lmap, her[k], builder)
        ok_flag, _, _ = solve([colint[c] for c in fg], target)
        tg = truth_generators(lmap, her[k], data_slot[k], check_k[k], builder)
        ok_truth, indep, comb = solve([colint[c] for c, _ in tg], target)
        res["in_flag_span"] += ok_flag
        res["in_truth_span"] += ok_truth
        if ok_truth and indep:
            res["independent"] += 1
            bits = [(comb >> i) & 1 for i in range(len(tg))]
            for (_, kind), b in zip(tg, bits, strict=True):
                fired[kind] += b
                total[kind] += 1
            if len(bits) >= 2:        # independence: first two generators of the shot
                pair_both += bits[0] & bits[1]
                pair_n += 1
    res["fire_rate"] = {kind: [fired[kind], total[kind], round(fired[kind] / total[kind], 4)] for kind in sorted(total)}
    res["pair_both_fired"] = [pair_both, pair_n, round(pair_both / max(pair_n, 1), 4)]
    res["mean_lost_data_per_shot"] = float((data_slot > 0).sum(axis=1).mean())
    res["mean_check_losses_per_shot"] = float((check_k > 0).sum(axis=(1, 2)).mean())
    return res


def forced_text(circuit, atom, cx_index):
    """The noiseless circuit with the atom lost for certain after its cx_index-th CX (1-based), with probes."""
    text, probe_layout, new_index = clifft_text(circuit, 0.0, probes=True)
    lines, count, out = text.splitlines(), 0, []
    done = False
    for line in lines:
        out.append(line)
        if line.startswith("CX ") and not done:
            qs = [int(x) for x in line.split()[1:]]
            if atom in qs:
                count += 1
                if count == cx_index:
                    out.append(f"LOSS(1) {atom}")
                    done = True
    assert done      # the LOSS(1) line sits between the CX and its probes, so the probes see the loss
    return "\n".join(out) + "\n", probe_layout, new_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", default="X")
    ap.add_argument("--s", type=float, default=1e-3)
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--forced", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    t0 = time.perf_counter()
    lmap = LossMap(a.basis, 0.004)          # columns are geometric: the same at every p
    colint = column_ints(lmap)
    circuit = kit_circuit(a.basis, 0)
    layout = Layout(circuit)
    out = {"basis": a.basis, "seed": a.seed, "shots": a.shots}
    if not a.forced:
        out["s"] = a.s
        text, pl, ni = clifft_text(circuit, a.s, probes=True)
        smp = sample_chunk(text, a.shots, a.seed)
        out |= analyse(lmap, colint, smp, layout, pl, ni)
        print(json.dumps(out, indent=1))
    else:
        own0, other0 = lmap.own[0], lmap.other[0]
        locations = [(own0, 6 * 5 + k, f"own check {own0}, cycle 5, after CX {k}") for k in range(1, 7)]
        locations += [(other0, 6 * 5 + k, f"other check {other0}, cycle 5, after CX {k}") for k in range(1, 7)]
        for q in (LEFT[0], RIGHT[0]):
            locations += [(q, j, f"data {q}, after CX {j}") for j in (1, 2, 3, 4, 5, 6, 31, 32, 33, 34, 35, 36, 71, 72)]
        out["locations"] = []
        for i, (atom, j, label) in enumerate(locations):
            text, pl, ni = forced_text(circuit, atom, j)
            smp = sample_chunk(text, a.shots, a.seed + i)
            r = analyse(lmap, colint, smp, layout, pl, ni)
            r["label"] = label
            out["locations"].append(r)
            print(label, {k: r[k] for k in ("shots", "in_flag_span", "in_truth_span", "independent")}, r["fire_rate"], flush=True)
    out["seconds"] = round(time.perf_counter() - t0, 1)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
