"""Second round: block-end residuals, idle faults at block boundaries, hunt criteria.
Uses only my own simulator / tracker (vlib, faultlib, tracker_large)."""
from __future__ import annotations

from collections import Counter

import numpy as np

import iceberg
from vlib import PAULI, apply, qubit_probs
from faultlib import Analysis, Fault, ROT, TWO_PAULIS, describe, pauli_masks, popc, push, qname
from tracker_large import TrackerOnly


def op_runs(an):
    """Run index of every op (a run = one labelled block instance) and last op index of every run."""
    run_of_data, r, prev = [], -1, None
    for name, cnt in an.enc.metadata["iceberg_blocks"]:
        r += 1
        run_of_data.extend([(r, name)] * cnt)
    op_run = [run_of_data[idx][0] for idx, *_ in an.ops]
    last = {}
    for j, r in enumerate(op_run):
        last[r] = j
    label = {r: lab for r, lab in set(run_of_data)}
    return op_run, last, label


def block_end(an, x, z, start, last_op):
    """Push a Pauli sitting before op `start` to just after op `last_op` (rotations skipped)."""
    for j in range(start, last_op + 1):
        _, name, qs, _ = an.ops[j]
        if name not in ROT:
            x, z = push(x, z, name, qs)
    return x, z


def annotate(an, f, x, z, start, home_last):
    """Residual at the end of the block where the fault occurred."""
    anc = ((1 << an.n) - 1) ^ an.code_mask
    xb, zb = block_end(an, x, z, start, home_last)
    f.blk_anc_x = xb & anc
    dec = an.logical_decode(xb, zb)
    f.blk_commutes = dec is not None
    f.blk_lx, f.blk_lz = dec if dec is not None else (-1, -1)
    f.blk_clean = bool(f.blk_commutes and f.blk_anc_x == 0)
    rot = {j: (lx, lz) for j, th, s, lx, lz in an.rot}
    if f.blk_clean:
        f.footprint = popc(f.blk_lx | f.blk_lz)
    else:
        sup = 0
        for j in f.flipped:
            sup |= rot[j][0] | rot[j][1]
        for i, cl in enumerate(an.lay.logical_clbits):
            if cl >= 0 and (f.shift >> cl) & 1:
                sup |= 1 << i
        f.footprint = popc(sup)
    return f


def all_faults(an, exact=True):
    """Gate faults, X/Y/Z before the first gate on every qubit, X/Y/Z idle on every qubit at
    every block boundary, readout flips.  Returns annotated Fault objects."""
    op_run, last, label = op_runs(an)
    n = an.n
    out = []
    tracker_only = isinstance(an, TrackerOnly)

    def finish(f, psi, x, z, start, home_last):
        if tracker_only:
            x_pre, z_pre, x_end, z_end, flipped = an.propagate(x, z, start)
            f.pred_kept = int(an.flips_kept(x_end))
            f.kept = float(f.pred_kept)
            f.shift = an.flips_to_shift(x_end)
            f.flipped = flipped
            f.status = "discarded" if not f.pred_kept else ("harmful" if (f.shift or flipped) else "harmless")
        else:
            an._finish(f, qubit_probs(psi), x, z, start)
        annotate(an, f, x, z, start, home_last)
        # init-state refinement for the tracker-only label
        if tracker_only and f.status == "harmful" and f.block == "init" and f.blk_clean:
            if (an.lay.init == "plus" and f.blk_lz == 0) or (an.lay.init == "zero" and f.blk_lx == 0):
                f.status = "harmless"
        out.append(f)

    def evolve(psi, start):
        for (_, nm, q2, _), mat in zip(an.ops[start:], an.mats[start:]):
            psi = apply(psi, mat, q2, n)
        return psi

    init_last = last[0]
    # before the first gate
    for q in range(n):
        for p in "XYZ":
            psi = None if tracker_only else evolve(apply(an.states[0], PAULI[p], [q], n), 0)
            x, z = pauli_masks(p, [q])
            finish(Fault("prep", q, -1, "reset", (q,), p, "init"), psi, x, z, 0, init_last)
    # after every gate
    for li, (idx, name, qs, params) in enumerate(an.ops):
        for p in (["X", "Y", "Z"] if len(qs) == 1 else TWO_PAULIS):
            psi = None
            if not tracker_only:
                psi = an.states[li + 1]
                for ch, q in zip(p, qs):
                    if ch != "I":
                        psi = apply(psi, PAULI[ch], [q], n)
                psi = evolve(psi, li + 1)
            x, z = pauli_masks(p, qs)
            finish(Fault("gate", li, idx, name, tuple(qs), p, an.op_block[li]), psi, x, z, li + 1, last[op_run[li]])
    # idle on every qubit at every block boundary (after the last op of each run but the last)
    for r in sorted(last):
        j = last[r]
        if j == len(an.ops) - 1:
            continue
        for q in range(n):
            for p in "XYZ":
                psi = None if tracker_only else evolve(apply(an.states[j + 1], PAULI[p], [q], n), j + 1)
                x, z = pauli_masks(p, [q])
                finish(Fault("idle", j, an.ops[j][0], "idle", (q,), p, label[r]), psi, x, z, j + 1, j)
    # readout flips
    for idx, q, c in an.meas:
        f = Fault("meas", q, idx, "measure", (q,), "flip", "measure")
        f.pred_kept = int(an.flips_kept(1 << c))
        f.kept = float(f.pred_kept)
        f.status = "discarded" if not f.pred_kept else "harmful"
        f.blk_anc_x, f.blk_clean, f.blk_commutes, f.footprint, f.blk_lx, f.blk_lz = 0, True, True, 0, 0, 0
        out.append(f)
    return out


def report(tag, an, faults, verbose=3):
    """One line + offenders.  Returns (n_faults, offenders)."""
    enc = an.enc
    exact = not isinstance(an, TrackerOnly)
    if exact:
        bad = [f for f in faults if not f.pred_ok]
        assert not bad, f"tracker mismatch: {describe(enc, bad[0])}"
        assert set(round(f.kept, 6) for f in faults) <= {0.0, 1.0}
    kept = [f for f in faults if f.kept > 0.5]
    harmful = [f for f in kept if f.status == "harmful"]
    per_block = Counter(f.block for f in harmful)
    bus_flipped = [f for f in kept if f.blk_anc_x]
    unclean = [f for f in kept if not f.blk_clean]
    wide = [f for f in harmful if f.footprint > 2]
    part = [f for f in harmful if f.block in ("init", "syndrome", "measure")]
    maxw = max((f.footprint for f in harmful), default=0)
    line = (f"{tag:46s} n={enc.num_qubits:2d} faults={len(faults):5d} kept={len(kept):4d} harmful={len(harmful):3d} "
            f"per block={dict(per_block)} max logical weight={maxw} | kept with bus flipped at block end={len(bus_flipped)} "
            f"kept with non-logical residual={len(unclean)} harmful >2 logical={len(wide)} harmful in init/syndrome/measure={len(part)}")
    if exact:
        ps = max((popc(f.pure_shift) for f in harmful if f.pure_shift >= 0), default=0)
        line += f" max exact XOR-shift weight={ps}"
    print(line, flush=True)
    offenders = {id(f): f for f in bus_flipped + unclean + wide + part}
    for f in list(offenders.values())[:verbose]:
        print("      OFFENDER:", f.kind, describe(enc, f), "| bus X at block end:", [qname(enc, q) for q in range(an.n) if f.blk_anc_x >> q & 1],
              "footprint", f.footprint)
    return len(faults), list(offenders.values())
