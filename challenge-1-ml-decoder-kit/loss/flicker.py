"""L5: do the checks next to a lost data atom "flicker" after the loss? The signal a loss-time tracker would use.

usage (from challenge-1-ml-decoder-kit/):
    python -m loss.flicker --data DATA.npz --out loss/results/L5_flicker_X_s2e-4

For every lost data atom q (true loss CX j from the probes, loss round tau = (j - 1) // 6) the 6 checks that
contain q (3 X checks, 3 Z checks) are read in every round r that has a detector on them. The table gives the
firing rate of those detectors by r - tau, for each check type.

Isolation: q is kept only if no other data atom that shares one of its 6 checks was lost, and none of its 6
check atoms was lost in any cycle. The baseline is the same statistic for data atoms that were not lost and pass
the same isolation test (every 7th data atom of every shot, to keep it cheap).
"""
import argparse
import json

import numpy as np

from loss.lossmodel import CYCLES, LEFT, N, RIGHT, X_ANC, Z_ANC, Layout, kit_circuit, load_dataset

DATA = (*LEFT, *RIGHT)
INDEX = {q: i for i, q in enumerate(DATA)}
BINS = ["<= -3", "-2", "-1", "0 (loss round)", "+1", "+2", ">= +3"]


def rel_bin(d):
    return 0 if d <= -3 else 6 if d >= 3 else d + 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = load_dataset(a.data)
    meta = data["meta"]
    lay = Layout(kit_circuit(meta["basis"], meta["p"]))
    det_of = {(q, r): d for d, (q, r) in enumerate(lay.det_check)}
    nbr = {q: sorted({partner for _, _, partner in lay.cx[q]}) for q in DATA}
    assert all(len(v) == 6 for v in nbr.values())
    data_of_check = {}
    for q, checks in nbr.items():
        for c in checks:
            data_of_check.setdefault(c, set()).add(q)
    check_index = {c: (c if c < N else c - 2 * N) for c in (*X_ANC, *Z_ANC)}     # row of check_k
    det, slot, ck = data["detectors"], data["data_slot"], data["check_k"]
    kinds = {"X": X_ANC, "Z": Z_ANC}
    fired = {k: np.zeros(7) for k in kinds}
    total = {k: np.zeros(7) for k in kinds}
    base_f = {k: np.zeros(CYCLES + 1) for k in kinds}
    base_t = {k: np.zeros(CYCLES + 1) for k in kinds}
    kept = dropped = 0
    for shot in range(len(det)):
        lost = {DATA[i] for i in np.nonzero(slot[shot])[0]}
        lost_checks = {c for c in (*X_ANC, *Z_ANC) if ck[shot, check_index[c]].any()}
        for q in DATA:
            is_lost = q in lost
            if not is_lost and (INDEX[q] + shot) % 7:
                continue
            others = {o for c in nbr[q] for o in data_of_check[c]} - {q}
            if others & lost or set(nbr[q]) & lost_checks:
                dropped += is_lost
                continue
            kept += is_lost
            tau = (int(slot[shot, INDEX[q]]) - 1) // 6 if is_lost else None
            for c in nbr[q]:
                kind = "X" if c in X_ANC else "Z"
                for r in range(CYCLES + 1):
                    d = det_of.get((c, r))
                    if d is None:
                        continue
                    v = det[shot, d]
                    if is_lost:
                        b = rel_bin(r - tau)
                        fired[kind][b] += v
                        total[kind][b] += 1
                    else:
                        base_f[kind][r] += v
                        base_t[kind][r] += 1
    rows = []
    for b, label in enumerate(BINS):
        row = {"rounds_after_loss": label}
        for k in kinds:
            row[f"{k}_rate"] = float(fired[k][b] / total[k][b]) if total[k][b] else None
            row[f"{k}_detector_reads"] = int(total[k][b])
        rows.append(row)
    base = {k: float(base_f[k].sum() / base_t[k].sum()) for k in kinds}
    out = {"dataset": meta, "lost_atoms_kept": kept, "lost_atoms_dropped_by_isolation": dropped,
           "baseline_rate": base, "baseline_reads": {k: int(base_t[k].sum()) for k in kinds}, "rows": rows}
    md = [f"Lost data atoms kept: {kept} (dropped by isolation: {dropped}). Baseline (unlost atoms): "
          f"X checks {base['X']:.4f}, Z checks {base['Z']:.4f}.", "",
          "| round - loss round | X-check detectors: firing rate (reads) | Z-check detectors: firing rate (reads) |",
          "|---|---|---|"]
    for r in rows:
        cells = [f"{r[f'{k}_rate']:.4f} ({r[f'{k}_detector_reads']})" if r[f"{k}_rate"] is not None else "none" for k in kinds]
        md.append(f"| {r['rounds_after_loss']} | {cells[0]} | {cells[1]} |")
    with open(a.out + ".json", "w") as f:
        json.dump(out, f, indent=1)
    with open(a.out + ".md", "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
