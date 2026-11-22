"""Where does the tracker lose against the ORACLE? Per-shot diagnostics on decoded shots (scoring only).

usage (from challenge-1-ml-decoder-kit/):
    python -m loss_tracker.diagnose --data VAL.npz --post POST.npz --parts 'PARTS/*.npz' --tracker tracker \
        --out loss_tracker/results/diagnose_val

For each decoded shot: the smallest and the mean posterior that the tracker gives to the TRUE loss round of the
shot's lost data atoms, and the number of lost atoms. The shots are grouped by which of `exact`, the tracker and
the ORACLE `oracle-round` failed. It reads the probes, to score; it changes nothing.
"""
import argparse
import json

import numpy as np

from loss.lossmodel import load_dataset
from loss.summarize import load_parts
from loss_tracker.accuracy import load_post
from loss_tracker.tracker import softmax, to_rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--parts", required=True, nargs="+")
    ap.add_argument("--tracker", default="tracker")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = load_dataset(a.data)
    f = load_post(a.post)
    per = load_parts(a.parts)
    n = min(len(per[c]["failed"]) for c in ("exact", a.tracker, "oracle-round"))
    rounds = to_rounds(softmax(f["log_prior"][None, :] + f["loglik"].astype(np.float64)))
    tc = (data["data_slot"][f["shot"], f["atom"]].astype(np.int64) - 1) // 6
    p_true = rounds[np.arange(len(tc)), tc]
    near = np.abs(rounds.argmax(axis=1) - tc) > 1
    mins, means, lost, far = np.ones(n), np.ones(n), np.zeros(n, int), np.zeros(n, int)
    for k in range(n):
        m = f["shot"] == k
        if m.any():
            mins[k], means[k], lost[k], far[k] = p_true[m].min(), p_true[m].mean(), m.sum(), near[m].sum()
    fe, ft, fo = (per[c]["failed"][:n].astype(bool) for c in ("exact", a.tracker, "oracle-round"))
    groups = {"all shots": np.ones(n, bool), "no decoder fails": ~fe & ~ft & ~fo,
              "exact fails, the tracker and the ORACLE do not (the tracker's gain)": fe & ~ft & ~fo,
              "exact and the tracker fail, the ORACLE does not (the gain left)": fe & ft & ~fo,
              "the tracker fails, exact and the ORACLE do not": ft & ~fe & ~fo,
              "all three fail": fe & ft & fo, "the ORACLE fails, the tracker does not": fo & ~ft}
    rows = []
    md = [f"{n} decoded shots of {a.data.split('/')[-1]}; tracker {f['meta'].get('mode')}, sweeps {f['meta'].get('sweeps')}, "
          f"explain {f['meta'].get('explain')}.", "",
          "| shots where | shots | lost data atoms per shot | mean P(true round) | smallest P(true round) in the shot, median | atoms more than one round off, per shot |",
          "|---|---|---|---|---|---|"]
    for g, m in groups.items():
        if not m.any():
            md.append(f"| {g} | 0 | | | | |")
            continue
        r = {"group": g, "shots": int(m.sum()), "lost_per_shot": float(lost[m].mean()), "mean_p_true": float(means[m].mean()),
             "median_min_p_true": float(np.median(mins[m])), "far_atoms_per_shot": float(far[m].mean())}
        rows.append(r)
        md.append(f"| {g} | {r['shots']} | {r['lost_per_shot']:.1f} | {r['mean_p_true']:.3f} | {r['median_min_p_true']:.3f} | {r['far_atoms_per_shot']:.2f} |")
    with open(a.out + ".json", "w") as fh:
        json.dump({"shots": n, "rows": rows}, fh, indent=1)
    with open(a.out + ".md", "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
