"""Step 8b: selector candidates from three decodes of the same shots, R (relay-BP), P1 (PriorTilt, seed 1 + chunk)
and P2 (PriorTilt, seed 1001 + chunk).

  python -I analyze_8b.py <results folder> <stem> [<stem> ...]

For each stem, merges results/<stem>.{relay,priortilt,priortilt_s1001}.cNNN.npz into selector_<stem>.npz (one row per
shot, in file order) and prints, per stem's basis and pooled over all stems:
  a. the paired table P1 against P2, and the eight right/wrong patterns of (R, P1, P2);
  b. how often the three predictions are not all equal, and how many failing shots of each decoder fall there;
  c. three selection rules (the truth is used only to score them):
     lowest weight   the candidate with the smallest sum of log((1 - prior) / prior) over its faults
     majority        the 12-bit prediction at least two candidates share; if all three differ, the lowest weight
     P1 unless       P1's answer, unless P1 and P2 differ: then the lowest weight of the three
     and, as an extra variant, the lowest weight among converged candidates (all three if none converged).
  Time per shot of a rule = the sum of the decoders it runs (P1-unless runs R only when P1 and P2 differ).
"""
import glob
import os
import sys

import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = {"R": "relay", "P1": "priortilt", "P2": "priortilt_s1001"}
FIELDS = ("predicted", "truth", "converged", "iterations", "n_faults", "weight", "seconds", "failed")


def merge(results, stem):
    out = {}
    for name, label in RUNS.items():
        files = sorted(glob.glob(os.path.join(results, f"{stem}.{label}.c*.npz")))
        parts = [np.load(f) for f in files]
        shot = np.concatenate([q["shot"] for q in parts])
        order = np.argsort(shot)
        if not np.array_equal(shot[order], np.arange(len(shot))):
            raise SystemExit(f"{stem} {label}: shots are not complete ({len(shot)} found)")
        for k in FIELDS:
            out[f"{k}_{name}"] = np.concatenate([q[k] for q in parts])[order]
        out[f"sha_{name}"] = sorted({str(q["data_sha256"]) for q in parts})
        out[f"settings_{name}"] = sorted({str(q["desc_settings"]) for q in parts})
    for name in ("P1", "P2"):
        if not np.array_equal(out["truth_R"], out[f"truth_{name}"]):
            raise SystemExit(f"{stem}: truth differs between runs")
    if len({tuple(out[f"sha_{n}"]) for n in RUNS}) != 1:
        raise SystemExit(f"{stem}: the runs decoded different files")
    merged = {"truth": out["truth_R"].astype(np.uint8), "data_sha256": out["sha_R"][0]}
    for name in RUNS:
        for k in FIELDS:
            if k != "truth":
                merged[f"{k}_{name}"] = out[f"{k}_{name}"]
        merged[f"settings_{name}"] = out[f"settings_{name}"][0]
    np.savez_compressed(os.path.join(results, f"selector_{stem}.npz"), **merged)
    return merged


def rules(m):
    """Chosen 12-bit predictions of each rule, and the per-shot time each rule costs."""
    names = list(RUNS)
    pred = np.stack([m[f"predicted_{n}"] for n in names])           # [3, shots, 12]
    weight = np.stack([m[f"weight_{n}"] for n in names])            # [3, shots]
    conv = np.stack([m[f"converged_{n}"] for n in names])
    sec = np.stack([m[f"seconds_{n}"] for n in names])
    shots = pred.shape[1]
    rows = np.arange(shots)
    lowest = weight.argmin(axis=0)
    w_conv = np.where(conv, weight, np.inf)
    lowest_conv = np.where(conv.any(axis=0), w_conv.argmin(axis=0), lowest)
    eq = lambda a, b: (pred[a] == pred[b]).all(axis=1)
    r_p1, r_p2, p1_p2 = eq(0, 1), eq(0, 2), eq(1, 2)
    majority = np.where(p1_p2 | r_p1, 1, np.where(r_p2, 0, lowest))
    p1_unless = np.where(p1_p2, 1, lowest)
    out = {
        "R alone": (pred[0], sec[0]),
        "P1 alone": (pred[1], sec[1]),
        "P2 alone": (pred[2], sec[2]),
        "lowest weight of R, P1, P2": (pred[lowest, rows], sec.sum(axis=0)),
        "majority (ties: lowest weight)": (pred[majority, rows], sec.sum(axis=0)),
        "P1 unless P1 != P2, then lowest weight": (pred[p1_unless, rows], sec[1] + sec[2] + np.where(p1_p2, 0, sec[0])),
        "extra: lowest weight among converged": (pred[lowest_conv, rows], sec.sum(axis=0)),
    }
    return out, (r_p1 & p1_p2)


def report(label, m):
    truth = m["truth"]
    wrong = {n: (m[f"predicted_{n}"] != truth).any(axis=1) for n in RUNS}
    n = len(truth)
    print(f"{label}: {n} shots")
    a, b = wrong["P1"], wrong["P2"]
    disc = int((a & ~b).sum() + (~a & b).sum())
    pv = binomtest(int((a & ~b).sum()), disc, 0.5).pvalue if disc else 1.0
    print(f"  a. P1 against P2: both fail {int((a & b).sum())}, P1 only {int((a & ~b).sum())}, P2 only {int((~a & b).sum())}, "
          f"both right {int((~a & ~b).sum())} (McNemar exact p = {pv:.3g})")
    pats = []
    for r in (False, True):
        for p1 in (False, True):
            for p2 in (False, True):
                k = int(((wrong["R"] == r) & (wrong["P1"] == p1) & (wrong["P2"] == p2)).sum())
                pats.append(f"R {'x' if r else 'ok'} P1 {'x' if p1 else 'ok'} P2 {'x' if p2 else 'ok'}: {k}")
    print("     patterns (ok = right, x = wrong): " + "; ".join(pats))
    chosen, all_equal = rules(m)
    differ = ~all_equal
    print(f"  b. predictions not all equal: {int(differ.sum())} shots ({differ.mean():.3%}); failing shots inside that set: "
          + ", ".join(f"{k} {int((wrong[k] & differ).sum())} of {int(wrong[k].sum())}" for k in RUNS))
    print("  c. rule: failures, logical error, mean time per shot (sum of the decoders it runs)")
    for name, (pred, sec) in chosen.items():
        f = int((pred != truth).any(axis=1).sum())
        print(f"     {name:42s} {f:4d}  {f / n:.3%}  {sec.mean() * 1e3:6.1f} ms")
    print(f"  unconverged: " + ", ".join(f"{k} {int((~m[f'converged_{k}']).sum())}" for k in RUNS)
          + f" | data sha256 {m['data_sha256']}")


def main():
    results = os.path.join(HERE, sys.argv[1])
    stems = sys.argv[2:]
    merged = {s: merge(results, s) for s in stems}
    groups = {}
    for s in stems:
        groups.setdefault(s.split("_")[-1] if s.endswith(("_X", "_Z")) else s, []).append(s)
    for key, members in groups.items():
        cat = {k: np.concatenate([merged[s][k] for s in members]) for k in merged[members[0]]
               if isinstance(merged[members[0]][k], np.ndarray)}
        cat["data_sha256"] = ", ".join(merged[s]["data_sha256"] for s in members)
        report(f"basis {key} ({' + '.join(members)})", cat)
    if len(groups) > 1:
        cat = {k: np.concatenate([merged[s][k] for s in stems]) for k in merged[stems[0]] if isinstance(merged[stems[0]][k], np.ndarray)}
        cat["data_sha256"] = "all"
        report(f"pooled ({' + '.join(stems)})", cat)


if __name__ == "__main__":
    main()
