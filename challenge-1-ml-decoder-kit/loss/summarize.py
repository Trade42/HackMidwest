"""Combine decode parts into one table per dataset, with Wilson intervals and paired comparisons.

usage (from challenge-1-ml-decoder-kit/):
    python -m loss.summarize --parts 'PARTS/*.npz' --reference blind --out loss/results/NAME

writes NAME.json and NAME.md. Paired comparisons are against the reference condition on the same shots:
ref_only = shots only the reference fails, cond_only = shots only the condition fails, the exact two-sided sign
test on those, and the failure ratio (reference failures / condition failures) with a 95% bootstrap interval
over shots (2,000 resamples, seed 0).
"""
import argparse
import glob
import json
from math import comb, sqrt

import numpy as np

CYCLES = 12


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def per_cycle(P):
    return 1 - (1 - P) ** (1 / CYCLES)


def sign_test(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def bootstrap_ratio(f_ref, f_cond, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(f_ref)
    idx = rng.integers(0, n, (reps, n))
    a, b = f_ref[idx].sum(axis=1).astype(float), f_cond[idx].sum(axis=1).astype(float)
    r = np.divide(a, b, out=np.full(reps, np.inf), where=b > 0)
    lo, hi = np.percentile(r, [2.5, 97.5], method="nearest")     # nearest: the ratios may include inf
    return float(lo), float(hi)


def load_parts(patterns):
    """{condition: {field: array over shots 0..n-1}} from every part file matching the globs.

    A condition may be decoded on a longer prefix of the shots than another; each must cover 0..n-1 exactly once.
    """
    files = sorted({f for pat in patterns for f in glob.glob(pat)})
    assert files, f"no parts match {patterns}"
    fields = ("failed", "converged", "iterations", "wrong_mask", "prior_ms", "build_ms", "decode_ms")
    per = {}
    for f in files:
        with np.load(f) as z:
            d = {k: z[k] for k in z.files}          # decompress each array once
        for i, c in enumerate(str(x) for x in d["conditions"]):
            for k, shot in enumerate(d["shots"]):
                assert int(shot) not in per.setdefault(c, {}), f"shot {shot} of {c} decoded twice"
                per[c][int(shot)] = tuple(d[fld][k, i] for fld in fields)
    out = {}
    for c, rows in per.items():
        shots = sorted(rows)
        assert shots == list(range(len(shots))), f"{c}: shots are not a prefix 0..n-1"
        out[c] = {fld: np.array([rows[s][j] for s in shots]) for j, fld in enumerate(fields)}
    return out


def summarize(per, reference, order=None):
    conds = order or list(per)
    rows = []
    for c in conds:
        r = per[c]
        f = r["failed"].astype(bool)
        n, k = len(f), int(f.sum())
        lo, hi = wilson(k, n)
        unconv = ~r["converged"].astype(bool)
        row = {"condition": c, "shots": n, "failures": k, "fail_fraction": k / n, "wilson95": [lo, hi],
               "per_cycle": per_cycle(k / n), "per_cycle95": [per_cycle(lo), per_cycle(hi)],
               "unconverged": int(unconv.sum()), "unconverged_but_right": int((unconv & ~f).sum()),
               "mean_iterations": float(r["iterations"].mean()),
               "mean_decode_ms": float(r["decode_ms"].mean()), "median_decode_ms": float(np.median(r["decode_ms"])),
               "mean_build_ms": float(r["build_ms"].mean()), "mean_prior_ms": float(r["prior_ms"].mean())}
        if reference in per and c != reference:
            m = min(n, len(per[reference]["failed"]))         # paired on the common prefix
            fr, fc = per[reference]["failed"][:m].astype(bool), f[:m]
            b, cc = int((fr & ~fc).sum()), int((fc & ~fr).sum())
            row |= {"paired_shots": m, "ref_failures": int(fr.sum()), "cond_failures": int(fc.sum()),
                    "ref_only": b, "cond_only": cc, "sign_test_p": sign_test(b, cc),
                    "ratio_ref_over_cond": (fr.sum() / fc.sum()) if fc.sum() else float("inf"),
                    "ratio95": bootstrap_ratio(fr, fc)}
        rows.append(row)
    return rows


def markdown(rows, reference, title):
    out = [f"### {title}", "",
           "| condition | failures / shots | fail fraction (95% Wilson) | per cycle | unconverged (of them right) | mean iterations |"
           f" paired shots: {reference} fails, cond fails | {reference}-only / cond-only (sign test) | {reference} ÷ cond (95% bootstrap) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lo, hi = r["wilson95"]
        if "ref_only" in r:
            paired = f"{r['paired_shots']}: {r['ref_failures']}, {r['cond_failures']}"
            disc = f"{r['ref_only']} / {r['cond_only']} (p={r['sign_test_p']:.2g})"
            ratio = f"{r['ratio_ref_over_cond']:.2f} ({r['ratio95'][0]:.2f}–{r['ratio95'][1]:.2f})"
        else:
            paired, disc, ratio = ("reference" if r["condition"] == reference else ""), "", ""
        out.append(f"| {r['condition']} | {r['failures']} / {r['shots']} | {r['fail_fraction']:.4f} ({lo:.4f}–{hi:.4f}) | "
                   f"{r['per_cycle']:.2e} | {r['unconverged']} ({r['unconverged_but_right']}) | {r['mean_iterations']:.0f} | "
                   f"{paired} | {disc} | {ratio} |")
    return "\n".join(out) + "\n"


def dataset_facts(path):
    """Meta, sha256 and loss counts of a dataset; cached beside it (the dataset never changes once written)."""
    import hashlib
    import os
    from loss.lossmodel import load_dataset
    cache = path + ".facts.json"
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(path):
        return json.load(open(cache))
    data = load_dataset(path)
    lost = (data["data_slot"] > 0).sum(axis=1)
    facts = {"file": path, "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(), **data["meta"],
             "lost_data_atoms_per_shot": float(lost.mean()),
             "check_heralds_per_shot": float(data["heralds"][:, :12 * 144].sum(axis=1).mean())}
    with open(cache, "w") as f:
        json.dump(facts, f)
    return facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True, nargs="+", help="globs of part files")
    ap.add_argument("--reference", default="blind")
    ap.add_argument("--order", help="comma-separated conditions, in table order")
    ap.add_argument("--title", default="")
    ap.add_argument("--data", help="the dataset file: its meta, sha256 and loss counts go beside the table")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    per = load_parts(a.parts)
    rows = summarize(per, a.reference, a.order.split(",") if a.order else None)
    with open(a.out + ".json", "w") as f:
        json.dump({"dataset": dataset_facts(a.data) if a.data else None, "reference": a.reference, "rows": rows}, f, indent=1)
    md = markdown(rows, a.reference, a.title or a.out)
    with open(a.out + ".md", "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
