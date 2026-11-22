"""Step 3 tables: every condition on the same shots, paired comparisons, and the share of the oracle's gain.

usage (from challenge-1-ml-decoder-kit/):
    python -m loss_tracker.summarize --parts 'PARTS/*/*.npz' --tracker tracker --out loss_tracker/results/NAME \
        [--order starter,exact,tracker,oracle-round] [--title TEXT]

Uses loss-gates' summary code (Wilson intervals, exact sign test, bootstrap of the failure ratio). Added here:
the share of the oracle's gain that a condition recovers,
    (failures of `exact` - failures of the condition) / (failures of `exact` - failures of `oracle-round`),
with a 95% bootstrap interval over shots (2,000 resamples, seed 0), and the same against `starter`.
All comparisons are on the common prefix of shots that every listed condition has decoded.

--cascade 'A>B' adds a row computed from the per-shot results, with no new decoding: condition A, and where A did
not converge, condition B (B's output is taken whether or not B converged; its iterations are added to A's).
The rule reads only the decoders' convergence flags.
"""
import argparse
import json

import numpy as np

from loss.summarize import load_parts, markdown, summarize

ORACLE = "oracle-round"


def gain_share(f_ref, f_cond, f_oracle, reps=2000, seed=0):
    """(share, lo, hi): (ref - cond) / (ref - oracle) in failures, with a bootstrap interval over shots."""
    def share(a, b, c):
        return (a - b) / (a - c) if a != c else float("nan")
    rng = np.random.default_rng(seed)
    n = len(f_ref)
    idx = rng.integers(0, n, (reps, n))
    a, b, c = (f[idx].sum(axis=1).astype(float) for f in (f_ref, f_cond, f_oracle))
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = (a - b) / (a - c)
    boot = boot[np.isfinite(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5]) if len(boot) else (float("nan"), float("nan"))
    return share(float(f_ref.sum()), float(f_cond.sum()), float(f_oracle.sum())), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True, nargs="+")
    ap.add_argument("--order", help="comma-separated conditions (default: all found)")
    ap.add_argument("--references", default="exact,starter", help="references of the paired tables and of the gain share")
    ap.add_argument("--title", default="")
    ap.add_argument("--cascade", action="append", default=[], help="'A>B': A, and B where A did not converge")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    per = load_parts(a.parts)
    order = a.order.split(",") if a.order else list(per)
    n = min(len(per[c]["failed"]) for c in order)
    per = {c: {k: v[:n] for k, v in per[c].items()} for c in order}          # the common prefix of shots
    for rule in a.cascade:
        first, second = rule.split(">")
        use = ~per[first]["converged"].astype(bool)                           # shots where the first did not converge
        per[rule] = {k: np.where(use, per[second][k], per[first][k]) for k in per[first]}
        per[rule]["iterations"] = per[first]["iterations"] + np.where(use, per[second]["iterations"], 0)
        for k in ("decode_ms", "build_ms", "prior_ms"):
            per[rule][k] = per[first][k] + np.where(use, per[second][k], 0)
        order.append(rule)
    out = {"shots": n, "conditions": order, "tables": {}, "gain_share": []}
    md = [f"## {a.title}" if a.title else "", f"Every condition on the same {n} shots.", ""]
    for ref in a.references.split(","):
        if ref not in per:
            continue
        rows = summarize(per, ref, order)
        out["tables"][ref] = rows
        md += [markdown(rows, ref, f"reference {ref}"), ""]
    if ORACLE in per:
        md += ["### The share of the oracle's gain", "",
               "| condition | reference | failures: reference, condition, ORACLE | share of the gain recovered (95% bootstrap) |", "|---|---|---|---|"]
        f_or = per[ORACLE]["failed"].astype(bool)
        for ref in a.references.split(","):
            if ref not in per:
                continue
            f_ref = per[ref]["failed"].astype(bool)
            for c in order:
                if c in (ref, ORACLE) or c in a.references.split(","):
                    continue
                f_c = per[c]["failed"].astype(bool)
                sh, lo, hi = gain_share(f_ref, f_c, f_or)
                out["gain_share"].append({"condition": c, "reference": ref, "ref_failures": int(f_ref.sum()), "cond_failures": int(f_c.sum()),
                                          "oracle_failures": int(f_or.sum()), "share": sh, "share95": [lo, hi]})
                md.append(f"| {c} | {ref} | {int(f_ref.sum())}, {int(f_c.sum())}, {int(f_or.sum())} | {sh:.2f} ({lo:.2f}–{hi:.2f}) |")
    with open(a.out + ".json", "w") as f:
        json.dump(out, f, indent=1)
    with open(a.out + ".md", "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
