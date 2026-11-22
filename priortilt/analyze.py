"""Paired comparison of relay-BP and PriorTilt on the same shots, from results/<stem>.<decoder>.cNNN.npz.

  python -I analyze.py <group label> <stem> [<stem> ...]      e.g.  analyze.py "p=0.004 X" train_X validation_X

Per stem and for the group: shots; failures per decoder; the 2 x 2 paired table; McNemar's exact test (two-sided
binomial test on the discordant pairs); the logical error per shot with a 95% Wilson interval; time per shot (mean,
median, p99); BP iterations (mean, median, p99); unconverged shots. Side-by-side process counts come from the
runner's job records.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest, norm

HERE = os.path.dirname(os.path.abspath(__file__))
DECODERS = ("relay", "priortilt")
RESULTS = os.path.join(HERE, "results")   # change with a first argument --results=<folder under this script's folder>


def wilson(k, n, conf=0.95):
    z = norm.ppf(0.5 + conf / 2)
    phat = k / n
    centre = (phat + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def load(stem, decoder):
    files = sorted(glob.glob(os.path.join(RESULTS, f"{stem}.{decoder}.c*.npz")))
    if not files:
        return None
    parts = [np.load(f) for f in files]
    shot = np.concatenate([q["shot"] for q in parts])
    order = np.argsort(shot)
    out = {k: np.concatenate([q[k] for q in parts])[order] for k in ("failed", "seconds", "iterations", "converged")}
    out["shot"] = shot[order]
    out["sha"] = {str(q["data_sha256"]) for q in parts}
    out["settings"] = {str(q["desc_settings"]) for q in parts}
    out["outs"] = {os.path.basename(f) for f in files}
    return out


def side_by_side(outs):
    vals = []
    for path in glob.glob(os.path.join(HERE, "runner_*.jsonl")):
        for line in open(path):
            r = json.loads(line)
            if r.get("out") in outs and r.get("exit") == 0:
                vals.append((r["side_by_side_at_start"], r["max_side_by_side"]))
    if not vals:
        return "n/a"
    lo = min(v[0] for v in vals)
    hi = max(v[1] for v in vals)
    return f"{lo}-{hi}"


def describe(label, runs):
    r, t = runs["relay"], runs["priortilt"]
    if r is None or t is None:
        return f"{label}: missing results for {'relay' if r is None else 'priortilt'}"
    if not np.array_equal(r["shot"], t["shot"]):
        common = np.intersect1d(r["shot"], t["shot"])
        rm, tm = np.isin(r["shot"], common), np.isin(t["shot"], common)
        r = {k: (v[rm] if isinstance(v, np.ndarray) else v) for k, v in r.items()}
        t = {k: (v[tm] if isinstance(v, np.ndarray) else v) for k, v in t.items()}
        note = f" (paired on the {len(common)} shots both have so far)"
    else:
        note = ""
    n = len(r["shot"])
    rf, tf = r["failed"], t["failed"]
    both, r_only, t_only, neither = int((rf & tf).sum()), int((rf & ~tf).sum()), int((~rf & tf).sum()), int((~rf & ~tf).sum())
    disc = r_only + t_only
    p_exact = binomtest(r_only, disc, 0.5).pvalue if disc else 1.0
    lines = [f"{label}: {n} shots{note}"]
    for name, d in (("relay-BP", r), ("PriorTilt", t)):
        k = int(d["failed"].sum())
        lo, hi = wilson(k, n)
        ms = d["seconds"] * 1e3
        it = d["iterations"]
        lines.append(f"  {name:9s} failures {k:4d}  logical error {k / n:.3%} [95% Wilson {lo:.3%}, {hi:.3%}]  "
                     f"time/shot mean {ms.mean():.1f} ms, median {np.median(ms):.1f}, p99 {np.percentile(ms, 99):.0f} ms  "
                     f"iterations mean {it.mean():.0f}, median {np.median(it):.0f}, p99 {np.percentile(it, 99):.0f}  "
                     f"unconverged {int((~d['converged']).sum())}  side-by-side processes {side_by_side(d['outs'])}")
    lines.append(f"  paired table (rows relay-BP, columns PriorTilt): both fail {both}, relay-BP only {r_only}, "
                 f"PriorTilt only {t_only}, both right {neither}")
    lines.append(f"  McNemar exact (two-sided binomial on {disc} discordant shots): p = {p_exact:.3g}")
    lines.append(f"  data sha256: {sorted(r['sha'] | t['sha'])}")
    return "\n".join(lines)


def main():
    global RESULTS
    argv = sys.argv[1:]
    if argv and argv[0].startswith("--results="):
        RESULTS = os.path.join(HERE, argv.pop(0).split("=", 1)[1])
    label, stems = argv[0], argv[1:]
    per = {s: {d: load(s, d) for d in DECODERS} for s in stems}
    for s in stems:
        print(describe(f"{label} / {s}", per[s]))
    if len(stems) > 1 and all(per[s][d] is not None for s in stems for d in DECODERS):
        merged = {}
        for d in DECODERS:
            merged[d] = {"shot": np.concatenate([per[s][d]["shot"] + 10**7 * i for i, s in enumerate(stems)]),
                         **{k: np.concatenate([per[s][d][k] for s in stems]) for k in ("failed", "seconds", "iterations", "converged")},
                         "sha": set().union(*(per[s][d]["sha"] for s in stems)),
                         "outs": set().union(*(per[s][d]["outs"] for s in stems))}
        print(describe(f"{label} / all files", merged))


if __name__ == "__main__":
    main()
