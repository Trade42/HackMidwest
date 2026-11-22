"""Apply the selection rules of RESULTS.md ("How the frozen decoder is chosen") to the validation results.

usage (from challenge-1-ml-decoder-kit/):
    python -m loss_tracker.choose --accuracy loss_tracker/results/accuracy_val.json \
        --variants 'VAL_PARTS/soft/*.npz' 'VAL_PARTS/variants/*.npz' --earlier-version order1_coupled2 \
        --record 'VAL_PARTS/explain_*/*.npz' --out loss_tracker/results/choice

Rule 1: the tracker version with the best mean log P(true round); versions within 0.005 nats of the best are a
        tie, and the one with the fewest sweeps is taken.
Rule 2: the decoder variant with the fewest failures; ties: fewer unconverged shots, then fewer mean iterations.
Record: the validation decode of the chosen version (soft posterior) against the earlier version's, paired.
Writes NAME.json, NAME.md and prints the two lines of runs/frozen.sh.
"""
import argparse
import json
import re

import numpy as np

from loss.summarize import load_parts, sign_test

TIE = 0.005


def version_args(name):
    """'order1_explain2' -> the arguments of track.py."""
    m = re.fullmatch(r"order([01])_(naive|drop|coupled|explain)(\d*)", name)
    order, kind, sweeps = int(m.group(1)), m.group(2), int(m.group(3) or 0)
    mode = "coupled" if kind == "explain" else kind
    return f"--mode {mode} --sweeps {sweeps} --order {order}" + (" --explain" if kind == "explain" else ""), sweeps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accuracy", required=True)
    ap.add_argument("--variants", required=True, nargs="+")
    ap.add_argument("--earlier-version", required=True, help="the tracker version the variants were decoded with")
    ap.add_argument("--record", nargs="*", default=[], help="parts of the chosen version's validation decode")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    acc = json.load(open(a.accuracy))["versions"]
    score = {name: v["all"]["mean_log_p_true"] for name, v in acc.items()}
    best = max(score.values())
    tied = [name for name in score if score[name] >= best - TIE]
    version = min(tied, key=lambda name: (version_args(name)[1], -score[name]))
    md = ["### Rule 1: the tracker version (4,000 validation shots)", "", "| version | mean log P(true round) | within 0.005 of the best | chosen |", "|---|---|---|---|"]
    md += [f"| {name} | {score[name]:.4f} | {'yes' if name in tied else ''} | {'**yes**' if name == version else ''} |"
           for name in sorted(score, key=score.get, reverse=True)]
    per = load_parts(a.variants)
    n = min(len(per[c]["failed"]) for c in per)
    rows = {c: (int(per[c]["failed"][:n].sum()), int((~per[c]["converged"][:n].astype(bool)).sum()), float(per[c]["iterations"][:n].mean()))
            for c in per}
    variant = min(rows, key=rows.get)
    md += ["", f"### Rule 2: the decoder variant (the first {n} validation shots, tracker version `{a.earlier_version}`)", "",
           "| variant | failures | unconverged | mean iterations | chosen |", "|---|---|---|---|---|"]
    md += [f"| {c} | {rows[c][0]} | {rows[c][1]} | {rows[c][2]:.0f} | {'**yes**' if c == variant else ''} |" for c in sorted(rows, key=rows.get)]
    out = {"tracker_version": version, "track_args": version_args(version)[0], "version_scores": score, "tied": tied,
           "variant": variant, "variant_rows": {c: {"failures": r[0], "unconverged": r[1], "mean_iterations": r[2]} for c, r in rows.items()},
           "validation_shots_decoded": n, "earlier_version": a.earlier_version}
    if a.record:
        rec = load_parts(a.record)
        md += ["", f"### Record: the chosen version `{version}` decoded on the same validation shots", "",
               "| variant | failures with the chosen version | with the earlier version | only the chosen fails / only the earlier fails | sign test |", "|---|---|---|---|---|"]
        out["record"] = {}
        for c in rec:
            if c not in per:
                continue
            m = min(n, len(rec[c]["failed"]))
            new, old = rec[c]["failed"][:m].astype(bool), per[c]["failed"][:m].astype(bool)
            b, cc = int((new & ~old).sum()), int((old & ~new).sum())
            p = sign_test(b, cc)
            out["record"][c] = {"shots": m, "chosen_version_failures": int(new.sum()), "earlier_version_failures": int(old.sum()),
                                "chosen_only": b, "earlier_only": cc, "sign_test_p": p, "lopsidedly_worse": bool(b > cc and p < 0.01)}
            md.append(f"| {c} | {int(new.sum())} / {m} | {int(old.sum())} / {m} | {b} / {cc} | p = {p:.2g} |")
    with open(a.out + ".json", "w") as f:
        json.dump(out, f, indent=1)
    with open(a.out + ".md", "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f'\nfrozen.sh:\nTRACK_NAME={version}\nTRACK_ARGS="{version_args(version)[0]}"\nVARIANT={variant}', flush=True)


if __name__ == "__main__":
    main()
