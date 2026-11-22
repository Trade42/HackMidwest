"""The combined L1 table over the loss rates, from results/L1_X_p0.004_s*.json (written by summarize.py).

usage (from challenge-1-ml-decoder-kit/):
    python -m loss.tables --out loss/results/L1_combined.md
"""
import argparse
import glob
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(glob.glob("loss/results/L1_X_p0.004_s*.json"), key=lambda f: float(f.split("_s")[-1][:-5]))
    lines = ["| s per atom per CX | lost data atoms per shot | check heralds per shot | blind: failures / shots (95% Wilson) "
             "| flag-aware: failures / shots (95% Wilson) | flag-aware, stop at first solution | paired shots: blind-only / aware-only "
             "| blind ÷ flag-aware (95% bootstrap) | unconverged: blind, aware |",
             "|---|---|---|---|---|---|---|---|---|"]
    for f in files:
        d = json.load(open(f))
        rows = {r["condition"]: r for r in d["rows"]}
        ds = d["dataset"]
        b, s1, st = rows.get("blind"), rows.get("starter"), rows.get("starter@0.5/stop1")

        def cell(r):
            if r is None:
                return "unmeasured"
            lo, hi = r["wilson95"]
            return f"{r['failures']} / {r['shots']} = {100 * r['fail_fraction']:.2f}% ({100 * lo:.2f}–{100 * hi:.2f})"

        paired = f"{s1['paired_shots']}: {s1['ref_only']} / {s1['cond_only']}" if s1 and "ref_only" in s1 else ""
        ratio = f"{s1['ratio_ref_over_cond']:.1f} ({s1['ratio95'][0]:.1f}–{s1['ratio95'][1]:.1f})" if s1 and "ref_only" in s1 else ""
        unc = f"{b['unconverged']} of {b['shots']}, {s1['unconverged']} of {s1['shots']}" if b and s1 else ""
        lines.append(f"| {ds['s']:g} | {ds['lost_data_atoms_per_shot']:.2f} | {ds['check_heralds_per_shot']:.2f} | {cell(b)} | "
                     f"{cell(s1)} | {cell(st)} | {paired} | {ratio} | {unc} |")
    with open(a.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
