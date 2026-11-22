"""Seed-to-seed variability of the evaluate stage.

Reads results/{run}/summary.json for seeds 0, 1, 2 of each stack (seed 0 is the
committed run: results/main, results/offline; seeds 1 and 2 come from
configs/<stack>_seed<k>.yaml) and writes results/seed_variability.md:

* per lane: mean and range (min to max) across seeds of EM single, EM vote,
  approximation ratio and kept share;
* per hypothesis H1-H5: verdict and point estimate for each seed, and whether the
  verdict changes across seeds.

    python scripts/seed_summary.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STACKS = {"main": "Main stack (Qwen2.5-1.5B, bge-small)", "offline": "Offline stack (t5-base ONNX, TF-IDF)"}
METRICS = [("em_single", "EM single"), ("em_vote", "EM vote"), ("approx_ratio", "Approx. ratio"),
           ("kept_share", "Kept share")]


def run_dir(stack: str, seed: int) -> Path:
    return ROOT / "results" / (stack if seed == 0 else f"{stack}_seed{seed}")


def load(stack: str, seeds: list[int]) -> dict[int, dict]:
    out = {}
    for s in seeds:
        p = run_dir(stack, s) / "summary.json"
        if p.exists():
            out[s] = json.loads(p.read_text(encoding="utf-8"))
    return out


def fmt(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def cell(vals: list[float]) -> str:
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return "-"
    m, lo, hi = sum(vals) / len(vals), min(vals), max(vals)
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def point(h: dict) -> str:
    """Point estimate of a hypothesis, with its bootstrap CI when present."""
    if h.get("mean_diff") is None:
        return "n/a"
    s = f"{h['mean_diff']:+.4f} [{fmt(h.get('lo'), 4)}, {fmt(h.get('hi'), 4)}]"
    if h["id"] == "H2" and h.get("hit_rate") is not None:
        s = f"hit {h['hit_rate']:.3f}; {s}"
    if h["id"] == "H4" and h.get("accuracy_gain") is not None:
        s += f" (EM gain {h['accuracy_gain']:+.4f})"
    return s


def stack_section(stack: str, runs: dict[int, dict]) -> tuple[list[str], list[str]]:
    seeds = sorted(runs)
    lines = [f"## {STACKS.get(stack, stack)}", "",
             f"Seeds: {', '.join(map(str, seeds))} ({runs[seeds[0]]['questions']} questions each). "
             "Cells: mean across seeds [min, max].", ""]
    lanes = list(runs[seeds[0]]["lanes"])
    lines += ["| Lane | " + " | ".join(n for _, n in METRICS) + " | EM single per seed |",
              "|---|" + "---|" * (len(METRICS) + 1)]
    for lane in lanes:
        row = [f"{lane} ({runs[seeds[0]]['lanes'][lane]['label']})"]
        for key, _ in METRICS:
            row.append(cell([runs[s]["lanes"].get(lane, {}).get(key, {}).get("mean", float("nan")) for s in seeds]))
        row.append(" / ".join(fmt(runs[s]["lanes"].get(lane, {}).get("em_single", {}).get("mean"), 3) for s in seeds))
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "### Hypotheses", "",
              "| Hypothesis | " + " | ".join(f"seed {s}" for s in seeds) + " | Verdict stable? |",
              "|---|" + "---|" * (len(seeds) + 1)]
    flips = []
    ids = [h["id"] for h in runs[seeds[0]]["hypotheses"]]
    for hid in ids:
        hs = {s: next((h for h in runs[s]["hypotheses"] if h["id"] == hid), None) for s in seeds}
        verdicts = {s: (h or {}).get("verdict", "missing") for s, h in hs.items()}
        stable = len(set(verdicts.values())) == 1
        if not stable:
            flips.append(f"{stack} {hid}: " + ", ".join(f"seed {s} {v}" for s, v in verdicts.items()))
        cells = [f"**{verdicts[s]}**: {point(hs[s]) if hs[s] else 'n/a'}" for s in seeds]
        lines.append(f"| {hid} ({hs[seeds[0]].get('quantity', '')}) | " + " | ".join(cells)
                     + f" | {'yes' if stable else '**NO**'} |")
    lines.append("")
    return lines, flips


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default=str(ROOT / "results" / "seed_variability.md"))
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]

    body, flips, missing = [], [], []
    for stack in STACKS:
        runs = load(stack, seeds)
        missing += [f"{stack} seed {s}" for s in seeds if s not in runs]
        if not runs:
            continue
        sec, f = stack_section(stack, runs)
        body += sec
        flips += f

    head = ["# Seed-to-seed variability", "",
            "The `seed` in the config drives QAOA angle tuning (random restarts), the shot sampling of "
            "every sampled lane (ideal, noisy, random control), simulated annealing, and the 64-shot "
            "subsampling. Scores, QUBOs, the reader (greedy decoding) and the bootstrap seed are fixed, "
            "so deterministic lanes (1a, 1b, 2a, 2b) should not move at all.", "",
            "Generated by `python scripts/seed_summary.py` from `results/{main,offline}[_seed1,_seed2]/summary.json`.", ""]
    if flips:
        head += ["**Verdict flips across seeds:**", ""] + [f"- {f}" for f in flips] + [""]
    else:
        head += ["**No hypothesis verdict changes across seeds** on either stack.", ""]
    if missing:
        head += ["Missing runs: " + ", ".join(missing), ""]
    Path(args.out).write_text("\n".join(head + body), encoding="utf-8")
    print(f"wrote {args.out}")
    for f in flips:
        print("flip:", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
