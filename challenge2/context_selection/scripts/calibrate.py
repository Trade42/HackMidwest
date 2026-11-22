"""Pick hop, tau and theta on the calibration questions (never on the evaluation set).

For each setting on a small grid, solve every calibration QUBO exactly and record how
often both gold paragraphs are selected and how many paragraphs are read. The
recommended setting maximises "both gold selected" among settings whose mean selection
size stays within the token budget.

    python scripts/calibrate.py --config configs/offline.yaml
    python scripts/calibrate.py --config configs/main.yaml --hops 0,1,2

The scorer comes from the config; ``--hops`` re-scores the calibration questions with
each second-hop weight. Paste the recommended values into the config's ``scores.scorer``
and ``qubo`` sections.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qctx.config import load_config  # noqa: E402
from qctx.data import read_questions  # noqa: E402
from qctx.scoring import make_scorer, paragraph_text  # noqa: E402
from qsel import build_qubo, optimum  # noqa: E402
from qsel.bits import selected  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/offline.yaml")
    ap.add_argument("--hops", default="0,1,2", help="comma-separated second-hop weights to try")
    ap.add_argument("--max-size", type=float, default=3.0, help="token budget: mean paragraphs selected")
    ap.add_argument("--min-edges", type=float, default=0.0, help="mean redundant edges per question")
    ap.add_argument("--out", default="results/calibration.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    questions = read_questions(cfg["data"]["calibration"])
    penalty_scale = float(cfg["qubo"].get("penalty_scale", 1.5))
    normalize = cfg["qubo"].get("normalize", "zscore")
    thetas = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5]
    quantiles = (0.5, 0.6, 0.7, 0.8, 0.9)

    rows = []
    for hop in [float(h) for h in args.hops.split(",")]:
        scorer = make_scorer({**cfg["scores"]["scorer"], "hop": hop})
        scores = {q.qid: scorer.score(q.question, [paragraph_text(p) for p in q.paragraphs]) for q in questions}
        pooled = np.concatenate([s[np.triu_indices(len(r), 1)] for r, s in scores.values()])
        top3 = float(np.mean([set(q.gold) <= set(np.argsort(-scores[q.qid][0])[:3].tolist()) for q in questions]))
        print(f"scorer {scorer.name}: top-3 by relevance selects both gold paragraphs in {top3:.3f} of questions")
        for qt in quantiles:
            tau = float(np.quantile(pooled, qt))
            for theta in thetas:
                both, recall, size, edges = [], [], [], []
                for q in questions:
                    r, s = scores[q.qid]
                    qubo = build_qubo(r, s, tau=tau, theta=theta, penalty_scale=penalty_scale, normalize=normalize)
                    picked = set(selected(optimum(qubo)[0]))
                    gold = set(q.gold)
                    both.append(float(gold <= picked))
                    recall.append(len(gold & picked) / len(gold))
                    size.append(len(picked))
                    edges.append(qubo.num_edges)
                rows.append({"hop": hop, "scorer": scorer.name, "tau_quantile": qt, "tau": tau, "theta": theta,
                             "both_gold": float(np.mean(both)), "gold_recall": float(np.mean(recall)),
                             "mean_size": float(np.mean(size)), "mean_edges": float(np.mean(edges)),
                             "top3_both_gold": top3})

    ok = [r for r in rows if r["mean_size"] <= args.max_size and r["mean_edges"] >= args.min_edges]
    best = max(ok or rows, key=lambda r: (r["both_gold"], r["gold_recall"], -r["mean_size"], -r["mean_edges"]))
    print(f"\n{len(questions)} calibration questions. Settings within the budget (mean size <= {args.max_size}):")
    print(f"{'hop':>4} {'tau':>12} {'theta':>6} {'both gold':>10} {'recall':>7} {'size':>5} {'edges':>6}")
    for r in sorted(ok, key=lambda r: -r["both_gold"])[:12]:
        mark = "  <-- chosen" if r is best else ""
        print(f"{r['hop']:4g} q{int(100 * r['tau_quantile'])}={r['tau']:.4f} {r['theta']:6.2f} {r['both_gold']:10.3f} "
              f"{r['gold_recall']:7.3f} {r['mean_size']:5.2f} {r['mean_edges']:6.1f}{mark}")
    print(f"\nrecommended: hop={best['hop']:g} tau={best['tau']:.4f} theta={best['theta']}  "
          f"(both gold {best['both_gold']:.3f}, mean size {best['mean_size']:.2f}, mean edges {best['mean_edges']:.1f}; "
          f"top-3 baseline {best['top3_both_gold']:.3f})")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"chosen": best, "grid": rows, "constraints": vars(args)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
