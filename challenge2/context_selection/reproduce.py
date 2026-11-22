"""One command reproduces every number and figure of an experiment.

    python reproduce.py --config configs/offline.yaml          # full run
    python reproduce.py --config configs/smoke.yaml            # CI smoke run
    python reproduce.py --config configs/offline.yaml --only figures

Stages: ``evaluate`` (all lanes, answers, hypotheses), ``studies`` (noise sweep, edge
sweep, fault budget, compile statistics, roadmap projection), ``figures``. Each stage
reads what the previous ones wrote, and finished work is cached on disk, so an
interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

STAGES = ("evaluate", "studies", "figures")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/offline.yaml")
    ap.add_argument("--only", choices=STAGES, action="append", help="run only these stages (repeatable)")
    ap.add_argument("--quick", action="store_true", help="smaller studies (fewer questions and noise levels)")
    args = ap.parse_args(argv)

    from qctx.config import load_config

    cfg = load_config(args.config)
    stages = args.only or list(STAGES)
    t0 = time.perf_counter()
    if "evaluate" in stages:
        from qctx.harness import run_experiment

        run_experiment(cfg)
    if "studies" in stages:
        from qctx.studies import run_studies

        run_studies(cfg, quick=args.quick)
    if "figures" in stages:
        from qctx.figures import make_figures

        for path in make_figures(cfg):
            print(f"figure: {path}")
    print(f"reproduce finished in {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
