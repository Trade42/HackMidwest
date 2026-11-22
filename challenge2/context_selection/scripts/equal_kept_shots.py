"""Equal-accepted-shot comparison: bare QAOA against bus-Iceberg QAOA with matched kept shots.

Lanes 3b and 3c give both tiers the same raw shots, so the error-detected tier ends with
fewer accepted shots. Here bare QAOA gets R raw shots and the error-detected tier gets
about R / kept_share raw shots, so both end with about R accepted shots; the raw-shot
multiplier is the price. Same local Pauli noise as the config's physical backend, same
QUBOs and angles as the evaluation (first N evaluation questions). No language model.

    python scripts/equal_kept_shots.py --config configs/main.yaml
    python scripts/equal_kept_shots.py --config configs/offline.yaml --questions 30 --shots 4000

Writes results/<run>/studies/equal_kept_shots.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qctx.config import load_config  # noqa: E402
from qctx.studies import equal_kept_shots, load_instances  # noqa: E402


def format_table(result: dict) -> str:
    lines = [f"{'Tier':<48} {'AR':>7} {'P(opt)':>8} {'opt found':>9} {'kept':>6} {'raw shots':>9} {'kept shots':>10} {'x raw':>6}"]
    for row in result["tiers"].values():
        lines.append(f"{row['label']:<48} {row['mean_approx_ratio']:7.4f} {row['p_optimum_per_kept_shot']:8.4f} "
                     f"{row['optimum_in_sample_share']:9.2f} {row['mean_kept_share']:6.3f} {row['mean_raw_shots']:9.0f} "
                     f"{row['mean_kept_shots']:10.0f} {row['raw_shot_multiplier']:6.2f}")
    for name, d in result["paired_differences"].items():
        lines.append(f"  {name:<50} {d['mean_diff']:+.4f}  [{d['lo']:+.4f}, {d['hi']:+.4f}]")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/offline.yaml")
    ap.add_argument("--questions", type=int, default=30)
    ap.add_argument("--shots", type=int, default=4000, help="R: raw shots of bare QAOA")
    ap.add_argument("--out", default=None, help="default: <output>/studies/equal_kept_shots.json")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    t0 = time.perf_counter()
    instances = load_instances(cfg, args.questions, seed=int(cfg["seed"]))
    result = equal_kept_shots(cfg, instances, shots=args.shots, seed=int(cfg["seed"]))
    result["seconds"] = time.perf_counter() - t0
    out = Path(args.out) if args.out else Path(cfg["output"]) / "studies" / "equal_kept_shots.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(format_table(result))
    print(f"wrote {out} ({result['seconds']:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
