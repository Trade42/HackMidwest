"""Regenerate every physics figure and a summary of the numbers they carry.

    python physics/make_all.py

Writes ``figures/physics/{grotrian,rydberg_scaling,stark_map,pair_potential}.png``,
a caption ``.txt`` and a numbers ``.json`` next to each, and ``summary.json`` with the
headline values and the measured runtime of each script. Scripts run one after the
other so only one ARC calculation is active at a time.
"""

from __future__ import annotations

import json
import time

import grotrian
import pair_potential
import rydberg_scaling
import stark_map
from _style import OUT_DIR

SCRIPTS = (grotrian, rydberg_scaling, stark_map, pair_potential)


def main() -> dict:
    summary: dict = {"scripts": {}}
    t_all = time.time()
    for module in SCRIPTS:
        name = module.__name__
        print(f"\n=== {name} ===")
        t0 = time.time()
        result = module.main()
        result["runtime_s"] = round(time.time() - t0, 1)
        summary["scripts"][name] = result
    summary["total_runtime_s"] = round(time.time() - t_all, 1)

    try:
        import arc
        summary["arc_version"] = getattr(arc, "__version__", "unknown")
    except Exception:  # pragma: no cover
        summary["arc_version"] = "unknown"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")

    print("\nRuntime per script (seconds):")
    for name, result in summary["scripts"].items():
        print(f"  {name:<18}{result['runtime_s']:>8.1f}")
    print(f"  {'total':<18}{summary['total_runtime_s']:>8.1f}")
    print(f"Figures, captions and numbers are in {OUT_DIR}")
    return summary


if __name__ == "__main__":
    main()
