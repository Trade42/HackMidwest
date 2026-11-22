"""Write cascade_X_s1/SCORES.md on the laptop-artifacts branch from every eval/step<N>_<label>.json there.

  python -I -B scores_table.py <laptop-artifacts checkout>
"""
import glob
import json
import os
import re
import sys

root = sys.argv[1]
rows = {}
for path in glob.glob(os.path.join(root, "cascade_X_s1", "eval", "step*_*.json")):
    m = re.match(r"step(\d+)_(.+)\.json$", os.path.basename(path))
    numbers = json.load(open(path))
    rows.setdefault(int(m.group(1)), {})[m.group(2)] = numbers
labels = sorted({label for r in rows.values() for label in r},
                key=lambda s: (s.split("_p")[-1], not s.startswith("validation"), s))
out = ["# Cascade X s1 on the kit's Stim shots, failures per shot (95% Wilson interval)", "",
       "Made by tools/scores_table.py from eval/*.json. One row per kept snapshot, one column per shot file.", "",
       "| Step | " + " | ".join(labels) + " |", "| --- |" + " --- |" * len(labels)]
for step in sorted(rows):
    cells = []
    for label in labels:
        n = rows[step].get(label)
        cells.append("-" if n is None else f"{n['failure_fraction']:.2%} [{n['failure_interval'][0]:.2%}, "
                                           f"{n['failure_interval'][1]:.2%}] of {n['shots']}")
    out.append(f"| {step} | " + " | ".join(cells) + " |")
open(os.path.join(root, "cascade_X_s1", "SCORES.md"), "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print("\n".join(out))
