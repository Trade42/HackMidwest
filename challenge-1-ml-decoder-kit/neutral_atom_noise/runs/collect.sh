#!/usr/bin/env bash
# Rebuild every committed table, summary and plot of results/ from the run folders on this PC (luminaty).
# From challenge-1-ml-decoder-kit/:   bash neutral_atom_noise/runs/collect.sh
# summarize.py reads only the per-part result files of run.py; it decodes nothing.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
S=/home/sal/hackMidwest/orchestration/scratch/infleqtion-noise
PY="$S/venv/bin/python"
R=neutral_atom_noise/results
mkdir -p "$R"

# collect NAME RUN-FOLDER: summarize the folder's results (with its fit.json, if any) and copy the outputs as NAME_*
collect() {
  local name=$1 run=$2 tmp
  tmp=$(mktemp -d)
  local fit=()
  [ -f "$run/fit.json" ] && fit=(--fit "$run/fit.json")
  "$PY" -m neutral_atom_noise.summarize --results "$run/results" "${fit[@]}" --out "$tmp" > /dev/null
  for f in tables.md summary.json rates.csv per_shot.npz; do cp "$tmp/$f" "$R/${name}_$f"; done
  [ -f "$run/fit.json" ] && cp "$run/fit.json" "$R/${name}_fit.json"
  for png in "$tmp"/*.png; do [ -e "$png" ] && cp "$png" "$R/${name}_$(basename "$png")"; done
  rm -rf "$tmp"
  echo "$name: $(ls "$R" | /usr/bin/grep -c "^${name}_") files"
}

collect reference "$S/runs/reference"
collect X "$S/runs/X_seed4100"
collect X_loss "$S/runs/X_loss_seed6100"
[ -d "$S/runs/Z_seed4200/results" ] && collect Z "$S/runs/Z_seed4200"
# the plot of question 3 keeps its plain name for README.md and RESULTS.md
mv "$R/X_failure_vs_factor_X.png" "$R/failure_vs_factor_X.png"
