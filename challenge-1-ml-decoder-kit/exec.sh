#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ "$#" -ne 2 ]; then
  echo "usage: MIDQ_PY=/path/to/python ./exec.sh {0.003|0.004|0.005} {mac|icrn|delta}" >&2
  exit 2
fi
p=$1
site=$2
case "$p" in
  0.003|0.004|0.005) ;;
  *) echo "unsupported physical error rate: $p" >&2; exit 2 ;;
esac
case "$site" in
  mac|icrn|delta) ;;
  *) echo "unsupported site: $site" >&2; exit 2 ;;
esac

MIDQ_PY=${MIDQ_PY:-.venv312/bin/python}
if [ ! -x "$MIDQ_PY" ]; then
  echo "Python environment missing or not executable: $MIDQ_PY" >&2
  exit 1
fi

out="results/$site"
options="$out/eval_options_p$p.json"
for path in "$out/eval_p$p" "$out/router_eval_p$p" "$out/gain_p$p" "$options"; do
  if [ -e "$path" ]; then
    echo "$path already exists; move the previous output aside before rerunning" >&2
    exit 1
  fi
done

# Check both frozen models before starting the long baseline evaluation.
"$MIDQ_PY" - "$p" <<'PY'
import sys
from circuit import GROSS_CIRCUITS, dem_matrices, load_circuit
from evaluation.adapters import Problem
from evaluation.router import load_frozen_router

p = float(sys.argv[1])
for basis in "XZ":
    circuit = load_circuit(GROSS_CIRCUITS[basis], p)
    H, L, priors = dem_matrices(circuit, basis)
    path = f"results/router_p{p:.3f}/model_{basis}.json"
    load_frozen_router(path, Problem(basis, H, L, priors, p, circuit))
    print(f"{basis}: frozen router is compatible", flush=True)
PY

PYTHON="$MIDQ_PY" OUT="$out" ./reproduce_eval.sh evaluate "$p"

printf '{"router_models":{"X":"results/router_p%s/model_X.json","Z":"results/router_p%s/model_Z.json"}}\n' \
  "$p" "$p" > "$options"

"$MIDQ_PY" -m evaluation.evaluate \
  --test-x "data/p$p/test_X.npz" --test-z "data/p$p/test_Z.npz" \
  --decoder evaluation.router:make_tiered_decoder \
  --plugin-options "$options" \
  --out "$out/router_eval_p$p"

"$MIDQ_PY" -m evaluation.compare_gain \
  --reference-results "$out/eval_p$p" \
  --candidate-results "$out/router_eval_p$p" \
  --candidate-decoder evaluation.router:make_tiered_decoder \
  --out "$out/gain_p$p"

echo "Done: $out/gain_p$p/comparison.csv"
