#!/bin/sh
set -eu
cd "$(dirname "$0")"
PYTHON=${PYTHON:-.venv312/bin/python}
OUT=${OUT:-results}
DATA=${DATA:-data}
stage=${1:-}
p=${2:-}
valid_p() {
  case "$1" in 0.003|0.004|0.005) ;; *) echo "p must be 0.003, 0.004, or 0.005" >&2; exit 2;; esac
}
generate() {
  destination=$1
  shift
  if [ -e "$destination" ]; then echo "refusing to overwrite $destination" >&2; exit 1; fi
  "$PYTHON" make_dataset.py "$@" --out "$destination"
}
case "$stage" in
  parse-log)
    "$PYTHON" -m evaluation.parse_baselines baselines.log --out-prefix "$OUT/baselines_parsed"
    ;;
  dataset)
    valid_p "$p"
    mkdir -p "$DATA/p$p"
    for basis in X Z; do
      if [ "$basis" = X ]; then offset=0; else offset=1; fi
      generate "$DATA/p$p/train_$basis.npz" --basis "$basis" --p "$p" --shots 20000 --seed "$((101+offset))"
      generate "$DATA/p$p/validation_$basis.npz" --basis "$basis" --p "$p" --shots 5000 --seed "$((201+offset))"
      generate "$DATA/p$p/test_$basis.npz" --basis "$basis" --p "$p" --shots 10000 --seed "$((301+offset))"
    done
    if [ "$p" = 0.004 ]; then
      generate "$DATA/p$p/go_X.npz" --basis X --p "$p" --shots 5000 --seed 401
    fi
    ;;
  evaluate)
    valid_p "$p"
    "$PYTHON" -m evaluation.evaluate \
      --train-x "$DATA/p$p/train_X.npz" --train-z "$DATA/p$p/train_Z.npz" \
      --validation-x "$DATA/p$p/validation_X.npz" --validation-z "$DATA/p$p/validation_Z.npz" \
      --test-x "$DATA/p$p/test_X.npz" --test-z "$DATA/p$p/test_Z.npz" \
      --decoder bposd --decoder relay-bp --decoder relay-first-solution \
      --decoder relay-first-leg-or-full --out "$OUT/eval_p$p"
    ;;
  plot)
    valid_p "$p"
    "$PYTHON" -m evaluation.plot_results "$OUT/eval_p$p/summary.csv" --out "$OUT/plots_p$p"
    ;;
  plot-all)
    set --
    for noise in 0.003 0.004 0.005; do
      if [ -f "$OUT/eval_p$noise/summary.csv" ]; then set -- "$@" "$OUT/eval_p$noise/summary.csv"; fi
    done
    "$PYTHON" -m evaluation.plot_results "$@" --out "$OUT/plots_all"
    ;;
  router-features)
    valid_p "$p"
    for basis in X Z; do
      for split in train validation; do
        "$PYTHON" -m evaluation.router_features --split "$split" --basis "$basis" \
          --data "$DATA/p$p/${split}_$basis.npz" --out "$OUT/router_p$p/${split}_$basis.npz"
      done
    done
    ;;
  router-train)
    valid_p "$p"
    : "${MAX_MISSED_WRONG:?set MAX_MISSED_WRONG to an explicit validation error budget}"
    for basis in X Z; do
      "$PYTHON" -m evaluation.router_train --train-features "$OUT/router_p$p/train_$basis.npz" \
        --validation-features "$OUT/router_p$p/validation_$basis.npz" \
        --max-missed-wrong "$MAX_MISSED_WRONG" --out "$OUT/router_p$p/model_$basis.json"
    done
    ;;
  router-sweep)
    valid_p "$p"
    shift 2
    if [ "$#" -eq 0 ]; then echo "supply one or more validation risk thresholds" >&2; exit 2; fi
    "$PYTHON" -m evaluation.router_sweep --validation-x "$DATA/p$p/validation_X.npz" \
      --validation-z "$DATA/p$p/validation_Z.npz" \
      --model-x "$OUT/router_p$p/model_X.json" --model-z "$OUT/router_p$p/model_Z.json" \
      --thresholds "$@" --out "$OUT/router_p$p/validation_sweep"
    ;;
  rows)
    "$PYTHON" -m evaluation.comparison_rows --out "$OUT/comparison_rows.json"
    ;;
  *)
    echo "usage: $0 {parse-log|dataset|evaluate|plot|plot-all|router-features|router-train|router-sweep|rows} [0.003|0.004|0.005] [thresholds...]" >&2
    exit 2
    ;;
esac
