#!/bin/bash
# The tracker's versions on the VALIDATION split (4,000 shots, seed 7200). track.py reads detectors and heralds.
# One hold of the lane, at most 4 processes at a time; a version whose file exists is kept.
source loss_tracker/runs/common.sh
mkdir -p "$S/post"
one() {   # one NAME MODE SWEEPS ORDER [--explain]
  local out="$S/post/val_$1.npz"
  [ -f "$out" ] && { echo "keep $out"; return 0; }
  "$PY" -m loss_tracker.track --data "$VAL" --tables "$TABLES" --mode "$2" --sweeps "$3" --order "$4" ${5:-} --out "$out.tmp.npz"
  mv "$out.tmp.npz" "$out"
}
export -f one; export S PY VAL TABLES
lane xargs -P 4 -L 1 bash -c 'one "$@"' _ <<'LIST'
order1_explain2 coupled 2 1 --explain
order1_coupled2 coupled 2 1
order0_naive naive 0 0
order0_coupled2 coupled 2 0
order1_naive naive 0 1
order1_drop drop 0 1
order1_coupled0 coupled 0 1
order1_coupled1 coupled 1 1
order1_coupled4 coupled 4 1
order1_explain0 coupled 0 1 --explain
order1_explain1 coupled 1 1 --explain
order1_explain4 coupled 4 1 --explain
LIST
