#!/bin/bash
# The three splits, X memory, p = 0.004, s = 1e-3 (Clifft, chunks of 2,000 shots, chunk k uses seed + k).
# training 20,000 shots (seeds 7100-7109), validation 4,000 (7200-7201), test 4,000 (7300-7301).
source loss_tracker/runs/common.sh
mkdir -p "$S/data"
sample() {   # sample OUT SHOTS SEED
  [ -f "$1" ] && { echo "keep $1"; return 0; }
  lane "$PY" -m loss.sample --basis X --p 0.004 --s 1e-3 --shots "$2" --seed "$3" --threads 4 --out "$1.tmp.npz"
  mv "$1.tmp.npz" "$1"
}
sample "$TRAIN" 20000 7100
sample "$VAL" 4000 7200
sample "$TEST" 4000 7300
sha256sum "$TRAIN" "$VAL" "$TEST"
