#!/bin/bash
# The tracker's versions on the VALIDATION split (4,000 shots, seed 7200). track.py reads detectors and heralds.
source loss_tracker/runs/common.sh
mkdir -p "$S/post"
run() {   # run NAME MODE SWEEPS ORDER
  [ -f "$S/post/val_$1.npz" ] && { echo "keep $S/post/val_$1.npz"; return 0; }
  lane "$PY" -m loss_tracker.track --data "$VAL" --tables "$TABLES" --mode "$2" --sweeps "$3" --order "$4" --out "$S/post/val_$1.tmp.npz"
  mv "$S/post/val_$1.tmp.npz" "$S/post/val_$1.npz"
}
run order1_coupled2 coupled 2 1
run order0_naive naive 0 0
run order0_coupled2 coupled 2 0
run order1_naive naive 0 1
run order1_drop drop 0 1
run order1_coupled0 coupled 0 1
run order1_coupled1 coupled 1 1
run order1_coupled4 coupled 4 1
