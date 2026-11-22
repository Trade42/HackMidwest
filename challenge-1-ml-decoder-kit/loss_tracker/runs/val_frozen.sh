#!/bin/bash
# The frozen pair (runs/frozen.sh) on the first 1,500 VALIDATION shots, for the record (rule 1 of RESULTS.md).
# usage: loss_tracker/runs/val_frozen.sh [FIRST LAST]   (default 0 1500, in parts of 250)
# Run on 2026-11-22 as two streams: `LANE=D ... 0 750` and `LANE=B ... 750 1500`.
source loss_tracker/runs/common.sh
source loss_tracker/runs/frozen.sh
POST=$S/post/val_$TRACK_NAME.npz
for ((a = ${1:-0}; a < ${2:-1500}; a += 250)); do
  part "$S/parts/val/frozen/$(printf %06d $a)-$(printf %06d $((a + 250))).npz" \
    "$PY" -m loss_tracker.decode --data "$VAL" --post "$POST" --conditions "$VARIANT" --start $a --stop $((a + 250)) --workers 4
done
