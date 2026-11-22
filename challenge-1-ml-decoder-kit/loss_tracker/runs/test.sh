#!/bin/bash
# Step 3 on the TEST split (4,000 shots, seed 7300): the frozen tracker decoder against `starter`, `exact` and the
# ORACLE `oracle-round`, every condition on every shot, in parts of 500 (any prefix is a complete paired result).
# The frozen tracker version and decoder variant are in runs/frozen.sh, written after the validation runs.
# usage: loss_tracker/runs/test.sh [FIRST_SHOT LAST_SHOT]   (default 0 4000)
# Run on 2026-11-22 as two streams: `LANE=D loss_tracker/runs/test.sh 0 2000` and `LANE=B loss_tracker/runs/test.sh 2000 4000`,
# after `loss_tracker/runs/test.sh 0 0` had written the tracker's output for the split.
source loss_tracker/runs/common.sh
source loss_tracker/runs/frozen.sh
POST=$S/post/test_$TRACK_NAME.npz
if [ ! -f "$POST" ]; then
  lane "$PY" -m loss_tracker.track --data "$TEST" --tables "$TABLES" $TRACK_ARGS --out "$POST.tmp.npz"
  mv "$POST.tmp.npz" "$POST"
fi
for ((a = ${1:-0}; a < ${2:-4000}; a += 500)); do
  part "$S/parts/test/$(printf %06d $a)-$(printf %06d $((a + 500))).npz" \
    "$PY" -m loss_tracker.decode --data "$TEST" --post "$POST" --conditions "starter,exact,$VARIANT,oracle-round" \
    --start $a --stop $((a + 500)) --workers 4
done
