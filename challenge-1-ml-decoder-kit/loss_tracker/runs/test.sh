#!/bin/bash
# Step 3 on the TEST split (4,000 shots, seed 7300): the frozen tracker decoder against `starter`, `exact` and the
# ORACLE `oracle-round`, every condition on every shot, in parts of 500 (any prefix is a complete paired result).
# The frozen variant is the one line of results/frozen_variant.txt, written after the validation runs.
# usage: loss_tracker/runs/test.sh [FIRST_SHOT LAST_SHOT]   (default 0 4000; two streams may take two halves)
source loss_tracker/runs/common.sh
FROZEN=$(cat loss_tracker/results/frozen_variant.txt)
POST=$S/post/test_order1_coupled2.npz
if [ ! -f "$POST" ]; then
  lane "$PY" -m loss_tracker.track --data "$TEST" --tables "$TABLES" --mode coupled --sweeps 2 --order 1 --out "$POST.tmp.npz"
  mv "$POST.tmp.npz" "$POST"
fi
for ((a = ${1:-0}; a < ${2:-4000}; a += 500)); do
  part "$S/parts/test/$(printf %06d $a)-$(printf %06d $((a + 500))).npz" \
    "$PY" -m loss_tracker.decode --data "$TEST" --post "$POST" --conditions "starter,exact,$FROZEN,oracle-round" \
    --start $a --stop $((a + 500)) --workers "${WORKERS:-4}"
done
