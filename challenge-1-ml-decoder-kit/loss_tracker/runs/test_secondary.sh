#!/bin/bash
# A SECONDARY row on the TEST split, decoded after the frozen run: the tracker version before the frozen one
# (order1_coupled2, soft posterior), which had fewer failures on the validation shots. Not the frozen decoder.
# usage: loss_tracker/runs/test_secondary.sh [FIRST LAST]   (default 0 4000, parts of 500)
source loss_tracker/runs/common.sh
POST=$S/post/test_order1_coupled2.npz
if [ ! -f "$POST" ]; then
  lane "$PY" -m loss_tracker.track --data "$TEST" --tables "$TABLES" --mode coupled --sweeps 2 --order 1 --out "$POST.tmp.npz"
  mv "$POST.tmp.npz" "$POST"
fi
for ((a = ${1:-0}; a < ${2:-4000}; a += 500)); do
  part "$S/parts/test_coupled2/$(printf %06d $a)-$(printf %06d $((a + 500))).npz" \
    "$PY" -m loss_tracker.decode --data "$TEST" --post "$POST" --conditions tracker --start $a --stop $((a + 500)) --workers 4
done
