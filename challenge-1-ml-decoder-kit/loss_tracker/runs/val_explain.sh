#!/bin/bash
# The soft posterior of the tracker version that explains shared bits, on the first 1,500 VALIDATION shots
# (the shots of val_references.sh and val_variants.sh). Needs track_val.sh first.
source loss_tracker/runs/common.sh
POST=$S/post/val_order1_explain2.npz
for a in 0 500 1000; do
  part "$S/parts/val/explain_${1:-tracker}/$(printf %06d $a)-$(printf %06d $((a + 500))).npz" \
    "$PY" -m loss_tracker.decode --data "$VAL" --post "$POST" --conditions "${1:-tracker}" --start $a --stop $((a + 500)) --workers 4
done
