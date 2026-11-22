#!/bin/bash
# The tracker's decoder variants on the first 1,500 VALIDATION shots (the shots of val_references.sh).
# The variant is chosen on these shots only. Group 1: the soft posterior. Group 2: the other variants.
source loss_tracker/runs/common.sh
POST=$S/post/val_order1_coupled2.npz
group() {   # group NAME CONDITIONS
  for a in 0 500 1000; do
    part "$S/parts/val/$1/$(printf %06d $a)-$(printf %06d $((a + 500))).npz" \
      "$PY" -m loss_tracker.decode --data "$VAL" --post "$POST" --conditions "$2" --start $a --stop $((a + 500)) --workers 4
  done
}
group soft tracker
group variants tracker-T2,tracker-T0.5,tracker-hard0.6,tracker-slot
