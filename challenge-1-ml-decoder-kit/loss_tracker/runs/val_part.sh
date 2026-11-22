#!/bin/bash
# One part of a validation decode with the tracker version order1_coupled2 (the version of val_variants.sh).
# usage: loss_tracker/runs/val_part.sh GROUP CONDITIONS START STOP
# Run on 2026-11-22 for the last 500 shots of the variants, as two parts on two lanes:
#   LANE=B loss_tracker/runs/val_part.sh variants tracker-T2,tracker-T0.5,tracker-hard0.6,tracker-slot 1000 1250
#   LANE=D loss_tracker/runs/val_part.sh variants tracker-T2,tracker-T0.5,tracker-hard0.6,tracker-slot 1250 1500
source loss_tracker/runs/common.sh
part "$S/parts/val/$1/$(printf %06d $3)-$(printf %06d $4).npz" \
  "$PY" -m loss_tracker.decode --data "$VAL" --post "$S/post/val_order1_coupled2.npz" --conditions "$2" --start $3 --stop $4 --workers 4
