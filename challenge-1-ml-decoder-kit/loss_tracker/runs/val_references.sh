#!/bin/bash
# The reference conditions on the first 1,500 VALIDATION shots (loss-gates' decoder, unchanged): `exact` and the
# ORACLE `oracle-round`. The tracker's variants are decoded on the same shots by val_variants.sh.
source loss_tracker/runs/common.sh
for a in 0 500 1000; do
  part "$S/parts/val/references/$(printf %06d $a)-$(printf %06d $((a + 500))).npz" \
    "$PY" -m loss.decode --data "$VAL" --conditions exact,oracle-round --start $a --stop $((a + 500)) --workers 4
done
