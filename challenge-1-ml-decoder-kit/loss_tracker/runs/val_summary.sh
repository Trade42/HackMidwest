#!/bin/bash
# The validation tables of step 3 and the choice of the frozen decoder, from the per-shot parts (no decoding).
# usage: loss_tracker/runs/val_summary.sh choose     the variants' table and the choice (prints the lines of frozen.sh)
#        loss_tracker/runs/val_summary.sh record     after frozen.sh and val_frozen.sh: the frozen pair on the same validation shots
source loss_tracker/runs/common.sh
P=$S/parts/val
if [ "${1:-choose}" = choose ]; then
  "$PY" -m loss_tracker.summarize --parts "$P/references/*.npz" "$P/soft/*.npz" "$P/variants/*.npz" --references exact \
    --order exact,tracker,tracker-T2,tracker-T0.5,tracker-hard0.6,tracker-slot,oracle-round --cascade 'tracker>exact' \
    --title "Validation (seed 7200), shots 0 to 1,499; the variants with the tracker version order1_coupled2" --out loss_tracker/results/val_variants
  "$PY" -m loss_tracker.choose --accuracy loss_tracker/results/accuracy_val.json --variants "$P/soft/*.npz" "$P/variants/*.npz" \
    --earlier-version order1_coupled2 --out loss_tracker/results/choice
else
  source loss_tracker/runs/frozen.sh
  "$PY" -m loss_tracker.choose --accuracy loss_tracker/results/accuracy_val.json --variants "$P/soft/*.npz" "$P/variants/*.npz" \
    --earlier-version order1_coupled2 --record "$P/frozen/*.npz" --out loss_tracker/results/choice
  # the 2-sweep explain version with the soft posterior, decoded before the choice (val_explain.sh): also a record
  "$PY" -m loss_tracker.summarize --parts "$P/references/*.npz" "$P/explain_tracker/*.npz" --references exact \
    --order exact,tracker,oracle-round --title "Validation (seed 7200), shots 0 to 1,499; tracker version order1_explain2, soft posterior (not the frozen pair)" \
    --out loss_tracker/results/val_explain2_soft
  "$PY" -m loss_tracker.summarize --parts "$P/references/*.npz" "$P/frozen/*.npz" --references exact \
    --order "exact,$VARIANT,oracle-round" --cascade "$VARIANT>exact" \
    --title "Validation (seed 7200), shots 0 to 1,499; the frozen pair: tracker version $TRACK_NAME, variant $VARIANT" --out loss_tracker/results/val_frozen
fi
