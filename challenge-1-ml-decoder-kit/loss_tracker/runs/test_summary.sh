#!/bin/bash
# The test tables of step 3 from the per-shot parts (no decoding). usage: loss_tracker/runs/test_summary.sh [PARTS_GLOB NAME TITLE]
source loss_tracker/runs/common.sh
source loss_tracker/runs/frozen.sh
"$PY" -m loss_tracker.summarize --parts "${1:-$S/parts/test/*.npz}" --references exact,starter \
  --order "starter,exact,$VARIANT,oracle-round" --cascade "$VARIANT>exact" --cascade "$VARIANT>starter" \
  --title "${3:-Test (seed 7300): the frozen decoder (tracker version $TRACK_NAME, variant $VARIANT) against the references}" \
  --out "loss_tracker/results/${2:-test}"
