#!/bin/bash
# The test tables of step 3 from the per-shot parts (no decoding).
# usage: loss_tracker/runs/test_summary.sh                 the frozen run: results/test.{md,json}
#        loss_tracker/runs/test_summary.sh secondary       with the secondary row (test_secondary.sh): results/test_secondary.{md,json}
source loss_tracker/runs/common.sh
source loss_tracker/runs/frozen.sh
P=$S/parts
if [ "${1:-frozen}" = frozen ]; then
  # second attempts, each computed from the per-shot results: the frozen decoder, then the uniform guess where it did
  # not converge; and the same second attempt for the references, so that the rows can be compared like with like
  "$PY" -m loss_tracker.summarize --parts "$P/test/*.npz" --references exact,starter \
    --order "starter,exact,$VARIANT,oracle-round" --cascade "$VARIANT>exact" --cascade "exact>starter" --cascade "oracle-round>exact" \
    --title "Test (seed 7300): the frozen decoder (tracker version $TRACK_NAME, variant $VARIANT) against the references" \
    --out loss_tracker/results/test
else
  "$PY" -m loss_tracker.summarize --parts "$P/test/*.npz" --references exact,starter \
    --order "starter,exact,$VARIANT,oracle-round" --also "tracker_order1_coupled2=tracker@$P/test_coupled2/*.npz" \
    --pair "$VARIANT,tracker_order1_coupled2" \
    --title "Test (seed 7300): with the SECONDARY row tracker_order1_coupled2 (the version before the frozen one; not the frozen decoder)" \
    --out loss_tracker/results/test_secondary
fi
