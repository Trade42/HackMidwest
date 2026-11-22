#!/bin/bash
# Where the frozen tracker loses against the ORACLE, per shot (scoring only, no decoding; about a second each).
source loss_tracker/runs/common.sh
source loss_tracker/runs/frozen.sh
"$PY" -m loss_tracker.diagnose --data "$VAL" --post "$S/post/val_$TRACK_NAME.npz" --parts "$S/parts/val/references/*.npz" \
  "$S/parts/val/frozen/*.npz" --out loss_tracker/results/diagnose_val
[ -f "$S/parts/test/003500-004000.npz" ] && "$PY" -m loss_tracker.diagnose --data "$TEST" --post "$S/post/test_$TRACK_NAME.npz" \
  --parts "$S/parts/test/*.npz" --out loss_tracker/results/diagnose_test
# the frozen tracker against the probes of the test split (scoring only; the decoding above did not use this)
[ -f "$S/post/test_$TRACK_NAME.npz" ] && "$PY" -m loss_tracker.accuracy --data "$TEST" --main "$TRACK_NAME" \
  --post "$TRACK_NAME=$S/post/test_$TRACK_NAME.npz" --out loss_tracker/results/accuracy_test
