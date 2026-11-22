#!/bin/bash
# Step 4: the tracker at another loss rate or in the other memory basis. Samples a training and a validation
# split, fits the tables, runs the tracker on the validation split and scores it (no decoding).
# usage: loss_tracker/runs/extra.sh NAME BASIS S TRAIN_SEED VAL_SEED
#   loss_tracker/runs/extra.sh X_p0.004_s2e-3 X 2e-3 7140 7240
#   loss_tracker/runs/extra.sh X_p0.004_s5e-4 X 5e-4 7160 7260
#   loss_tracker/runs/extra.sh Z_p0.004_s1e-3 Z 1e-3 7120 7220
# The tracker is the frozen version of step 3 (runs/frozen.sh), with tables fitted at this point.
source loss_tracker/runs/common.sh
source loss_tracker/runs/frozen.sh
NAME=$1; BASIS=$2; RATE=$3; TRAIN_SEED=$4; VAL_SEED=$5
TR=$S/data/train_${NAME}_seed$TRAIN_SEED.npz
VA=$S/data/val_${NAME}_seed$VAL_SEED.npz
TB=loss_tracker/tables/$NAME.npz
mkdir -p "$S/data" "$S/post"
[ -f "$TR" ] || { lane "$PY" -m loss.sample --basis "$BASIS" --p 0.004 --s "$RATE" --shots 20000 --seed "$TRAIN_SEED" --threads 4 --out "$TR.tmp.npz"; mv "$TR.tmp.npz" "$TR"; }
[ -f "$VA" ] || { lane "$PY" -m loss.sample --basis "$BASIS" --p 0.004 --s "$RATE" --shots 4000 --seed "$VAL_SEED" --threads 4 --out "$VA.tmp.npz"; mv "$VA.tmp.npz" "$VA"; }
sha256sum "$TR" "$VA"
lane "$PY" -m loss_tracker.fit --data "$TR" --out "$TB"
lane "$PY" -m loss_tracker.track --data "$VA" --tables "$TB" $TRACK_ARGS --out "$S/post/val_$NAME.npz"
ARGS=(tracker="$S/post/val_$NAME.npz")
if [ "$BASIS" = X ] && [ "$NAME" != X_p0.004_s1e-3 ]; then
  # the same shots with the tables fitted at s = 1e-3: do the tables carry over to another loss rate?
  lane "$PY" -m loss_tracker.track --data "$VA" --tables "$TABLES" $TRACK_ARGS --out "$S/post/val_${NAME}_tables1e-3.npz"
  ARGS+=(tables_of_1e-3="$S/post/val_${NAME}_tables1e-3.npz")
fi
"$PY" -m loss_tracker.accuracy --data "$VA" --main tracker --out "loss_tracker/results/accuracy_val_$NAME" --post "${ARGS[@]}"
