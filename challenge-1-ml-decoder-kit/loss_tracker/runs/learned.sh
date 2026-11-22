#!/bin/bash
# Step 4: the learned tracker. Trains the network on the TRAINING split (CPU, 4 threads, seed 0), runs it on the
# VALIDATION split and scores it beside the model of step 2.
source loss_tracker/runs/common.sh
MODEL=loss_tracker/tables/learned_X_p0.004_s1e-3.pt
[ -f "$MODEL" ] || lane "$PY" -m loss_tracker.learned train --data "$TRAIN" --out "$MODEL"
lane "$PY" -m loss_tracker.learned track --data "$VAL" --model "$MODEL" --out "$S/post/val_learned.npz"
"$PY" -m loss_tracker.accuracy --data "$VAL" --main learned --out loss_tracker/results/accuracy_val_learned --post \
  learned="$S/post/val_learned.npz" order1_explain2="$S/post/val_order1_explain2.npz" order1_coupled2="$S/post/val_order1_coupled2.npz" \
  order1_drop="$S/post/val_order1_drop.npz" order1_naive="$S/post/val_order1_naive.npz"
