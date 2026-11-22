#!/bin/bash
# Step 1 (the signal, on the TRAINING split) and step 2 (the tracker's accuracy on the VALIDATION split).
# Needs fit.sh and track_val.sh first.
source loss_tracker/runs/common.sh
lane "$PY" -m loss_tracker.signal --data "$TRAIN" --tables "$TABLES" --out loss_tracker/results/signal_train
P=$S/post
lane "$PY" -m loss_tracker.accuracy --data "$VAL" --main order1_coupled2 --out loss_tracker/results/accuracy_val --post \
  order0_naive=$P/val_order0_naive.npz order0_coupled2=$P/val_order0_coupled2.npz \
  order1_naive=$P/val_order1_naive.npz order1_drop=$P/val_order1_drop.npz order1_coupled0=$P/val_order1_coupled0.npz \
  order1_coupled1=$P/val_order1_coupled1.npz order1_coupled2=$P/val_order1_coupled2.npz order1_coupled4=$P/val_order1_coupled4.npz
