#!/bin/bash
# The plumbing check of step 3: this pipeline with the TRUE loss round against loss-gates' ORACLE `oracle-round`,
# on the first 500 shots of loss-gates' dataset (seed 2300) and its per-shot parts.
source loss_tracker/runs/common.sh
G=/home/sal/hackMidwest/orchestration/scratch/loss-gates
part "$S/parts/plumbing/000000-000500.npz" "$PY" -m loss_tracker.decode --data "$G/data/X_p0.004_s1e-3_seed2300.npz" \
  --conditions plumb-oracle-round --start 0 --stop 500 --workers 4
"$PY" -m loss_tracker.plumbing --mine "$S/parts/plumbing/*.npz" --theirs "$G/parts/X_p0.004_s1e-3/aware/*.npz" --out loss_tracker/results/plumbing
