#!/bin/bash
# Step 4: the tracker's cost per shot. One process pinned to CPU 0 (one thread of a physical core; its sibling,
# CPU 1, runs no other job of ours), under the lane-D lock, on the first 300 VALIDATION shots. Numpy on one thread.
source loss_tracker/runs/common.sh
flock "$LOCK" env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 15 taskset -c 0 \
  "$PY" -m loss_tracker.timing --data "$VAL" --tables "$TABLES" --shots 300 --out loss_tracker/results/timing
