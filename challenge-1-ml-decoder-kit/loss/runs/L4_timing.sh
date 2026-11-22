#!/bin/bash
# L4: one process on CPU 0 under the lane D lock (so CPU 1, its sibling thread, stays idle). Run from challenge-1-ml-decoder-kit/.
source loss/runs/lane.sh
flock "$RUN/sim-d.lock" nice -n 15 taskset -c 0 "$PY" -m loss.timing \
  --loss-data $S/data/X_p0.004_s1e-3_seed2300.npz --clean-data $S/data/X_p0.004_s0_seed1000.npz \
  --out loss/results/L4_timing > loss/logs/L4_timing.log 2>&1
