#!/bin/bash
# L2 on this PC: leave-one-out from the starter's map on the first 500 shots of the s = 1e-3 dataset, then the
# cap sweep on the first 500. The full L2 (1,000 shots) runs on the second machine (job D of loss/job.py, seed 5300).
# Run from challenge-1-ml-decoder-kit/.
source loss/runs/lane.sh
D=$S/data
parts X_p0.004_s1e-3/ablate $D/X_p0.004_s1e-3_seed2300.npz starter-nofinal,starter-nohook,starter-ratebase 500 250
parts X_p0.004_s1e-3/cap $D/X_p0.004_s1e-3_seed2300.npz exact@0.4,oracle-round@0.4,exact@0.25,oracle-round@0.25 500 250
