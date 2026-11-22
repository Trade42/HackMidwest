#!/bin/bash
# L2: leave-one-out from the starter's map, and the cap sweep, on the first 1,000 shots of the s = 1e-3 dataset.
# Run from challenge-1-ml-decoder-kit/.
source loss/runs/lane.sh
D=$S/data
parts X_p0.004_s1e-3/ablate $D/X_p0.004_s1e-3_seed2300.npz starter-nofinal,starter-nohook,starter-ratebase 1000 250
parts X_p0.004_s1e-3/cap $D/X_p0.004_s1e-3_seed2300.npz exact@0.4,oracle-round@0.4,exact@0.25,oracle-round@0.25 1000 250
