#!/bin/bash
# L3, second part: the round-level oracle on the starter's own map (starter-oracle-round), on the first 2,000
# shots of the s = 1e-3 dataset (paired with the L3 group of L0_L1_s1e-3.sh). The slot-level oracle and the
# other 2,000 shots run on the second machine (job A of loss/job.py, seed 5300). Run from challenge-1-ml-decoder-kit/.
source loss/runs/lane.sh
D=$S/data
parts X_p0.004_s1e-3/oracle-starter $D/X_p0.004_s1e-3_seed2300.npz starter-oracle-round 2000 500
