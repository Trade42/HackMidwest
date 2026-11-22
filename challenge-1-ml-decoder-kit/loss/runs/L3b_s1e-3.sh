#!/bin/bash
# L3, second part: the round-level oracle on the starter's own map (starter-oracle-round), on the same 4,000 shots
# as the L3 group of L0_L1_s1e-3.sh, and the slot-level oracle on the first 2,000. Run from challenge-1-ml-decoder-kit/.
source loss/runs/lane.sh
D=$S/data
parts X_p0.004_s1e-3/oracle-starter $D/X_p0.004_s1e-3_seed2300.npz starter-oracle-round 4000 500
parts X_p0.004_s1e-3/slot $D/X_p0.004_s1e-3_seed2300.npz oracle-slot 2000 500
