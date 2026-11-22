#!/bin/bash
# L0 (no loss) and L1/L3 at s = 1e-3, X memory, p = 0.004. Run from challenge-1-ml-decoder-kit/.
source loss/runs/lane.sh
D=$S/data
dataset $D/X_p0.004_s0_seed1000.npz --basis X --p 0.004 --s 0 --shots 10000 --seed 1000
dataset $D/X_p0.004_s1e-3_seed2300.npz --basis X --p 0.004 --s 1e-3 --shots 20000 --seed 2300
parts L0_X_p0.004_s0/blind $D/X_p0.004_s0_seed1000.npz blind 10000 2500
parts X_p0.004_s1e-3/blind $D/X_p0.004_s1e-3_seed2300.npz blind 300 100
parts X_p0.004_s1e-3/aware $D/X_p0.004_s1e-3_seed2300.npz starter,exact,oracle-round,starter@0.5/stop1 4000 400
