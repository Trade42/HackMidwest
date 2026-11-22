#!/bin/bash
# L1 at the other loss rates, X memory, p = 0.004: blind on a prefix sized for about 100 failures, the starter's
# flag-aware priors (and the stop-at-first-solution column) on a longer prefix. Run from challenge-1-ml-decoder-kit/.
# This is what ran on this PC. The orchestrator stopped s = 5e-4 at 2,000 flag-aware shots and s = 2e-4 before
# any decode (2026-11-22 06:00 UTC): server jobs B (s = 2e-4) and C (s = 5e-4) measure them on more shots
# (loss/job.py; command lines in RESULTS.md, L1).
source loss/runs/lane.sh
D=$S/data
dataset $D/X_p0.004_s2e-3_seed2400.npz --basis X --p 0.004 --s 2e-3 --shots 20000 --seed 2400
dataset $D/X_p0.004_s5e-4_seed2200.npz --basis X --p 0.004 --s 5e-4 --shots 20000 --seed 2200
dataset $D/X_p0.004_s2e-4_seed2100.npz --basis X --p 0.004 --s 2e-4 --shots 20000 --seed 2100
parts X_p0.004_s2e-3/blind $D/X_p0.004_s2e-3_seed2400.npz blind 150 50
parts X_p0.004_s2e-3/aware $D/X_p0.004_s2e-3_seed2400.npz starter,starter@0.5/stop1 750 250
parts X_p0.004_s5e-4/blind $D/X_p0.004_s5e-4_seed2200.npz blind 400 100
parts X_p0.004_s5e-4/aware $D/X_p0.004_s5e-4_seed2200.npz starter,starter@0.5/stop1 2000 1000
