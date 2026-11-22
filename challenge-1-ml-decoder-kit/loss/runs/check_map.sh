#!/bin/bash
# Launched as: flock /home/sal/hackMidwest/dq/run/sim-d.lock nice -n 15 taskset -c 0-3 loss/runs/check_map.sh
# Map checks: loss-only Clifft shots, both bases, two loss rates, and forced single losses. Lane D, 4 processes.
cd /home/sal/hackMidwest/worktrees/loss-gates/challenge-1-ml-decoder-kit
export PYTHONDONTWRITEBYTECODE=1
V=/home/sal/hackMidwest/orchestration/scratch/loss-gates/venv
L=loss/logs
$V/bin/python -m loss.check_map --basis X --s 1e-3 --shots 1000 --seed 101 --out loss/results/check_map_X_s1e-3.json > $L/check_map_X_s1e-3.log 2>&1 &
$V/bin/python -m loss.check_map --basis Z --s 1e-3 --shots 1000 --seed 102 --out loss/results/check_map_Z_s1e-3.json > $L/check_map_Z_s1e-3.log 2>&1 &
$V/bin/python -m loss.check_map --basis X --s 2e-3 --shots 1000 --seed 103 --out loss/results/check_map_X_s2e-3.json > $L/check_map_X_s2e-3.log 2>&1 &
$V/bin/python -m loss.check_map --basis Z --s 2e-3 --shots 1000 --seed 104 --out loss/results/check_map_Z_s2e-3.json > $L/check_map_Z_s2e-3.log 2>&1 &
wait
$V/bin/python -m loss.check_map --basis X --forced --shots 400 --seed 201 --out loss/results/check_forced_X.json > $L/check_forced_X.log 2>&1 &
$V/bin/python -m loss.check_map --basis Z --forced --shots 400 --seed 301 --out loss/results/check_forced_Z.json > $L/check_forced_Z.log 2>&1 &
wait
