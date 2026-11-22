#!/bin/bash
# What ran on lane D after the test, one job after the other (2026-11-22): the secondary test row, the rerun check
# of done-when 4, then step 4 in the brief's order (the learned tracker; s = 2e-3 and 5e-4; Z memory; the cost).
set -uo pipefail
loss_tracker/runs/test_summary.sh > loss_tracker/logs/test_summary.log 2>&1
loss_tracker/runs/diagnose.sh > loss_tracker/logs/diagnose.log 2>&1
loss_tracker/runs/step1_step2.sh > loss_tracker/logs/step1_step2.log 2>&1    # a second run of steps 1 and 2: same tables?
loss_tracker/runs/test_secondary.sh > loss_tracker/logs/test_secondary.log 2>&1
loss_tracker/runs/test_summary.sh secondary > loss_tracker/logs/test_summary_secondary.log 2>&1
loss_tracker/runs/rerun_check.sh > loss_tracker/logs/rerun_check.log 2>&1
loss_tracker/runs/learned.sh > loss_tracker/logs/learned.log 2>&1
loss_tracker/runs/extra.sh X_p0.004_s2e-3 X 2e-3 7140 7240 > loss_tracker/logs/extra_X_s2e-3.log 2>&1
loss_tracker/runs/extra.sh X_p0.004_s5e-4 X 5e-4 7160 7260 > loss_tracker/logs/extra_X_s5e-4.log 2>&1
loss_tracker/runs/extra.sh Z_p0.004_s1e-3 Z 1e-3 7120 7220 > loss_tracker/logs/extra_Z_s1e-3.log 2>&1
loss_tracker/runs/timing.sh > loss_tracker/logs/timing.log 2>&1
