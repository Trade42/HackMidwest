#!/bin/bash
# The short validation jobs, one after the other. Run on 2026-11-22 from 06:52 UTC as `LANE=B loss_tracker/runs/queue_short_jobs.sh`,
# while lane D decoded the variants (val_variants.sh).
set -euo pipefail
loss_tracker/runs/track_val.sh > loss_tracker/logs/track_val.log 2>&1
loss_tracker/runs/val_explain.sh > loss_tracker/logs/val_explain.log 2>&1
loss_tracker/runs/step1_step2.sh > loss_tracker/logs/step1_step2.log 2>&1
loss_tracker/runs/plumbing.sh > loss_tracker/logs/plumbing.log 2>&1
