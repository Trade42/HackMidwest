#!/bin/bash
# The smoke job of job.py: 200 shots sampled (seed 9100), the first 40 decoded under four conditions.
# The same command runs on the second machine; comparing the two printouts says whether Clifft's shots and
# relay-BP's outputs are the same on both machines.
source loss_tracker/runs/common.sh
lane "$PY" -m loss_tracker.job --seed 9100 --shots 200 --decode-shots 40 --part 20 --workers 4 --out "$S/jobs/smoke_seed9100" \
  --tables "$TABLES" --conditions starter,exact,tracker,oracle-round
