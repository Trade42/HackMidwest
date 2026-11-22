#!/bin/bash
# Fit the emission tables on the TRAINING split (20,000 shots, seed 7100). The only step that reads the probes
# of a dataset for the tracker.
source loss_tracker/runs/common.sh
lane "$PY" -m loss_tracker.fit --data "$TRAIN" --out "$TABLES"
sha256sum "$TABLES"
