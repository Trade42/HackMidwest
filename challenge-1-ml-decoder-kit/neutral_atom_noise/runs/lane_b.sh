#!/usr/bin/env bash
# What ran on this PC (luminaty) for RESULTS.md. Source it from challenge-1-ml-decoder-kit/, then call a function.
# The tools themselves need none of these paths: see README.md, "Run it".
#
# Compute (brief): lane B, 4 processes on CPUs 8-11, niced, shared with two other workers, so no hold of the
# lock lasts over 20 minutes. `hold` is one hold: job.py stops before a new part once 12 minutes are spent, and
# a part of 125 shots takes at most a few minutes. `stream` repeats holds, releasing the lock in between, until
# the job is done. job.py keeps finished parts, so a stopped stream resumes where it was.
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1
S=/home/sal/hackMidwest/orchestration/scratch/infleqtion-noise
PY="$S/venv/bin/python"
LOCK=${LOCK:-/home/sal/hackMidwest/dq/run/sim-b.lock}
CPUS=${CPUS:-8-11}
OUT=${OUT:-$S/runs/X_seed4100}
SEED=${SEED:-4100}
BASIS=${BASIS:-X}

hold() {
  flock "$LOCK" nice -n 15 taskset -c "$CPUS" "$PY" -m neutral_atom_noise.job --basis "$BASIS" --shots 2000 --seed "$SEED" \
    --workers 4 --part 125 --budget-minutes 12 --out "$OUT" "$@"
}

stream() {
  while true; do
    hold "$@"; code=$?
    [ $code -eq 0 ] && return 0
    [ $code -eq 3 ] || { echo "job failed with exit code $code"; return $code; }
    sleep 10      # let the other workers of the lane in
  done
}

# Question 1 and 3, in the order agreed with the orchestrator (2026-11-22 07:35 UTC).
q1_matched()        { stream --factors 0.6,0.7,0.8 --decoders relay,bposd --rows matched; }
q1_matched_high()   { stream --factors 0.9,1 --decoders relay,bposd --rows matched --first "${FIRST:-500}"; }
q1_generic()        { stream --factors 0.8,0.7 --decoders relay,bposd --rows generic-2q --first "${FIRST:-500}"; }
q1_fit()            { stream --factors 0.8,0.7 --decoders relay,bposd --rows generic-fit --first "${FIRST:-500}"; }

# The pipeline check: the kit's own uniform noise at p = 0.004, decoded by this folder's loop with the kit's priors.
# The kit's README has relay-BP at 45 of 10,000 X shots and BP-OSD at 253 of 10,000.
reference() {
  "$PY" -m neutral_atom_noise.make_shots --basis X --uniform-p 0.004 --shots 2000 --seed "$SEED" --out "$S/runs/reference/shots"
  for decoder in relay bposd; do
    flock "$LOCK" nice -n 15 taskset -c "$CPUS" "$PY" -m neutral_atom_noise.run --shots "$S/runs/reference/shots/X_p0.004_seed$SEED.npz" \
      --decoder $decoder --priors uniform:0.004 --workers 4 --out "$S/runs/reference/results"
    sleep 5
  done
  "$PY" -m neutral_atom_noise.summarize --results "$S/runs/reference/results" --out "$S/runs/reference"
}

# Question 2: our loss loop against loss/'s own job on the kit's noise, 100 shots (the orchestrator's condition).
loss_check() {
  flock "$LOCK" nice -n 15 taskset -c "$CPUS" "$PY" -m neutral_atom_noise.loss_check --shots 100 --p 0.004 --s 1e-3 --seed 5300 \
    --workers 4 --out "$S/runs/loss_check"
}
