#!/usr/bin/env bash
# What ran on this PC (luminaty) for RESULTS.md. From challenge-1-ml-decoder-kit/:
#     bash -c 'source neutral_atom_noise/runs/lane_b.sh; reference'
# The tools themselves need none of these paths: see README.md, "Run it".
#
# Compute (brief): lane B (CPUs 8-11, sim-b.lock) or lane C (LANE=C: CPUs 12-15, sim.lock); 4 processes, niced.
# The lanes are shared in turns that the orchestrator sets (from 2026-11-22 08:50 UTC lane B rotates between
# workers, one hold each of at most 19 minutes). So a call is ONE hold: `hold` stops before a new part once
# BUDGET minutes (default 12) are spent, and a part (50 relay-BP shots, 125 BP-OSD shots) takes at most about
# 4 minutes. Set BUDGET to end the hold in time. REPEAT=1 repeats holds instead, for a lane that is ours alone.
# job.py keeps finished parts, so the next call resumes where the last one stopped.
# Every call appends to neutral_atom_noise/logs/<name>.log.
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1
S=/home/sal/hackMidwest/orchestration/scratch/infleqtion-noise
PY="$S/venv/bin/python"
# lane: set LOCK and CPUS from LANE (B or C) at the time of the call
lane() {
  case "${LANE:-B}" in
    B) LOCK=/home/sal/hackMidwest/dq/run/sim-b.lock; CPUS=8-11 ;;
    C) LOCK=/home/sal/hackMidwest/dq/run/sim.lock; CPUS=12-15 ;;
    D) LOCK=/home/sal/hackMidwest/dq/run/sim-d.lock; CPUS=0-3 ;;
    *) echo "LANE must be B, C or D"; return 1 ;;
  esac
}
OUT=${OUT:-$S/runs/X_seed4100}      # every row of question 1 (X memory), so one summary holds them all
SEED=${SEED:-4100}                   # test shots
TRAIN_SEED=${TRAIN_SEED:-5100}       # train shots of the fitted generic p: never the test seed
LOGS=neutral_atom_noise/logs

hold() {
  lane || return 1
  local budget=${BUDGET:-12}
  if [ -n "${UNTIL:-}" ]; then          # UNTIL=HH:MM (UTC): end the hold by then (parts start up to 3 minutes before)
    local left=$(( ($(date -u -d "$UNTIL" +%s) - $(date -u +%s)) / 60 - 3 ))
    [ $left -lt $budget ] && budget=$left
    [ $budget -lt 1 ] && { echo "=== $(date -u +%FT%TZ) no time left before $UNTIL"; return 4; }
  fi
  echo "=== $(date -u +%FT%TZ) hold: lane ${LANE:-B}, lock $LOCK, CPUs $CPUS, head $(git rev-parse --short HEAD): job $*"
  flock "$LOCK" nice -n 15 taskset -c "$CPUS" "$PY" -m neutral_atom_noise.job --basis "${BASIS:-X}" --shots 2000 --seed $SEED \
    --train-seed $TRAIN_SEED --train-shots 250 --workers 4 --part 125 --relay-part 50 --budget-minutes "$budget" --out "$OUT" "$@"
}

# stream NAME JOB-ARGUMENTS...   one hold (REPEAT=1: holds until the rows are done); output appended to logs/NAME.log
stream() {
  local name=$1; shift
  mkdir -p "$LOGS"
  {
    while true; do
      hold "$@"; code=$?
      [ $code -eq 4 ] && break
      [ $code -eq 0 ] && { echo "=== $(date -u +%FT%TZ) $name done"; break; }
      [ $code -eq 3 ] || { echo "=== $(date -u +%FT%TZ) $name: job failed with exit code $code"; break; }
      [ "${REPEAT:-0}" = 1 ] || { echo "=== $(date -u +%FT%TZ) $name: hold over, more to do"; break; }
      sleep 20      # let the other workers of the lane in
    done
  } 2>&1 | tee -a "$LOGS/$name.log"
}

# The order agreed with the orchestrator (dq-orch-3, 2026-11-22 07:47 UTC), with the advisor's changes:
# 0. the pipeline on the kit's own noise; 1. matched rows everywhere (question 3); 2. generic rows where failures
# are: BP-OSD on all 2,000 shots at 0.7-0.9, relay-BP at 0.9 (where matched fails) and 0.8; 3. the fitted generic p.
q1_matched()        { stream q1_matched --factors 0.6,0.7,0.8 --decoders relay,bposd --rows matched; }
q1_matched_bposd()  { stream q1_matched_bposd --factors 0.9,1 --decoders bposd --rows matched; }
q1_matched_relay()  { stream q1_matched_relay --factors 0.9 --decoders relay --rows matched --first 500 &&
                      stream q1_matched_relay --factors 1 --decoders relay --rows matched --first 250; }
q1_generic_bposd()  { stream q1_generic_bposd --factors 0.8,0.9,0.7 --decoders bposd --rows generic-2q; }
q1_generic_relay()  { stream q1_generic_relay --factors 0.9 --decoders relay --rows generic-2q --first 250 &&
                      stream q1_generic_relay --factors 0.8 --decoders relay --rows generic-2q --first 500; }
q1_fit_bposd()      { stream q1_fit_bposd --factors 0.8 --decoders bposd --rows generic-fit; }
q1_fit_bposd_09()   { stream q1_fit_bposd --factors 0.9 --decoders bposd --rows generic-fit --train-seed 5101 \
                        --grid 1.96,2.744,3.8416 --first 625; }
# widen relay-BP's matched rows at 0.9 to 2,000 shots and at 1 as far as time allows (FIRST)
q1_widen_relay()    { stream q1_matched_relay --factors 0.9 --decoders relay --rows matched &&
                      stream q1_matched_relay --factors 1 --decoders relay --rows matched --first "${FIRST:-750}"; }
q1_fit_relay()      { stream q1_fit_relay --factors 0.8 --decoders relay --rows generic-fit --first 500 --train-first 100 \
                        --grid 1,1.96,3.8416; }

# Question 2 (loss), X memory at factor 0.4 (atom loss 1e-3 per atom per CX, the rate loss/ studied most), its own
# folder and seeds. Flag-aware (loss/'s `starter`, the published recipe) with matched and with generic Pauli priors;
# and matched Pauli priors without the flags (`blind`), on fewer shots because most of its shots run relay-BP to
# its cap of 18,080 iterations.
LOSS_OUT=$S/runs/X_loss_seed6100
q2_starter()  { OUT=$LOSS_OUT SEED=6100 stream q2_loss --loss --factors 0.4 --decoders relay --rows starter/matched &&
                OUT=$LOSS_OUT SEED=6100 stream q2_loss --loss --factors 0.4 --decoders relay --rows starter/generic-2q --first "${FIRST:-1000}"; }
q2_blind()    { OUT=$LOSS_OUT SEED=6100 stream q2_loss --loss --factors 0.4 --decoders relay --rows blind/matched --first "${FIRST:-150}"; }
# factor 0.6 with loss: where flag-aware decoding fails often enough to compare matched with generic Pauli priors
q2_06()       { OUT=$LOSS_OUT SEED=6100 stream q2_loss --loss --factors 0.6 --decoders relay --rows starter/matched --first 1000 &&
                OUT=$LOSS_OUT SEED=6100 stream q2_loss --loss --factors 0.6 --decoders relay --rows starter/generic-2q --first 150; }

# The pipeline on the kit's own noise at p = 0.004: this folder's sampler and decode loop, with the kit's priors.
# The kit's README has relay-BP at 45 and BP-OSD at 253 failures of 10,000 X shots.
reference() {
  lane || return 1
  mkdir -p "$LOGS"
  {
    echo "=== $(date -u +%FT%TZ) reference: lane ${LANE:-B}, head $(git rev-parse --short HEAD)"
    "$PY" -m neutral_atom_noise.make_shots --basis X --uniform-p 0.004 --shots 2000 --seed $SEED --out "$S/runs/reference/shots"
    for decoder in bposd relay; do
      flock "$LOCK" nice -n 15 taskset -c "$CPUS" "$PY" -m neutral_atom_noise.run --decoder $decoder --priors uniform:0.004 \
        --shots "$S/runs/reference/shots/X_p0.004_seed$SEED.npz" --workers 4 --out "$S/runs/reference/results"
      sleep 20
    done
    "$PY" -m neutral_atom_noise.summarize --results "$S/runs/reference/results" --out "$S/runs/reference"
    echo "=== $(date -u +%FT%TZ) reference done"
  } 2>&1 | tee -a "$LOGS/reference.log"
}

# Checks of the pipeline (advisor, 10:50 UTC): relay-BP on a second seed of the kit's own noise at p = 0.004, and
# BP-OSD on the first reference file with inflated priors (p = 0.011), to see whether BP-OSD prefers a larger p on
# any noise or only on the neutral-atom shape.
checks() {
  lane || return 1
  mkdir -p "$LOGS"
  {
    echo "=== $(date -u +%FT%TZ) checks: lane ${LANE:-B}, head $(git rev-parse --short HEAD)"
    "$PY" -m neutral_atom_noise.make_shots --basis X --uniform-p 0.004 --shots 2000 --seed 4101 --out "$S/runs/reference/shots"
    flock "$LOCK" nice -n 15 taskset -c "$CPUS" "$PY" -m neutral_atom_noise.run --decoder relay --priors uniform:0.004 \
      --shots "$S/runs/reference/shots/X_p0.004_seed4101.npz" --workers 4 --out "$S/runs/reference/results"
    flock "$LOCK" nice -n 15 taskset -c "$CPUS" "$PY" -m neutral_atom_noise.run --decoder bposd --priors uniform:0.011 \
      --shots "$S/runs/reference/shots/X_p0.004_seed4100.npz" --workers 4 --out "$S/runs/reference/results"
    "$PY" -m neutral_atom_noise.summarize --results "$S/runs/reference/results" --out "$S/runs/reference"
    echo "=== $(date -u +%FT%TZ) checks done"
  } 2>&1 | tee -a "$LOGS/checks.log"
}

# Question 2: our loss loop against loss/'s own job on the kit's noise, 100 shots (the orchestrator's condition).
loss_check() {
  lane || return 1
  mkdir -p "$LOGS"
  {
    echo "=== $(date -u +%FT%TZ) loss_check: lane ${LANE:-B}, head $(git rev-parse --short HEAD)"
    flock "$LOCK" nice -n 15 taskset -c "$CPUS" "$PY" -m neutral_atom_noise.loss_check --shots 100 --p 0.004 --s 1e-3 \
      --seed 5300 --workers 4 --out "$S/runs/loss_check"
    echo "=== $(date -u +%FT%TZ) loss_check exit code $?"
  } 2>&1 | tee -a "$LOGS/loss_check.log"
}

# From 2026-11-22 ~10:10 UTC lane B is ours alone (dq-orch-4): the question-1 streams back to back, each to the end.
lane_b_alone() {
  REPEAT=1 q1_matched
  REPEAT=1 q1_fit_bposd
  REPEAT=1 q1_matched_relay
  REPEAT=1 q1_generic_relay
}
# Lane C after dq-hierarchy's ablation logits: the fitted generic p for BP-OSD again (10:39 UTC: the first fit's best
# point was the edge of its grid, so job.py now extends the grid), the rest of BP-OSD's generic rows, the 100-shot
# loss check, question 2.
lane_c() {
  LANE=C REPEAT=1 q1_fit_bposd
  LANE=C REPEAT=1 q1_generic_bposd
  LANE=C loss_check
  LANE=C REPEAT=1 q2_starter
  LANE=C REPEAT=1 q2_blind
}

# Lane D, ours from 10:50 UTC (dq-orch-4): the pipeline checks, relay-BP's matched rows widened, question 2 at
# factor 0.6, BP-OSD's fitted generic p at factor 0.9.
lane_d() {
  LANE=D checks
  LANE=D REPEAT=1 q1_widen_relay
  LANE=D REPEAT=1 q2_06
  LANE=D REPEAT=1 q1_fit_bposd_09
}

# Z memory, if time allows (brief): its own folder and seed. Matched rows at 0.7 and 0.8, both decoders; generic at
# the same two-qubit error at 0.8 (BP-OSD on 2,000 shots, relay-BP on 500).
Z_OUT=$S/runs/Z_seed4200
z_matched()   { BASIS=Z OUT=$Z_OUT SEED=4200 stream z_memory --factors 0.7,0.8 --decoders relay,bposd --rows matched; }
z_generic()   { BASIS=Z OUT=$Z_OUT SEED=4200 stream z_memory --factors 0.8 --decoders bposd --rows generic-2q &&
                BASIS=Z OUT=$Z_OUT SEED=4200 stream z_memory --factors 0.8 --decoders relay --rows generic-2q --first 500; }
