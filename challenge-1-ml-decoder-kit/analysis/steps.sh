#!/usr/bin/env bash
# The gate measurements, step by step, exactly as they were run. From challenge-1-ml-decoder-kit/:
#
#   V=/path/to/venv S=/path/to/scratch D004=/path/to/data/p0.004 analysis/steps.sh <step>
#
# V: a Python 3.12 venv with requirements.txt (+ tesseract-decoder for the probe).
# S: a scratch folder: datasets at p = 0.003 and 0.005, the per-chunk cache, logs.
# D004: the team's committed p = 0.004 files (main: challenge-1-ml-decoder-kit/data/p0.004);
#       only train_* and validation_* are opened. Test and go files are refused by the loader.
# LOCK: the lane prefix; on luminaty lanes B and C, 8 processes.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
: "${V:?set V}" "${S:?set S}" "${D004:?set D004}"
PY="$V/bin/python"
LOCK=${LOCK:-"flock /home/sal/hackMidwest/dq/run/sim-b.lock flock /home/sal/hackMidwest/dq/run/sim.lock nice -n 15 taskset -c 8-15"}
P004="$D004/validation_X.npz $D004/validation_Z.npz $D004/train_X.npz $D004/train_Z.npz"
P003="$S/data/p0.003/explore_X.npz $S/data/p0.003/explore_Z.npz"
P005="$S/data/p0.005/explore_X.npz $S/data/p0.005/explore_Z.npz"
R=analysis/results
L=analysis/logs
mkdir -p "$S/logs" "$R" "$L"

case "${1:?step}" in
  datasets)   # the exploration sets at p = 0.003 and 0.005 (100,000 and 20,000 shots per basis)
    $PY -m analysis.datasets make --out-dir "$S/data" --manifest $R/datasets_exploration.json | tee $L/datasets_make.log
    $PY -m analysis.datasets manifest $P004 --out $R/datasets_p0.004.json | tee $L/datasets_p0.004.log ;;
  step1)      # G1 at p = 0.004: train and validation, 25,000 shots per basis
    $LOCK $PY -m analysis.g1_relay_failures run --data $P004 --cache "$S/cache" --processes 8 --log $L/g1_run_p0.004.log ;;
  summarize-g1)
    $PY -m analysis.g1_relay_failures summarize --data $P004 $P003 $P005 --cache "$S/cache" \
        --max-shots "${G1_MAX_SHOTS:-100000}" --out $R | tee $L/g1_summary.log ;;
  step2)      # the process-count check, the Tesseract probe, and G2's BP-OSD and re-roll rows at 0.004
    $PY -m analysis.g2_headroom select --pershot $R/pershot --p 0.004 --out $R/g2_selection_p0.004.npz | tee $L/g2_select_p0.004.log
    $LOCK bash -c "
      $PY -m analysis.check_processes --data $D004/validation_X.npz --cache '$S/cache' --chunks 2 3 4 5 \
          --out $R/check_processes.json > $L/check_processes.log 2>&1 &
      $PY -m analysis.tesseract_probe --data $D004/validation_X.npz --pershot $R/pershot/g1_p0.004_X_validation.npz \
          --shots 3 --timeout 60 --out $R/tesseract_probe.json > $L/tesseract_probe.log 2>&1 &   # rerun alone after a fix, see RESULTS.md
      $PY -m analysis.g2_headroom run --selection $R/g2_selection_p0.004.npz --data $P004 --cache '$S/cache' \
          --processes 4 --only bposd10 bposd_hi kit_fresh --log $L/g2_run_step2.log > /dev/null 2>&1 &
      wait" ;;
  step2b)     # G2 item 4: Tesseract on every failure and control shot at p = 0.004
    $LOCK $PY -m analysis.g2_headroom run --selection $R/g2_selection_p0.004.npz --data $P004 --cache "$S/cache" \
        --processes 8 --only tesseract --log $L/g2_run_step2b.log ;;
  step3)      # G1 at p = 0.003: the first 50,000 shots of each exploration file
    $LOCK $PY -m analysis.g1_relay_failures run --data $P003 --cache "$S/cache" --processes 8 --max-shots 50000 \
        --log $L/g1_run_p0.003.log ;;
  step4)      # G2 item 5 at p = 0.003: joint decoding, kit settings and 600 legs, every failure and 150 controls per basis
    $PY -m analysis.g2_headroom select --pershot $R/pershot --p 0.003 --control-min 0.003:150 \
        --out $R/g2_selection_p0.003.npz | tee $L/g2_select_p0.003.log
    $LOCK $PY -m analysis.g2_headroom run --selection $R/g2_selection_p0.003.npz --data $P003 --cache "$S/cache" \
        --processes 8 --only joint joint600 --log $L/g2_run_step4.log ;;
  step5)      # G2 item 5 at p = 0.004 on a seeded sample: 50 failures and 100 controls per basis
    $LOCK $PY -m analysis.g2_headroom run --selection $R/g2_selection_p0.004.npz --data $P004 --cache "$S/cache" \
        --processes 8 --only joint joint600 --caps "${CAPS:-joint@0.004=50/100/all,joint600@0.004=50/100/all}" \
        --log $L/g2_run_step5.log ;;
  step6)      # G1 at p = 0.005: the first 5,000 shots of each exploration file
    $LOCK $PY -m analysis.g1_relay_failures run --data $P005 --cache "$S/cache" --processes 8 --max-shots 5000 \
        --log $L/g1_run_p0.005.log ;;
  step7)      # G2 items 1 to 3 at p = 0.005 (a seeded 100 failures and 100 controls per basis). Items 1 and 2 at
              # p = 0.004 ran on another machine: python -m analysis.remote g2 (see RESULTS.md)
    $PY -m analysis.g2_headroom select --pershot $R/pershot --p 0.005 --max-failed 0.005:100 \
        --out $R/g2_selection_p0.005.npz | tee $L/g2_select_p0.005.log
    $LOCK $PY -m analysis.g2_headroom run --selection $R/g2_selection_p0.004.npz $R/g2_selection_p0.005.npz \
        --data $P004 $P005 --cache "$S/cache" --processes 8 --only big cand16 bposd10 bposd_hi kit_fresh --p 0.005 \
        --caps "${CAPS7:-cand16@0.004=all/all/10,cand16@0.005=all/all/5}" --log $L/g2_run_step7.log ;;
  step7b)     # G2 item 5 at p = 0.005, setting (a) only, on a seeded 10 failures and 20 controls per basis.
              # DROPPED by the orchestrator at 2026-11-22 06:0x UTC (deadline; lanes B and C needed elsewhere): not run.
    echo "step7b was dropped and not run"; exit 0
    $LOCK $PY -m analysis.g2_headroom run --selection $R/g2_selection_p0.005.npz --data $P005 --cache "$S/cache" \
        --processes 8 --only joint --caps "${CAPS7B:-joint@0.005=10/20/all}" --log $L/g2_run_step7b.log ;;
  summarize-g2)
    $PY -m analysis.g2_headroom summarize --selection $R/g2_selection_p0.00[345].npz --data $P004 $P003 $P005 \
        --cache "$S/cache" --caps "${CAPS:-joint@0.004=50/100/all,joint600@0.004=50/100/all},${CAPS7:-cand16@0.004=all/all/10,cand16@0.005=all/all/5},${CAPS7B:-joint@0.005=10/20/all}" \
        --out $R | tee $L/g2_summary.log ;;
  step8)      # G3: the 16-schedule frontier on the validation files at p = 0.004. It ran on another machine as
              # python -m analysis.remote g3 (see RESULTS.md); this is the same run under luminaty's lanes
    $LOCK $PY -m analysis.g3_schedules run --data $D004/validation_X.npz $D004/validation_Z.npz --cache "$S/cache" \
        --processes 8 --log $L/g3_run_p0.004.log ;;
  summarize-g3)
    $PY -m analysis.g3_schedules tail --pershot $R/pershot --out $R | tee $L/g3_tail.log
    $PY -m analysis.g3_schedules summarize --data $D004/validation_X.npz $D004/validation_Z.npz --cache "$S/cache" \
        --tag frontier_p0.004 --out $R | tee $L/g3_frontier_p0.004.log ;;
  *) echo "unknown step $1"; exit 2 ;;
esac
