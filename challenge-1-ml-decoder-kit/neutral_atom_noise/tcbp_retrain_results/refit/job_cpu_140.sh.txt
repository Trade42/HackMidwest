#!/bin/bash
# cpu 140 (the cpu lane is a queue; after cpu 130): task tcbp-retrain, the REFIT (exploration on branch tcbp-retrain; Sal via dq-orch-6, 18:46 UTC:
# "accuracy pushed"). Declared in neutral_atom_noise/tcbp_retrain_refit_declared.md (2144eac3) before any number: the retrained
# head's knobs refitted on TRAIN shots (seed 5800, 1,500 shots; arm m once, then alpha 0.1-1.0 x eps 1e-4, 1e-6 with the matched
# base), then ONE test of the chosen knobs on fresh shots (seed 4900, 4,000 shots; arms m, r at the frozen knobs, rf at the chosen
# ones; primary rf vs m). Two pythons, as cpu/090: the runner's (torch + CUDA) for the network in fp32, the teammates' TCBP env
# (read and run only) for TCBP-JOBS, one process, 1 CPU thread. Writes only under ~/mlBranchMidwest; the log folder holds no logits.
# Fill in A (the laptop-artifacts commit holding tcbp-retrain-refit/ and predecoder_X_na0.8_s1/). STOP is computed at the start.
B=$HOME/mlBranchMidwest
STOP=$(date -u -d "+60 min" +%Y-%m-%dT%H:%M:%SZ)   # declared: the job's start + 60 minutes, for the fit and the test
RSHA=9a1aaa3735bb6c3072010e2faf5e7612f8516273   # tcbp-retrain: tcbp_retrain.py fit and the rf arm; evaluate --base-priors
A=ca34c0f17726392d88f177db2bd36b1718ee9a58                     # laptop-artifacts: tcbp-retrain-refit/ (inputs, two plans) and the head and trunk
RHEAD=b42d71d584c95da84aa4f086c8c23fa49d846f559407ae11fd7a4d89abf091fa
WR=$B/tcbp-retrain-r${RSHA:0:7}
R=$B/runs/tcbp-retrain-refit; IN=$R/in; O=$R/out-${RSHA:0:7}; F=$B/logsync/files/tcbp-retrain-refit
[ -e "$O" ] && { echo "$O exists: refusing"; exit 1; }
[ -e "$F" ] && { echo "$F exists: refusing"; exit 1; }
date -u; echo "STOP=$STOP"
fetch() { for i in 1 2 3; do git -C "$B/repo" fetch -q origin "$1" && return 0; sleep 20; done; return 1; }
git -C "$B/repo" cat-file -e "$RSHA^{commit}" 2>/dev/null || fetch tcbp-retrain || { echo "cannot fetch tcbp-retrain"; exit 1; }
git -C "$B/repo" cat-file -e "$A^{commit}" 2>/dev/null || fetch laptop-artifacts || { echo "cannot fetch laptop-artifacts"; exit 1; }
[ -d "$WR" ] || git -C "$B/repo" worktree add -q --detach "$WR" "$RSHA" || exit 1
[ "$(git -C "$WR" rev-parse HEAD)" = "$RSHA" ] || { echo "checkout is not $RSHA"; exit 1; }
mkdir -p "$IN/train5800" "$IN/test4900" "$F" && mkdir "$O" || exit 1
for f in SHA256SUMS plan_train.json plan_test.json \
  train5800/X_f0.8_seed5800.npz train5800/X_f0.8_seed5800_kitformat_p0.0326371.npz train5800/priors_generic.npy \
  train5800/priors_matched.npy train5800/transfer.json \
  test4900/X_f0.8_seed4900.npz test4900/X_f0.8_seed4900_kitformat_p0.0326371.npz test4900/priors_generic.npy \
  test4900/priors_matched.npy test4900/transfer.json; do
  git -C "$B/repo" cat-file blob "$A:tcbp-retrain-refit/$f" > "$IN/$f" || exit 1
done
git -C "$B/repo" cat-file blob "$A:predecoder_X_na0.8_s1/head.pt" > "$IN/head_X_na0.8.pt" || exit 1
git -C "$B/repo" cat-file blob "$A:cascade_X_s1/checkpoint_final_step120000.pt" > "$IN/trunk_X.pt" || exit 1
(cd "$IN" && sha256sum -c SHA256SUMS && printf '%s\n' \
  "$RHEAD  head_X_na0.8.pt" \
  "d8d1c70f552233300f1c10997b69292092042eb8517f5b6526b607c1f6dc4933  trunk_X.pt" \
  "d94a67d144a1ea09d16eadfdedf1b92ec700e455c3c47178224683e3ab6b201a  plan_train.json" \
  "4e719fe201ac0a9ad2436a0e5f85c260ae791110275c9f65066039083ef860b4  plan_test.json" \
  "cb298991b34b44b44ff6a54651ccc185c186a310131fd8d698947cbd34ad804b  train5800/X_f0.8_seed5800.npz" \
  "8f61f30b65740f6306be4a846de77d7addc13633de1ee424b690e19bad072dc9  test4900/X_f0.8_seed4900.npz" \
  "45704fcb9f2bde6fdbc6a7f0fa975f7e343c5283179e65a4cf0af45a741fc3ac  train5800/priors_matched.npy" \
  "45704fcb9f2bde6fdbc6a7f0fa975f7e343c5283179e65a4cf0af45a741fc3ac  test4900/priors_matched.npy" | sha256sum -c -) || exit 1
cd "$WR/challenge-1-ml-decoder-kit" || exit 1
echo "7697010678a369c77949a6a4233a3087824fa4d2753bcadc4869f2eaee91b0d8  predecoder/logs/fit_tcbp_X_p0.005.json" | sha256sum -c - || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYT=$(command -v python); PY=/home/blakew3/envs/tcbp-py312/bin/python; echo "PYT=$PYT"; date -u
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
# 1. the network: the retrained head on the matched base it was trained with, fp32 on the GPU, one thread, both files
for x in train5800:f0.8-seed5800-train:5800 test4900:f0.8-seed4900:4900; do
  d=${x%%:*}; rest=${x#*:}; task=${rest%%:*}; seed=${rest##*:}
  "$PYT" -m predecoder.evaluate --head "$IN/head_X_na0.8.pt" --trunk "$IN/trunk_X.pt" \
    --data "$IN/$d/X_f0.8_seed${seed}_kitformat_p0.0326371.npz" --device cuda --precision fp32 --threads 1 --time-shots 100 \
    --base-priors "$IN/$d/priors_matched.npy" --out "$O/logits/$task/retrained" > "$F/evaluate_$d.log" 2>&1
  echo "evaluate $d exit $?"; cp "$O/logits/$task/retrained/meta.json" "$F/logits_meta_$d.json" 2>/dev/null
done
date -u
# 2. the fit on the train shots: arm m once, then the 14 settings of FIT_GRID; the declared rule chooses
PYTHONPATH="$WR/challenge-1-ml-decoder-kit" "$PY" -m neutral_atom_noise.tcbp_retrain fit --plan "$IN/plan_train.json" \
  --root "$IN" --logits-root "$O/logits" --stop-at "$STOP" --out "$O/fit/fit.json" > "$F/fit.log" 2>&1; rc=$?
echo "fit exit $rc"; mkdir -p "$O/fit"; cp "$O/fit/fit.json" "$F/" 2>/dev/null; date -u
# 3. ONE test of the chosen knobs on the test shots: arms m, r (frozen knobs), rf (the fit's); no new block after $STOP
if [ $rc -eq 0 ] && "$PY" -c "import json,sys; sys.exit(json.load(open('$O/fit/fit.json'))['chosen'] is None)"; then
  PYTHONPATH="$WR/challenge-1-ml-decoder-kit" "$PY" -m neutral_atom_noise.tcbp_retrain decode --plan "$IN/plan_test.json" \
    --root "$IN" --logits-root "$O/logits" --settings predecoder/logs/fit_tcbp_X_p0.005.json --stop-at "$STOP" \
    --refit "$O/fit/fit.json" --out "$O/decode" > "$F/decode.log" 2>&1; rc=$?; echo "decode exit $rc"
  mkdir -p "$F/f0.8-seed4900" && cp "$O/decode/f0.8-seed4900/per_shot.npz" "$O/decode/f0.8-seed4900/summary.json" "$F/f0.8-seed4900/" 2>/dev/null
else
  echo "the fit chose nothing: no test (declared)"; rc=1
fi
tail -3 "$F"/*.log; date -u; exit $rc
