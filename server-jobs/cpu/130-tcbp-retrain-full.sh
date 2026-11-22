#!/bin/bash
# DRAFT for the orchestrator: RUN 2 of amendment 3 (task tcbp-retrain; exploration on branch tcbp-retrain; Sal 17:34 UTC: "improve it!!! dont stop").
# The TCBP pipeline with the head RETRAINED on the neutral-atom-shaped noise (factor 0.8, X memory), against the kit-trained
# head, matched and generic priors: TCBP-JOBS arms declared in neutral_atom_noise/tcbp_retrain_declared.md (7d8c80e7) and, after
# a cut for a deadline (amendment 1, 163ff2ba), restored in full (amendment 2, bf5cf0a4); run 1 was cpu/120 (amendment 1); this is run 2 (amendment 3,
# cb739fbe), sent whatever run 1 shows. All declared before any number. Two shot files:
# seed 4800 (4,000 shots, primary, fresh; arms m, g, h, hm, r) and seed 4100 (2,000 shots, phase 1's file, secondary; m, h, hm, r).
# Two pythons, as cpu/090: the runner's (torch + CUDA) for the network in fp32, the teammates' TCBP env (read and run only) for
# TCBP-JOBS, one process, 1 CPU thread. Writes only under ~/mlBranchMidwest; the log folder holds no logits.
# Fill in A (the laptop-artifacts commit holding tcbp-retrain-full/ and predecoder_X_na0.8_s1/) and STOP (declared: the job's start
# + 45 minutes; no new block after it). The decode needs about 35 minutes; a stop keeps each task's longest prefix of whole blocks.
B=$HOME/mlBranchMidwest
RSHA=cb739fbe9b2d76742bdd15f9b961b450e981b788   # tcbp-retrain: predecoder/evaluate.py --base-priors, neutral_atom_noise/tcbp_retrain.py
A=3ed14b5b950f7355d2db8b4b46cd9354c1c42f52                     # laptop-artifacts: tcbp-retrain-full/ (inputs, plan.json) and the two heads and the trunk
RHEAD=b42d71d584c95da84aa4f086c8c23fa49d846f559407ae11fd7a4d89abf091fa
STOP=$(date -u -d "+45 min" +%Y-%m-%dT%H:%M:%SZ)                          # the job's start + 45 minutes, e.g. 2026-11-22T19:25:00Z
WR=$B/tcbp-retrain-r${RSHA:0:7}
R=$B/runs/tcbp-retrain-full; IN=$R/in; O=$R/out-${RSHA:0:7}; F=$B/logsync/files/tcbp-retrain-full
[ -e "$O" ] && { echo "$O exists: refusing"; exit 1; }
[ -e "$F" ] && { echo "$F exists: refusing"; exit 1; }
date -u
fetch() { for i in 1 2 3; do git -C "$B/repo" fetch -q origin "$1" && return 0; sleep 20; done; return 1; }
git -C "$B/repo" cat-file -e "$RSHA^{commit}" 2>/dev/null || fetch tcbp-retrain || { echo "cannot fetch tcbp-retrain"; exit 1; }
git -C "$B/repo" cat-file -e "$A^{commit}" 2>/dev/null || fetch laptop-artifacts || { echo "cannot fetch laptop-artifacts"; exit 1; }
[ -d "$WR" ] || git -C "$B/repo" worktree add -q --detach "$WR" "$RSHA" || exit 1
[ "$(git -C "$WR" rev-parse HEAD)" = "$RSHA" ] || { echo "checkout is not $RSHA"; exit 1; }
mkdir -p "$IN/seed4800" "$IN/seed4100" "$F" && mkdir "$O" || exit 1
for f in SHA256SUMS plan.json \
  seed4800/X_f0.8_seed4800.npz seed4800/X_f0.8_seed4800_kitformat_p0.0326371.npz seed4800/priors_generic.npy \
  seed4800/priors_matched.npy seed4800/transfer.json \
  seed4100/X_f0.8_seed4100_kitformat_p0.0326371.npz seed4100/priors_generic.npy seed4100/priors_matched.npy \
  seed4100/transfer.json; do
  git -C "$B/repo" cat-file blob "$A:tcbp-retrain-full/$f" > "$IN/$f" || exit 1
done
git -C "$B/repo" cat-file blob "$A:predecoder_X_s1/head.pt" > "$IN/head_X.pt" || exit 1
git -C "$B/repo" cat-file blob "$A:predecoder_X_na0.8_s1/head.pt" > "$IN/head_X_na0.8.pt" || exit 1
git -C "$B/repo" cat-file blob "$A:cascade_X_s1/checkpoint_final_step120000.pt" > "$IN/trunk_X.pt" || exit 1
(cd "$IN" && sha256sum -c SHA256SUMS && printf '%s\n' \
  "38ab667e251c4e7b991e26e9b660c12e80a2d1206589e638c5b372806a648a0d  head_X.pt" \
  "$RHEAD  head_X_na0.8.pt" \
  "d8d1c70f552233300f1c10997b69292092042eb8517f5b6526b607c1f6dc4933  trunk_X.pt" \
  "0b3ec31626364f36d336ba98f5842e2f20e44b23d35b0026425bef7eefc2b9a1  plan.json" | sha256sum -c -) || exit 1
(cd "$IN" && printf '%s\n' \
  "b2c023bbf24108ee771d6ebea4405971888164732aad0ab0da53be4848bc1162  seed4800/X_f0.8_seed4800.npz" \
  "32fc7703d19ba4e7e1c1c058a459afe1356f0662d789afb030d07353324ac94a  seed4800/X_f0.8_seed4800_kitformat_p0.0326371.npz" \
  "a251dfe065d8c4ab427671930e6e13f66182a1144acee6980f3b9d3329f11141  seed4100/X_f0.8_seed4100_kitformat_p0.0326371.npz" \
  "b8d59cb850ee6842011283d559b33e560c07ceecd2edfed2f10617ce6557a9ea  seed4800/priors_generic.npy" \
  "45704fcb9f2bde6fdbc6a7f0fa975f7e343c5283179e65a4cf0af45a741fc3ac  seed4800/priors_matched.npy" \
  "b8d59cb850ee6842011283d559b33e560c07ceecd2edfed2f10617ce6557a9ea  seed4100/priors_generic.npy" \
  "45704fcb9f2bde6fdbc6a7f0fa975f7e343c5283179e65a4cf0af45a741fc3ac  seed4100/priors_matched.npy" | sha256sum -c -) || exit 1
cd "$WR/challenge-1-ml-decoder-kit" || exit 1
echo "7697010678a369c77949a6a4233a3087824fa4d2753bcadc4869f2eaee91b0d8  predecoder/logs/fit_tcbp_X_p0.005.json" | sha256sum -c - || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYT=$(command -v python); PY=/home/blakew3/envs/tcbp-py312/bin/python; echo "PYT=$PYT"; date -u
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
# 1. the network: three logits folders per task (task names as in plan.json), fp32 on the GPU, one thread
for t in seed4800 seed4100; do
  D=$IN/$t/X_f0.8_${t}_kitformat_p0.0326371.npz; L=$O/logits/f0.8-$t
  "$PYT" -m predecoder.evaluate --head "$IN/head_X.pt" --trunk "$IN/trunk_X.pt" --data "$D" --device cuda --precision fp32 \
    --threads 1 --time-shots 100 --out "$L/kit" > "$F/evaluate_${t}_kit.log" 2>&1; echo "evaluate $t kit exit $?"
  "$PYT" -m predecoder.evaluate --head "$IN/head_X.pt" --trunk "$IN/trunk_X.pt" --data "$D" --device cuda --precision fp32 \
    --threads 1 --time-shots 100 --base-priors "$IN/$t/priors_matched.npy" --out "$L/kit-matched" \
    > "$F/evaluate_${t}_kit-matched.log" 2>&1; echo "evaluate $t kit-matched exit $?"
  "$PYT" -m predecoder.evaluate --head "$IN/head_X_na0.8.pt" --trunk "$IN/trunk_X.pt" --data "$D" --device cuda --precision fp32 \
    --threads 1 --time-shots 100 --base-priors "$IN/$t/priors_matched.npy" --out "$L/retrained" \
    > "$F/evaluate_${t}_retrained.log" 2>&1; echo "evaluate $t retrained exit $?"
  for a in kit kit-matched retrained; do cp "$L/$a/meta.json" "$F/logits_meta_${t}_$a.json" 2>/dev/null; done
done
date -u
# 2. TCBP-JOBS: the declared arms, blocks of 100, tasks in turn, one model; no new block after $STOP
PYTHONPATH="$WR/challenge-1-ml-decoder-kit" "$PY" -m neutral_atom_noise.tcbp_retrain decode --plan "$IN/plan.json" \
  --root "$IN" --logits-root "$O/logits" --settings predecoder/logs/fit_tcbp_X_p0.005.json --stop-at "$STOP" \
  --out "$O/decode" > "$F/decode.log" 2>&1; rc=$?; echo "decode exit $rc"
for t in f0.8-seed4800 f0.8-seed4100; do
  mkdir -p "$F/$t" && cp "$O/decode/$t/per_shot.npz" "$O/decode/$t/summary.json" "$F/$t/" 2>/dev/null
done
tail -3 "$F"/evaluate_*.log "$F/decode.log"; date -u; exit $rc
