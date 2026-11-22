#!/bin/bash
# cpu 070 (the queue only; it uses the A100): branch predecoder-tcbp, job 4, the second lever (Sal, 15:28 UTC): the
# network on the A100 in float32 (no fp16 autocast, no TF32), timed one shot at a time on the first 1,000 shots of
# data/p0.005/validation_X.npz; its logits compared with luminaty's CPU float32 logits; those 1,000 shots decoded with
# TCBP-JOBS and the frozen knobs from the GPU logits; an fp16 run for its time only. VALIDATION shots only. Two
# pythons: the runner's (torch + CUDA) for the network, the teammates' TCBP env (read and run only) for TCBP-JOBS.
# Writes only under ~/mlBranchMidwest; the folder copied to the log branch holds no logits.
B=$HOME/mlBranchMidwest; SHA=8c25c16dd6fd04cc8e248fb2163700e68519b566; W=$B/pred-tcbp-8c25c16; C=666f10bf27939a74df6c6c68bf332af7a8a6aa77
if [ ! -d "$W" ]; then
  git -C "$B/repo" cat-file -e "$SHA^{commit}" 2>/dev/null || git -C "$B/repo" fetch -q origin predecoder-tcbp
  git -C "$B/repo" worktree add -q --detach "$W" "$SHA" || exit 1
fi
[ "$(git -C "$W" rev-parse HEAD)" = "$SHA" ] || { echo "checkout is not $SHA"; exit 1; }
git -C "$B/repo" cat-file -e "$C^{commit}" 2>/dev/null || git -C "$B/repo" fetch -q origin laptop-artifacts
IN=$B/runs/predecoder-tcbp/in; O=$B/runs/predecoder-tcbp/gpu_network; F=$B/logsync/files/predecoder-tcbp/gpu_network; mkdir -p "$O" "$F"
git -C "$B/repo" show "$C:predecoder_X_s1/head.pt" > "$IN/head_X.pt"; git -C "$B/repo" show "$C:cascade_X_s1/checkpoint_final_step120000.pt" > "$IN/trunk_X.pt"
(cd "$IN" && printf '%s\n' "38ab667e251c4e7b991e26e9b660c12e80a2d1206589e638c5b372806a648a0d  head_X.pt" "d8d1c70f552233300f1c10997b69292092042eb8517f5b6526b607c1f6dc4933  trunk_X.pt" | sha256sum -c -) || exit 1
(cd "$IN/validation_X_p0.005" && sha256sum -c ../INNER_SHA256SUMS_validation) || exit 1
cd "$W/challenge-1-ml-decoder-kit" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYT=$(command -v python); PY=/home/blakew3/envs/tcbp-py312/bin/python; echo "PYT=$PYT"; date -u
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
"$PYT" -m predecoder.evaluate --head "$IN/head_X.pt" --trunk "$IN/trunk_X.pt" --data data/p0.005/validation_X.npz --limit 1000 --device cuda --precision fp32 --threads 1 --time-shots 1000 --out "$O/logits_val1000_fp32" > "$O/evaluate_fp32.log" 2>&1; echo "evaluate fp32 exit $?"
"$PY" -m predecoder.tcbp logits-diff --a "$IN/validation_X_p0.005" --b "$O/logits_val1000_fp32" --limit 1000 --out "$O/logits_diff_fp32.json" > "$O/logits_diff_fp32.log" 2>&1; echo "diff exit $?"
"$PY" -m predecoder.tcbp decode --data data/p0.005/validation_X.npz --logits "$O/logits_val1000_fp32" --settings predecoder/logs/fit_tcbp_X_p0.005.json --arms head --limit 1000 --out "$O/decode_val1000_fp32" > "$O/decode_fp32.log" 2>&1; echo "decode exit $?"
"$PYT" -m predecoder.evaluate --head "$IN/head_X.pt" --trunk "$IN/trunk_X.pt" --data data/p0.005/validation_X.npz --limit 1000 --device cuda --precision auto --threads 1 --time-shots 1000 --out "$O/logits_val1000_fp16" > "$O/evaluate_fp16.log" 2>&1; echo "evaluate fp16 (for the time only) exit $?"
"$PY" -m predecoder.tcbp logits-diff --a "$IN/validation_X_p0.005" --b "$O/logits_val1000_fp16" --limit 1000 --out "$O/logits_diff_fp16.json" > "$O/logits_diff_fp16.log" 2>&1
cp "$O"/*.log "$O"/*.json "$F/" 2>/dev/null; cp "$O/logits_val1000_fp32/meta.json" "$F/meta_fp32.json" 2>/dev/null; cp "$O/logits_val1000_fp16/meta.json" "$F/meta_fp16.json" 2>/dev/null
cp "$O/decode_val1000_fp32/per_shot.npz" "$F/decode_val1000_fp32_per_shot.npz" 2>/dev/null; cp "$O/decode_val1000_fp32/summary.json" "$F/decode_val1000_fp32_summary.json" 2>/dev/null
tail -3 "$O"/*.log; date -u
