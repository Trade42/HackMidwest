#!/bin/bash
# ctl 130: branch predecoder-tcbp, job 3: VALIDATION of TCBP-JOBS with the head's per-shot priors, knobs frozen before
# this run (log-odds alpha 0.4, eps 1e-4, predecoder/logs/fit_tcbp_X_p0.005.json at 86259291). data/p0.005/validation_X
# (5,000 shots): the head arm, then the fresh arm, then the batch check (the kit's priors built once, decode_batch, on
# the same model). Runs on the A100 beside gpu 075's tail (the slice has room for two); 1 CPU thread; about 12 min.
# Inputs unpacked and checked by gpu 070 are re-checked. Writes only under ~/mlBranchMidwest; the log is mirrored.
B=$HOME/mlBranchMidwest; SHA=8625929149b7af31a4d06fb52fe93193511961f1; W=$B/pred-tcbp-8625929
if [ ! -d "$W" ]; then
  git -C "$B/repo" cat-file -e "$SHA^{commit}" 2>/dev/null || git -C "$B/repo" fetch -q origin predecoder-tcbp
  git -C "$B/repo" worktree add -q --detach "$W" "$SHA" || exit 1
fi
[ "$(git -C "$W" rev-parse HEAD)" = "$SHA" ] || { echo "checkout is not $SHA"; exit 1; }
IN=$B/runs/predecoder-tcbp/in
(cd "$IN/validation_X_p0.005" && sha256sum -c ../INNER_SHA256SUMS_validation) || { echo "validation logits do not match"; exit 1; }
cd "$W/challenge-1-ml-decoder-kit" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/blakew3/envs/tcbp-py312/bin/python; O=$B/runs/predecoder-tcbp; F=$B/logsync/files/predecoder-tcbp
mkdir -p "$O" "$F"
sha256sum predecoder/logs/fit_tcbp_X_p0.005.json; date -u
"$PY" -m predecoder.tcbp decode --data data/p0.005/validation_X.npz --logits "$IN/validation_X_p0.005" \
  --settings predecoder/logs/fit_tcbp_X_p0.005.json --arms head fresh --batch-check --out "$O/validation_X_p0.005_job3" > "$O/validation_head.log" 2>&1 &
pid=$!
while kill -0 "$pid" 2>/dev/null; do cp "$O/validation_head.log" "$F/validation_head.log" 2>/dev/null; sleep 60; done
wait "$pid"; rc=$?; echo "decode exit $rc"
cp "$O/validation_head.log" "$F/"
cp "$O/validation_X_p0.005_job3/per_shot.npz" "$F/validation_X_p0.005_per_shot.npz" 2>/dev/null
cp "$O/validation_X_p0.005_job3/summary.json" "$F/validation_X_p0.005_summary.json" 2>/dev/null
tail -8 "$O/validation_head.log"; date -u; exit $rc
