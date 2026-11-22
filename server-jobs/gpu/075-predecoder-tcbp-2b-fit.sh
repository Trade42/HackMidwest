#!/bin/bash
# gpu 075: branch predecoder-tcbp, job 2b: resumes the knob fit of gpu 070, which stopped because TCBP-JOBS refuses a
# prior above 0.5. Commit 83563dbe caps every shot's priors at 0.5 after mixing (declared before any further fit
# number); the rows already done are unchanged by the cap and their parts are reused. TRAIN subsets only (p = 0.005).
# The inputs gpu 070 unpacked and checked are re-checked here. The A100 and 1 CPU thread; about 20 minutes.
# Writes only under ~/mlBranchMidwest. fit.log is copied to the log branch every minute while it runs.
B=$HOME/mlBranchMidwest; SHA=83563dbe5c8395651a1513a4d8c0c1a9e796e43d; W=$B/pred-tcbp-83563db
if [ ! -d "$W" ]; then
  git -C "$B/repo" cat-file -e "$SHA^{commit}" 2>/dev/null || git -C "$B/repo" fetch -q origin predecoder-tcbp
  git -C "$B/repo" worktree add -q --detach "$W" "$SHA" || exit 1
fi
[ "$(git -C "$W" rev-parse HEAD)" = "$SHA" ] || { echo "checkout is not $SHA"; exit 1; }
IN=$B/runs/predecoder-tcbp/in; cd "$IN" || exit 1
sha256sum -c SHA256SUMS && sha256sum -c INNER_SHA256SUMS_train && (cd validation_X_p0.005 && sha256sum -c ../INNER_SHA256SUMS_validation) || { echo "inputs do not match"; exit 1; }
cd "$W/challenge-1-ml-decoder-kit" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/blakew3/envs/tcbp-py312/bin/python; O=$B/runs/predecoder-tcbp; F=$B/logsync/files/predecoder-tcbp
mkdir -p "$O" "$F"
[ -f "$O/fit.log" ] && mv "$O/fit.log" "$O/fit_attempt1.log" && cp "$O/fit_attempt1.log" "$F/"
echo "parts reused: $(ls "$O/fit_tcbp_X_p0.005.parts" 2>/dev/null | wc -l)"; date -u
"$PY" -m predecoder.tcbp fit --data "$IN/train_X_T_failed.npz" "$IN/train_X_T_right.npz" --weights 1 9.805 \
  --logits "$IN/train_X_T_failed" "$IN/train_X_T_right" --out "$O/fit_tcbp_X_p0.005.json" > "$O/fit.log" 2>&1 &
pid=$!
while kill -0 "$pid" 2>/dev/null; do cp "$O/fit.log" "$F/fit.log" 2>/dev/null; sleep 60; done
wait "$pid"; rc=$?; echo "fit exit $rc"
cp "$O/fit.log" "$F/"; cp "$O/fit_tcbp_X_p0.005.json" "$F/" 2>/dev/null
tar czf "$F/fit_tcbp_X_p0.005.parts.tar.gz" -C "$O" fit_tcbp_X_p0.005.parts 2>/dev/null
tail -20 "$O/fit.log"; date -u; exit $rc
