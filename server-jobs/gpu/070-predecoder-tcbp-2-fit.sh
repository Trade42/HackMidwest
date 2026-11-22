#!/bin/bash
# gpu 070: branch predecoder-tcbp, job 2: the mixing knobs of the head's priors refitted for TCBP-JOBS, on TRAIN subsets
# only (data/p0.005/train_X.npz: the 390 shots TCBP-JOBS fails + 2,000 it gets right, seed 5; declared at 14:48 UTC
# before any number). The A100 and 1 CPU thread; about 30 minutes; resumable from fit_tcbp_X_p0.005.parts/.
# Inputs: luminaty's CPU float32 logits and the subset files, from laptop-artifacts at a literal commit, every sha256
# checked. Our code from a detached checkout of predecoder-tcbp at a literal commit, in the teammates' TCBP env (read
# and run only). Writes only under ~/mlBranchMidwest. fit.log is copied to the log branch every minute while it runs.
B=$HOME/mlBranchMidwest; SHA=abb056077e6d285a5453403fc5de4d34020b7ac5; W=$B/pred-tcbp-abb0560; C=666f10bf27939a74df6c6c68bf332af7a8a6aa77
if [ ! -d "$W" ]; then
  git -C "$B/repo" cat-file -e "$SHA^{commit}" 2>/dev/null || git -C "$B/repo" fetch -q origin predecoder-tcbp
  git -C "$B/repo" worktree add -q --detach "$W" "$SHA" || exit 1
fi
[ "$(git -C "$W" rev-parse HEAD)" = "$SHA" ] || { echo "checkout is not $SHA"; exit 1; }
git -C "$B/repo" cat-file -e "$C^{commit}" 2>/dev/null || git -C "$B/repo" fetch -q origin laptop-artifacts
IN=$B/runs/predecoder-tcbp/in; mkdir -p "$IN"; cd "$IN" || exit 1
for f in SHA256SUMS INNER_SHA256SUMS_validation INNER_SHA256SUMS_train train_X_T_failed.npz train_X_T_right.npz train_X_T_subsets.json logits_train_X_T_p0.005.tar.gz logits_validation_X_p0.005.tar.gz; do
  git -C "$B/repo" show "$C:predecoder-tcbp/$f" > "$f" || { echo "cannot read $f at $C"; exit 1; }
done
sha256sum -c SHA256SUMS || { echo "a transferred file does not match its sha256"; exit 1; }
tar xzf logits_train_X_T_p0.005.tar.gz && sha256sum -c INNER_SHA256SUMS_train || exit 1
mkdir -p validation_X_p0.005 && tar xzf logits_validation_X_p0.005.tar.gz -C validation_X_p0.005 && (cd validation_X_p0.005 && sha256sum -c ../INNER_SHA256SUMS_validation) || exit 1
cd "$W/challenge-1-ml-decoder-kit" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/blakew3/envs/tcbp-py312/bin/python; O=$B/runs/predecoder-tcbp; F=$B/logsync/files/predecoder-tcbp
mkdir -p "$O" "$F"; date -u
"$PY" -m predecoder.tcbp fit --data "$IN/train_X_T_failed.npz" "$IN/train_X_T_right.npz" --weights 1 9.805 \
  --logits "$IN/train_X_T_failed" "$IN/train_X_T_right" --out "$O/fit_tcbp_X_p0.005.json" > "$O/fit.log" 2>&1 &
pid=$!
while kill -0 "$pid" 2>/dev/null; do cp "$O/fit.log" "$F/fit.log" 2>/dev/null; sleep 60; done
wait "$pid"; rc=$?; echo "fit exit $rc"
cp "$O/fit.log" "$F/"; cp "$O/fit_tcbp_X_p0.005.json" "$F/" 2>/dev/null
tar czf "$F/fit_tcbp_X_p0.005.parts.tar.gz" -C "$O" fit_tcbp_X_p0.005.parts 2>/dev/null
tail -20 "$O/fit.log"; date -u; exit $rc
