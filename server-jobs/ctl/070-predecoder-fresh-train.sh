#!/bin/bash
# ctl 070: for dq-predecoder's prior fit, relay-BP with the kit's priors built per shot ("fresh" arm of
# predecoder.compare) on the team's EXPLORATION file data/p0.004/train_X.npz (20,000 shots). No test file is
# touched. From a frozen checkout of branch predecoder at c2a6db5, 6 single-thread workers (about 15 to 30 minutes;
# the notebook used about 3 of its 12 CPUs at 09:18 UTC). It is stopped at 10:10 UTC at the latest so the run on the
# test shots has the CPUs; the tool is resumable (parts/). Files: files/predecoder/fresh_train_p0.004/ on the log branch.
B=$HOME/mlBranchMidwest; SHA=c2a6db519c35e3d4deab5a947eaefa0c8c271520; W=$B/pred-c2a6db5
O=$B/runs/predecoder/fresh_train_p0.004; F=$B/logsync/files/predecoder/fresh_train_p0.004
date -u
if [ ! -d "$W" ]; then
  git -C "$B/repo" cat-file -e "$SHA^{commit}" 2>/dev/null || git -C "$B/repo" fetch -q origin predecoder
  git -C "$B/repo" worktree add -q --detach "$W" "$SHA" || { echo "the checkout could not be made"; exit 1; }
fi
[ "$(git -C "$W" rev-parse HEAD)" = "$SHA" ] || { echo "the checkout is not at $SHA"; exit 1; }
echo "checkout: $(git -C "$W" log --oneline -1)"
cd "$W/challenge-1-ml-decoder-kit" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
left=$(( $(date -u -d '2026-11-22 10:10:00' +%s) - $(date +%s) ))
[ "$left" -gt 300 ] || { echo "too late to start (the run on the test shots needs the CPUs)"; exit 0; }
mkdir -p "$O" "$F"
timeout "$left" python -m predecoder.compare --data data/p0.004/train_X.npz --arms fresh --workers 6 --out "$O"
rc=$?
echo "predecoder.compare exit $rc (124 = stopped at 10:10 UTC; it resumes from parts/ if started again)"
cp "$O"/per_shot.npz "$O"/summary.json "$F/" 2>/dev/null
ls -la "$O" | head -20
date -u
exit $rc
