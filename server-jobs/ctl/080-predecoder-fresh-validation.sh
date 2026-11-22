#!/bin/bash
# ctl 080: for dq-predecoder, relay-BP with the kit's priors built per shot ("fresh" arm of predecoder.compare) on
# the team's EXPLORATION files data/p0.005/validation_X.npz, then data/p0.003/validation_X.npz. No test file is
# touched. From a frozen checkout of branch predecoder at 372d050, 6 single-thread workers. HARD STOP at 10:10 UTC:
# the run on the test shots (gpu 050) starts its CPU checks right after the Z training ends (about 10:07 to 10:11),
# and nothing of ours may then share its CPUs. The tool is resumable: if stopped, parts/ is copied back instead.
# Files: files/predecoder/fresh_validation_p0.00{5,3}/ on the log branch.
B=$HOME/mlBranchMidwest; SHA=372d050d021939d288a94b6abcb30a5d87308f4c; W=$B/pred-372d050
STOP_AT='2026-11-22 10:10:00'
date -u
if [ ! -d "$W" ]; then
  git -C "$B/repo" cat-file -e "$SHA^{commit}" 2>/dev/null || git -C "$B/repo" fetch -q origin predecoder
  git -C "$B/repo" worktree add -q --detach "$W" "$SHA" || { echo "the checkout could not be made"; exit 1; }
fi
[ "$(git -C "$W" rev-parse HEAD)" = "$SHA" ] || { echo "the checkout is not at $SHA"; exit 1; }
echo "checkout: $(git -C "$W" log --oneline -1)"
cd "$W/challenge-1-ml-decoder-kit" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
rc=0
for p in 0.005 0.003; do
  O=$B/runs/predecoder/fresh_validation_p$p; F=$B/logsync/files/predecoder/fresh_validation_p$p
  left=$(( $(date -u -d "$STOP_AT" +%s) - $(date +%s) ))
  if [ "$left" -lt 120 ]; then echo "p = $p: not started (under 2 minutes left before $STOP_AT UTC)"; continue; fi
  mkdir -p "$O" "$F"
  timeout "$left" python -m predecoder.compare --data "data/p$p/validation_X.npz" --arms fresh --workers 6 --out "$O"
  r=$?; echo "p = $p: predecoder.compare exit $r (124 = stopped at $STOP_AT UTC; it resumes from parts/)"; [ $r -ne 0 ] && rc=$r
  if [ -s "$O/per_shot.npz" ]; then cp "$O"/per_shot.npz "$O"/summary.json "$F/" 2>/dev/null
  else mkdir -p "$F/parts"; cp -r "$O"/parts/. "$F/parts/" 2>/dev/null; fi
  date -u
done
exit $rc
