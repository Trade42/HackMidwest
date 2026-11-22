#!/bin/bash
# gpu 040: TCBP-JOBS as an ensemble member on the VALIDATION files of p = 0.003 and p = 0.005 (X and Z, 5,000 shots
# each). Batch mode: predictions only, no latency. The GPU is shared with the Z training while this runs.
# Files: files/tcbp_p0.003/ and files/tcbp_p0.005/ on the log branch.
B=$HOME/mlBranchMidwest; K=$B/jobrepo-gpu/challenge-1-ml-decoder-kit
PY=/home/blakew3/envs/tcbp-py312/bin/python
cd "$K" || exit 1
echo "checkout: $(git -C "$B/jobrepo-gpu" log --oneline -1)"; date -u
export PYTHONDONTWRITEBYTECODE=1
worst=0
for p in 0.003 0.005; do
  OUT=$B/runs/tcbp_p$p; F=$B/logsync/files/tcbp_p$p
  mkdir -p "$OUT" "$F"
  "$PY" -m hierarchy.tcbp_member --data "data/p$p/validation_X.npz" "data/p$p/validation_Z.npz" --out "$OUT" --batch
  rc=$?
  [ $rc -ne 0 ] && worst=$rc
  cp "$OUT"/*.tcbp.npz "$F"/ 2>/dev/null
  echo "p = $p: batch run exit $rc"; date -u
done
exit $worst
