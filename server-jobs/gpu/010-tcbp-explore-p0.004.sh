#!/bin/bash
# 010: TCBP-JOBS as an ensemble member on the p = 0.004 exploration files (train and validation, X and Z).
# Batch mode: predictions only, no latency. The GPU is shared with the Z training while this runs.
B=$HOME/mlBranchMidwest; K=$B/jobrepo-gpu/challenge-1-ml-decoder-kit
OUT=$B/runs/tcbp_p0.004; F=$B/logsync/files/tcbp_p0.004
PY=/home/blakew3/envs/tcbp-py312/bin/python
FILES="data/p0.004/validation_X.npz data/p0.004/train_X.npz data/p0.004/validation_Z.npz data/p0.004/train_Z.npz"
cd "$K" || exit 1
echo "checkout: $(git -C "$B/jobrepo-gpu" log --oneline -1)"; date -u
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT" "$F"
"$PY" -m hierarchy.tcbp_member --data $FILES --out "$OUT" --batch
rc=$?
cp "$OUT"/*.tcbp.npz "$F"/ 2>/dev/null
echo "batch run exit $rc"; date -u
if [ $rc -ne 0 ]; then
  echo "the batch run failed: the serial run follows (its times are inflated by the shared GPU)"
  mkdir -p "${OUT}_serial" "${F}_serial"
  "$PY" -m hierarchy.tcbp_member --data $FILES --out "${OUT}_serial"
  rc=$?
  cp "${OUT}_serial"/*.tcbp.npz "${F}_serial"/ 2>/dev/null
  echo "serial run exit $rc"; date -u
fi
exit $rc
