#!/bin/bash
# gpu 047: TCBP-JOBS as an ensemble member on the TRAIN files of p = 0.005 (X and Z, 20,000 shots each), batch mode
# (predictions only), for the training of a learned selector among the candidates. The GPU is shared with the Z
# training while this runs. Files: files/tcbp_p0.005/train_{X,Z}.tcbp.npz on the log branch.
B=$HOME/mlBranchMidwest; K=$B/jobrepo-gpu/challenge-1-ml-decoder-kit
OUT=$B/runs/tcbp_p0.005; F=$B/logsync/files/tcbp_p0.005
PY=/home/blakew3/envs/tcbp-py312/bin/python
cd "$K" || exit 1
echo "checkout: $(git -C "$B/jobrepo-gpu" log --oneline -1)"; date -u
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT" "$F"
"$PY" -m hierarchy.tcbp_member --data data/p0.005/train_X.npz data/p0.005/train_Z.npz --out "$OUT" --batch
rc=$?
cp "$OUT"/train_*.tcbp.npz "$F"/ 2>/dev/null
echo "batch run exit $rc"; date -u
exit $rc
