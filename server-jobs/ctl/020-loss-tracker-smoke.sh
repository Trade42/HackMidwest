#!/bin/bash
# ctl 020: a 2-minute smoke run of the loss tracker's job on the server (4 workers), to learn whether Clifft gives the
# same shots here as on luminaty for the same seed. There: dataset sha256 d69bc1518bc4..., failures starter 0, exact 2,
# tracker 0, oracle-round 0 of 40. It runs after the joint-decoding job (the lane's order).
# Files: files/loss_tracker/smoke_seed9100/ on the log branch.
B=$HOME/mlBranchMidwest; K=$B/jobrepo-ctl/challenge-1-ml-decoder-kit; OUT=$B/runs/loss_tracker/smoke_seed9100
F=$B/logsync/files/loss_tracker/smoke_seed9100
cd "$K" || exit 1
date -u; echo "checkout: $(git -C "$B/jobrepo-ctl" log --oneline -1)"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m loss_tracker.job --seed 9100 --shots 200 --decode-shots 40 --part 20 --workers 4 --out "$OUT" \
  --tables loss_tracker/tables/X_p0.004_s1e-3.npz --conditions starter,exact,tracker,oracle-round
rc=$?
mkdir -p "$F"; cp "$OUT/tables.md" "$OUT/accuracy.md" "$F/" 2>/dev/null
echo "smoke run exit $rc"; date -u
exit $rc
