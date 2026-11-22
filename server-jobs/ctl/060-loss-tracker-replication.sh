#!/bin/bash
# ctl 060: the loss tracker's replication on a second seed (7310, 4,000 shots), with its frozen version, on 4 workers:
# starter, exact, tracker and the ORACLE on the same shots. 45 to 90 minutes. Then, only if it is still before
# 09:20 UTC, the earlier tracker version on the same shots (a labelled secondary row; about a quarter of the time).
# Loss shots are sampled here from the seed (Clifft gives the same shots as on luminaty: ctl 020). No test file of
# the required sweep is touched. Files: files/loss_tracker/rep_seed7310/ and rep_seed7310_coupled2/ on the log branch.
B=$HOME/mlBranchMidwest; K=$B/jobrepo-ctl/challenge-1-ml-decoder-kit; O=$B/runs/loss_tracker; F=$B/logsync/files/loss_tracker
cd "$K" || exit 1
date -u; echo "checkout: $(git -C "$B/jobrepo-ctl" log --oneline -1)"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
back() {  # copies a run's small files to the log branch: name
  mkdir -p "$F/$1/parts"
  cp "$O/$1"/tables.md "$O/$1"/tables.json "$O/$1"/accuracy.md "$O/$1"/accuracy.json "$F/$1/" 2>/dev/null
  cp "$O/$1"/parts/*.npz "$F/$1/parts/" 2>/dev/null
}
python -m loss_tracker.job --seed 7310 --shots 4000 --workers 4 --mode coupled --sweeps 1 --order 1 --explain \
  --out "$O/rep_seed7310" --tables loss_tracker/tables/X_p0.004_s1e-3.npz --conditions starter,exact,tracker,oracle-round
rc=$?
back rep_seed7310
echo "job 1 (the frozen tracker, seed 7310): exit $rc"; date -u
if [ $rc -eq 0 ] && [ "$(date -u +%H%M)" -lt 0920 ]; then
  python -m loss_tracker.job --seed 7310 --shots 4000 --workers 4 --mode coupled --sweeps 2 --order 1 \
    --out "$O/rep_seed7310_coupled2" --tables loss_tracker/tables/X_p0.004_s1e-3.npz --conditions tracker
  rc2=$?
  back rep_seed7310_coupled2
  echo "job 2 (the earlier tracker version, the same shots): exit $rc2"; date -u
else
  echo "job 2 is not started (job 1 failed, or it is past 09:20 UTC: the run on the test shots needs the CPUs)"
fi
exit $rc
