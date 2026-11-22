#!/bin/bash
# ctl 030: the three candidates (relay-BP, PriorTilt under two seeds) of the p = 0.005 validation_Z shots, on 8
# processes, while the notebook's CPUs are idle (before loss job D and the timed rows start). Job cpu 030 names the
# same output folder: it keeps the finished parts and only prints its table.
B=$HOME/mlBranchMidwest; K=$B/jobrepo-ctl/challenge-1-ml-decoder-kit; H=$B/runs/hier; F=$B/logsync/files/hier
cd "$K" || exit 1
date -u; echo "checkout: $(git -C "$B/jobrepo-ctl" log --oneline -1)"
export PRIORTILT_REPO=$HOME/PriorTiltDecoder PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m hierarchy.candidates --data data/p0.005/validation_Z.npz --out "$H/cand_p0.005_valZ" --processes 8
rc=$?
mkdir -p "$F/cand_p0.005"; cp "$H"/cand_p0.005_valZ/*.chunks.npz "$F/cand_p0.005/" 2>/dev/null
echo "candidates exit $rc"; date -u
exit $rc
