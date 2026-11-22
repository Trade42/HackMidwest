#!/bin/bash
# 030: the three candidates (relay-BP, PriorTilt under two seeds) of the exploration shots, 4 processes, in the
# background: the rest of p = 0.004 (validation_Z, train_X, train_Z), then p = 0.005 Z (validation_Z, train_Z).
# The laptop decodes the p = 0.005 X files. Printout: runs/cand_explore.out. Files: files/hier/ on the log branch.
B=$HOME/mlBranchMidwest; C=$B/explore; H=$B/runs/hier; F=$B/logsync/files/hier
sha=$(git -C "$B/jobrepo-cpu" rev-parse HEAD)
[ -d "$C" ] || git -C "$B/repo" worktree add -q --detach "$C" "$sha"
echo "checkout for the exploration runs: $C at $(git -C "$C" log --oneline -1)"; date -u
mkdir -p "$F/cand_p0.004" "$F/cand_p0.005"
nohup sh -c "
cd '$C/challenge-1-ml-decoder-kit' || exit 1
export PRIORTILT_REPO=\$HOME/PriorTiltDecoder PYTHONDONTWRITEBYTECODE=1
python -m hierarchy.candidates --data data/p0.004/validation_Z.npz data/p0.004/train_X.npz data/p0.004/train_Z.npz --out '$H/cand_p0.004_rest' --processes 4
cp '$H'/cand_p0.004_rest/*.chunks.npz '$F/cand_p0.004/'
date -u
python -m hierarchy.candidates --data data/p0.005/validation_Z.npz --out '$H/cand_p0.005_valZ' --processes 4
cp '$H'/cand_p0.005_valZ/*.chunks.npz '$F/cand_p0.005/'
date -u
python -m hierarchy.candidates --data data/p0.005/train_Z.npz --out '$H/cand_p0.005_trainZ' --processes 4
cp '$H'/cand_p0.005_trainZ/*.chunks.npz '$F/cand_p0.005/'
echo \"\$(date -u +%FT%TZ) EXPLORE CANDIDATES DONE\"
" > "$B/runs/cand_explore.out" 2>&1 &
echo "started in the background, pid $!"
sleep 30; tail -3 "$B/runs/cand_explore.out"
