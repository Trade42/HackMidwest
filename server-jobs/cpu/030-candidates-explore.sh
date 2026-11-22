#!/bin/bash
# 030 (rewritten before it ran): in the background, on 4 processes beside the timed rows: loss job C (L1 at s = 5e-4,
# the same command as queued before but with 4 workers; the output does not depend on the worker count), then the three
# candidates of the p = 0.005 validation_Z shots. Printout: runs/explore2.out.
B=$HOME/mlBranchMidwest; C=$B/explore; H=$B/runs/hier; F=$B/logsync/files/hier
sha=$(git -C "$B/jobrepo-cpu" rev-parse HEAD)
[ -d "$C" ] || git -C "$B/repo" worktree add -q --detach "$C" "$sha"
echo "checkout for the candidates: $C at $(git -C "$C" log --oneline -1)"; date -u
mkdir -p "$F/cand_p0.005"
nohup sh -c "
cd '$B/loss/challenge-1-ml-decoder-kit' && PYTHONDONTWRITEBYTECODE=1 python -m loss.job --basis X --p 0.004 --s 5e-4 --shots 10000 --seed 5200 --workers 4 --out '$B/runs/loss/L1_s5e-4_seed5200' --group blind:800 --group starter,starter@0.5/stop1:10000 --table blind:blind,starter,starter@0.5/stop1 --table starter:starter,starter@0.5/stop1
echo \"\$(date -u +%FT%TZ) LOSS JOB C END exit \$?\"
cd '$C/challenge-1-ml-decoder-kit' || exit 1
export PRIORTILT_REPO=\$HOME/PriorTiltDecoder PYTHONDONTWRITEBYTECODE=1
python -m hierarchy.candidates --data data/p0.005/validation_Z.npz --out '$H/cand_p0.005_valZ' --processes 4
cp '$H'/cand_p0.005_valZ/*.chunks.npz '$F/cand_p0.005/'
echo \"\$(date -u +%FT%TZ) EXPLORE2 END\"
" > "$B/runs/explore2.out" 2>&1 &
echo "started in the background, pid $!"
sleep 20; tail -3 "$B/runs/explore2.out"
