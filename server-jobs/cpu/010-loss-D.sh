#!/bin/bash
# 010: waits for the loss chain (jobs A, B, C; pid 32825) and for the hierarchy chain to end, then runs loss job D:
# the L2 pieces on job A's dataset (seed 5300), 10 workers, from the loss worktree at 6ea830c.
B=$HOME/mlBranchMidwest
echo "waiting for the loss chain and the hierarchy chain"; date -u
while kill -0 32825 2>/dev/null || pgrep -f 'runs/hier/cand_p0.003' > /dev/null; do sleep 30; done
echo "both ended"; date -u
cd "$B/loss/challenge-1-ml-decoder-kit" || exit 1
echo "checkout: $(git log --oneline -1)"
export PYTHONDONTWRITEBYTECODE=1
OUT=$B/runs/loss
python -m loss.job --basis X --p 0.004 --s 1e-3 --shots 4000 --seed 5300 --workers 10 --out "$OUT/L3_s1e-3_seed5300" \
  --group starter-nofinal,starter-nohook,starter-ratebase:1000 \
  --group exact@0.4,oracle-round@0.4,exact@0.25,oracle-round@0.25:1000 \
  --table starter:starter,starter-nofinal,starter-nohook,starter-ratebase,exact \
  --table exact@0.4:exact@0.4,oracle-round@0.4,exact \
  --table exact@0.25:exact@0.25,oracle-round@0.25,exact \
  --table oracle-round:oracle-round,oracle-round@0.4,oracle-round@0.25
rc=$?
echo "job D exit $rc"; date -u
exit $rc
