#!/bin/bash
# 020 (after the TCBP run): (1) starts a third runner lane, ctl, for control and CPU jobs that must not wait behind the
# cpu lane; (2) stops the loss chain's wrapper so that its job C does not start by itself after job B (the orchestrator
# re-queues C with fewer workers); (3) writes the ensemble's selection lists on the p = 0.004 exploration files;
# (4) saves the Z network's logits on the p = 0.004 Z files from a copy of its latest checkpoint.
B=$HOME/mlBranchMidwest; K=$B/jobrepo-gpu/challenge-1-ml-decoder-kit; R=$B/runs; F=$B/logsync/files
date -u; echo "checkout: $(git -C "$B/jobrepo-gpu" log --oneline -1)"
# (1)
if pgrep -f 'jobs/runner.sh ctl' > /dev/null; then echo "ctl lane already running"; else (cd "$B" && nohup jobs/runner.sh ctl > runs/runner_ctl.out 2>&1 &); echo "ctl lane started"; fi
# (2)
if kill -0 32825 2>/dev/null && tr '\0' ' ' < /proc/32825/cmdline | grep -q 'loss.job'; then
  kill 32825 && echo "loss chain wrapper 32825 stopped: the running job's own processes go on to their end; job C will not start by itself"
else
  echo "pid 32825 is not the loss chain (any more): nothing stopped"
fi
echo "loss.job processes now: $(pgrep -fc 'loss[.]job')"
# (3)
cd "$K" || exit 1
export PYTHONDONTWRITEBYTECODE=1
git -C "$B/repo" fetch -q origin laptop-artifacts
mkdir -p "$R/lap" "$R/sel_p0.004" "$F/sel_p0.004"
for s in validation_X train_X validation_Z train_Z; do
  git -C "$B/repo" show "origin/laptop-artifacts:priortilt/results_8b/selector_$s.npz" > "$R/lap/selector_$s.npz"
  python -m hierarchy.ensemble --members "$R/lap/selector_$s.npz" "$R/tcbp_p0.004/$s.tcbp.npz" --write-selection "$R/sel_p0.004/$s.selection.npz"
  echo "ensemble on $s exit $?"
done
cp "$R"/sel_p0.004/*.selection.npz "$F/sel_p0.004/" 2>/dev/null
date -u
# (4)
mkdir -p "$R/netZ" "$F/netZ"
cp "$R/cascade_Z_s1/checkpoint.pt" "$R/netZ/z_checkpoint_copy.pt"; sha256sum "$R/netZ/z_checkpoint_copy.pt"
for f in validation_Z train_Z; do
  python -m neural.evaluate_stim --checkpoint "$R/netZ/z_checkpoint_copy.pt" --data "data/p0.004/$f.npz" --save "$R/netZ/netZ_${f}_p0.004"
  echo "evaluate_stim on $f exit $?"
done
cp "$R"/netZ/netZ_*.npz "$R"/netZ/netZ_*.json "$R"/netZ/netZ_*.log "$F/netZ/" 2>/dev/null
date -u
