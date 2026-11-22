#!/bin/bash
# ctl 010: joint relay-BP (all 1,728 detectors) on the shots of the p = 0.004 exploration files where PriorTilt under
# two seeds and TCBP-JOBS are not unanimous. Waits for the selection lists and until the loss job's processes have
# ended, then uses 9 processes (with the hierarchy chain's 2 and the training's 1 that fills the 12 CPUs).
B=$HOME/mlBranchMidwest; K=$B/jobrepo-ctl/challenge-1-ml-decoder-kit; R=$B/runs; F=$B/logsync/files; S=$R/sel_p0.004
echo "waiting for the selection lists and for the loss job's processes to end"; date -u
while [ ! -f "$S/train_Z.selection.npz" ] || pgrep -f 'loss[.]job' > /dev/null; do sleep 20; done
date -u; echo "checkout: $(git -C "$B/jobrepo-ctl" log --oneline -1)"
cd "$K" || exit 1
export PYTHONDONTWRITEBYTECODE=1
python -m hierarchy.joint_member \
  --data data/p0.004/validation_X.npz data/p0.004/train_X.npz data/p0.004/validation_Z.npz data/p0.004/train_Z.npz \
  --selection "$S/validation_X.selection.npz" "$S/train_X.selection.npz" "$S/validation_Z.selection.npz" "$S/train_Z.selection.npz" \
  --out "$R/joint_p0.004" --processes 9
rc=$?
mkdir -p "$F/joint_p0.004"; cp "$R"/joint_p0.004/*.joint.npz "$F/joint_p0.004/" 2>/dev/null
echo "joint_member exit $rc"; date -u
exit $rc
