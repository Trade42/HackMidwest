#!/bin/bash
# ctl 040: the ensemble's exploration tables at the two other noise levels, on VALIDATION files only, from a frozen
# checkout. It starts two background scripts and returns:
#   p = 0.003: when the hierarchy chain's candidates and gpu 040's TCBP-JOBS files exist: the selection lists, joint
#              relay-BP on the listed shots, the ensemble with rule f, the pair score;
#   p = 0.005: when gpu 040's files exist: the ensemble without joint decoding, for X from the laptop's candidates and
#              for Z from ctl 030's.
# Printouts: runs/ens_explore_p0.003.out and runs/ens_explore_p0.005.out. Files: files/{sel,joint,ens}_p0.003/ and
# files/{sel,ens}_p0.005/ on the log branch. Each script gives up 90 minutes after its start.
B=$HOME/mlBranchMidwest; C=$B/ens-explore
sha=$(git -C "$B/jobrepo-ctl" rev-parse HEAD)
if [ ! -d "$C" ]; then
  git -C "$B/repo" cat-file -e "$sha^{commit}" 2>/dev/null || git -C "$B/repo" fetch -q origin integration
  git -C "$B/repo" worktree add -q --detach "$C" "$sha" || { echo "the checkout could not be made"; exit 1; }
fi
echo "checkout: $C at $(git -C "$C" log --oneline -1)"; date -u
mkdir -p "$B/jobs" "$B/runs/lap"
git -C "$B/repo" fetch -q origin laptop-artifacts
git -C "$B/repo" show origin/laptop-artifacts:priortilt/results_p005main/selector_validation_X.npz > "$B/runs/lap/selector_p0.005_validation_X.npz" \
  || echo "the laptop's p = 0.005 candidates could not be read from laptop-artifacts"
ls -la "$B/runs/lap/selector_p0.005_validation_X.npz"

cat > "$B/jobs/ens_explore_common.sh" <<'EOS'
B=$HOME/mlBranchMidwest; K=$B/ens-explore/challenge-1-ml-decoder-kit; R=$B/runs; F=$B/logsync/files
export PRIORTILT_REPO=$HOME/PriorTiltDecoder PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
END_AT=$(( $(date +%s) + 5400 ))
wait_for() {  # waits until every named file exists and is not empty; gives up at END_AT
  echo "$(date -u +%FT%TZ) waiting for: $*"
  while :; do
    local missing=""
    for f in "$@"; do [ -s "$f" ] || missing="$missing $f"; done
    [ -z "$missing" ] && break
    if [ "$(date +%s)" -gt "$END_AT" ]; then echo "$(date -u +%FT%TZ) GAVE UP, still missing:$missing"; return 1; fi
    sleep 20
  done
  sleep 15   # a file that has just appeared may still be being written
  echo "$(date -u +%FT%TZ) inputs ready"
}
EOS

cat > "$B/jobs/ens_explore_p0.003.sh" <<'EOS'
#!/bin/bash
. "$HOME/mlBranchMidwest/jobs/ens_explore_common.sh"
cd "$K" || exit 1
wait_for "$R/hier/cand_p0.003/validation_X.chunks.npz" "$R/hier/cand_p0.003/validation_Z.chunks.npz" \
         "$R/tcbp_p0.003/validation_X.tcbp.npz" "$R/tcbp_p0.003/validation_Z.tcbp.npz" || exit 1
mkdir -p "$R/sel_p0.003" "$R/ens_p0.003" "$F/sel_p0.003" "$F/joint_p0.003" "$F/ens_p0.003"
for b in X Z; do
  python -m hierarchy.ensemble --members "$R/hier/cand_p0.003/validation_$b.chunks.npz" "$R/tcbp_p0.003/validation_$b.tcbp.npz" \
    --write-selection "$R/sel_p0.003/validation_$b.selection.npz"
  echo "selection list $b: exit $?"
done
procs=9; pgrep -f 'evaluation[.]evaluate' > /dev/null && procs=3   # few processes if timed rows are running
echo "$(date -u +%FT%TZ) joint decoding on $procs processes"
python -m hierarchy.joint_member --data data/p0.003/validation_X.npz data/p0.003/validation_Z.npz \
  --selection "$R/sel_p0.003/validation_X.selection.npz" "$R/sel_p0.003/validation_Z.selection.npz" \
  --out "$R/joint_p0.003" --processes "$procs"
echo "joint decoding: exit $?"
for b in X Z; do
  python -m hierarchy.ensemble --members "$R/hier/cand_p0.003/validation_$b.chunks.npz" "$R/tcbp_p0.003/validation_$b.tcbp.npz" \
    "$R/joint_p0.003/validation_$b.joint.npz" --rule f --save "$R/ens_p0.003/validation_$b.json"
  echo "ensemble $b: exit $?"
done
python -m hierarchy.ensemble --pair "$R/ens_p0.003/validation_X.npz" "$R/ens_p0.003/validation_Z.npz"
echo "pair score: exit $?"
cp "$R"/sel_p0.003/*.npz "$F/sel_p0.003/" 2>/dev/null
cp "$R"/joint_p0.003/*.joint.npz "$F/joint_p0.003/" 2>/dev/null
cp "$R"/ens_p0.003/* "$F/ens_p0.003/" 2>/dev/null
echo "$(date -u +%FT%TZ) ENS-EXPLORE p0.003 END"
EOS

cat > "$B/jobs/ens_explore_p0.005.sh" <<'EOS'
#!/bin/bash
. "$HOME/mlBranchMidwest/jobs/ens_explore_common.sh"
cd "$K" || exit 1
mkdir -p "$R/sel_p0.005" "$R/ens_p0.005" "$F/sel_p0.005" "$F/ens_p0.005"
if wait_for "$R/lap/selector_p0.005_validation_X.npz" "$R/tcbp_p0.005/validation_X.tcbp.npz"; then
  python -m hierarchy.ensemble --members "$R/lap/selector_p0.005_validation_X.npz" "$R/tcbp_p0.005/validation_X.tcbp.npz" \
    --no-joint --rule f --write-selection "$R/sel_p0.005/validation_X.selection.npz" --save "$R/ens_p0.005/validation_X.json"
  echo "ensemble X: exit $?"
  cp "$R"/sel_p0.005/*.npz "$F/sel_p0.005/" 2>/dev/null; cp "$R"/ens_p0.005/* "$F/ens_p0.005/" 2>/dev/null
fi
if wait_for "$R/hier/cand_p0.005_valZ/validation_Z.chunks.npz" "$R/tcbp_p0.005/validation_Z.tcbp.npz"; then
  python -m hierarchy.ensemble --members "$R/hier/cand_p0.005_valZ/validation_Z.chunks.npz" "$R/tcbp_p0.005/validation_Z.tcbp.npz" \
    --no-joint --rule f --write-selection "$R/sel_p0.005/validation_Z.selection.npz" --save "$R/ens_p0.005/validation_Z.json"
  echo "ensemble Z: exit $?"
  python -m hierarchy.ensemble --pair "$R/ens_p0.005/validation_X.npz" "$R/ens_p0.005/validation_Z.npz"
  echo "pair score: exit $?"
  cp "$R"/sel_p0.005/*.npz "$F/sel_p0.005/" 2>/dev/null; cp "$R"/ens_p0.005/* "$F/ens_p0.005/" 2>/dev/null
fi
echo "$(date -u +%FT%TZ) ENS-EXPLORE p0.005 END"
EOS

chmod +x "$B/jobs/ens_explore_p0.003.sh" "$B/jobs/ens_explore_p0.005.sh"
nohup "$B/jobs/ens_explore_p0.003.sh" > "$B/runs/ens_explore_p0.003.out" 2>&1 < /dev/null &
echo "p = 0.003 script started, pid $!"
nohup "$B/jobs/ens_explore_p0.005.sh" > "$B/runs/ens_explore_p0.005.out" 2>&1 < /dev/null &
echo "p = 0.005 script started, pid $!"
sleep 5; head -3 "$B/runs/ens_explore_p0.003.out" "$B/runs/ens_explore_p0.005.out"
