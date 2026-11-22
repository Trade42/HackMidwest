#!/bin/bash
# 030: in the background, on 4 processes beside the timed rows: the three candidates (relay-BP, PriorTilt under two
# seeds) of the p = 0.005 validation_Z shots. It waits until the joint-decoding job (ctl 010) has ended.
# Printout: runs/explore2.out.
# An earlier form of this script ran loss job C first. That job ran by itself at 06:26 UTC (the loss chain started it
# before gpu job 020 stopped the chain), so it is not repeated here.
B=$HOME/mlBranchMidwest; C=$B/explore; H=$B/runs/hier; F=$B/logsync/files/hier
sha=$(git -C "$B/jobrepo-cpu" rev-parse HEAD)
if [ ! -d "$C" ]; then
  git -C "$B/repo" cat-file -e "$sha^{commit}" 2>/dev/null || git -C "$B/repo" fetch -q origin integration
  git -C "$B/repo" worktree add -q --detach "$C" "$sha" || { echo "the checkout could not be made"; exit 1; }
fi
echo "checkout for the candidates: $C at $(git -C "$C" log --oneline -1)"; date -u
mkdir -p "$F/cand_p0.005" "$B/jobs"
cat > "$B/jobs/explore2.sh" <<'EOS'
#!/bin/bash
B=$HOME/mlBranchMidwest; C=$B/explore; H=$B/runs/hier; F=$B/logsync/files/hier
while pgrep -f 'hierarchy[.]joint_member' > /dev/null; do sleep 20; done
echo "$(date -u +%FT%TZ) EXPLORE2 START"
cd "$C/challenge-1-ml-decoder-kit" || exit 1
export PRIORTILT_REPO=$HOME/PriorTiltDecoder PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m hierarchy.candidates --data data/p0.005/validation_Z.npz --out "$H/cand_p0.005_valZ" --processes 4
echo "$(date -u +%FT%TZ) candidates exit $?"
cp "$H"/cand_p0.005_valZ/*.chunks.npz "$F/cand_p0.005/"
echo "$(date -u +%FT%TZ) EXPLORE2 END"
EOS
chmod +x "$B/jobs/explore2.sh"
nohup "$B/jobs/explore2.sh" > "$B/runs/explore2.out" 2>&1 < /dev/null &
echo "started in the background, pid $!"
sleep 20; tail -3 "$B/runs/explore2.out"
