#!/bin/bash
# ctl 170: tcbp-speed job 2 (branch tcbp-speed 958ffe9b; an improvement branch, merged nowhere): the decoder free moved off the
# answer path (freed after the answer / on a background thread), checked for the same answers against dense on 1,000
# validation_X p = 0.005 shots, timed the same way; timing-only probes of four decoder options. Two processes, one after
# the other, each under timeout -k 30 900. A100, 1 CPU thread each. Writes only under ~/mlBranchMidwest.
B=$HOME/mlBranchMidwest; SHA=958ffe9bbab5185d209bb4c3191cfdb3c5276c12; W=$B/tcbp-speed-958ffe9; C=64163747dc5f0cb4f3b596fb0a3842fd3684095b
if [ ! -d "$W" ]; then git -C "$B/repo" cat-file -e "$SHA^{commit}" 2>/dev/null || git -C "$B/repo" fetch -q origin tcbp-speed; git -C "$B/repo" worktree add -q --detach "$W" "$SHA" || exit 1; fi
[ "$(git -C "$W" rev-parse HEAD)" = "$SHA" ] || exit 1
IN=$B/runs/tcbp-speed/in; F=$B/logsync/files/tcbp-speed/job2; OA=$B/runs/tcbp-speed/job2a; OB=$B/runs/tcbp-speed/job2b
for o in "$OA" "$OB"; do [ -e "$o" ] && { echo "refusing: $o exists (one run per output folder)"; exit 1; }; done
mkdir -p "$IN" "$F"; cd "$IN" || exit 1
if ! (cd validation_X_p0.005 2>/dev/null && sha256sum -c ../INNER_SHA256SUMS_validation); then
  git -C "$B/repo" cat-file -e "$C^{commit}" 2>/dev/null || git -C "$B/repo" fetch -q origin laptop-artifacts
  for f in SHA256SUMS INNER_SHA256SUMS_validation logits_validation_X_p0.005.tar.gz; do git -C "$B/repo" show "$C:predecoder-tcbp/$f" > "$f" || exit 1; done
  grep " logits_validation_X_p0.005.tar.gz\$" SHA256SUMS | sha256sum -c - || exit 1
  mkdir -p validation_X_p0.005 && tar xzf logits_validation_X_p0.005.tar.gz -C validation_X_p0.005 && (cd validation_X_p0.005 && sha256sum -c ../INNER_SHA256SUMS_validation) || exit 1
fi
cd "$W/challenge-1-ml-decoder-kit" || exit 1
printf '%s\n' "25e9ce8e28714c985e20f9f9a017dd49ff25a26200626a1b830cfee0cc045312  data/p0.005/validation_X.npz" "7697010678a369c77949a6a4233a3087824fa4d2753bcadc4869f2eaee91b0d8  predecoder/logs/fit_tcbp_X_p0.005.json" | sha256sum -c - || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/blakew3/envs/tcbp-py312/bin/python; date -u; nvidia-smi -L
timeout -k 30 900 "$PY" -X faulthandler -m predecoder.tcbp_speed run --data data/p0.005/validation_X.npz --logits "$IN/validation_X_p0.005" \
    --settings predecoder/logs/fit_tcbp_X_p0.005.json --limit 1000 --arms dense csr csr_after --profile-builds 50 --profile-set options --out "$OA" > "$OA.log" 2>&1
ra=$?; echo "A exit $ra"; date -u
mkdir -p "$F/a"; cp "$OA.log" "$F/a/job2a.log"; cp "$OA/summary.json" "$OA/check_per_shot.npz" "$OA/profile_dense.txt" "$F/a/" 2>/dev/null
tail -8 "$OA.log"
timeout -k 30 900 "$PY" -X faulthandler -m predecoder.tcbp_speed run --data data/p0.005/validation_X.npz --logits "$IN/validation_X_p0.005" \
    --settings predecoder/logs/fit_tcbp_X_p0.005.json --limit 1000 --arms dense csr csr_thread --profile-builds 0 --queue-depth 2 --out "$OB" > "$OB.log" 2>&1
rb=$?; echo "B exit $rb"; date -u
mkdir -p "$F/b"; cp "$OB.log" "$F/b/job2b.log"; cp "$OB/summary.json" "$OB/check_per_shot.npz" "$F/b/" 2>/dev/null
tail -8 "$OB.log"
echo "exit codes: A $ra, B $rb"; [ "$ra" = 0 ] && [ "$rb" = 0 ]
