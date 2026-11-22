#!/bin/bash
# ctl 160: tcbp-speed job 1 (branch tcbp-speed e5354788; an improvement branch, merged nowhere): where TCBP-JOBS's 42 ms per-shot
# build goes (cProfile of 50 builds, timed variants: dense / CSR / CSC H, presets), then a check on 1,000 validation_X p = 0.005
# shots that each faster arm gives the SAME 12 bits, faults and correction hash as the current dense path. One process,
# 1 CPU thread, A100, about 7 min. Logits: laptop-artifacts 64163747 predecoder-tcbp/ (sha256 checked). Writes only under ~/mlBranchMidwest.
B=$HOME/mlBranchMidwest; SHA=e5354788808fbc4713ff46c3989db19a2031359d; W=$B/tcbp-speed-e535478; C=64163747dc5f0cb4f3b596fb0a3842fd3684095b
if [ ! -d "$W" ]; then git -C "$B/repo" cat-file -e "$SHA^{commit}" 2>/dev/null || git -C "$B/repo" fetch -q origin tcbp-speed; git -C "$B/repo" worktree add -q --detach "$W" "$SHA" || exit 1; fi
[ "$(git -C "$W" rev-parse HEAD)" = "$SHA" ] || exit 1
git -C "$B/repo" cat-file -e "$C^{commit}" 2>/dev/null || git -C "$B/repo" fetch -q origin laptop-artifacts
IN=$B/runs/tcbp-speed/in; O=$B/runs/tcbp-speed/job1; F=$B/logsync/files/tcbp-speed
[ -e "$O" ] && { echo "refusing: $O exists (one run per output folder)"; exit 1; }
mkdir -p "$IN" "$F"; cd "$IN" || exit 1
for f in SHA256SUMS INNER_SHA256SUMS_validation logits_validation_X_p0.005.tar.gz; do git -C "$B/repo" show "$C:predecoder-tcbp/$f" > "$f" || exit 1; done
grep " logits_validation_X_p0.005.tar.gz\$" SHA256SUMS | sha256sum -c - || exit 1
mkdir -p validation_X_p0.005 && tar xzf logits_validation_X_p0.005.tar.gz -C validation_X_p0.005 && (cd validation_X_p0.005 && sha256sum -c ../INNER_SHA256SUMS_validation) || exit 1
cd "$W/challenge-1-ml-decoder-kit" || exit 1
printf '%s\n' "25e9ce8e28714c985e20f9f9a017dd49ff25a26200626a1b830cfee0cc045312  data/p0.005/validation_X.npz" "7697010678a369c77949a6a4233a3087824fa4d2753bcadc4869f2eaee91b0d8  predecoder/logs/fit_tcbp_X_p0.005.json" | sha256sum -c - || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/blakew3/envs/tcbp-py312/bin/python; date -u; nvidia-smi -L
"$PY" -m predecoder.tcbp_speed run --data data/p0.005/validation_X.npz --logits "$IN/validation_X_p0.005" \
    --settings predecoder/logs/fit_tcbp_X_p0.005.json --limit 1000 --out "$O" > "$O.log" 2>&1
rc=$?; echo "tcbp_speed exit $rc"
cp "$O.log" "$F/job1.log"; cp "$O/summary.json" "$O/check_per_shot.npz" "$O/profile_dense.txt" "$O/profile_dense.pstats" "$F/" 2>/dev/null
tail -14 "$O.log"; date -u; exit $rc
