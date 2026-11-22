#!/bin/bash
# ctl 180: tcbp-fastpath job 1 (branch tcbp-fastpath; exploration): the TCBP version (head priors) and TCBP-JOBS with the kit's
# priors (fresh) + batch check on VALIDATION_X p = 0.004 (5,000 shots), knobs frozen at p = 0.005, code main e6c2a5a0; for a fast
# path rule declared before its numbers. Logits: laptop-artifacts 424909f8 predecoder-tcbp/fastpath_p0.004/ (sha256 checked).
# Waits (up to 15 min) until no tcbp_speed process runs, so the A100 slice holds at most two TCBP processes. 1 CPU thread.
W0=$(date +%s); while pgrep -u "$(id -un)" -f 'predecoder[.]tcbp_speed' > /dev/null; do [ $(( $(date +%s) - W0 )) -ge 900 ] && { echo "tcbp_speed still running after 15 min: not starting"; exit 1; }; sleep 15; done; echo "waited $(( $(date +%s) - W0 )) s"
B=$HOME/mlBranchMidwest; SHA=e6c2a5a06e0534cd212568e948e3de8b84aaa94f; W=$B/pred-tcbp-e6c2a5a; C=424909f8767832291d1d1be05fb5926d2271bdd8
if [ ! -d "$W" ]; then git -C "$B/repo" cat-file -e "$SHA^{commit}" 2>/dev/null || git -C "$B/repo" fetch -q origin main; git -C "$B/repo" worktree add -q --detach "$W" "$SHA" || exit 1; fi
[ "$(git -C "$W" rev-parse HEAD)" = "$SHA" ] || exit 1
git -C "$B/repo" cat-file -e "$C^{commit}" 2>/dev/null || git -C "$B/repo" fetch -q origin laptop-artifacts
IN=$B/runs/tcbp-fastpath/in; mkdir -p "$IN"; cd "$IN" || exit 1
for f in SHA256SUMS INNER_SHA256SUMS_validation_p0.004 logits_validation_X_p0.004.tar.gz; do git -C "$B/repo" show "$C:predecoder-tcbp/fastpath_p0.004/$f" > "$f" || exit 1; done
sha256sum -c SHA256SUMS || { echo "a transferred file does not match its sha256"; exit 1; }
mkdir -p validation_X_p0.004 && tar xzf logits_validation_X_p0.004.tar.gz -C validation_X_p0.004 && (cd validation_X_p0.004 && sha256sum -c ../INNER_SHA256SUMS_validation_p0.004) || exit 1
cd "$W/challenge-1-ml-decoder-kit" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/blakew3/envs/tcbp-py312/bin/python; O=$B/runs/tcbp-fastpath; F=$B/logsync/files/tcbp-fastpath
mkdir -p "$O" "$F"
sha256sum data/p0.004/validation_X.npz predecoder/logs/fit_tcbp_X_p0.005.json; date -u
"$PY" -m predecoder.tcbp decode --data data/p0.004/validation_X.npz --logits "$IN/validation_X_p0.004" \
    --settings predecoder/logs/fit_tcbp_X_p0.005.json --arms head fresh --batch-check --out "$O/validation_X_p0.004" > "$O/validation_X_p0.004.log" 2>&1
rc=$?; echo "decode exit $rc"
cp "$O/validation_X_p0.004.log" "$F/"; cp "$O/validation_X_p0.004/per_shot.npz" "$F/validation_X_p0.004_per_shot.npz"; cp "$O/validation_X_p0.004/summary.json" "$F/validation_X_p0.004_summary.json"
tail -8 "$O/validation_X_p0.004.log"; date -u; exit $rc
