#!/bin/bash
# ctl 190: tcbp-fastpath job 2 (branch tcbp-fastpath; exploration): the ONE held-out check of the frozen fast-path rule C9
# (CRITERION_4.md 7bcd7058): the TCBP version (head priors) and TCBP-JOBS with kit priors + batch check on 5,000 FRESH X shots
# at p = 0.005 (seed 2026112251, made after the rule was frozen). Logits + shots: laptop-artifacts a0cec813
# predecoder-tcbp/fastpath_holdout/ (sha256 checked). On the ctl lane after ctl/180 (two TCBP processes at most). 1 thread.
B=$HOME/mlBranchMidwest; SHA=e6c2a5a06e0534cd212568e948e3de8b84aaa94f; W=$B/pred-tcbp-e6c2a5a; C=a0cec813c61140c813cd92e9b59a9d1a47cac1c4
if [ ! -d "$W" ]; then git -C "$B/repo" cat-file -e "$SHA^{commit}" 2>/dev/null || git -C "$B/repo" fetch -q origin main; git -C "$B/repo" worktree add -q --detach "$W" "$SHA" || exit 1; fi
[ "$(git -C "$W" rev-parse HEAD)" = "$SHA" ] || exit 1
git -C "$B/repo" cat-file -e "$C^{commit}" 2>/dev/null || git -C "$B/repo" fetch -q origin laptop-artifacts
IN=$B/runs/tcbp-fastpath/in_holdout; mkdir -p "$IN"; cd "$IN" || exit 1
for f in SHA256SUMS INNER_SHA256SUMS_holdout logits_holdout_X_p0.005.tar.gz holdout_X_p0.005_s2026112251.npz; do git -C "$B/repo" show "$C:predecoder-tcbp/fastpath_holdout/$f" > "$f" || exit 1; done
sha256sum -c SHA256SUMS || { echo "a transferred file does not match its sha256"; exit 1; }
mkdir -p logits_holdout && tar xzf logits_holdout_X_p0.005.tar.gz -C logits_holdout && (cd logits_holdout && sha256sum -c ../INNER_SHA256SUMS_holdout) || exit 1
cd "$W/challenge-1-ml-decoder-kit" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/blakew3/envs/tcbp-py312/bin/python; O=$B/runs/tcbp-fastpath; F=$B/logsync/files/tcbp-fastpath
mkdir -p "$O" "$F"
sha256sum "$IN/holdout_X_p0.005_s2026112251.npz" predecoder/logs/fit_tcbp_X_p0.005.json; date -u
"$PY" -m predecoder.tcbp decode --data "$IN/holdout_X_p0.005_s2026112251.npz" --logits "$IN/logits_holdout" \
    --settings predecoder/logs/fit_tcbp_X_p0.005.json --arms head fresh --batch-check --out "$O/holdout_X_p0.005" > "$O/holdout_X_p0.005.log" 2>&1
rc=$?; echo "decode exit $rc"
cp "$O/holdout_X_p0.005.log" "$F/"; cp "$O/holdout_X_p0.005/per_shot.npz" "$F/holdout_X_p0.005_per_shot.npz"; cp "$O/holdout_X_p0.005/summary.json" "$F/holdout_X_p0.005_summary.json"
tail -8 "$O/holdout_X_p0.005.log"; date -u; exit $rc
