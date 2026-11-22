#!/bin/bash
# cpu 110 (the cpu lane is a queue; it uses the A100). Template: server-jobs/ctl/140 and job 6.
# Branch tcbp-zlowp: REGISTERED RUN 4 (an extension; declared in predecoder/logs/tcbp_zlowp_declared.md at a8e24119):
# TCBP-JOBS with the Z head's per-shot priors, X's frozen TCBP knobs (76970106..., no Z fit), Z at p = 0.004 then 0.003,
# head arm only, once per file (--final), no validation and no gate (as run 2 for X). Designed after the test results
# were seen; no validation at these p. Logits: luminaty CPU float32 bytes (Z head 083147d2..., Z trunk fad24e80...,
# 16 threads), from laptop-artifacts <C> predecoder-tcbp/zlowp/ (every sha256 checked).
# ONE TCBP-JOBS process at a time (p = 0.004, then 0.003), 1 CPU thread, own worktree; for the cpu lane (cpu/110), beside ctl/160.
# A100; about 2 min of model fit and ~45 ms a shot per p, so ~20 min in all.
# Writes only under ~/mlBranchMidwest. Outputs: logsync files/predecoder-tcbp/test_Z_p0.004/ and test_Z_p0.003/.
B=$HOME/mlBranchMidwest; SHA=f6482040e80d4a7ae07bedb136341a1d2ce6dc5c; W=$B/pred-tcbp-zlowp-${SHA:0:7}; C=b4ef168ae8dd2ec5397d980a880d39e3365101d8
fetch() { for t in 1 2 3 4 5; do git -C "$B/repo" fetch -q origin "$1" && return 0; echo "fetch $1 failed (try $t), retrying in 30 s"; sleep 30; done; return 1; }
if [ ! -d "$W" ]; then git -C "$B/repo" cat-file -e "$SHA^{commit}" 2>/dev/null || fetch tcbp-zlowp || exit 1; git -C "$B/repo" worktree add -q --detach "$W" "$SHA" || exit 1; fi
[ "$(git -C "$W" rev-parse HEAD)" = "$SHA" ] || exit 1
git -C "$B/repo" cat-file -e "$C^{commit}" 2>/dev/null || fetch laptop-artifacts || exit 1
git -C "$B/repo" cat-file -e "$C^{commit}" || { echo "laptop-artifacts commit $C not found"; exit 1; }
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/blakew3/envs/tcbp-py312/bin/python; IN=$B/runs/predecoder-tcbp/in_zlowp; rc=0
for P in 0.004 0.003; do
  O=$B/runs/predecoder-tcbp/test_Z_p$P; F=$B/logsync/files/predecoder-tcbp/test_Z_p$P
  [ -e "$O" ] && { echo "refusing: $O exists (this run happens once)"; rc=1; continue; }
  mkdir -p "$IN" "$F"; cd "$IN" || exit 1
  for f in ZLOWP_SHA256SUMS INNER_SHA256SUMS_test_Z_p$P logits_test_Z_p$P.tar.gz.part-00 logits_test_Z_p$P.tar.gz.part-01; do git -C "$B/repo" show "$C:predecoder-tcbp/zlowp/$f" > "$f" || exit 1; done
  grep "p$P.tar.gz.part-" ZLOWP_SHA256SUMS | sha256sum -c - || exit 1
  grep " INNER_SHA256SUMS_test_Z_p$P\$" ZLOWP_SHA256SUMS | sha256sum -c - || exit 1
  cat logits_test_Z_p$P.tar.gz.part-* > logits_test_Z_p$P.tar.gz && grep " logits_test_Z_p$P.tar.gz\$" ZLOWP_SHA256SUMS | sha256sum -c - || exit 1
  mkdir -p logits_test_Z_p$P && tar xzf logits_test_Z_p$P.tar.gz -C logits_test_Z_p$P && (cd logits_test_Z_p$P && sha256sum -c ../INNER_SHA256SUMS_test_Z_p$P) || exit 1
  cd "$W/challenge-1-ml-decoder-kit" || exit 1
  case $P in 0.004) D=b61c1325de8852385d19ee5ce4eb41ea939b668f83fb3e994ce0f590218b3d56;; 0.003) D=f5ac5b336f4155870fd8a6792ad61e21d5d76f04e05a9c09419879b6512e7f49;; esac
  printf '%s\n' "$D  data/p$P/test_Z.npz" "7697010678a369c77949a6a4233a3087824fa4d2753bcadc4869f2eaee91b0d8  predecoder/logs/fit_tcbp_X_p0.005.json" | sha256sum -c - || exit 1
  date -u
  "$PY" -m predecoder.tcbp decode --data data/p$P/test_Z.npz --logits "$IN/logits_test_Z_p$P" --settings predecoder/logs/fit_tcbp_X_p0.005.json --arms head --out "$O" --final > "$O.log" 2>&1
  r=$?; echo "p $P decode exit $r"; [ $r -ne 0 ] && rc=$r
  cp "$O.log" "$F/decode.log"; cp "$O/per_shot.npz" "$O/summary.json" "$F/" 2>/dev/null
  tail -4 "$O.log"
done
date -u; exit $rc
