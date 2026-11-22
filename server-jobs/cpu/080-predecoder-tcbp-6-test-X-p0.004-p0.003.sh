#!/bin/bash
# cpu 080 (the cpu lane is a queue; it uses the A100): branch predecoder-tcbp, job 6 = the registered TEST RUN 2
# (tcbp_test_registered_spec.md at dad4fd50, an extension): TH, head arm only, X at p = 0.004 then 0.003, knobs
# fitted at p = 0.005 applied unchanged, once each. Designed after the test results were seen; a separate branch.
# It waits (up to 15 min) for gpu 080 to have started its decode, so the two never fetch or add the checkout at the
# same time. Logits from laptop-artifacts b2d236c705ae643d15e6bf947e26b17e6d93bd2a (parts, every sha256 checked). Refuses an existing out folder.
# A100, 1 CPU thread, about 2 x 9 min. Writes only under ~/mlBranchMidwest; logs mirrored every minute.
B=$HOME/mlBranchMidwest; SHA=dad4fd50073bc18f582565da403867fb6d9afb5d; W=$B/pred-tcbp-dad4fd5; C=b2d236c705ae643d15e6bf947e26b17e6d93bd2a
for i in $(seq 1 30); do [ -e "$B/runs/predecoder-tcbp/test_X_p0.005.log" ] && break; sleep 30; done
for i in 1 2 3 4 5; do git -C "$B/repo" cat-file -e "$SHA^{commit}" 2>/dev/null && git -C "$B/repo" cat-file -e "$C^{commit}" 2>/dev/null && break; git -C "$B/repo" fetch -q origin predecoder-tcbp laptop-artifacts || sleep 20; done
git -C "$B/repo" cat-file -e "$C^{commit}" || { echo "laptop-artifacts commit $C missing"; exit 1; }
if [ ! -d "$W" ]; then git -C "$B/repo" worktree add -q --detach "$W" "$SHA" || exit 1; fi
[ "$(git -C "$W" rev-parse HEAD)" = "$SHA" ] || { echo "checkout is not $SHA"; exit 1; }
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/blakew3/envs/tcbp-py312/bin/python; IN=$B/runs/predecoder-tcbp/in_test2; rc=0
for P in 0.004 0.003; do
  O=$B/runs/predecoder-tcbp/test_X_p$P; F=$B/logsync/files/predecoder-tcbp/test_X_p$P
  [ -e "$O" ] && { echo "refusing: $O exists (this run happens once)"; rc=1; continue; }
  mkdir -p "$IN" "$F"; cd "$IN" || exit 1
  for f in TEST_SHA256SUMS INNER_SHA256SUMS_test_p$P logits_test_X_p$P.tar.gz.part-00 logits_test_X_p$P.tar.gz.part-01; do git -C "$B/repo" show "$C:predecoder-tcbp/test/$f" > "$f" || exit 1; done
  grep "p$P.tar.gz.part-" TEST_SHA256SUMS | sha256sum -c - || exit 1
  cat logits_test_X_p$P.tar.gz.part-00 logits_test_X_p$P.tar.gz.part-01 > logits_test_X_p$P.tar.gz && grep " logits_test_X_p$P.tar.gz\$" TEST_SHA256SUMS | sha256sum -c - || exit 1
  mkdir -p logits_test_X_p$P && tar xzf logits_test_X_p$P.tar.gz -C logits_test_X_p$P && (cd logits_test_X_p$P && sha256sum -c ../INNER_SHA256SUMS_test_p$P) || exit 1
  cd "$W/challenge-1-ml-decoder-kit" || exit 1
  case $P in 0.004) D=53a7cee301600d36f3dad806a88f59ad6d84b681044b2548268c2ea9b5689e94;; 0.003) D=f7bc181399a0a8fd1324950c98a8419c0082d7c34940c03dfb34c9e2cefc1ff3;; esac
  printf '%s\n' "$D  data/p$P/test_X.npz" "7697010678a369c77949a6a4233a3087824fa4d2753bcadc4869f2eaee91b0d8  predecoder/logs/fit_tcbp_X_p0.005.json" | sha256sum -c - || exit 1
  date -u
  "$PY" -m predecoder.tcbp decode --data data/p$P/test_X.npz --logits "$IN/logits_test_X_p$P" --settings predecoder/logs/fit_tcbp_X_p0.005.json --arms head --out "$O" --final > "$O.log" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do cp "$O.log" "$F/decode.log" 2>/dev/null; sleep 60; done
  wait "$pid"; r=$?; echo "p $P decode exit $r"; [ $r -ne 0 ] && rc=$r
  cp "$O.log" "$F/decode.log"; cp "$O/per_shot.npz" "$O/summary.json" "$F/" 2>/dev/null
  tail -4 "$O.log"
done
date -u; exit $rc
