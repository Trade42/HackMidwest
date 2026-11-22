#!/bin/bash
# gpu 080: branch predecoder-tcbp, job 5 = the registered TEST RUN 1 (predecoder/logs/tcbp_test_registered_spec.md at
# dad4fd50): TCBP-JOBS with the head's per-shot priors (arm (a) TH, knobs frozen on train, passed on validation 33 vs 91)
# on data/p0.005/test_X.npz, once; head arm first, then fresh, then the batch check. Designed after the test results
# were seen; a separate branch. Logits: luminaty CPU float32, laptop-artifacts b2d236c705ae643d15e6bf947e26b17e6d93bd2a predecoder-tcbp/test/ (parts,
# every sha256 checked). Refuses if its out folder exists. A100, 1 CPU thread, about 20 min. Writes only under
# ~/mlBranchMidwest; the log is mirrored to logsync/files/predecoder-tcbp/test_X_p0.005/decode.log every minute.
B=$HOME/mlBranchMidwest; SHA=dad4fd50073bc18f582565da403867fb6d9afb5d; W=$B/pred-tcbp-dad4fd5; C=b2d236c705ae643d15e6bf947e26b17e6d93bd2a
for i in 1 2 3 4 5; do git -C "$B/repo" cat-file -e "$SHA^{commit}" 2>/dev/null && git -C "$B/repo" cat-file -e "$C^{commit}" 2>/dev/null && break; git -C "$B/repo" fetch -q origin predecoder-tcbp laptop-artifacts || sleep 20; done
git -C "$B/repo" cat-file -e "$C^{commit}" || { echo "laptop-artifacts commit $C missing"; exit 1; }
if [ ! -d "$W" ]; then git -C "$B/repo" worktree add -q --detach "$W" "$SHA" || exit 1; fi
[ "$(git -C "$W" rev-parse HEAD)" = "$SHA" ] || { echo "checkout is not $SHA"; exit 1; }
P=0.005; IN=$B/runs/predecoder-tcbp/in_test; O=$B/runs/predecoder-tcbp/test_X_p$P; F=$B/logsync/files/predecoder-tcbp/test_X_p$P
[ -e "$O" ] && { echo "refusing: $O exists (this run happens once)"; exit 1; }
mkdir -p "$IN" "$F"; cd "$IN" || exit 1
for f in TEST_SHA256SUMS INNER_SHA256SUMS_test_p$P logits_test_X_p$P.tar.gz.part-00 logits_test_X_p$P.tar.gz.part-01; do git -C "$B/repo" show "$C:predecoder-tcbp/test/$f" > "$f" || exit 1; done
grep "p$P.tar.gz.part-" TEST_SHA256SUMS | sha256sum -c - || exit 1
cat logits_test_X_p$P.tar.gz.part-00 logits_test_X_p$P.tar.gz.part-01 > logits_test_X_p$P.tar.gz && grep " logits_test_X_p$P.tar.gz\$" TEST_SHA256SUMS | sha256sum -c - || exit 1
mkdir -p logits_test_X_p$P && tar xzf logits_test_X_p$P.tar.gz -C logits_test_X_p$P && (cd logits_test_X_p$P && sha256sum -c ../INNER_SHA256SUMS_test_p$P) || exit 1
cd "$W/challenge-1-ml-decoder-kit" || exit 1
printf '%s\n' "1b13521cddb2bf5e5d4c7f1e85a8ecaaa3c72a0b99754a3f76b4a5a4a912fc8b  data/p0.005/test_X.npz" "7697010678a369c77949a6a4233a3087824fa4d2753bcadc4869f2eaee91b0d8  predecoder/logs/fit_tcbp_X_p0.005.json" | sha256sum -c - || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/blakew3/envs/tcbp-py312/bin/python; date -u
"$PY" -m predecoder.tcbp decode --data data/p0.005/test_X.npz --logits "$IN/logits_test_X_p$P" --settings predecoder/logs/fit_tcbp_X_p0.005.json --arms head fresh --batch-check --out "$O" --final > "$O.log" 2>&1 &
pid=$!
while kill -0 "$pid" 2>/dev/null; do cp "$O.log" "$F/decode.log" 2>/dev/null; sleep 60; done
wait "$pid"; rc=$?; echo "decode exit $rc"
cp "$O.log" "$F/decode.log"; cp "$O/per_shot.npz" "$O/summary.json" "$F/" 2>/dev/null
tail -8 "$O.log"; date -u; exit $rc
