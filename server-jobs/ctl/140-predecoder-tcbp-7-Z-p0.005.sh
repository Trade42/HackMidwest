#!/bin/bash
# ctl 140: branch predecoder-tcbp, job 7 = the registered TEST RUN 3 (an extension; Z declarations at 76521b78):
# TCBP-JOBS with the Z head's per-shot priors, X's frozen TCBP knobs (no Z fit), Z at p = 0.005, with a gate inside:
# step 1 validation_Z head arm and its paired score vs the stored TCBP-JOBS validation_Z file; step 2 test_Z (--final)
# ONLY IF step 1 has fewer failures, paired p < 0.05; else SKIPPED and test_Z is not opened. Designed after the test
# results were seen; a separate branch. Logits: laptop-artifacts 690bbd716e0dda2962a38e7453327410446f4706 predecoder-tcbp/z/ (every sha256 checked).
# It first waits (up to 40 min) for gpu 080 (job 5) to END, so at most two TCBP-JOBS processes share the A100 slice
# and no two jobs fetch or add a checkout at once. A100, 1 CPU thread, about 15 min. Writes only under ~/mlBranchMidwest.
for i in $(seq 1 80); do [ -e "$HOME/mlBranchMidwest/runs/predecoder-tcbp/test_X_p0.005/summary.json" ] && break; sleep 30; done
sleep 30
B=$HOME/mlBranchMidwest; SHA=76521b78fb537a5db483ed3a0472e2c6fb0ddb21; W=$B/pred-tcbp-76521b7; C=690bbd716e0dda2962a38e7453327410446f4706
if [ ! -d "$W" ]; then git -C "$B/repo" cat-file -e "$SHA^{commit}" 2>/dev/null || git -C "$B/repo" fetch -q origin predecoder-tcbp; git -C "$B/repo" worktree add -q --detach "$W" "$SHA" || exit 1; fi
[ "$(git -C "$W" rev-parse HEAD)" = "$SHA" ] || exit 1
git -C "$B/repo" cat-file -e "$C^{commit}" 2>/dev/null || git -C "$B/repo" fetch -q origin laptop-artifacts
IN=$B/runs/predecoder-tcbp/in_z; OV=$B/runs/predecoder-tcbp/validation_Z_p0.005; OT=$B/runs/predecoder-tcbp/test_Z_p0.005; F=$B/logsync/files/predecoder-tcbp/z
[ -e "$OV" ] || [ -e "$OT" ] && { echo "refusing: an output folder exists (this run happens once)"; exit 1; }
mkdir -p "$IN" "$F"; cd "$IN" || exit 1
for f in Z_SHA256SUMS INNER_SHA256SUMS_validation_Z INNER_SHA256SUMS_test_Z logits_validation_Z_p0.005.tar.gz logits_test_Z_p0.005.tar.gz.part-00 logits_test_Z_p0.005.tar.gz.part-01; do git -C "$B/repo" show "$C:predecoder-tcbp/z/$f" > "$f" || exit 1; done
grep -v ' logits_test_Z_p0.005.tar.gz$' Z_SHA256SUMS | sha256sum -c - || exit 1
cat logits_test_Z_p0.005.tar.gz.part-00 logits_test_Z_p0.005.tar.gz.part-01 > logits_test_Z_p0.005.tar.gz && grep ' logits_test_Z_p0.005.tar.gz$' Z_SHA256SUMS | sha256sum -c - || exit 1
mkdir -p logits_validation_Z logits_test_Z && tar xzf logits_validation_Z_p0.005.tar.gz -C logits_validation_Z && tar xzf logits_test_Z_p0.005.tar.gz -C logits_test_Z || exit 1
(cd logits_validation_Z && sha256sum -c ../INNER_SHA256SUMS_validation_Z) && (cd logits_test_Z && sha256sum -c ../INNER_SHA256SUMS_test_Z) || exit 1
VT=$B/runs/tcbp_p0.005/validation_Z.tcbp.npz
echo "b0ba027a704c734a8c77b55e557b698dfc851e27e273b07df623cbbf76b27619  $VT" | sha256sum -c - || { git -C "$B/repo" fetch -q origin server-logs && git -C "$B/repo" show origin/server-logs:files/tcbp_p0.005/validation_Z.tcbp.npz > "$IN/validation_Z.tcbp.npz" && VT=$IN/validation_Z.tcbp.npz && echo "b0ba027a704c734a8c77b55e557b698dfc851e27e273b07df623cbbf76b27619  $VT" | sha256sum -c - || exit 1; }
cd "$W/challenge-1-ml-decoder-kit" || exit 1
printf '%s\n' "24f12efa48d92b7a97767a36679be9216d75d5e6b88502e39f684ec17417774e  data/p0.005/validation_Z.npz" "b86ae320d13fa555d8358fe399156d28d4e52b01fabf8eafa1b30e60bdcd0cef  data/p0.005/test_Z.npz" "7697010678a369c77949a6a4233a3087824fa4d2753bcadc4869f2eaee91b0d8  predecoder/logs/fit_tcbp_X_p0.005.json" | sha256sum -c - || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/blakew3/envs/tcbp-py312/bin/python; date -u
# step 1: validation_Z, the head arm, X's knobs
"$PY" -m predecoder.tcbp decode --data data/p0.005/validation_Z.npz --logits "$IN/logits_validation_Z" --settings predecoder/logs/fit_tcbp_X_p0.005.json --arms head --out "$OV" > "$OV.log" 2>&1 || { echo "step 1 failed"; cp "$OV.log" "$F/"; exit 1; }
"$PY" -m predecoder.tcbp score --data data/p0.005/validation_Z.npz --tcbp-head "$OV/per_shot.npz" --tcbp "$VT" --out "$OV/score.json" > "$OV.score.log" 2>&1 || exit 1
cp "$OV.log" "$OV.score.log" "$F/"; cp "$OV/per_shot.npz" "$F/validation_Z_per_shot.npz"; cp "$OV/summary.json" "$F/validation_Z_summary.json"; cp "$OV/score.json" "$F/validation_Z_score.json"
cat "$OV.score.log"
GATE=$("$PY" -c "import json; q=json.load(open('$OV/score.json'))['pairs']['TCBP-JOBS, head priors per shot | TCBP-JOBS alone (stored)']; print('GO' if q['only_first'] < q['only_second'] and q['p'] < 0.05 else 'NO')")
echo "gate (validation_Z: TH fewer failures than the stored TCBP-JOBS, paired, p < 0.05): $GATE"; date -u
if [ "$GATE" != GO ]; then echo "SKIPPED: test_Z is not opened"; exit 0; fi
# step 2: test_Z, the head arm, once
"$PY" -m predecoder.tcbp decode --data data/p0.005/test_Z.npz --logits "$IN/logits_test_Z" --settings predecoder/logs/fit_tcbp_X_p0.005.json --arms head --out "$OT" --final > "$OT.log" 2>&1
rc=$?; echo "step 2 exit $rc"
cp "$OT.log" "$F/test_Z.log"; cp "$OT/per_shot.npz" "$F/test_Z_per_shot.npz"; cp "$OT/summary.json" "$F/test_Z_summary.json"
tail -4 "$OT.log"; date -u; exit $rc
