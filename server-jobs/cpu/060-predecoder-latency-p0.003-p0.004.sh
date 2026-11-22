#!/bin/bash
# cpu 060: the CLEAN latency of the learned predecoder at p = 0.003 and p = 0.004 (Sal, 15:13 UTC: "so how much faster
# is it on the 0.003 compared to our pipline then?"). Exactly as cpu 050 (which measured p = 0.005): one process, one
# thread, on this server, the first 1,000 shots of the EXPLORATION file data/p<p>/validation_X.npz; no test file is
# decoded again. Times only. Network + head on the CPU in float32; relay-BP built per shot with the head's priors
# (knobs predecoder/logs/fit_X_p0.004.json, as in every test run). Writes only under ~/mlBranchMidwest.
B=$HOME/mlBranchMidwest; SHA=e96cc020f3da414fe3ac3f9cd8f9c030349f0b50; W=$B/pred-lat-e96cc02
HEAD_SHA=38ab667e251c4e7b991e26e9b660c12e80a2d1206589e638c5b372806a648a0d
TRUNK=$B/runs/test/ckpt/x_final.pt; TRUNK_SHA=d8d1c70f552233300f1c10997b69292092042eb8517f5b6526b607c1f6dc4933
H=$B/runs/predecoder/latency_validation_X_p0.005/head_X.pt
date -u
[ "$(git -C "$W" rev-parse HEAD 2>/dev/null)" = "$SHA" ] || { echo "cpu 050's checkout $W is not at $SHA"; exit 1; }
[ "$(sha256sum "$H" | cut -d' ' -f1)" = "$HEAD_SHA" ] || { echo "head.pt sha256 differs"; exit 1; }
[ "$(sha256sum "$TRUNK" | cut -d' ' -f1)" = "$TRUNK_SHA" ] || { echo "trunk sha256 differs"; exit 1; }
cd "$W/challenge-1-ml-decoder-kit" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "machine: load $(cut -d' ' -f1-3 /proc/loadavg)"
rc=0
for p in 0.003 0.004; do
  O=$B/runs/predecoder/latency_validation_X_p$p; F=$B/logsync/files/predecoder/latency_validation_X_p$p; mkdir -p "$O" "$F"
  python -m predecoder.evaluate --head "$H" --trunk "$TRUNK" --data "data/p$p/validation_X.npz" --limit 1000 \
    --out "$O/logits" --device cpu --threads 1 --time-shots 1000 > "$O/evaluate.log" 2>&1; r1=$?
  python -m predecoder.compare --data "data/p$p/validation_X.npz" --logits "$O/logits" --settings predecoder/logs/fit_X_p0.004.json \
    --arms head --workers 1 --chunk 1000 --limit 1000 --out "$O/cmp" > "$O/compare.log" 2>&1; r2=$?
  python - "$O/cmp/per_shot.npz" > "$O/latency.txt" 2>&1 <<'PY'
import sys, numpy as n
d = n.load(sys.argv[1])
t = d["seconds_build_head"] + d["seconds_decode_head"]
print("relay-BP with the head's priors, build + decode per shot, seconds: mean %.4f median %.4f p99 %.4f (n = %d)" % (n.mean(t), n.median(t), n.percentile(t, 99), t.size))
if "converged_head" in d.files:
    c = d["converged_head"].astype(bool)
    print("converged %d of %d; mean on converged %.4f, on unconverged %.4f" % (c.sum(), c.size, n.mean(t[c]), n.mean(t[~c]) if (~c).any() else float("nan")))
PY
  echo "p = $p: evaluate exit $r1, compare exit $r2"; cat "$O/latency.txt"; /usr/bin/env grep -h 'one shot at a time' "$O/evaluate.log"
  cp "$O/evaluate.log" "$O/compare.log" "$O/latency.txt" "$F/"; cp "$O/cmp/per_shot.npz" "$O/cmp/summary.json" "$F/" 2>/dev/null
  [ $r1 -eq 0 ] && [ $r2 -eq 0 ] || rc=1
done
date -u; exit $rc
