#!/bin/bash
# cpu 050: a CLEAN latency measurement of the learned predecoder (Sal, 14:50 UTC: "did we include the latency from the
# predecoder as well?"). One process, one thread, on this server, like the timed evaluator rows. On the first 1,000
# shots of the EXPLORATION file data/p0.005/validation_X.npz (same distribution as the test shots; no test file is
# decoded again). Times only: the accuracy of these shots is already measured.
#   1. predecoder.evaluate: network + head on the CPU in float32, one shot at a time (mean, median, p99 per shot);
#   2. predecoder.compare --arms head: relay-BP built per shot with the head's priors (build + decode per shot).
# From a detached checkout of branch predecoder at a literal commit. The X head from laptop-artifacts ed20439 and the
# X trunk from gpu 050's copy, both with their sha256 checked. Writes only under ~/mlBranchMidwest.
B=$HOME/mlBranchMidwest; SHA=e96cc020f3da414fe3ac3f9cd8f9c030349f0b50; W=$B/pred-lat-e96cc02
HEAD_COMMIT=ed204392e0d01001b3b70a1d0b0316a0057a6d9f; HEAD_SHA=38ab667e251c4e7b991e26e9b660c12e80a2d1206589e638c5b372806a648a0d
TRUNK=$B/runs/test/ckpt/x_final.pt; TRUNK_SHA=d8d1c70f552233300f1c10997b69292092042eb8517f5b6526b607c1f6dc4933
O=$B/runs/predecoder/latency_validation_X_p0.005; F=$B/logsync/files/predecoder/latency_validation_X_p0.005
date -u
if [ ! -d "$W" ]; then
  git -C "$B/repo" cat-file -e "$SHA^{commit}" 2>/dev/null || git -C "$B/repo" fetch -q origin predecoder
  git -C "$B/repo" worktree add -q --detach "$W" "$SHA" || { echo "the checkout could not be made"; exit 1; }
fi
[ "$(git -C "$W" rev-parse HEAD)" = "$SHA" ] || { echo "the checkout is not at $SHA"; exit 1; }
mkdir -p "$O" "$F"
git -C "$B/repo" cat-file -e "$HEAD_COMMIT^{commit}" 2>/dev/null || git -C "$B/repo" fetch -q origin laptop-artifacts
git -C "$B/repo" show "$HEAD_COMMIT:predecoder_X_s1/head.pt" > "$O/head_X.pt" || { echo "head.pt could not be read"; exit 1; }
[ "$(sha256sum "$O/head_X.pt" | cut -d' ' -f1)" = "$HEAD_SHA" ] || { echo "head.pt sha256 differs"; exit 1; }
[ "$(sha256sum "$TRUNK" | cut -d' ' -f1)" = "$TRUNK_SHA" ] || { echo "trunk sha256 differs"; exit 1; }
echo "checkout $(git -C "$W" log --oneline -1); head $HEAD_SHA; trunk $TRUNK_SHA"
cd "$W/challenge-1-ml-decoder-kit" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "machine: $(nproc) visible CPUs, load $(cut -d' ' -f1-3 /proc/loadavg)"
python -m predecoder.evaluate --head "$O/head_X.pt" --trunk "$TRUNK" --data data/p0.005/validation_X.npz --limit 1000 \
  --out "$O/logits" --device cpu --threads 1 --time-shots 1000 > "$O/evaluate.log" 2>&1; r1=$?; echo "evaluate exit $r1"
python -m predecoder.compare --data data/p0.005/validation_X.npz --logits "$O/logits" --settings predecoder/logs/fit_X_p0.004.json \
  --arms head --workers 1 --chunk 1000 --limit 1000 --out "$O/cmp" > "$O/compare.log" 2>&1; r2=$?; echo "compare exit $r2"
python - "$O/cmp/per_shot.npz" > "$O/latency.txt" 2>&1 <<'PY'
import sys, numpy as n
d = n.load(sys.argv[1])
t = d["seconds_build_head"] + d["seconds_decode_head"]
print("relay-BP with the head's priors, build + decode per shot, seconds: mean %.4f median %.4f p99 %.4f (n = %d)" % (n.mean(t), n.median(t), n.percentile(t, 99), t.size))
if "converged_head" in d.files:
    c = d["converged_head"].astype(bool)
    print("converged %d of %d; mean on converged %.4f, on unconverged %.4f" % (c.sum(), c.size, n.mean(t[c]), n.mean(t[~c]) if (~c).any() else float("nan")))
PY
cp "$O/evaluate.log" "$O/compare.log" "$O/latency.txt" "$F/"; cp "$O/cmp/per_shot.npz" "$O/cmp/summary.json" "$F/" 2>/dev/null
cat "$O/latency.txt"; tail -5 "$O/evaluate.log"; date -u
[ $r1 -eq 0 ] && [ $r2 -eq 0 ]
