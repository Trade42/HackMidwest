#!/bin/bash
# gpu (uses the A100): task tcbp-noise PHASE 2b, exploration on branch infleqtion-noise (dq-orch-6's yes, 16:45 UTC). SI1000 at
# p = 0.006 and 0.008 (3,000 shots each, seeds 7101 and 7102), arms g / m / h, declared in neutral_atom_noise/tcbp_check2b_declared.md
# at 0a780d77 before any number of them: the strengths were chosen after phase 2's p = 0.004 failed 0 in every arm (a bracket).
# One TCBP model for both, blocks of 100 taken in turn, no new block after 17:35 UTC. It first WAITS until the loss job (ctl/150) has
# ended (~/mlBranchMidwest/logsync/files/tcbp-loss/summary.json exists and no process of this user has 'tcbp_loss' in its command
# line), at most 30 minutes, and does nothing if that never happens: the GPU slice must never hold more than two TCBP processes.
# Two pythons: the runner's (torch + CUDA) for the network in fp32, the teammates' TCBP env (read and run only) for TCBP-JOBS,
# one process, 1 CPU thread. Writes only under ~/mlBranchMidwest; the folder copied to the log branch holds no logits.
B=$HOME/mlBranchMidwest
PSHA=095c8d228d81aa46632560ac4ef800b8e27d1507    # predecoder-tcbp: predecoder/tcbp.py, evaluate.py, the frozen knobs
NSHA=0a780d77588d27451af13ca2327ec4b43948e930    # infleqtion-noise: neutral_atom_noise/tcbp_check.py decode-many
A=a07a315be8b180ad03eb23122f6358cda172985d       # laptop-artifacts: the head and trunk (as cpu/090)
A2=53942fd4d2bef61a02eb7b81b1f03c3024dbee1e # laptop-artifacts: tcbp-noise-2b/ (plan, shots, priors)
WP=$B/tcbp-noise-p095c8d2; WN=$B/tcbp-noise-n0a780d7
R=$B/runs/tcbp-noise; IN=$R/in2b; O=$R/out2b-0a780d7; F=$B/logsync/files/tcbp-noise-2b
[ -e "$O" ] && { echo "$O exists: refusing"; exit 1; }
[ -e "$F" ] && { echo "$F exists: refusing"; exit 1; }
[ -e "$IN" ] && { echo "$IN exists: refusing"; exit 1; }
date -u
W0=$(date +%s)
until [ -f "$B/logsync/files/tcbp-loss/summary.json" ] && ! pgrep -u "$(id -un)" -f 'tcbp_loss' > /dev/null; do
  [ $(( $(date +%s) - W0 )) -ge 1800 ] && { echo "the loss job has not ended after 30 min: not running (slice limit)"; exit 1; }
  sleep 30
done
echo "waited $(( $(date +%s) - W0 )) s for the loss job to end"; date -u
fetch() { for _ in 1 2 3; do git -C "$B/repo" fetch -q origin "$1" && return 0; sleep 20; done; return 1; }
git -C "$B/repo" cat-file -e "$PSHA^{commit}" 2>/dev/null || fetch predecoder-tcbp || { echo "cannot fetch predecoder-tcbp"; exit 1; }
git -C "$B/repo" cat-file -e "$NSHA^{commit}" 2>/dev/null || fetch infleqtion-noise || { echo "cannot fetch infleqtion-noise"; exit 1; }
git -C "$B/repo" cat-file -e "$A^{commit}" 2>/dev/null || fetch laptop-artifacts || { echo "cannot fetch laptop-artifacts"; exit 1; }
git -C "$B/repo" cat-file -e "$A2^{commit}" 2>/dev/null || fetch laptop-artifacts || { echo "cannot fetch laptop-artifacts"; exit 1; }
[ -d "$WP" ] || git -C "$B/repo" worktree add -q --detach "$WP" "$PSHA" || exit 1
[ -d "$WN" ] || git -C "$B/repo" worktree add -q --detach "$WN" "$NSHA" || exit 1
[ "$(git -C "$WP" rev-parse HEAD)" = "$PSHA" ] || { echo "checkout is not $PSHA"; exit 1; }
[ "$(git -C "$WN" rev-parse HEAD)" = "$NSHA" ] || { echo "checkout is not $NSHA"; exit 1; }
mkdir -p "$IN/si1000-p0.006" "$IN/si1000-p0.008" "$F" && mkdir "$O" || exit 1
for f in plan.json SHA256SUMS \
         si1000-p0.006/si1000-p0.006_X_kitformat_p0.006.npz si1000-p0.006/priors_generic.npy si1000-p0.006/priors_matched.npy si1000-p0.006/transfer.json \
         si1000-p0.008/si1000-p0.008_X_kitformat_p0.008.npz si1000-p0.008/priors_generic.npy si1000-p0.008/priors_matched.npy si1000-p0.008/transfer.json; do
  git -C "$B/repo" cat-file blob "$A2:tcbp-noise-2b/$f" > "$IN/$f" || exit 1
done
git -C "$B/repo" cat-file blob "$A:predecoder_X_s1/head.pt" > "$IN/head_X.pt" || exit 1
git -C "$B/repo" cat-file blob "$A:cascade_X_s1/checkpoint_final_step120000.pt" > "$IN/trunk_X.pt" || exit 1
(cd "$IN" && sha256sum -c SHA256SUMS && printf '%s\n' \
  "38ab667e251c4e7b991e26e9b660c12e80a2d1206589e638c5b372806a648a0d  head_X.pt" \
  "d8d1c70f552233300f1c10997b69292092042eb8517f5b6526b607c1f6dc4933  trunk_X.pt" \
  "c67712a5ff0ca90f95bd7bb830572940ca0a4ab0e76dd885f2af1c53fa949b58  plan.json" \
  "52393629898c4e148da6c21d7a07eeed611153c7daabe6b73d0da0ae3eb90635  si1000-p0.006/priors_generic.npy" \
  "c6e19dae49ede1b8531ae4ed0b996ec8698961411671cffcdd197eebe1268d6d  si1000-p0.006/priors_matched.npy" \
  "747dfdf598edffe70bf1cbe9a1479f5e8c43d46134112f5b23609eed17cad16b  si1000-p0.006/si1000-p0.006_X_kitformat_p0.006.npz" \
  "667ede80c48591a53d7733f315b9b667f40f1b9a6d09f4d69ffede24bed1c9d6  si1000-p0.006/transfer.json" \
  "37845de4b2cadad14f9016bef48aea529f6892fcd974ca36cb699fb2dcbd9b61  si1000-p0.008/priors_generic.npy" \
  "a5c2a0f5b867b36850eec77fffea4f659799b66480d85b4156e53d8b90f855ea  si1000-p0.008/priors_matched.npy" \
  "20a819d501b760de0b640053177ad3cb37412287174f720fd170a178d354bc4d  si1000-p0.008/si1000-p0.008_X_kitformat_p0.008.npz" \
  "66b363d9fe984f120a22605f4e8d616693c2040ae615655744637c4d6a757070  si1000-p0.008/transfer.json" | sha256sum -c -) || exit 1
cd "$WP/challenge-1-ml-decoder-kit" || exit 1
echo "7697010678a369c77949a6a4233a3087824fa4d2753bcadc4869f2eaee91b0d8  predecoder/logs/fit_tcbp_X_p0.005.json" | sha256sum -c - || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYT=$(command -v python); PY=/home/blakew3/envs/tcbp-py312/bin/python; echo "PYT=$PYT"; date -u
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
for t in si1000-p0.006:si1000-p0.006_X_kitformat_p0.006.npz si1000-p0.008:si1000-p0.008_X_kitformat_p0.008.npz; do
  name=${t%%:*}; file=${t#*:}
  "$PYT" -m predecoder.evaluate --head "$IN/head_X.pt" --trunk "$IN/trunk_X.pt" --data "$IN/$name/$file" --device cuda \
    --precision fp32 --threads 1 --time-shots 0 --out "$O/logits/$name" > "$F/evaluate_$name.log" 2>&1; echo "evaluate $name exit $?"
  cp "$O/logits/$name/meta.json" "$F/logits_meta_$name.json" 2>/dev/null
done
PYTHONPATH="$WP/challenge-1-ml-decoder-kit:$WN/challenge-1-ml-decoder-kit" "$PY" -m neutral_atom_noise.tcbp_check decode-many \
  --plan "$IN/plan.json" --root "$IN" --logits-root "$O/logits" --settings predecoder/logs/fit_tcbp_X_p0.005.json \
  --stop-at 2026-11-22T17:35:00Z --out "$O/decode" --mirror "$F" > "$F/decode.log" 2>&1; rc=$?; echo "decode-many exit $rc"
tail -2 "$F"/evaluate_*.log; tail -5 "$F/decode.log"; date -u; exit $rc
