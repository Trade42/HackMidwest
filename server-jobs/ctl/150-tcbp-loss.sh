#!/bin/bash
# ctl (uses the A100; ONE TCBP-JOBS process): task tcbp-loss, exploration on branch infleqtion-noise (Sal, 16:24 and 16:27 UTC:
# "it does fit we can do it", "like this stuff is needed they will def be looking at all of this"). The TCBP pipeline with
# atom loss: TCBP-JOBS on question 2's X loss shot files (seed 6100, f0.4 and f0.6, first 1,000 shots each), arms declared in
# neutral_atom_noise/tcbp_loss_declared.md at d733437b before any TCBP number: (bm) matched priors, loss-blind; (fm) matched +
# loss/'s 'starter' flag raise; (fh) the head's priors at the frozen TCBP knobs + the same raise; (bh) head, loss-blind (pass 2).
# Two pythons: the runner's (torch + CUDA) for the network in fp32, the teammates' TCBP env (read and run only) for TCBP-JOBS,
# one process, 1 CPU thread, one model for both files. No new block after 17:35 UTC. Writes only under ~/mlBranchMidwest;
# the folder copied to the log branch holds no logits.
B=$HOME/mlBranchMidwest
PSHA=095c8d228d81aa46632560ac4ef800b8e27d1507    # predecoder-tcbp: predecoder/tcbp.py, evaluate.py, the frozen knobs (as cpu/090)
NSHA=211cfbbfc4f4974bc3d065def481aa8d52ff37ab    # infleqtion-noise: neutral_atom_noise/tcbp_loss.py
A=0b5d6d833dd5ed751dac8d1c62169d5c955ec6e6       # laptop-artifacts: tcbp-loss/ (converted shots, matched vectors, flag factors)
WP=$B/tcbp-loss-p095c8d2; WN=$B/tcbp-loss-n211cfbb
R=$B/runs/tcbp-loss; IN=$R/in; O=$R/out-211cfbb; F=$B/logsync/files/tcbp-loss
[ -e "$O" ] && { echo "$O exists: refusing"; exit 1; }
[ -e "$F" ] && { echo "$F exists: refusing"; exit 1; }
date -u
[ "$(date -u +%s)" -lt "$(date -u -d 2026-11-22T17:35:00Z +%s)" ] || { echo "after 17:35 UTC: not starting"; exit 1; }
fetch() { for i in 1 2 3; do git -C "$B/repo" fetch -q origin "$1" && return 0; sleep 20; done; return 1; }
git -C "$B/repo" cat-file -e "$PSHA^{commit}" 2>/dev/null || fetch predecoder-tcbp || { echo "cannot fetch predecoder-tcbp"; exit 1; }
git -C "$B/repo" cat-file -e "$NSHA^{commit}" 2>/dev/null || fetch infleqtion-noise || { echo "cannot fetch infleqtion-noise"; exit 1; }
git -C "$B/repo" cat-file -e "$A^{commit}" 2>/dev/null || fetch laptop-artifacts || { echo "cannot fetch laptop-artifacts"; exit 1; }
[ -d "$WP" ] || git -C "$B/repo" worktree add -q --detach "$WP" "$PSHA" || exit 1
[ -d "$WN" ] || git -C "$B/repo" worktree add -q --detach "$WN" "$NSHA" || exit 1
[ "$(git -C "$WP" rev-parse HEAD)" = "$PSHA" ] || { echo "checkout is not $PSHA"; exit 1; }
[ "$(git -C "$WN" rev-parse HEAD)" = "$NSHA" ] || { echo "checkout is not $NSHA"; exit 1; }
mkdir -p "$IN" "$F" && mkdir "$O" || exit 1
for f in X_f0.4_loss_seed6100_first1000_kitformat_p0.0059563.npz X_f0.6_loss_seed6100_first1000_kitformat_p0.0089274.npz \
         priors_matched_f0.4.npy priors_matched_f0.6.npy flags_f0.4.npz flags_f0.6.npz transfer.json SHA256SUMS; do
  git -C "$B/repo" cat-file blob "$A:tcbp-loss/$f" > "$IN/$f" || exit 1
done
git -C "$B/repo" cat-file blob "$A:predecoder_X_s1/head.pt" > "$IN/head_X.pt" || exit 1
git -C "$B/repo" cat-file blob "$A:cascade_X_s1/checkpoint_final_step120000.pt" > "$IN/trunk_X.pt" || exit 1
(cd "$IN" && sha256sum -c SHA256SUMS && printf '%s\n' \
  "dccb3df5a4d96494de9e97b17ede03007593c196be7335506ddcb29a2ae2dc9a  SHA256SUMS" \
  "38ab667e251c4e7b991e26e9b660c12e80a2d1206589e638c5b372806a648a0d  head_X.pt" \
  "d8d1c70f552233300f1c10997b69292092042eb8517f5b6526b607c1f6dc4933  trunk_X.pt" \
  "4f3ddc6ba7c318d61abf2f230c17462c584798a9e2093a9d3706ab2d392a472d  flags_f0.4.npz" \
  "5d4629bb3168b99bac25a6f637400c28b4653a94cecc1017ac15776330ef66a4  flags_f0.6.npz" \
  "1539aa06c4cf839191424de372a1785238c32d5aeece283b4b343ff822e9819f  X_f0.4_loss_seed6100_first1000_kitformat_p0.0059563.npz" \
  "97f0ef3644d45a06627519e01788e677480b6af86b278697f04fddc7d6c7fe34  X_f0.6_loss_seed6100_first1000_kitformat_p0.0089274.npz" \
  "f918a92a4e70fafcbb9d30b14098080e3cdd1697805102f61a9b52f511952fde  priors_matched_f0.4.npy" \
  "2f0e3823912ed1cdcc2150ed5b69d22945496205ad0f9b8aa68bd392445d4625  priors_matched_f0.6.npy" \
  "ce358b741bf12a44f67a25b53307a38a36a44bfeaf9bfb7b333ac9bcf92b3193  transfer.json" | sha256sum -c -) || exit 1
cd "$WP/challenge-1-ml-decoder-kit" || exit 1
echo "7697010678a369c77949a6a4233a3087824fa4d2753bcadc4869f2eaee91b0d8  predecoder/logs/fit_tcbp_X_p0.005.json" | sha256sum -c - || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYT=$(command -v python); PY=/home/blakew3/envs/tcbp-py312/bin/python; echo "PYT=$PYT"; date -u
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
for t in f0.4 f0.6; do
  D=$(ls "$IN"/X_${t}_loss_seed6100_first1000_kitformat_p*.npz)
  "$PYT" -m predecoder.evaluate --head "$IN/head_X.pt" --trunk "$IN/trunk_X.pt" --data "$D" --device cuda --precision fp32 \
    --threads 1 --time-shots 100 --out "$O/logits_$t" > "$F/evaluate_$t.log" 2>&1; echo "evaluate $t fp32 exit $?"
  cp "$O/logits_$t/meta.json" "$F/logits_meta_$t.json" 2>/dev/null
done
PYTHONPATH="$WP/challenge-1-ml-decoder-kit:$WN/challenge-1-ml-decoder-kit" "$PY" -m neutral_atom_noise.tcbp_loss decode \
  --transfer "$IN/transfer.json" --logits "$O/logits_f0.4" "$O/logits_f0.6" \
  --settings predecoder/logs/fit_tcbp_X_p0.005.json --stop-at 2026-11-22T17:35:00Z \
  --out "$O/decode" --mirror "$F" > "$F/decode.log" 2>&1; echo "decode exit $?"
cp "$O/decode/per_shot_f0.4.npz" "$O/decode/per_shot_f0.6.npz" "$O/decode/summary.json" "$F/" 2>/dev/null
tail -3 "$F/evaluate_f0.4.log" "$F/evaluate_f0.6.log" "$F/decode.log"; date -u
