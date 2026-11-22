#!/bin/bash
# cpu (the queue only; it uses the A100): task tcbp-noise PHASE 2, exploration on branch infleqtion-noise (Sal, 16:24 UTC, via
# dq-orch-6). The TCBP pipeline (network + head -> per-shot priors -> TCBP-JOBS) on three more pictures of the noise, declared in
# neutral_atom_noise/tcbp_check2_declared.md at 7f44d465 before any number: wrong-p (the kit's validation_X p0.005, first 4,000
# shots, decoded as if p = 0.004), si1000 (p = 0.004, no resonator idle) and biased (Z, eta 100, p = 0.003); arms g / m / h.
# One TCBP model for all three (same H), blocks of 100 taken in turn, no new block after 17:35 UTC.
# Two pythons: the runner's (torch + CUDA) for the network in fp32, the teammates' TCBP env (read and run only) for TCBP-JOBS,
# one process, 1 CPU thread. Writes only under ~/mlBranchMidwest; the folder copied to the log branch holds no logits.
B=$HOME/mlBranchMidwest
PSHA=095c8d228d81aa46632560ac4ef800b8e27d1507    # predecoder-tcbp: predecoder/tcbp.py, evaluate.py, the frozen knobs
NSHA=7f44d4657f51e901e8d1ab56c6903833ad38bb65    # infleqtion-noise: neutral_atom_noise/tcbp_check.py decode-many
A=a07a315be8b180ad03eb23122f6358cda172985d       # laptop-artifacts: the head and trunk (as cpu/090)
A2=8c479023a68b060679f88854912d9102498b3d51  # laptop-artifacts: tcbp-noise-2/ (plan, shots, priors)
WP=$B/tcbp-noise-p095c8d2; WN=$B/tcbp-noise-n7f44d46
R=$B/runs/tcbp-noise; IN=$R/in2; O=$R/out2-7f44d46; F=$B/logsync/files/tcbp-noise-2
[ -e "$O" ] && { echo "$O exists: refusing"; exit 1; }
[ -e "$F" ] && { echo "$F exists: refusing"; exit 1; }
[ -e "$IN" ] && { echo "$IN exists: refusing"; exit 1; }
date -u
fetch() { for _ in 1 2 3; do git -C "$B/repo" fetch -q origin "$1" && return 0; sleep 20; done; return 1; }
git -C "$B/repo" cat-file -e "$PSHA^{commit}" 2>/dev/null || fetch predecoder-tcbp || { echo "cannot fetch predecoder-tcbp"; exit 1; }
git -C "$B/repo" cat-file -e "$NSHA^{commit}" 2>/dev/null || fetch infleqtion-noise || { echo "cannot fetch infleqtion-noise"; exit 1; }
git -C "$B/repo" cat-file -e "$A^{commit}" 2>/dev/null || fetch laptop-artifacts || { echo "cannot fetch laptop-artifacts"; exit 1; }
git -C "$B/repo" cat-file -e "$A2^{commit}" 2>/dev/null || fetch laptop-artifacts || { echo "cannot fetch laptop-artifacts"; exit 1; }
[ -d "$WP" ] || git -C "$B/repo" worktree add -q --detach "$WP" "$PSHA" || exit 1
[ -d "$WN" ] || git -C "$B/repo" worktree add -q --detach "$WN" "$NSHA" || exit 1
[ "$(git -C "$WP" rev-parse HEAD)" = "$PSHA" ] || { echo "checkout is not $PSHA"; exit 1; }
[ "$(git -C "$WN" rev-parse HEAD)" = "$NSHA" ] || { echo "checkout is not $NSHA"; exit 1; }
mkdir -p "$IN/wrong-p" "$IN/si1000" "$IN/biased" "$F" && mkdir "$O" || exit 1
for f in plan.json SHA256SUMS \
         wrong-p/wrong-p_X_kitformat_p0.004.npz wrong-p/priors_generic.npy wrong-p/priors_matched.npy wrong-p/transfer.json \
         si1000/si1000_X_kitformat_p0.004.npz si1000/priors_generic.npy si1000/priors_matched.npy si1000/transfer.json \
         biased/biased_X_kitformat_p0.005991.npz biased/priors_generic.npy biased/priors_matched.npy biased/transfer.json; do
  git -C "$B/repo" cat-file blob "$A2:tcbp-noise-2/$f" > "$IN/$f" || exit 1
done
git -C "$B/repo" cat-file blob "$A:predecoder_X_s1/head.pt" > "$IN/head_X.pt" || exit 1
git -C "$B/repo" cat-file blob "$A:cascade_X_s1/checkpoint_final_step120000.pt" > "$IN/trunk_X.pt" || exit 1
(cd "$IN" && sha256sum -c SHA256SUMS && printf '%s\n' \
  "38ab667e251c4e7b991e26e9b660c12e80a2d1206589e638c5b372806a648a0d  head_X.pt" \
  "d8d1c70f552233300f1c10997b69292092042eb8517f5b6526b607c1f6dc4933  trunk_X.pt" \
  "f31e1d36631d5e7d064a24b91ed323fb3671cc9f76e4e71d5763a8e0e66318d9  plan.json" \
  "d5837d3e8d6e1e3cec65f03a023e8dd1b597c6ea22444476b3d63f067eaea690  wrong-p/wrong-p_X_kitformat_p0.004.npz" \
  "e5c5f764699a61713bfbf3b1bab537df8f49f254b5afee509f94fb5f8012450b  wrong-p/priors_generic.npy" \
  "147d7ef4bf7bb467930395fee39d57d23dd26833f634862aa4eb0b467db9dfc7  wrong-p/priors_matched.npy" \
  "0c2749e91159e788891d31d63c3e6ea8506b58c2cd71de03456a9ba3f644b3d8  wrong-p/transfer.json" \
  "353f291794919fd821ae62917dd6d8dfd893cdf6b522e671e7ee8966ceead5dd  si1000/si1000_X_kitformat_p0.004.npz" \
  "e5c5f764699a61713bfbf3b1bab537df8f49f254b5afee509f94fb5f8012450b  si1000/priors_generic.npy" \
  "433f8003a573f2f9d991c22fe4ab1ab9660863ff20b21f14b0391e2bbaea28b6  si1000/priors_matched.npy" \
  "6d7ca5a6361ef6c8ffa2c8edeaa908a092670b780d2f543746df0a5384a02f26  si1000/transfer.json" \
  "a36b59595c835a608ec72de8fd819cceaf7ef73b872397dc22dee3229fbcea0e  biased/biased_X_kitformat_p0.005991.npz" \
  "822835f794c1d3b88e34385cdbc64ee2561f9496cd0c96e44aa0b5b70dc1c234  biased/priors_generic.npy" \
  "5eb6ab0360be366e489f6f4a369e27ba06669bdcd03e2cf40fc4e5adf6faaf3e  biased/priors_matched.npy" \
  "21e2b0a878470544807e125da56aee63f74b391a28f98f6d63e6a416f47d9504  biased/transfer.json" | sha256sum -c -) || exit 1
cd "$WP/challenge-1-ml-decoder-kit" || exit 1
echo "7697010678a369c77949a6a4233a3087824fa4d2753bcadc4869f2eaee91b0d8  predecoder/logs/fit_tcbp_X_p0.005.json" | sha256sum -c - || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYT=$(command -v python); PY=/home/blakew3/envs/tcbp-py312/bin/python; echo "PYT=$PYT"; date -u
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
for t in wrong-p:wrong-p_X_kitformat_p0.004.npz si1000:si1000_X_kitformat_p0.004.npz biased:biased_X_kitformat_p0.005991.npz; do
  name=${t%%:*}; file=${t#*:}
  "$PYT" -m predecoder.evaluate --head "$IN/head_X.pt" --trunk "$IN/trunk_X.pt" --data "$IN/$name/$file" --device cuda \
    --precision fp32 --threads 1 --time-shots 0 --out "$O/logits/$name" > "$F/evaluate_$name.log" 2>&1; echo "evaluate $name exit $?"
  cp "$O/logits/$name/meta.json" "$F/logits_meta_$name.json" 2>/dev/null
done
PYTHONPATH="$WP/challenge-1-ml-decoder-kit:$WN/challenge-1-ml-decoder-kit" "$PY" -m neutral_atom_noise.tcbp_check decode-many \
  --plan "$IN/plan.json" --root "$IN" --logits-root "$O/logits" --settings predecoder/logs/fit_tcbp_X_p0.005.json \
  --stop-at 2026-11-22T17:35:00Z --out "$O/decode" --mirror "$F" > "$F/decode.log" 2>&1; rc=$?; echo "decode-many exit $rc"
tail -2 "$F"/evaluate_*.log; tail -5 "$F/decode.log"; date -u; exit $rc
