#!/bin/bash
# cpu (the queue only; it uses the A100): task tcbp-noise, exploration on branch infleqtion-noise (Sal, 15:46 UTC: "even with
# the noise we can test our updated tcbp full pipeline"). The TCBP pipeline (network + head -> per-shot priors -> TCBP-JOBS)
# on question 1's neutral-atom-shaped X shot file at factor 0.8 (2,000 shots, seed 4100), three arms declared in
# neutral_atom_noise/tcbp_check_declared.md at c4b82a0a before any TCBP number: (g) the kit's uniform priors at BP-OSD's
# fitted p 0.0326371, (m) matched priors, (h) the head's priors at the frozen TCBP knobs (no refit).
# Two pythons: the runner's (torch + CUDA) for the network in fp32, the teammates' TCBP env (read and run only) for TCBP-JOBS,
# one process, 1 CPU thread. Waits for the GPU slice. Writes only under ~/mlBranchMidwest; the folder copied to the log
# branch holds no logits.
B=$HOME/mlBranchMidwest
PSHA=095c8d228d81aa46632560ac4ef800b8e27d1507    # predecoder-tcbp: predecoder/tcbp.py, evaluate.py, the frozen knobs
NSHA=6569bd7bfd7db8db8f682b853c7f77a16c67da97    # infleqtion-noise: neutral_atom_noise/tcbp_check.py
A=a07a315be8b180ad03eb23122f6358cda172985d       # laptop-artifacts: tcbp-noise/ (shots, priors) and the head and trunk
WP=$B/tcbp-noise-p095c8d2; WN=$B/tcbp-noise-n6569bd7
R=$B/runs/tcbp-noise; IN=$R/in; O=$R/out-6569bd7; F=$B/logsync/files/tcbp-noise
[ -e "$O" ] && { echo "$O exists: refusing"; exit 1; }
[ -e "$F" ] && { echo "$F exists: refusing"; exit 1; }
# the GPU slice holds at most two TCBP-JOBS processes: wait for job 6 (test_X p = 0.003) to finish, at most 25 minutes
W0=$(date +%s)
until [ -f "$B/runs/predecoder-tcbp/test_X_p0.003/summary.json" ] || [ $(( $(date +%s) - W0 )) -ge 1500 ]; do sleep 30; done
echo "waited $(( $(date +%s) - W0 )) s"; date -u
fetch() { for i in 1 2 3; do git -C "$B/repo" fetch -q origin "$1" && return 0; sleep 20; done; return 1; }
git -C "$B/repo" cat-file -e "$PSHA^{commit}" 2>/dev/null || fetch predecoder-tcbp || { echo "cannot fetch predecoder-tcbp"; exit 1; }
git -C "$B/repo" cat-file -e "$NSHA^{commit}" 2>/dev/null || fetch infleqtion-noise || { echo "cannot fetch infleqtion-noise"; exit 1; }
git -C "$B/repo" cat-file -e "$A^{commit}" 2>/dev/null || fetch laptop-artifacts || { echo "cannot fetch laptop-artifacts"; exit 1; }
[ -d "$WP" ] || git -C "$B/repo" worktree add -q --detach "$WP" "$PSHA" || exit 1
[ -d "$WN" ] || git -C "$B/repo" worktree add -q --detach "$WN" "$NSHA" || exit 1
[ "$(git -C "$WP" rev-parse HEAD)" = "$PSHA" ] || { echo "checkout is not $PSHA"; exit 1; }
[ "$(git -C "$WN" rev-parse HEAD)" = "$NSHA" ] || { echo "checkout is not $NSHA"; exit 1; }
mkdir -p "$IN" "$F" && mkdir "$O" || exit 1
for f in X_f0.8_seed4100.npz X_f0.8_seed4100_kitformat_p0.0326371.npz priors_generic.npy priors_matched.npy transfer.json SHA256SUMS; do
  git -C "$B/repo" cat-file blob "$A:tcbp-noise/$f" > "$IN/$f" || exit 1
done
git -C "$B/repo" cat-file blob "$A:predecoder_X_s1/head.pt" > "$IN/head_X.pt" || exit 1
git -C "$B/repo" cat-file blob "$A:cascade_X_s1/checkpoint_final_step120000.pt" > "$IN/trunk_X.pt" || exit 1
(cd "$IN" && sha256sum -c SHA256SUMS && printf '%s\n' \
  "38ab667e251c4e7b991e26e9b660c12e80a2d1206589e638c5b372806a648a0d  head_X.pt" \
  "d8d1c70f552233300f1c10997b69292092042eb8517f5b6526b607c1f6dc4933  trunk_X.pt" \
  "4e6e2867557ced29a25b1c0380b1aa2b6f2d9922bdb4fe1300e2de0a409365e8  X_f0.8_seed4100.npz" \
  "a251dfe065d8c4ab427671930e6e13f66182a1144acee6980f3b9d3329f11141  X_f0.8_seed4100_kitformat_p0.0326371.npz" \
  "b8d59cb850ee6842011283d559b33e560c07ceecd2edfed2f10617ce6557a9ea  priors_generic.npy" \
  "45704fcb9f2bde6fdbc6a7f0fa975f7e343c5283179e65a4cf0af45a741fc3ac  priors_matched.npy" \
  "97ad92c19f5d478e0f90590dc3494b8c343f634c83da4e9218ad0571c6d63897  transfer.json" | sha256sum -c -) || exit 1
cd "$WP/challenge-1-ml-decoder-kit" || exit 1
echo "7697010678a369c77949a6a4233a3087824fa4d2753bcadc4869f2eaee91b0d8  predecoder/logs/fit_tcbp_X_p0.005.json" | sha256sum -c - || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYT=$(command -v python); PY=/home/blakew3/envs/tcbp-py312/bin/python; echo "PYT=$PYT"; date -u
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
D=$IN/X_f0.8_seed4100_kitformat_p0.0326371.npz
"$PYT" -m predecoder.evaluate --head "$IN/head_X.pt" --trunk "$IN/trunk_X.pt" --data "$D" --device cuda --precision fp32 \
  --threads 1 --time-shots 100 --out "$O/logits" > "$F/evaluate.log" 2>&1; echo "evaluate fp32 exit $?"
cp "$O/logits/meta.json" "$F/logits_meta.json" 2>/dev/null
PYTHONPATH="$WP/challenge-1-ml-decoder-kit:$WN/challenge-1-ml-decoder-kit" "$PY" -m neutral_atom_noise.tcbp_check decode \
  --data "$D" --generic "$IN/priors_generic.npy" --matched "$IN/priors_matched.npy" --transfer "$IN/transfer.json" \
  --logits "$O/logits" --settings predecoder/logs/fit_tcbp_X_p0.005.json --stop-at 2026-11-22T17:12:00Z \
  --out "$O/decode" --mirror "$F" > "$F/decode.log" 2>&1; rc=$?; echo "decode exit $rc"
cp "$O/decode/per_shot.npz" "$O/decode/summary.json" "$F/" 2>/dev/null
tail -3 "$F/evaluate.log" "$F/decode.log"; date -u; exit $rc
