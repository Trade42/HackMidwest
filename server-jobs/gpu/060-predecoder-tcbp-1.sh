#!/bin/bash
# gpu 060: branch predecoder-tcbp, job 1 (the team's request, 14:39 UTC: the network's per-shot priors fed to TCBP-JOBS
# instead of relay-BP, on a separate branch). The A100 and 1 CPU thread; about 9 minutes. VALIDATION shots only.
#   1. the cheaper-path probe (prints only);
#   2. the "fresh" arm on data/p0.005/validation_X.npz: TCBP-JOBS with the kit's priors, a decoder built per shot,
#      decode() one shot at a time, to check it gives the stored 91 (built once, decode_batch). No head, no knobs.
# Runs our code from a detached checkout of the branch at a literal commit, in the teammates' TCBP env (read and run
# only, never written). Writes only under ~/mlBranchMidwest: runs/predecoder-tcbp/ and logsync/files/predecoder-tcbp/.
B=$HOME/mlBranchMidwest; SHA=abb056077e6d285a5453403fc5de4d34020b7ac5; W=$B/pred-tcbp-abb0560
if [ ! -d "$W" ]; then
  git -C "$B/repo" cat-file -e "$SHA^{commit}" 2>/dev/null || git -C "$B/repo" fetch -q origin predecoder-tcbp
  git -C "$B/repo" worktree add -q --detach "$W" "$SHA" || exit 1
fi
[ "$(git -C "$W" rev-parse HEAD)" = "$SHA" ] || { echo "checkout is not $SHA"; exit 1; }
cd "$W/challenge-1-ml-decoder-kit" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/blakew3/envs/tcbp-py312/bin/python; O=$B/runs/predecoder-tcbp; F=$B/logsync/files/predecoder-tcbp
mkdir -p "$O" "$F"; date -u
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
"$PY" -m predecoder.tcbp probe --data data/p0.005/validation_X.npz > "$O/probe.log" 2>&1; echo "probe exit $?"
"$PY" -m predecoder.tcbp decode --data data/p0.005/validation_X.npz --arms fresh --out "$O/validation_X_p0.005" > "$O/validation_fresh.log" 2>&1; rc=$?; echo "decode exit $rc"
cp "$O/probe.log" "$O/validation_fresh.log" "$F/"
cp "$O/validation_X_p0.005/per_shot.npz" "$F/validation_X_p0.005_fresh_per_shot.npz"
cp "$O/validation_X_p0.005/summary.json" "$F/validation_X_p0.005_fresh_summary.json"
tail -5 "$O/validation_fresh.log"; date -u; exit $rc
