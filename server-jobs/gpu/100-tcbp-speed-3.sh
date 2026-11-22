#!/bin/bash
# gpu 100: tcbp-speed job 3 (branch tcbp-speed fba1ca00; an improvement branch, merged nowhere): (a) the network + head at batch
# one in fp32: eager / cached / dense / CUDA graph / torch.compile, logits compared with eager, the graph arm decoded with
# TCBP-JOBS on 1,000 shots; (b) a probe for changing a built decoder's error rates; (c) the tcbp preset (no OSD/JOBS) with
# the head's priors on 2,000 validation_X p = 0.005 shots (accuracy experiment). VALIDATION shots only; each step under
# its own timeout; 1 CPU thread; A100. Writes only under ~/mlBranchMidwest.
# the GPU slice holds at most two TCBP-JOBS processes: wait (up to 15 min) for tcbp-speed job 2 (ctl/170) to end
W0=$(date +%s); while pgrep -u "$(id -un)" -f 'predecoder[.]tcbp_speed' > /dev/null; do [ $(( $(date +%s) - W0 )) -ge 900 ] && { echo "job 2 still running after 15 min: not starting"; exit 1; }; sleep 15; done; echo "waited $(( $(date +%s) - W0 )) s"
B=$HOME/mlBranchMidwest; SHA=fba1ca00c741a13627c4d8f53d36fcddb951cd54; W=$B/tcbp-speed-fba1ca0; C=666f10bf27939a74df6c6c68bf332af7a8a6aa77
if [ ! -d "$W" ]; then git -C "$B/repo" cat-file -e "$SHA^{commit}" 2>/dev/null || git -C "$B/repo" fetch -q origin tcbp-speed; git -C "$B/repo" worktree add -q --detach "$W" "$SHA" || exit 1; fi
[ "$(git -C "$W" rev-parse HEAD)" = "$SHA" ] || exit 1
IN=$B/runs/tcbp-speed/in; O=$B/runs/tcbp-speed/job3; F=$B/logsync/files/tcbp-speed/job3
[ -e "$O" ] && { echo "refusing: $O exists"; exit 1; }
mkdir -p "$O" "$F" "$IN"
if ! (cd "$IN" && printf '%s\n' "38ab667e251c4e7b991e26e9b660c12e80a2d1206589e638c5b372806a648a0d  head_X.pt" "d8d1c70f552233300f1c10997b69292092042eb8517f5b6526b607c1f6dc4933  trunk_X.pt" | sha256sum -c - 2>/dev/null); then
  git -C "$B/repo" cat-file -e "$C^{commit}" 2>/dev/null || git -C "$B/repo" fetch -q origin laptop-artifacts
  git -C "$B/repo" show "$C:predecoder_X_s1/head.pt" > "$IN/head_X.pt"; git -C "$B/repo" show "$C:cascade_X_s1/checkpoint_final_step120000.pt" > "$IN/trunk_X.pt"
  (cd "$IN" && printf '%s\n' "38ab667e251c4e7b991e26e9b660c12e80a2d1206589e638c5b372806a648a0d  head_X.pt" "d8d1c70f552233300f1c10997b69292092042eb8517f5b6526b607c1f6dc4933  trunk_X.pt" | sha256sum -c -) || exit 1
fi
(cd "$IN/validation_X_p0.005" && sha256sum -c ../INNER_SHA256SUMS_validation) || exit 1
cd "$W/challenge-1-ml-decoder-kit" || exit 1
printf '%s\n' "25e9ce8e28714c985e20f9f9a017dd49ff25a26200626a1b830cfee0cc045312  data/p0.005/validation_X.npz" "7697010678a369c77949a6a4233a3087824fa4d2753bcadc4869f2eaee91b0d8  predecoder/logs/fit_tcbp_X_p0.005.json" | sha256sum -c - || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYT=$(command -v python); PY=/home/blakew3/envs/tcbp-py312/bin/python; echo "PYT=$PYT"; date -u; nvidia-smi -L
timeout -k 30 300 "$PYT" -X faulthandler -m predecoder.fast_network --head "$IN/head_X.pt" --trunk "$IN/trunk_X.pt" --data data/p0.005/validation_X.npz --limit 1000 --arms eager cached dense graph --out "$O/net" > "$O/net.log" 2>&1; ra=$?; echo "a network exit $ra"; date -u
cp "$O/net.log" "$O/net/summary.json" "$F/" 2>/dev/null; tail -6 "$O/net.log"
timeout -k 30 240 "$PYT" -X faulthandler -m predecoder.fast_network --head "$IN/head_X.pt" --trunk "$IN/trunk_X.pt" --data data/p0.005/validation_X.npz --limit 1000 --arms compile --reference "$O/net/eager" --out "$O/net_compile" > "$O/net_compile.log" 2>&1; rc=$?; echo "a compile exit $rc"; date -u
cp "$O/net_compile.log" "$F/"; cp "$O/net_compile/summary.json" "$F/summary_compile.json" 2>/dev/null; tail -4 "$O/net_compile.log"
timeout -k 30 120 "$PY" -m predecoder.tcbp_speed probe-rates --data data/p0.005/validation_X.npz --out "$O/probe_rates.json" > "$O/probe_rates.log" 2>&1; rb=$?; echo "b probe exit $rb"
cp "$O/probe_rates.log" "$O/probe_rates.json" "$F/" 2>/dev/null; tail -3 "$O/probe_rates.log"
timeout -k 30 300 "$PY" -m predecoder.tcbp decode --data data/p0.005/validation_X.npz --logits "$IN/validation_X_p0.005" --settings predecoder/logs/fit_tcbp_X_p0.005.json --arms head --preset tcbp --h-format csr --limit 2000 --out "$O/bp_only" > "$O/bp_only.log" 2>&1; rcc=$?; echo "c bp-only exit $rcc"; date -u
cp "$O/bp_only.log" "$F/"; cp "$O/bp_only/per_shot.npz" "$F/bp_only_per_shot.npz" 2>/dev/null; cp "$O/bp_only/summary.json" "$F/bp_only_summary.json" 2>/dev/null; tail -3 "$O/bp_only.log"
timeout -k 30 300 "$PY" -m predecoder.tcbp decode --data data/p0.005/validation_X.npz --logits "$O/net/graph" --settings predecoder/logs/fit_tcbp_X_p0.005.json --arms head --h-format csr --limit 1000 --out "$O/decode_graph" > "$O/decode_graph.log" 2>&1; rd=$?; echo "a graph-logits decode exit $rd"; date -u
cp "$O/decode_graph.log" "$F/"; cp "$O/decode_graph/per_shot.npz" "$F/decode_graph_per_shot.npz" 2>/dev/null; cp "$O/decode_graph/summary.json" "$F/decode_graph_summary.json" 2>/dev/null; tail -3 "$O/decode_graph.log"
echo "exit codes: network $ra, compile $rc, probe $rb, bp-only $rcc, graph decode $rd"
