#!/bin/bash
# ctl 050: a DRY RUN on VALIDATION shots only, with today's checkpoints (X: the laptop's step-70,000 snapshot; Z: the
# copy of step 22,000 that gpu 020 made), of what the run on the test shots will do first:
#   1. the decoder's preflight with TCBP-JOBS in serial mode on the GPU (it shares the GPU with the Z training here);
#   2. the preflight with the side-by-side option (TCBP-JOBS replaced by its stand-in);
#   3. a rehearsal of the three timed rows through the team's evaluator on the p = 0.003 validation files.
# The stages run one after the other, on at most 4 CPUs, beside the timed baseline rows (6) and the training (1).
# No test file is opened. Files: files/clean/rehearsal_dry_validation_p0.003_<row>/ on the log branch.
B=$HOME/mlBranchMidwest; K=$B/jobrepo-ctl/challenge-1-ml-decoder-kit; R=$B/runs; D=$R/preflight_dry; F=$B/logsync/files/clean
cd "$K" || exit 1
date -u; echo "checkout: $(git -C "$B/jobrepo-ctl" log --oneline -1)"
export PRIORTILT_REPO=$HOME/PriorTiltDecoder PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$D"
X=$D/x_step70000.pt; Z=$R/netZ/z_checkpoint_copy.pt
if [ ! -s "$X" ]; then
  git -C "$B/repo" fetch -q origin laptop-artifacts
  git -C "$B/repo" show d4971bac0344dc22c975343e3e01d5b518ca8a31:cascade_X_s1/checkpoint_step70000.pt > "$X" || { echo "the X snapshot could not be read"; exit 1; }
fi
got=$(sha256sum "$X" | cut -d' ' -f1)
[ "$got" = 692a796f94401dbdd7d522967fb407e14f75a2d6537613263edfd0f4896b12f8 ] || { echo "the X snapshot's sha256 is $got, not the laptop's"; exit 1; }
ls -la "$X" "$Z" || exit 1
printf '{"rule": {"X": "network", "Z": "network"}, "seeds": [1, 1001], "checkpoints": {"X": "%s", "Z": "%s"}, "device": "cpu", "torch_threads": 1}\n' "$X" "$Z" > "$D/options.json"
printf '{"rule": {"X": "network", "Z": "network"}, "seeds": [1, 1001], "checkpoints": {"X": "%s", "Z": "%s"}, "device": "cpu", "torch_threads": 1, "parallel": true}\n' "$X" "$Z" > "$D/options_parallel.json"
cat "$D/options.json" "$D/options_parallel.json"
worst=0
stamp=$(date -u +%H%M%S)

echo; echo "===== 1. the preflight, TCBP-JOBS in serial mode on the GPU ====="; date -u
python -m hierarchy.preflight --options "$D/options.json" --x data/p0.004/validation_X.npz --z data/p0.004/validation_Z.npz --rule f \
  --tcbp-python /home/blakew3/envs/tcbp-py312/bin/python --out "$D/out_$stamp"
rc=$?; echo "preflight 1: exit $rc"; [ $rc -ne 0 ] && worst=$rc

echo; echo "===== 2. the preflight with the side-by-side option, TCBP-JOBS as its stand-in ====="; date -u
python -m hierarchy.preflight --options "$D/options_parallel.json" --x data/p0.004/validation_X.npz --z data/p0.004/validation_Z.npz --rule f \
  --tcbp-stand-in --out "$D/out_parallel_$stamp"
rc=$?; echo "preflight 2: exit $rc"; [ $rc -ne 0 ] && worst=$rc

echo; echo "===== 3. the three timed rows through the evaluator, p = 0.003 validation files ====="; date -u
reh() {  # row name, decoder spec, options file
  python -m evaluation.evaluate --test-x data/p0.003/validation_X.npz --test-z data/p0.003/validation_Z.npz --decoder "$2" --plugin-options "$3" \
    --out "$D/rehearsal_${stamp}_$1" > "$D/rehearsal_${stamp}_$1.log" 2>&1
}
reh hierarchy hierarchy.plugin:make_decoder "$D/options.json" & r1=$!
reh hierarchy-parallel hierarchy.plugin:make_decoder "$D/options_parallel.json" & r2=$!
reh priortilt hierarchy.priortilt_plugin:make_decoder "$D/options.json" & r3=$!
wait $r1; c1=$?; wait $r2; c2=$?; wait $r3; c3=$?
echo "rehearsal exit codes (hierarchy, hierarchy-parallel, priortilt): $c1 $c2 $c3"
[ "$c1$c2$c3" = 000 ] || worst=1
for n in hierarchy hierarchy-parallel priortilt; do
  echo "--- $n"; cat "$D/rehearsal_${stamp}_$n/summary.csv" 2>/dev/null || tail -n 12 "$D/rehearsal_${stamp}_$n.log"
  mkdir -p "$F/rehearsal_dry_validation_p0.003_$n"
  cp "$D/rehearsal_${stamp}_$n"/summary.json "$D/rehearsal_${stamp}_$n"/summary.csv "$F/rehearsal_dry_validation_p0.003_$n/" 2>/dev/null
done
echo; echo "dry run: worst exit $worst"; date -u
exit $worst
