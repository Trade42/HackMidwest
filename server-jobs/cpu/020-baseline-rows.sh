#!/bin/bash
# 020: the baseline rows on the saved TEST shots, through the team's evaluator (evaluation.evaluate), from a frozen
# checkout: BP-OSD, relay-BP and the teammates' frozen router at p = 0.003, 0.004, 0.005. One process per row, one
# thread each, at most 6 side by side. It starts them in the background and returns; their START and END lines go to
# runs/clean_rows.out, each row's summary and per-shot file to the log branch under files/clean/.
# These decoders are fixed (the kit's settings, the teammates' frozen models): nothing is tuned here.
B=$HOME/mlBranchMidwest; C=$B/clean-base; OUT=$B/runs/clean
sha=$(git -C "$B/jobrepo-cpu" rev-parse HEAD)
[ -d "$C" ] || git -C "$B/repo" worktree add -q --detach "$C" "$sha"
echo "frozen checkout: $C at $(git -C "$C" log --oneline -1)"; date -u
mkdir -p "$OUT"
echo '{"router_models":{"X":"results/router_p0.005/model_X.json","Z":"results/router_p0.005/model_Z.json"}}' > "$OUT/router_options_p0.005.json"
cat > "$OUT/run_row.sh" <<'EOS'
#!/bin/bash
# one evaluator row: p, decoder spec, row name, plugin options file or "-"
p=$1; dec=$2; name=$3; opt=$4
B=$HOME/mlBranchMidwest; K=$B/clean-base/challenge-1-ml-decoder-kit; OUT=$B/runs/clean; F=$B/logsync/files/clean
cd "$K" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
d="$OUT/eval_p${p}_$name"
if [ -e "$d" ]; then echo "$(date -u +%FT%TZ) SKIP p=$p $name: $d exists"; exit 0; fi
args=(--test-x "data/p$p/test_X.npz" --test-z "data/p$p/test_Z.npz" --decoder "$dec" --out "$d")
[ "$opt" != "-" ] && args+=(--plugin-options "$opt")
echo "$(date -u +%FT%TZ) START p=$p $name with $(pgrep -fc 'evaluation.evaluate') evaluator process(es) already running"
python -m evaluation.evaluate "${args[@]}" > "$OUT/eval_p${p}_$name.log" 2>&1
rc=$?
echo "$(date -u +%FT%TZ) END p=$p $name exit $rc: $(tail -n +2 "$d/summary.csv" 2>/dev/null | cut -d, -f1,7-11,12 | head -1)"
mkdir -p "$F/eval_p${p}_$name"
cp "$d"/summary.json "$d"/summary.csv "$d"/*_shots.npz "$F/eval_p${p}_$name/" 2>/dev/null
[ $rc -ne 0 ] && tail -5 "$OUT/eval_p${p}_$name.log"
exit 0
EOS
chmod +x "$OUT/run_row.sh"
cat > "$OUT/rows.txt" <<EOS
0.005 relay-bp relay-bp -
0.005 bposd bposd -
0.005 evaluation.router:make_tiered_decoder router $OUT/router_options_p0.005.json
0.004 relay-bp relay-bp -
0.004 bposd bposd -
0.004 evaluation.router:make_tiered_decoder router results/router_p0.004/eval_options.json
0.003 bposd bposd -
0.003 relay-bp relay-bp -
0.003 evaluation.router:make_tiered_decoder router results/router_p0.003/eval_options.json
EOS
nohup sh -c "xargs -P 6 -L 1 '$OUT/run_row.sh' < '$OUT/rows.txt'; echo \"\$(date -u +%FT%TZ) ALL ROWS DONE\"" > "$B/runs/clean_rows.out" 2>&1 &
echo "rows started in the background, pid $!"
sleep 20; cat "$B/runs/clean_rows.out"
