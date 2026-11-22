#!/bin/bash
# cpu 040: a labelled CONFIRMATION of the p = 0.003 baseline times (BP-OSD, relay-BP, the teammates' router) on the
# test shots, in a quiet window. Their first times are clean by the corrected throttle rule (at most 1.2 to 2.5% of
# one process's time); these rows only confirm them. The same frozen checkout (clean-base) and the same command
# lines as job cpu 020 (its runs/clean/run_row.sh), new folders eval_p0.003_<row>_retime. Answers are deterministic:
# dq-report checks the failure masks equal the first run's. A background waiter starts the three rows side by side
# only after the run on the test shots has ended (TEST RUN END in runs/test_run.out) and job 020's last row too
# (ALL ROWS DONE in runs/clean_rows.out), and gives up at 16:30 UTC. Records in runs/clean_retime.out.
B=$HOME/mlBranchMidwest; OUT=$B/runs/clean
[ -x "$OUT/run_row.sh" ] || { echo "no $OUT/run_row.sh (job cpu 020 writes it)"; exit 1; }
mkdir -p "$B/jobs"
cat > "$B/jobs/retime_p0.003.sh" <<'EOS'
#!/bin/bash
B=$HOME/mlBranchMidwest; OUT=$B/runs/clean
stop=$(date -u -d '2026-11-22 16:30:00' +%s)
echo "$(date -u +%FT%TZ) waiting for TEST RUN END and ALL ROWS DONE"
until grep -q 'TEST RUN END' "$B/runs/test_run.out" 2>/dev/null && grep -q 'ALL ROWS DONE' "$B/runs/clean_rows.out" 2>/dev/null; do
  [ "$(date +%s)" -ge "$stop" ] && { echo "$(date -u +%FT%TZ) gave up: 16:30 UTC reached"; exit 0; }
  sleep 60
done
sleep 60
echo "$(date -u +%FT%TZ) quiet window: the three p = 0.003 rows side by side (one process each)"
"$OUT/run_row.sh" 0.003 bposd bposd_retime - &
sleep 5
"$OUT/run_row.sh" 0.003 relay-bp relay-bp_retime - &
sleep 5
"$OUT/run_row.sh" 0.003 evaluation.router:make_tiered_decoder router_retime results/router_p0.003/eval_options.json &
wait
echo "$(date -u +%FT%TZ) RETIME ROWS DONE"
EOS
chmod +x "$B/jobs/retime_p0.003.sh"
nohup "$B/jobs/retime_p0.003.sh" > "$B/runs/clean_retime.out" 2>&1 < /dev/null &
echo "started the waiter, pid $!"; sleep 3; cat "$B/runs/clean_retime.out"
