#!/bin/bash
# ctl 100: copy the step logs of the run on the test shots to the log branch, for the audit (the END lines of
# test_rows.out keep only each log's last line). A file copy only, no CPU or GPU work: a background loop copies
# runs/test/logs/*.log and runs/test/rehearsal/*/*.log (and the rehearsal folders' .log files) every 5 minutes into
# logsync/files/test-logs/, and once more after "TEST RUN END" appears in runs/test_run.out (or after 8 hours).
B=$HOME/mlBranchMidwest; L=$B/jobs/copy_test_logs.sh
mkdir -p "$B/jobs"
cat > "$L" <<'EOS'
#!/bin/bash
B=$HOME/mlBranchMidwest; T=$B/runs/test; D=$B/logsync/files/test-logs
copy() {
  mkdir -p "$D/logs" "$D/rehearsal"
  cp "$T"/logs/*.log "$D/logs/" 2>/dev/null
  for r in "$T"/rehearsal/*/; do
    [ -d "$r" ] || continue
    n=$(basename "$r"); mkdir -p "$D/rehearsal/$n"
    find "$r" -maxdepth 2 -name '*.log' -exec cp {} "$D/rehearsal/$n/" \; 2>/dev/null
  done
  cp "$T"/*.json "$D/" 2>/dev/null
  date -u > "$D/copied_at.txt"
}
end=$(( $(date +%s) + 8*3600 ))
while [ "$(date +%s)" -lt "$end" ]; do
  copy
  grep -q 'TEST RUN END' "$B/runs/test_run.out" 2>/dev/null && { sleep 30; copy; echo "final copy after TEST RUN END: $(date -u)"; exit 0; }
  sleep 300
done
copy; echo "stopped after 8 hours: $(date -u)"
EOS
chmod +x "$L"
nohup "$L" > "$B/runs/copy_test_logs.out" 2>&1 < /dev/null &
echo "started the copy loop, pid $!"; sleep 5; ls "$B/logsync/files/test-logs/logs" 2>/dev/null | head -30
