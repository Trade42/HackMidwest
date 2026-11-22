#!/bin/bash
# gpu 030: starts a one-line-per-minute record of how busy the notebook is, runs/cpu_monitor.out (the log branch gets it
# with every sync). Per line: the CPUs used on average over the last minute, from the cgroup's own counter (the quota is
# 12); how long and how often the quota throttled the notebook in that minute; how many of our processes of each kind
# are alive. It only reads. It ends by itself after 30 hours, or when ~/mlBranchMidwest/jobs/MONITOR-STOP exists.
B=$HOME/mlBranchMidwest
if pgrep -f 'jobs/cpu_monitor[.]sh' > /dev/null; then echo "the monitor is already running"; exit 0; fi
cat > "$B/jobs/cpu_monitor.sh" <<'EOS'
#!/bin/bash
B=$HOME/mlBranchMidwest; S=${CPU_STAT:-/sys/fs/cgroup/cpu.stat}; EVERY=${EVERY:-60}
get() { awk -v k="$1" '$1==k{print $2; f=1} END{if(!f) print 0}' "$S" 2>/dev/null || echo 0; }
n() { pgrep -fc "$1"; }
echo "# one line per minute. cpus: CPUs used on average over the minute (the quota is 12). thr_s and thr_n: seconds and periods the quota throttled the notebook in the minute."
echo "# time_utc cpus thr_s thr_n | evaluator joint loss_job candidates tcbp_member evaluate_stim python_all | load_1m"
u0=$(get usage_usec); t0=$(get throttled_usec); k0=$(get nr_throttled); s0=$(date +%s); end=$(( s0 + 108000 ))
while [ "$(date +%s)" -lt "$end" ] && [ ! -e "$B/jobs/MONITOR-STOP" ]; do
  sleep "$EVERY"
  u1=$(get usage_usec); t1=$(get throttled_usec); k1=$(get nr_throttled); s1=$(date +%s)
  c="$(n 'evaluation[.]evaluate') $(n 'hierarchy[.]joint_member') $(n 'loss[.]job') $(n 'hierarchy[.]candidates') $(n 'hierarchy[.]tcbp_member') $(n 'neural[.]evaluate_stim') $(n 'python')"
  awk -v u0="$u0" -v u1="$u1" -v t0="$t0" -v t1="$t1" -v k0="$k0" -v k1="$k1" -v s0="$s0" -v s1="$s1" \
      -v d="$(date -u +%FT%TZ)" -v c="$c" -v l="$(cut -d' ' -f1 /proc/loadavg)" \
      'BEGIN{dt=s1-s0; if(dt<1) dt=1; printf "%s %.1f %.1f %d | %s | %s\n", d, (u1-u0)/1e6/dt, (t1-t0)/1e6, k1-k0, c, l}'
  u0=$u1; t0=$t1; k0=$k1; s0=$s1
done
EOS
chmod +x "$B/jobs/cpu_monitor.sh"
nohup "$B/jobs/cpu_monitor.sh" > "$B/runs/cpu_monitor.out" 2>&1 < /dev/null &
echo "monitor started, pid $!"; date -u
echo "cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>&1)"
echo "cpu.stat:"; cat /sys/fs/cgroup/cpu.stat 2>&1 | head -12
