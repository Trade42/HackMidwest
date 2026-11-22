#!/bin/bash
# 000: how old is this notebook (ICRN ends one 24 hours after its start), what does it have, and a faster log sync.
B=$HOME/mlBranchMidwest
echo "now: $(date -u)"
echo "uptime: $(uptime)"
echo "pid 1: $(ps -o lstart=,etime=,args= -p 1 | cut -c1-160)"
echo "oldest processes of this user:"; ps -u "$(id -un)" -o pid,lstart,etime,args --sort=start_time | head -9 | cut -c1-170
echo "cpus visible: $(nproc); cgroup cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"
free -g | sed -n '1,2p'; df -h "$HOME" | tail -1
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null
echo "python: $(which python) $(python -V 2>&1)"
echo "runners: $(pgrep -af 'jobs/runner.sh' | cut -c1-80 | tr '\n' ';')"
echo "loss chain alive (pid 32825): $(kill -0 32825 2>/dev/null && echo yes || echo no)"
cd "$B/logsync" || exit 1
old=$(cat sync.pid 2>/dev/null)
[ -n "$old" ] && kill "$old" 2>/dev/null
sleep 5
[ -f .git/index.lock ] && ! pgrep -f 'git .*(commit|add|push)' > /dev/null && rm -f .git/index.lock
sed -i 's/sleep 300/sleep 60/' sync.sh
grep -n 'sleep' sync.sh
nohup ./sync.sh > sync.out 2>&1 &
echo $! > sync.pid
echo "sync loop restarted with pid $(cat sync.pid): every 60 s from now"
