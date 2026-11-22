#!/bin/bash
# Pushes training logs, job output and small result files to the server-logs branch every 5 minutes.
cd ~/mlBranchMidwest/logsync || exit 1
mkdir -p files
while true; do
  for d in ~/mlBranchMidwest/runs/*/; do
    n=$(basename "$d")
    [ -f "$d/train.log" ] && cp "$d/train.log" "$n.train.log"
  done
  cp ~/mlBranchMidwest/runs/*.out . 2>/dev/null
  (cd ~/mlBranchMidwest/runs && find . -maxdepth 3 -type f \( -name '*.json' -o -name '*.csv' -o -name 'tables.md' \) -size -1000k -exec cp --parents {} ~/mlBranchMidwest/logsync/files/ \;) 2>/dev/null
  nvidia-smi > nvidia-smi.txt 2>&1
  date -u > updated.txt
  git add *.train.log *.out files nvidia-smi.txt updated.txt sync.sh 2>/dev/null
  git commit -q -m "logs $(date -u +%H:%M)" && git push -q origin server-logs
  sleep 300
done
