#!/bin/bash
# Copies the training logs to the server-logs branch every 5 minutes.
cd ~/mlBranchMidwest/logsync || exit 1
while true; do
  for d in ~/mlBranchMidwest/runs/*/; do
    n=$(basename "$d")
    [ -f "$d/train.log" ] && cp "$d/train.log" "$n.train.log"
  done
  cp ~/mlBranchMidwest/runs/*.out . 2>/dev/null
  nvidia-smi > nvidia-smi.txt 2>&1
  date -u > updated.txt
  git add *.train.log *.out nvidia-smi.txt updated.txt sync.sh 2>/dev/null
  git commit -q -m "logs $(date -u +%H:%M)" && git push -q origin server-logs
  sleep 300
done
