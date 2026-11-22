#!/bin/bash
# ctl 090: put the final Z checkpoint on the log branch, so that the submission can name a commit for it as it does
# for the X checkpoint (laptop-artifacts). A file copy only: no CPU or GPU work. The file is the one the run on the
# test shots copied (runs/test/ckpt/z_final.pt, sha256 fad24e80...0700, step 100,000), checked before and after.
B=$HOME/mlBranchMidwest; SRC=$B/runs/test/ckpt/z_final.pt; D=$B/logsync/files/ckpt
WANT=fad24e80ec4e2c5ea7b54717c6b309b244d90ee9fb50a60128da997840230700
date -u
got=$(sha256sum "$SRC" | cut -d' ' -f1)
[ "$got" = "$WANT" ] || { echo "the source's sha256 is $got, not $WANT: nothing copied"; exit 1; }
mkdir -p "$D"
cp "$SRC" "$D/cascade_Z_s1_final_step100000.pt.tmp" && mv "$D/cascade_Z_s1_final_step100000.pt.tmp" "$D/cascade_Z_s1_final_step100000.pt"
cp "$B/runs/cascade_Z_s1/train.log" "$D/cascade_Z_s1_train.log"
( cd "$D" && sha256sum cascade_Z_s1_final_step100000.pt cascade_Z_s1_train.log > SHA256SUMS_Z && cat SHA256SUMS_Z )
ls -la "$D"
date -u
