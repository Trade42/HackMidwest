#!/bin/bash
# The server replication (seed 7310), from the branch origin/server-logs: copy its files, rebuild its tables from
# its per-shot parts here, and pool it with the test (seed 7300). No decoding.
# usage: loss_tracker/runs/replication.sh [COMMIT_OF_SERVER_LOGS]   (default origin/server-logs)
source loss_tracker/runs/common.sh
source loss_tracker/runs/frozen.sh
REF=${1:-origin/server-logs}
R=$S/replication/rep_seed7310
mkdir -p "$R/parts" "$S/replication/rep_seed7310_coupled2/parts"
TOP=$(git rev-parse --show-toplevel)            # paths on the branch are relative to the repository's root
git -C "$TOP" fetch -q origin server-logs
for f in $(git -C "$TOP" ls-tree -r --name-only "$REF" files/loss_tracker/rep_seed7310/ files/loss_tracker/rep_seed7310_coupled2/); do
  git -C "$TOP" show "$REF:$f" > "$S/replication/${f#files/loss_tracker/}"
done
git -C "$TOP" show "$REF:job_ctl_060-loss-tracker-replication.out" > loss_tracker/logs/server_job_ctl_060.out
echo "server-logs at $(git -C "$TOP" rev-parse "$REF")"
cp "$R/accuracy.md" loss_tracker/results/replication_seed7310_accuracy.md
cp "$R/accuracy.json" loss_tracker/results/replication_seed7310_accuracy.json
cp "$R/tables.md" loss_tracker/results/replication_seed7310_server_tables.md
# its tables rebuilt here from its parts, with the same rows as the test's
"$PY" -m loss_tracker.summarize --parts "$R/parts/*.npz" --references exact,starter --order "starter,exact,$VARIANT,oracle-round" \
  --cascade "$VARIANT>exact" --cascade "exact>starter" --cascade "oracle-round>exact" \
  --title "Replication on the team's server (seed 7310, 4,000 shots): the frozen decoder against the references" \
  --out loss_tracker/results/replication_seed7310
# the secondary row (the earlier tracker version) on the same replication shots
"$PY" -m loss_tracker.summarize --parts "$R/parts/*.npz" --references exact,starter --order "starter,exact,$VARIANT,oracle-round" \
  --also "tracker_order1_coupled2=tracker@$S/replication/rep_seed7310_coupled2/parts/*.npz" --pair "$VARIANT,tracker_order1_coupled2" \
  --title "Replication (seed 7310): with the SECONDARY row tracker_order1_coupled2 (not the frozen decoder)" \
  --out loss_tracker/results/replication_seed7310_secondary
# pooled with the test
"$PY" -m loss_tracker.pool --set "$S/parts/test/*.npz" --set "$R/parts/*.npz" --scratch "$S/replication/pooled_parts" -- \
  --references exact,starter --order "starter,exact,$VARIANT,oracle-round" --cascade "$VARIANT>exact" \
  --title "Pooled: the test (seed 7300, this PC) and the replication (seed 7310, the server), 8,000 shots" --out loss_tracker/results/pooled_7300_7310
