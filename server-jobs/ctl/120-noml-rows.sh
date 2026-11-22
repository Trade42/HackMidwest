#!/bin/bash
# ctl 120: the hierarchy WITHOUT the network, as timed rows on the held-out test shots, at the team's request (13:52
# UTC): a clean run of our pipeline at the three noise levels with no ML part, to show how much the network does.
# An ablation run AFTER gpu 050's test run (TEST RUN END 13:05:38 UTC), not part of its frozen plan,
# and labelled so. The decoder is the frozen plugin (hierarchy/ at 921f2e8, in gpu 050's checkout test-frozen at
# fd3aed3) with a hand-written tie-break rule in place of the network; nothing is fitted or chosen:
#   converged  the lowest weight among the converged candidates (the best hand-written rule): one core at p = 0.005
#              (gpu 050 timed it at 0.003 and 0.004), and the two PriorTilt decodes side by side on two cores at all p;
#   weight     the plain lowest weight: one core at all three p.
# It reuses gpu 050's helpers (jobs/test/lib.sh, step.sh, row.sh): the same frozen checkout, single-thread env, CPU gate
# (tokens and measured CPUs), START/END lines in runs/test_rows.out, and copies to files/clean/eval_p<p>_<row>/.
# The driver runs in the background; its record is runs/noml.out; it ends with a "NOML ROWS END" line in test_rows.out.
B=$HOME/mlBranchMidwest; J=$B/jobs/test; T=$B/runs/test
[ -x "$J/row.sh" ] && [ -x "$J/step.sh" ] && [ -s "$J/lib.sh" ] && [ -s "$T/GO" ] || { echo "gpu 050's helpers or its GO file are missing: nothing started"; exit 1; }
grep -q 'TEST RUN END' "$B/runs/test_rows.out" || { echo "gpu 050's test run has not ended: nothing started"; exit 1; }
[ -e "$T/NOML_STARTED" ] && { echo "ctl 120 was started before ($(cat "$T/NOML_STARTED")): nothing started"; exit 0; }
# the same checkout assertions as gpu 050's stage 0, before anything is marked or started
. "$J/lib.sh"
[ "$(git -C "$C" rev-parse HEAD)" = "$FROZEN" ] || { echo "the frozen checkout is at $(git -C "$C" rev-parse HEAD), not at $FROZEN: nothing started"; exit 1; }
[ -z "$(git -C "$C" status --porcelain --untracked-files=no)" ] || { echo "the frozen checkout has local changes: nothing started"; exit 1; }
echo "frozen checkout: $(git -C "$C" log --oneline -1)"
date -u +%FT%TZ > "$T/NOML_STARTED"
cat > "$J/noml.sh" <<'EOS'
#!/bin/bash
J=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$J/lib.sh"
cd "$K" || exit 1
echo '{"rule": "converged", "seeds": [1, 1001]}' > "$T/converged_options.json"
echo '{"rule": "converged", "seeds": [1, 1001], "parallel": true}' > "$T/converged_options_parallel.json"
echo '{"rule": "weight", "seeds": [1, 1001]}' > "$T/weight_options.json"
say "NOML: the hierarchy without the network, after TEST RUN END; frozen $FROZEN [$(state)]"
rowlog "NOML START: the hierarchy without the network (the team's request, 13:52 UTC), an ablation after TEST RUN END; frozen $FROZEN; rules converged and weight"
start_when_room 1 eval_p0.005_hierarchy-converged "$J/row.sh" 0.005 hierarchy.plugin:make_decoder hierarchy-converged "$T/converged_options.json" -
start_when_room 2 eval_p0.005_hierarchy-converged-parallel "$J/row.sh" 0.005 hierarchy.plugin:make_decoder hierarchy-converged-parallel "$T/converged_options_parallel.json" -
start_when_room 1 eval_p0.005_hierarchy-weight "$J/row.sh" 0.005 hierarchy.plugin:make_decoder hierarchy-weight "$T/weight_options.json" -
for p in 0.004 0.003; do
  start_when_room 2 "eval_p${p}_hierarchy-converged-parallel" "$J/row.sh" "$p" hierarchy.plugin:make_decoder hierarchy-converged-parallel "$T/converged_options_parallel.json" -
  start_when_room 1 "eval_p${p}_hierarchy-weight" "$J/row.sh" "$p" hierarchy.plugin:make_decoder hierarchy-weight "$T/weight_options.json" -
done
steps="eval_p0.005_hierarchy-converged"
for p in 0.005 0.004 0.003; do steps="$steps eval_p${p}_hierarchy-converged-parallel eval_p${p}_hierarchy-weight"; done
for n in $steps; do until is_done "$n"; do sleep 30; done; done
say "NOML ROWS END. Exit codes:"
for n in $steps; do echo "  $n: $(cat "$DONE/$n.rc")"; done
rowlog "NOML ROWS END [$(state)]"
EOS
chmod +x "$J/noml.sh"
nohup "$J/noml.sh" > "$B/runs/noml.out" 2>&1 < /dev/null &
echo "noml driver started, pid $!"
sleep 20; cat "$B/runs/noml.out"
