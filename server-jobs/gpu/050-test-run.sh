#!/bin/bash
# gpu 050: THE RUN ON THE HELD-OUT TEST SHOTS, once. It writes its scripts under ~/mlBranchMidwest/jobs/test<TAG>/,
# starts the driver in the background and returns. The driver's record is runs/test_run<TAG>.out; every step's START
# and END line is in runs/test_rows<TAG>.out; files go to the log branch under files/clean<TAG>/. With DRY=1 (the dry-run
# job) it stops at GO: nothing of the test shots is opened.
#
# Order, and what protects the test shots:
#   0. a frozen checkout at a literal commit, asserted;
#   1. the two final checkpoints: X from laptop-artifacts with its sha256 checked, Z after its training has ended;
#   2. on EXPLORATION shots only: the final networks' logits, and from them the plugin's rule per basis;
#   3. the preflight and a rehearsal of the timed rows, both on validation shots. If either does not pass, nothing
#      below runs and no test file is opened;
#   4. on the test shots: TCBP-JOBS in serial mode on the idle GPU, then the networks' logits; the timed evaluator
#      rows; the members' candidates; per noise level the selection lists, joint decoding where planned, the ensemble
#      with the frozen rule and the pair score.
# Every step runs once (a finished step is skipped if the driver is started again), on one thread per process, and
# starts only when the notebook's 12 CPUs have room for it.
# ---- the values of this run (the dry-run job is this file with the block below changed) ----
TAG=                                   # "" for the real run; "-dry" for a dry run
DRY=0                                  # 1: stop at GO, before any test file is opened
FROZEN=fd3aed3c0d1f2e9e8bd181bd22c9853a19e2dfc5        # integration: the first commit that holds the frozen hierarchy/
HIER_FROZEN=921f2e8f8e05eff4ca3f60b565e48bbc08d42628   # the decoder's frozen commit (branch hierarchy)
ROWS_BASE=472b18f56775025ccb85db467005f72801fffbd0     # the checkout the timed baseline rows run from (job cpu 020)
X_COMMIT=2f752e9367f80e1180711ded6b6ca854a5c0c147                  # the laptop-artifacts commit with the final X checkpoint
X_PATH=cascade_X_s1/checkpoint_final_step120000.pt
X_SHA256=d8d1c70f552233300f1c10997b69292092042eb8517f5b6526b607c1f6dc4933
Z_GIVEN=                               # dry run only: a Z checkpoint file to use instead of waiting for the training
Z_CUTOFF='2026-11-22 10:20:00'            # if the Z training still runs then (UTC), it is stopped by its STOP file
WAIT_S=1800                            # the longest wait for CPU room before a stage-3 step goes on anyway
# ---------------------------------------------------------------------------------------------
B=$HOME/mlBranchMidwest; J=$B/jobs/test$TAG; T=$B/runs/test$TAG
mkdir -p "$J" "$T"
if [ -e "$T/STARTED" ]; then echo "this run was started before ($(cat "$T/STARTED")); this job starts nothing"; exit 0; fi
date -u +%FT%TZ > "$T/STARTED"

cat > "$J/config.sh" <<EOS
TAG='$TAG'
DRY=$DRY
FROZEN=$FROZEN
HIER_FROZEN=$HIER_FROZEN
ROWS_BASE=$ROWS_BASE
X_COMMIT=$X_COMMIT
X_PATH=$X_PATH
X_SHA256=$X_SHA256
Z_GIVEN='$Z_GIVEN'
Z_CUTOFF='$Z_CUTOFF'
WAIT_S=$WAIT_S
RULE=f                                 # the ensemble's frozen rule
Z_RUN=\$HOME/mlBranchMidwest/runs/cascade_Z_s1
Z_PID=51025                            # the Z training's python process (nvidia-smi lists it)
TCBP_PY=/home/blakew3/envs/tcbp-py312/bin/python
PS="0.005 0.004 0.003"                 # the longest first
JOINT_PS="0.003 0.004"                 # joint decoding only at these noise levels
BUDGET_CPUS=10.8                       # no step starts if the measured CPU use plus its weight would pass this
BUDGET_TOKENS=10                       # the weights of our own running steps never sum above this
EOS

cat > "$J/lib.sh" <<'EOS'
. "$J/config.sh"
B=$HOME/mlBranchMidwest; T=$B/runs/test$TAG; R=$B/runs; F=$B/logsync/files/clean$TAG
C=$B/test-frozen$TAG; K=$C/challenge-1-ml-decoder-kit
export PRIORTILT_REPO=$HOME/PriorTiltDecoder PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
CPU_STAT=${CPU_STAT:-/sys/fs/cgroup/cpu.stat}
TOK=$T/tokens; DONE=$T/done; LOGS=$T/logs; ROWS=$R/test_rows$TAG.out
mkdir -p "$TOK" "$DONE" "$LOGS" "$F"
now() { date -u +%FT%TZ; }
say() { echo "$(now) $*"; }
rowlog() { echo "$(now) $*" >> "$ROWS"; }
cpus_used() {  # the CPUs the notebook used on average over the last $1 seconds, from the cgroup's counter
  local a b
  a=$(awk '$1=="usage_usec"{print $2}' "$CPU_STAT" 2>/dev/null); sleep "$1"
  b=$(awk '$1=="usage_usec"{print $2}' "$CPU_STAT" 2>/dev/null)
  awk -v a="${a:-0}" -v b="${b:-0}" -v s="$1" 'BEGIN{printf "%.1f", (b-a)/1e6/s}'
}
throttled() { awk '$1=="throttled_usec"{printf "%.1f", $2/1e6; f=1} END{if(!f) printf "0.0"}' "$CPU_STAT" 2>/dev/null; }
tokens() {  # the summed weights of our steps that are still running
  local s=0 f pid
  for f in "$TOK"/[0-9]*; do
    [ -e "$f" ] || continue
    pid=${f##*/}
    if kill -0 "$pid" 2>/dev/null; then s=$(( s + $(cat "$f") )); else rm -f "$f"; fi
  done
  echo "$s"
}
state() { echo "CPUs in use $(cpus_used 5), tokens $(tokens), throttled so far $(throttled) s, load $(cut -d' ' -f1 /proc/loadavg)"; }
is_done() { [ -e "$DONE/$1.rc" ]; }
wait_done() {  # waits until every named step has ended; fails if one of them did not exit 0
  local n ok=0
  for n in "$@"; do until is_done "$n"; do sleep 20; done; [ "$(cat "$DONE/$n.rc")" = 0 ] || ok=1; done
  return $ok
}
wait_room() {  # weight [seconds]: waits until the measured CPU use leaves room for this weight, at most the given seconds
  local w=$1 end=$(( $(date +%s) + ${2:-$WAIT_S} )) u
  while :; do
    u=$(cpus_used 15)
    awk -v u="$u" -v w="$w" -v bu="$BUDGET_CPUS" 'BEGIN{exit !(u+w<=bu)}' && return 0
    [ "$(date +%s)" -ge "$end" ] && { say "no room for weight $w after waiting (CPUs in use $u): going on"; return 0; }
    sleep 15
  done
}
gpu_procs() { nvidia-smi 2>/dev/null | awk '/Processes:/{f=1} f && /MiB/{printf "pid %s %s; ", $5, $(NF-1); n++} END{if(!n) printf "none"}'; }
start_when_room() {  # weight name command...: starts the command in the background as a step, once there is room
  local w=$1 name=$2; shift 2
  if is_done "$name"; then say "SKIP $name: it ended before with exit $(cat "$DONE/$name.rc")"; return 0; fi
  (
    flock 9
    while :; do
      t=$(tokens); u=$(cpus_used 15)
      awk -v t="$t" -v u="$u" -v w="$w" -v bt="$BUDGET_TOKENS" -v bu="$BUDGET_CPUS" 'BEGIN{exit !((t+w<=bt) && (u+w<=bu))}' && break
      sleep 15
    done
    nohup "$J/step.sh" "$name" "$@" > "$LOGS/$name.log" 2>&1 < /dev/null 9>&- &
    echo "$w" > "$TOK/$!"
    rowlog "START $name: weight $w, pid $!; before it: tokens $t, CPUs in use $u, throttled so far $(throttled) s"
    sleep 25   # let it show in the CPU counter before the next start is measured
  ) 9> "$T/room.lock"
}
EOS

cat > "$J/step.sh" <<'EOS'
#!/bin/bash
# runs one step from the kit folder of the frozen checkout; records its exit code and its END line
J=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$J/lib.sh"
name=$1; shift
cd "$K" || exit 1
"$@"
rc=$?
echo "$rc" > "$DONE/$name.rc"
rowlog "END $name exit $rc: $(tail -n 1 "$LOGS/$name.log" 2>/dev/null | cut -c1-160) [$(state)]"
exit $rc
EOS

cat > "$J/row.sh" <<'EOS'
#!/bin/bash
# one evaluator row on the test shots: p, decoder spec, row name, options file or "-", "metrics" to copy *_metrics.json
J=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$J/lib.sh"
p=$1; dec=$2; name=$3; opt=$4; metrics=$5
d="$T/eval_p${p}_$name"
if [ -s "$d/summary.csv" ]; then echo "SKIP p=$p $name: $d holds a finished row"; exit 0; fi
if [ -e "$d" ]; then echo "STOP p=$p $name: $d exists without a summary (an unfinished earlier attempt): not decoded again without the orchestrator's word"; exit 2; fi
args=(--test-x "data/p$p/test_X.npz" --test-z "data/p$p/test_Z.npz" --decoder "$dec" --out "$d")
[ "$opt" != "-" ] && args+=(--plugin-options "$opt")
python -m evaluation.evaluate "${args[@]}"
rc=$?
mkdir -p "$F/eval_p${p}_$name"
cp "$d"/summary.json "$d"/summary.csv "$d"/*_shots.npz "$F/eval_p${p}_$name/" 2>/dev/null
[ "$metrics" = metrics ] && cp "$d"/*_metrics.json "$F/eval_p${p}_$name/" 2>/dev/null
echo "row p=$p $name exit $rc: $(tail -n +2 "$d/summary.csv" 2>/dev/null | cut -d, -f1,7-11,12 | head -1)"
exit $rc
EOS

cat > "$J/commands.sh" <<'EOS'
# ---- the commands on the test files, as the decoder's author gave them at the freeze. Run from the kit folder. ----
# (README of the frozen commit, section "The run on the held-out shots: frozen commit and exact commands", steps 2 to 4;
#  the process counts are ours: the README's 12 would take the whole machine)
cand_file() { echo "$T/cand_p$1/test_$2.chunks.npz"; }                                       # p basis
cmd_candidates() { python -m hierarchy.candidates --final --data "data/p$1/test_X.npz" "data/p$1/test_Z.npz" --out "$T/cand_p$1" --processes "$2"; }   # p processes; chunk mode: relay-BP on every shot
cmd_tcbp() { "$TCBP_PY" -m hierarchy.tcbp_member --final --data "data/p$1/test_X.npz" "data/p$1/test_Z.npz" --out "$T/tcbp_p$1"; }   # serial mode (no --batch): a time per shot
cmd_logits() {   # p: both bases, on the GPU
  mkdir -p "$T/net_p$1"
  python -m neural.evaluate_stim --final --checkpoint "$T/ckpt/x_final.pt" --data "data/p$1/test_X.npz" --save "$T/net_p$1/netX_test_X" &&
  python -m neural.evaluate_stim --final --checkpoint "$T/ckpt/z_final.pt" --data "data/p$1/test_Z.npz" --save "$T/net_p$1/netZ_test_Z"
}
cmd_selection() { python -m hierarchy.ensemble --members "$(cand_file "$1" "$2")" "$T/tcbp_p$1/test_$2.tcbp.npz" --write-selection "$T/sel_p$1/test_$2.selection.npz"; }   # p basis
cmd_joint() { python -m hierarchy.joint_member --final --data "data/p$1/test_X.npz" "data/p$1/test_Z.npz" --selection "$T/sel_p$1/test_X.selection.npz" "$T/sel_p$1/test_Z.selection.npz" --out "$T/joint_p$1" --processes 4; }
cmd_ensemble() {   # p basis joint|nojoint
  local members=("$(cand_file "$1" "$2")" "$T/tcbp_p$1/test_$2.tcbp.npz") extra=()
  if [ "$3" = joint ]; then members+=("$T/joint_p$1/test_$2.joint.npz"); else extra=(--no-joint --write-selection "$T/sel_p$1/test_$2.selection.npz"); fi
  python -m hierarchy.ensemble --members "${members[@]}" --logits "$T/net_p$1/net$2_test_$2.npz" --rule "$RULE" "${extra[@]}" --save "$T/ensemble_p$1/test_$2.json"
}
cmd_pair() { python -m hierarchy.ensemble --pair "$T/ensemble_p$1/test_X.npz" "$T/ensemble_p$1/test_Z.npz"; }
EOS

cat > "$J/assemble.sh" <<'EOS'
#!/bin/bash
# one noise level: the selection lists, joint decoding where planned, the ensemble with the frozen rule, the pair score
J=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$J/lib.sh"; . "$J/commands.sh"
p=$1
cd "$K" || exit 1
wait_done "cand_p$p" "tcbp_p$p" "net_p$p" || { echo "an input step of p = $p failed: no ensemble at this noise level"; exit 1; }
mkdir -p "$T/sel_p$p" "$T/ensemble_p$p"
rc=0
if echo " $JOINT_PS " | grep -q " $p "; then
  for b in X Z; do cmd_selection "$p" "$b" || rc=1; done
  [ $rc = 0 ] || { echo "a selection list of p = $p failed"; exit 1; }
  start_when_room 4 "joint_p$p" "$J/call.sh" cmd_joint "$p"
  wait_done "joint_p$p" || { echo "joint decoding of p = $p failed"; exit 1; }
  for b in X Z; do cmd_ensemble "$p" "$b" joint || rc=1; done
else
  for b in X Z; do cmd_ensemble "$p" "$b" nojoint || rc=1; done
fi
[ $rc = 0 ] && { cmd_pair "$p" || rc=1; }
mkdir -p "$F/cand_p$p"; cp "$T/cand_p$p"/*.chunks.npz "$F/cand_p$p/" 2>/dev/null
for d in sel joint net ensemble; do
  [ -d "$T/${d}_p$p" ] && { mkdir -p "$F/${d}_p$p"; cp "$T/${d}_p$p"/*.npz "$T/${d}_p$p"/*.json "$F/${d}_p$p/" 2>/dev/null; }
done
echo "ensemble of p = $p: exit $rc"
exit $rc
EOS

cat > "$J/call.sh" <<'EOS'
#!/bin/bash
# calls one function of commands.sh with its arguments (so that a step can be a function)
J=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$J/lib.sh"; . "$J/commands.sh"
"$@"
EOS

cat > "$J/gpu_chain.sh" <<'EOS'
#!/bin/bash
# the GPU's work on the test shots, one thing at a time: TCBP-JOBS in serial mode (its times are the clean ones:
# nothing else is on the GPU), then the two networks' logits
J=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$J/lib.sh"; . "$J/commands.sh"
cd "$K" || exit 1
echo 1 > "$TOK/$$"
for p in $PS; do
  n="tcbp_p$p"
  if is_done "$n"; then say "SKIP $n"; continue; fi
  w0=$(date +%s); while [ "$(gpu_procs)" != none ] && [ $(( $(date +%s) - w0 )) -lt 1200 ]; do sleep 20; done
  g=$(gpu_procs); [ "$g" = none ] && g="none: the GPU is idle" || g="$g NOT IDLE after 20 minutes of waiting: its times are not clean"
  rowlog "START $n (serial mode). GPU processes: $g [$(state)]"
  cmd_tcbp "$p" > "$LOGS/$n.log" 2>&1; rc=$?
  echo "$rc" > "$DONE/$n.rc"
  mkdir -p "$F/tcbp_p$p"; cp "$T/tcbp_p$p"/*.tcbp.npz "$F/tcbp_p$p/" 2>/dev/null
  rowlog "END $n exit $rc: $(grep -E 'failures' "$LOGS/$n.log" | tr '\n' ' ' | cut -c1-220) [$(state)]"
done
for p in $PS; do
  n="net_p$p"
  if is_done "$n"; then say "SKIP $n"; continue; fi
  rowlog "START $n"
  cmd_logits "$p" > "$LOGS/$n.log" 2>&1; rc=$?
  echo "$rc" > "$DONE/$n.rc"
  rowlog "END $n exit $rc: $(grep -E 'memory: failures' "$LOGS/$n.log" | tr '\n' ' ' | cut -c1-200)"
done
EOS

cat > "$J/driver.sh" <<'EOS'
#!/bin/bash
J=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . "$J/lib.sh"; . "$J/commands.sh"
fail() { say "ABORT: $*"; echo "$(now) $*" > "$T/NOGO"; rowlog "ABORT $*"; exit 1; }
say "TEST RUN DRIVER START: frozen $FROZEN, rule $RULE, X checkpoint $X_COMMIT:$X_PATH"

# ---- 0. the frozen checkout
if [ ! -d "$C" ]; then
  git -C "$B/repo" cat-file -e "$FROZEN^{commit}" 2>/dev/null || git -C "$B/repo" fetch -q origin integration hierarchy
  git -C "$B/repo" worktree add -q --detach "$C" "$FROZEN" || fail "the frozen checkout could not be made"
fi
[ "$(git -C "$C" rev-parse HEAD)" = "$FROZEN" ] || fail "the frozen checkout is at $(git -C "$C" rev-parse HEAD), not at $FROZEN"
[ -z "$(git -C "$C" status --porcelain --untracked-files=no)" ] || fail "the frozen checkout has local changes"
cd "$K" || fail "the checkout has no kit folder"
kit=challenge-1-ml-decoder-kit
git -C "$C" diff --quiet "$HIER_FROZEN" HEAD -- "$kit/hierarchy" ":!$kit/hierarchy/README.md" || fail "hierarchy/ in the checkout is not the frozen code $HIER_FROZEN"
git -C "$C" diff --quiet "$HIER_FROZEN" HEAD -- "$kit/neural" || fail "neural/ in the checkout differs from the frozen commit's"
git -C "$C" diff --quiet "$ROWS_BASE" HEAD -- "$kit/evaluation" "$kit/decoders.py" "$kit/circuit.py" "$kit/make_circuit.py" "$kit/data" "$kit/results" \
  || fail "the evaluator, the kit's files or the shot files differ from the baseline rows' checkout $ROWS_BASE"
say "0. frozen checkout: $(git -C "$C" log --oneline -1); hierarchy/ and neural/ equal $HIER_FROZEN; the evaluator, the kit and the shot files equal the baseline rows' checkout"

# ---- 1. the final checkpoints
mkdir -p "$T/ckpt" "$T/final"
if [ ! -s "$T/ckpt/x_final.pt" ]; then
  git -C "$B/repo" cat-file -e "$X_COMMIT^{commit}" 2>/dev/null || git -C "$B/repo" fetch -q origin laptop-artifacts
  git -C "$B/repo" show "$X_COMMIT:$X_PATH" > "$T/ckpt/x_final.pt.tmp" || fail "the X checkpoint could not be read from $X_COMMIT"
  mv "$T/ckpt/x_final.pt.tmp" "$T/ckpt/x_final.pt"
fi
got=$(sha256sum "$T/ckpt/x_final.pt" | cut -d' ' -f1)
[ "$got" = "$X_SHA256" ] || fail "the X checkpoint's sha256 is $got, not $X_SHA256"
say "1. X checkpoint: sha256 $got"
if [ ! -s "$T/ckpt/z_final.pt" ] && [ -n "$Z_GIVEN" ]; then
  [ "$DRY" = 1 ] || fail "Z_GIVEN is for a dry run only"
  cp "$Z_GIVEN" "$T/ckpt/z_final.pt" || fail "the given Z checkpoint $Z_GIVEN could not be copied"
  say "1. DRY RUN: the Z checkpoint is the given file $Z_GIVEN"
fi
if [ ! -s "$T/ckpt/z_final.pt" ]; then
  cutoff=$(date -u -d "$Z_CUTOFF" +%s) || fail "the Z cutoff time cannot be read"
  say "1. waiting for the Z training to end (it is stopped at $Z_CUTOFF UTC if it still runs); process $Z_PID: $(tr '\0' ' ' < /proc/$Z_PID/cmdline 2>/dev/null | cut -c1-200)"
  z_alive() { [ -r "/proc/$Z_PID/cmdline" ] && tr '\0' ' ' < "/proc/$Z_PID/cmdline" | grep -q "${Z_RUN##*/}"; }
  z_alive || say "NOTE: process $Z_PID is not the Z training (any more)"
  while z_alive; do
    if [ "$(date +%s)" -ge "$cutoff" ] && [ ! -e "$Z_RUN/STOP" ]; then
      say "the cutoff is reached at: $(grep -E '^step [0-9]+ of' "$Z_RUN/train.log" | tail -n 1 | cut -c1-60): stopping the training with its STOP file"
      touch "$Z_RUN/STOP" "$T/z_stopped_by_us"
    fi
    sleep 20
  done
  sleep 15
  last=$(grep -E '^step [0-9]+ of' "$Z_RUN/train.log" | tail -n 1)
  say "the Z training has ended at: $(echo "$last" | cut -c1-60)"
  echo "$last" | awk '{exit !($2 == $4)}' || [ -e "$T/z_stopped_by_us" ] \
    || fail "the Z training is neither at its last step nor stopped by this run (its last line: $(echo "$last" | cut -c1-40)): is process $Z_PID really the training?"
  ls -la --time-style=+%FT%T "$Z_RUN/checkpoint.pt"
  cp "$Z_RUN/checkpoint.pt" "$T/ckpt/z_final.pt.tmp" && mv "$T/ckpt/z_final.pt.tmp" "$T/ckpt/z_final.pt" || fail "the Z checkpoint could not be copied"
fi
say "1. Z checkpoint: sha256 $(sha256sum "$T/ckpt/z_final.pt" | cut -d' ' -f1)"

# ---- 2. exploration shots only: the final networks' logits, then the plugin's rule per basis
#         (stages 2 and 3 run once: after GO the options files are never written again)
if [ ! -e "$T/GO" ]; then
for s in validation train; do
  [ -s "$T/final/netX_${s}_X.npz" ] || python -m neural.evaluate_stim --checkpoint "$T/ckpt/x_final.pt" --data "data/p0.004/${s}_X.npz" --save "$T/final/netX_${s}_X" > "$LOGS/final_netX_$s.log" 2>&1 || fail "the X logits on ${s}_X failed"
  [ -s "$T/final/netZ_${s}_Z.npz" ] || python -m neural.evaluate_stim --checkpoint "$T/ckpt/z_final.pt" --data "data/p0.004/${s}_Z.npz" --save "$T/final/netZ_${s}_Z" > "$LOGS/final_netZ_$s.log" 2>&1 || fail "the Z logits on ${s}_Z failed"
done
grep -h -E '^checkpoint|memory: failures' "$LOGS"/final_net*.log
popt=(--x-checkpoint "$T/ckpt/x_final.pt" --x-candidates "$R/lap/selector_validation_X.npz" "$R/lap/selector_train_X.npz" --x-logits "$T/final/netX_validation_X.npz" "$T/final/netX_train_X.npz"
      --z-checkpoint "$T/ckpt/z_final.pt" --z-candidates "$R/lap/selector_validation_Z.npz" "$R/lap/selector_train_Z.npz" --z-logits "$T/final/netZ_validation_Z.npz" "$T/final/netZ_train_Z.npz")
python -m hierarchy.plugin_options --out "$T/hierarchy_options.json" "${popt[@]}" || fail "the plugin's options could not be written"
python -m hierarchy.plugin_options --parallel --out "$T/hierarchy_options_parallel.json" "${popt[@]}" > "$LOGS/plugin_options_parallel.log" 2>&1 || fail "the side-by-side options could not be written"
say "2. the plugin's options: $(cat "$T/hierarchy_options.json")"
# the reviewer's check: a rule object or a checkpoints object that lacks a basis is not caught by the plugin or the preflight
python - "$T/hierarchy_options.json" "$T/hierarchy_options_parallel.json" <<'PY' || fail "an options file is not complete: no test file is opened"
import json, os, sys
files = sys.argv[1:3]
opts = [json.load(open(f)) for f in files]
for path, o in zip(files, opts):
    r, c = o.get("rule"), o.get("checkpoints")
    assert isinstance(r, dict) and set(r) == {"X", "Z"}, f"{path}: 'rule' must name X and Z: {r!r}"
    assert all(v in ("network", "converged", "weight") for v in r.values()), f"{path}: unknown rule in {r!r}"
    assert isinstance(c, dict) and set(c) == {"X", "Z"}, f"{path}: 'checkpoints' must name X and Z: {c!r}"
    for f in c.values():
        assert os.path.isfile(f), f"{path}: no checkpoint file {f}"
    assert o.get("device") == "cpu" and o.get("torch_threads") == 1, f"{path}: device and torch_threads must be cpu and 1"
    print("options ok:", path, r, "side by side" if o.get("parallel") else "sequential")
plain, par = opts
assert not plain.get("parallel") and par.get("parallel") is True, "the second file must be the side-by-side one"
strip = lambda o: {k: v for k, v in o.items() if k != "parallel"}
assert strip(plain) == strip(par), "the two options files differ in more than 'parallel'"
PY

# ---- 3. the preflight and a rehearsal, on validation shots. No test file is opened before both have passed.
  rm -f "$T/NOGO"
  # 3a. the author's preflight with the exact options file (TCBP-JOBS on the GPU, both networks asked), then with the
  #     side-by-side options (TCBP-JOBS replaced by its stand-in: the GPU part has just been checked)
  for o in hierarchy_options hierarchy_options_parallel; do
    wait_room 2
    extra=(--tcbp-python "$TCBP_PY"); [ "$o" = hierarchy_options_parallel ] && extra=(--tcbp-stand-in)
    python -m hierarchy.preflight --options "$T/$o.json" --x data/p0.004/validation_X.npz --z data/p0.004/validation_Z.npz --rule "$RULE" \
      "${extra[@]}" --out "$T/preflight_${o}_$(date -u +%H%M%S)" > "$LOGS/preflight_$o.log" 2>&1
    rc=$?
    tail -n 3 "$LOGS/preflight_$o.log"
    { [ $rc -eq 0 ] && tail -n 1 "$LOGS/preflight_$o.log" | grep -q '^PREFLIGHT PASSED'; } || fail "the preflight with $o.json did not pass (exit $rc): no test file is opened"
  done
  # the author's check that the side-by-side mode returns what the sequential one returns (300 validation shots per basis:
  # the reviewer's advice, so that relay-BP is reached in both bases)
  wait_room 2
  python -m hierarchy.parallel_check --options "$T/hierarchy_options.json" --x data/p0.004/validation_X.npz --z data/p0.004/validation_Z.npz --shots 300 \
    > "$LOGS/parallel_check.log" 2>&1 || { tail -n 5 "$LOGS/parallel_check.log"; fail "the side-by-side mode does not equal the sequential one: no test file is opened"; }
  tail -n 2 "$LOGS/parallel_check.log"
  # 3b. a rehearsal of the three timed rows through the team's evaluator, on the p = 0.003 VALIDATION files (10,000
  #     shots, about ten minutes): the same command lines as on the test files, with the same options files
  mkdir -p "$T/rehearsal"
  reh() {  # row name, decoder spec, options file
    [ -s "$T/rehearsal/eval_p0.003_$1/summary.csv" ] && return 0
    rm -rf "$T/rehearsal/eval_p0.003_$1"
    python -m evaluation.evaluate --test-x data/p0.003/validation_X.npz --test-z data/p0.003/validation_Z.npz --decoder "$2" --plugin-options "$3" \
      --out "$T/rehearsal/eval_p0.003_$1" > "$LOGS/rehearsal_$1.log" 2>&1
  }
  # one after the other, each when there is room: they share the notebook with the timed baseline rows
  wait_room 1; reh hierarchy hierarchy.plugin:make_decoder "$T/hierarchy_options.json"; c1=$?
  wait_room 2; reh hierarchy-parallel hierarchy.plugin:make_decoder "$T/hierarchy_options_parallel.json"; c2=$?
  wait_room 1; reh priortilt hierarchy.priortilt_plugin:make_decoder "$T/hierarchy_options.json"; c3=$?
  for n in hierarchy hierarchy-parallel priortilt; do
    say "rehearsal on validation shots, p = 0.003, $n: $(tail -n +2 "$T/rehearsal/eval_p0.003_$n/summary.csv" 2>/dev/null | cut -d, -f1,7-11,12 | head -1)"
    mkdir -p "$F/rehearsal_validation_p0.003_$n"; cp "$T/rehearsal/eval_p0.003_$n"/summary.json "$T/rehearsal/eval_p0.003_$n"/summary.csv "$F/rehearsal_validation_p0.003_$n/" 2>/dev/null
  done
  [ "$c1$c2$c3" = 000 ] || fail "a rehearsal row failed (exit codes $c1 $c2 $c3: hierarchy, hierarchy-parallel, priortilt): no test file is opened"
  for n in hierarchy hierarchy-parallel priortilt; do [ -s "$T/rehearsal/eval_p0.003_$n/summary.csv" ] || fail "the rehearsal row $n wrote no summary"; done
  now > "$T/GO"
fi
say "3. PREFLIGHT AND REHEARSAL PASSED. [$(state)]"
rowlog "GO: the preflight and the rehearsal passed; frozen $FROZEN; options $(cat "$T/hierarchy_options.json")"
if [ "$DRY" = 1 ]; then say "DRY RUN END: stages 0 to 3 passed; no test file was opened"; rowlog "DRY RUN END"; exit 0; fi

# ---- 4. the test shots
nohup "$J/gpu_chain.sh" > "$LOGS/gpu_chain.log" 2>&1 < /dev/null &
sleep 2
for p in $PS; do start_when_room 1 "eval_p${p}_hierarchy" "$J/row.sh" "$p" hierarchy.plugin:make_decoder hierarchy "$T/hierarchy_options.json" metrics; done
for p in $PS; do is_done "ensemble_p$p" || nohup "$J/step.sh" "ensemble_p$p" "$J/assemble.sh" "$p" > "$LOGS/ensemble_p$p.log" 2>&1 < /dev/null & done
for p in $PS; do start_when_room 2 "eval_p${p}_hierarchy-parallel" "$J/row.sh" "$p" hierarchy.plugin:make_decoder hierarchy-parallel "$T/hierarchy_options_parallel.json" -; done
start_when_room 2 cand_p0.004 "$J/call.sh" cmd_candidates 0.004 2
start_when_room 1 cand_p0.003 "$J/call.sh" cmd_candidates 0.003 1
start_when_room 4 cand_p0.005 "$J/call.sh" cmd_candidates 0.005 4
for p in $PS; do start_when_room 1 "eval_p${p}_priortilt" "$J/row.sh" "$p" hierarchy.priortilt_plugin:make_decoder priortilt "$T/hierarchy_options.json" -; done
# the hierarchy without learning, as a timed row where it is cheap (at p = 0.005 it would take three more hours: the
# ablation there comes from the ensemble's tables on the same candidates)
echo '{"rule": "converged", "seeds": [1, 1001]}' > "$T/converged_options.json"
for p in 0.004 0.003; do start_when_room 1 "eval_p${p}_hierarchy-converged" "$J/row.sh" "$p" hierarchy.plugin:make_decoder hierarchy-converged "$T/converged_options.json" -; done

# ---- 5. the end
steps=""
for p in $PS; do steps="$steps eval_p${p}_hierarchy eval_p${p}_hierarchy-parallel eval_p${p}_priortilt cand_p$p tcbp_p$p net_p$p ensemble_p$p"; done
steps="$steps eval_p0.004_hierarchy-converged eval_p0.003_hierarchy-converged"
for n in $steps; do until is_done "$n"; do sleep 30; done; done
say "TEST RUN END. Exit codes:"
for n in $steps $(for p in $JOINT_PS; do echo "joint_p$p"; done); do echo "  $n: $(cat "$DONE/$n.rc" 2>/dev/null || echo 'not run')"; done
say "[$(state)]"
rowlog "TEST RUN END"
EOS

chmod +x "$J"/*.sh
nohup "$J/driver.sh" > "$B/runs/test_run$TAG.out" 2>&1 < /dev/null &
echo "driver started, pid $!"
sleep 15; cat "$B/runs/test_run$TAG.out"
