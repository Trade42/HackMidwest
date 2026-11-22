# Shared helpers for the run scripts. Source it from challenge-1-ml-decoder-kit/.
# Compute (brief): lanes A and D, at most 8 processes; one hold of sim-a.lock lasts at most about 20 minutes.
# Each part takes lane A as well if it is free (8 workers on cores 0-7), else lane D alone (4 workers, cores 0-3).
# A part's result does not depend on the worker count (every shot gets a fresh, fixed-seed decoder).
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
V=/home/sal/hackMidwest/orchestration/scratch/loss-gates/venv
S=/home/sal/hackMidwest/orchestration/scratch/loss-gates
RUN=/home/sal/hackMidwest/dq/run
PY="$V/bin/python"

# lanes CMD...   run CMD with WORKERS set to 8 (lanes A+D) or 4 (lane D), niced and pinned
lanes() {
  exec 8>"$RUN/sim-a.lock"
  if flock -n 8; then
    WORKERS=8 flock "$RUN/sim-d.lock" env WORKERS=8 nice -n 15 taskset -c 0-7 "$@"
    flock -u 8
  else
    flock "$RUN/sim-d.lock" env WORKERS=4 nice -n 15 taskset -c 0-3 "$@"
  fi
  exec 8>&-
  sleep 5
}

# part OUT DATA CONDITIONS START STOP   decode one part unless it exists
part() {
  local out=$1 data=$2 conds=$3 start=$4 stop=$5
  [ -f "$out" ] && { echo "keep $out"; return 0; }
  mkdir -p "$(dirname "$out")"
  lanes bash -c "\"$PY\" -m loss.decode --data \"$data\" --conditions \"$conds\" --start $start --stop $stop --workers \$WORKERS --out \"$out.tmp.npz\""
  mv "$out.tmp.npz" "$out"
}

# parts NAME DATA CONDITIONS STOP SIZE   decode shots 0..STOP in parts of SIZE (resumable)
parts() {
  local name=$1 data=$2 conds=$3 stop=$4 size=$5
  for ((a = 0; a < stop; a += size)); do
    local b=$((a + size < stop ? a + size : stop))
    part "$S/parts/$name/$(printf %06d $a)-$(printf %06d $b).npz" "$data" "$conds" $a $b
  done
}

# dataset OUT ARGS...   sample a dataset unless it exists (4 threads on lane D)
dataset() {
  local out=$1; shift
  [ -f "$out" ] && { echo "keep $out"; return 0; }
  flock "$RUN/sim-d.lock" nice -n 15 taskset -c 0-3 "$PY" -m loss.sample "$@" --threads 4 --out "$out"
}
