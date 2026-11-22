# Shared settings of the run scripts. Source it from challenge-1-ml-decoder-kit/.
# Compute: lane D of this PC (4 logical CPUs = 2 physical cores), at most 4 processes, niced.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
S=${S:-/home/sal/hackMidwest/orchestration/scratch/loss-tracker}     # datasets, posteriors, per-shot parts
V=${V:-$S/venv}
PY="$V/bin/python"
LOCK=/home/sal/hackMidwest/dq/run/sim-d.lock
lane() { flock "$LOCK" nice -n 15 taskset -c 0-3 "$@"; }
TABLES=loss_tracker/tables/X_p0.004_s1e-3.npz
TRAIN=$S/data/train_X_p0.004_s1e-3_seed7100.npz
VAL=$S/data/val_X_p0.004_s1e-3_seed7200.npz
TEST=$S/data/test_X_p0.004_s1e-3_seed7300.npz

# part OUT CMD...   run CMD on lane D unless OUT exists; CMD writes OUT.tmp.npz (resumable, one lock hold per part)
part() {
  local out=$1; shift
  [ -f "$out" ] && { echo "keep $out"; return 0; }
  mkdir -p "$(dirname "$out")"
  lane "$@" --out "$out.tmp.npz"
  mv "$out.tmp.npz" "$out"
}
