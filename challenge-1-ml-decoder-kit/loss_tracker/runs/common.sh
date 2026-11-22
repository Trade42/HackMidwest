# Shared settings of the run scripts. Source it from challenge-1-ml-decoder-kit/.
# Compute: lane D of this PC (4 logical CPUs = 2 physical cores), at most 4 processes, niced. LANE=B takes lane B
# instead (CPUs 8-11), which the orchestrator granted beside lane D on 2026-11-22 at 06:52 UTC.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
S=${S:-/home/sal/hackMidwest/orchestration/scratch/loss-tracker}     # datasets, posteriors, per-shot parts
V=${V:-$S/venv}
PY="$V/bin/python"
LANE=${LANE:-D}
case "$LANE" in
  D) LOCK=/home/sal/hackMidwest/dq/run/sim-d.lock; CPUS=0-3 ;;
  B) LOCK=/home/sal/hackMidwest/dq/run/sim-b.lock; CPUS=8-11 ;;
  *) echo "unknown lane $LANE" >&2; exit 1 ;;
esac
lane() { flock "$LOCK" nice -n 15 taskset -c "$CPUS" "$@"; }
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
