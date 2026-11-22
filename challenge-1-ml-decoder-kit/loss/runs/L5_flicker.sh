#!/bin/bash
# L5: firing of the 6 checks next to a lost data atom, by round relative to its true loss round. No decoding.
# Run from challenge-1-ml-decoder-kit/. Lane D, or lane A with LANE=A.
source loss/runs/lane.sh
D=$S/data
if [ "$LANE" = A ]; then LOCK=$RUN/sim-a.lock; CORES=4-7; else LOCK=$RUN/sim-d.lock; CORES=0-3; fi
for spec in 2e-4:2100 1e-3:2300; do
  s=${spec%:*}; seed=${spec#*:}
  flock "$LOCK" nice -n 15 taskset -c $CORES "$PY" -m loss.flicker --data $D/X_p0.004_s${s}_seed$seed.npz \
    --out loss/results/L5_flicker_X_p0.004_s$s > loss/logs/L5_flicker_s$s.log 2>&1
done
