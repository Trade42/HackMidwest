#!/bin/bash
# Rebuild every table in results/ from the decode parts (cheap: no decoding). Run from challenge-1-ml-decoder-kit/.
source loss/runs/lane.sh
R=loss/results
D=$S/data
P=$S/parts
sum() { "$PY" -m loss.summarize "$@" | tail -n +3; }
if [ -d $P/L0_X_p0.004_s0 ]; then
  "$PY" -m loss.l0_check --data $D/X_p0.004_s0_seed1000.npz --parts "$P/L0_X_p0.004_s0/blind/*.npz" --out $R/L0_X_p0.004 > /dev/null
  sum --parts "$P/L0_X_p0.004_s0/*/*.npz" --data $D/X_p0.004_s0_seed1000.npz --reference blind \
      --title "L0: X memory, p = 0.004, no loss" --out $R/L0_X_p0.004_s0_table
fi
for spec in 2e-4:2100 5e-4:2200 1e-3:2300 2e-3:2400; do
  s=${spec%:*}; seed=${spec#*:}
  [ -d $P/X_p0.004_s$s ] || continue
  conds=blind,blind-rate,starter,starter@0.5/stop1
  ls $P/X_p0.004_s$s/*/*.npz >/dev/null 2>&1 || continue
  avail=$("$PY" -c "import glob,numpy as np; print(','.join(sorted({str(c) for f in glob.glob('$P/X_p0.004_s$s/*/*.npz') for c in np.load(f)['conditions']})))")
  order=$(echo $conds | tr , '\n' | grep -Fx -f <(echo $avail | tr , '\n') | paste -sd,)
  sum --parts "$P/X_p0.004_s$s/*/*.npz" --data $D/X_p0.004_s${s}_seed$seed.npz --reference blind --order $order \
      --title "L1: X memory, p = 0.004, s = $s: blind against flag-aware" --out $R/L1_X_p0.004_s$s
done
if [ -d $P/X_p0.004_s1e-3/aware ]; then
  pick() { local want=$1; echo $want | tr , '\n' | grep -Fx -f <(echo $avail | tr , '\n') | paste -sd,; }
  avail=$("$PY" -c "import glob,numpy as np; print(','.join(sorted({str(c) for f in glob.glob('$P/X_p0.004_s1e-3/*/*.npz') for c in np.load(f)['conditions']})))")
  sum --parts "$P/X_p0.004_s1e-3/*/*.npz" --data $D/X_p0.004_s1e-3_seed2300.npz --reference exact \
      --order $(pick exact,oracle-round,oracle-slot) \
      --title "L3: the oracle on the loss time, same map as exact, s = 1e-3 (ORACLE rows use the true loss times)" --out $R/L3_X_p0.004_s1e-3
  sum --parts "$P/X_p0.004_s1e-3/*/*.npz" --data $D/X_p0.004_s1e-3_seed2300.npz --reference starter \
      --order $(pick starter,starter-oracle-round,oracle-round,oracle-slot,exact,starter@0.5/stop1) \
      --title "L3: the oracle against the starter's flag-aware decoder, s = 1e-3 (ORACLE rows use the true loss times)" --out $R/L3_starter_X_p0.004_s1e-3
  sum --parts "$P/X_p0.004_s1e-3/*/*.npz" --data $D/X_p0.004_s1e-3_seed2300.npz --reference starter \
      --order $(pick starter,starter-nofinal,starter-nohook,starter-ratebase,starter-noown,exact) \
      --title "L2: leave-one-out from the starter's map, and the location weights (exact), s = 1e-3" --out $R/L2_X_p0.004_s1e-3
  for cap in 0.4 0.25; do
    if echo $avail | tr , '\n' | grep -Fxq "exact@$cap"; then
      sum --parts "$P/X_p0.004_s1e-3/*/*.npz" --data $D/X_p0.004_s1e-3_seed2300.npz --reference exact@$cap \
          --order $(pick exact@$cap,oracle-round@$cap,exact) \
          --title "L2/L3 at cap $cap, s = 1e-3" --out $R/L2L3_cap${cap}_X_p0.004_s1e-3
    fi
  done
fi
"$PY" -m loss.tables --out $R/L1_combined.md > /dev/null
