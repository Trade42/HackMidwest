#!/bin/bash
# Done-when 4: a second run gives the same per-shot results. Decodes the first 250 validation shots of the frozen
# pair again with 2 worker processes instead of 4, in one part instead of the original's, and compares every field.
source loss_tracker/runs/common.sh
source loss_tracker/runs/frozen.sh
OUT=$S/parts/rerun/val_frozen_000000-000250_workers2.npz
[ -f "$OUT" ] || lane "$PY" -m loss_tracker.decode --data "$VAL" --post "$S/post/val_$TRACK_NAME.npz" --conditions "$VARIANT" \
  --start 0 --stop 250 --workers 2 --out "$OUT"
"$PY" - "$OUT" "$S/parts/val/frozen/000000-000250.npz" <<'PY'
import json, sys
import numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
fields = ("shots", "failed", "converged", "iterations", "wrong_mask")
out = {f: bool(np.array_equal(a[f], b[f])) for f in fields}
out["shots_compared"] = int(len(a["shots"]))
print(json.dumps(out))
json.dump(out, open("loss_tracker/results/rerun_check.json", "w"), indent=1)
PY
