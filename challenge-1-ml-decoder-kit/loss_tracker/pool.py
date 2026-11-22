"""Pool the part files of several decoded datasets into one table, the shots of each dataset numbered after the last.

usage (from challenge-1-ml-decoder-kit/):
    python -m loss_tracker.pool --set 'TEST_PARTS/*.npz' --set 'REPLICATION_PARTS/*.npz' --scratch DIR -- \
        --order starter,exact,tracker,oracle-round --references exact,starter --out loss_tracker/results/pooled

Writes renumbered copies of the parts into DIR (set k's shots are moved up by the shots of the sets before it),
then runs loss_tracker.summarize on them with the arguments after `--`.
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np


def main():
    argv = sys.argv[1:]
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True)
    ap.add_argument("--scratch", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.scratch, exist_ok=True)
    offset = 0
    for k, pattern in enumerate(a.set):
        files = sorted(glob.glob(pattern))
        assert files, pattern
        top = 0
        for f in files:
            with np.load(f) as z:
                d = {key: z[key] for key in z.files}
            top = max(top, int(d["shots"].max()) + 1)
            d["shots"] = d["shots"] + offset
            np.savez_compressed(os.path.join(a.scratch, f"set{k}_{os.path.basename(f)}"), **d)
        print(f"set {k}: {len(files)} part files, shots {offset}..{offset + top - 1}", flush=True)
        offset += top
    subprocess.run([sys.executable, "-m", "loss_tracker.summarize", "--parts", os.path.join(a.scratch, "*.npz"), *rest], check=True)


if __name__ == "__main__":
    main()
