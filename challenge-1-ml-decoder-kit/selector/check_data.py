"""Physics check: every member file was made from a committed data file, named by its sha256.

Run from challenge-1-ml-decoder-kit/:  python -m selector.check_data <member files or folders>

For each member file it prints the data file of data/p*/ whose sha256 is the one the member file
records. Only train_* and validation_* data files are hashed: no test_* or go_* file is opened.
It exits with status 1 if a member file names no such data file.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    known = {}
    for data in sorted(Path("data").glob("p*/*.npz")):
        if data.name.startswith(("train_", "validation_")):
            known[hashlib.sha256(data.read_bytes()).hexdigest()] = str(data)
    files = []
    for name in argv:
        path = Path(name)
        files += sorted(path.rglob("*.npz")) if path.is_dir() else [path]
    bad = 0
    for path in files:
        with np.load(path, allow_pickle=False) as file:
            recorded = str(file["data_sha256"])
            note = f"  (records {file['data_name']})" if "data_name" in file.files else "  (records no name)"
        match = known.get(recorded)
        bad += match is None
        print(f"{recorded[:16]}...  {match or 'NO COMMITTED train/validation DATA FILE'}  <-  {path}{note}")
    print(f"{len(files) - bad} of {len(files)} member files name a committed train or validation data file")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
