#!/usr/bin/env python3
"""Compare exported hardware verdicts with the kit's exact NumPy/SciPy rule."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--generated", default="generated")
    args = p.parse_args()
    model = json.loads(Path(args.model).read_text())
    with np.load(args.features) as data:
        features = data["features"]
        easy = data["first_leg_converged"]
    w = np.asarray(model["weights"])
    z = w[0] + ((features - model["mean"]) / model["scale"]) @ w[1:]
    software = (~easy) & (expit(z) > model["threshold"])
    expected = np.frombuffer((Path(args.generated) / "expected.bin").read_bytes(), dtype=np.uint8)
    if expected.size != len(features) * 8:
        raise ValueError("expected.bin length does not match feature rows")
    hardware = expected.reshape(-1, 8)[:, 7].astype(bool)
    mismatch = np.flatnonzero(software != hardware)
    print(f"{len(features)} saved shots: software escalates {int(software.sum())}, "
          f"fixed hardware escalates {int(hardware.sum())}, mismatches {len(mismatch)}")
    if len(mismatch):
        print("first mismatch indices:", mismatch[:10].tolist())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
