"""The relay-BP authors' published runs on this code: separate (xyz=False) against joint (xyz=True) decoding.

  curl -sSo bb144.csv https://raw.githubusercontent.com/trmue/relay/main/examples/data/bicycle_bivariate_144_12_12_fine_ler_sweep.csv
  python -m analysis.authors_joint bb144.csv

These are the authors' numbers, read from their CSV, not ours. In their notebook
(examples/BivariateBicycleCodeAnalysis.ipynb), xyz=False keeps only the detectors of the memory
basis (testdata.filter_detectors_by_basis), which is what the kit's baselines decode; xyz=True
decodes all detectors. Their "seconds" is sinter's total decoder time over all shots.
"""
import csv
import json
import sys

from analysis.common import sha256, wilson


def main():
    path = sys.argv[1]
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(path))]
    table = {}
    for r in rows:
        m = json.loads(r["json_metadata"])
        if r["decoder"].strip() != "relay-bp":
            continue
        key = (m["p"], m["xyz"])
        table.setdefault(key, {})[m["circuit"][-1]] = (int(r["shots"]), int(r["errors"]), float(r["seconds"]),
                                                      json.dumps(m["decoder_params"], sort_keys=True))
    print(f"file {path} sha256 {sha256(path)}")
    print("| p | decoding | shots per basis | failures X, Z | rate per shot X [95% Wilson] | Z | seconds per shot X, Z |")
    print("|---|---|---|---|---|---|---|")
    params = set()
    for (p, xyz), d in sorted(table.items()):
        (nx, ex, sx, px), (nz, ez, sz, pz) = d["X"], d["Z"]
        params |= {px, pz}
        lx, hx = wilson(ex, nx)
        lz, hz = wilson(ez, nz)
        print(f"| {p} | {'joint (xyz)' if xyz else 'separate'} | {nx:,} | {ex}, {ez} | {ex / nx:.2e} [{lx:.1e}, {hx:.1e}] | "
              f"{ez / nz:.2e} [{lz:.1e}, {hz:.1e}] | {sx / nx:.3f}, {sz / nz:.3f} |")
    print("decoder settings in these rows:", params)


if __name__ == "__main__":
    main()
