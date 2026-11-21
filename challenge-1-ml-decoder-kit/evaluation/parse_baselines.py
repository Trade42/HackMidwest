"""Read baseline.py's text output without modifying the active log."""
import argparse
import csv
import json
import re
from pathlib import Path

HEADER = re.compile(r"^(\d+) shots at p=([0-9.eE+-]+), (\d+) cycles, failures listed as \[X Z\]$")
ENTRY = re.compile(r"^(\S+)\s+failures \[(\d+)\s+(\d+)\] of (\d+)\s+per cycle ([0-9.eE+-]+) \+- ([0-9.eE+-]+)\s+mean ([0-9.eE+-]+) ms\s+p99 ([0-9.eE+-]+) ms$")
DECODERS = ("bposd", "relay-bp")
FIELDS = ("run", "status", "p", "shots", "cycles", "decoder", "failures_X", "failures_Z",
          "reported_per_cycle_rate", "reported_standard_error", "reported_mean_ms", "reported_p99_ms")


def parse(text):
    runs = []
    current = None
    for line in text.splitlines(keepends=True):
        if not line.endswith("\n"):
            if current is not None:
                current["partial_tail"] = line
            break
        line = line.rstrip("\r\n")
        if match := HEADER.fullmatch(line):
            shots, p, cycles = match.groups()
            current = {"run": len(runs) + 1, "p": float(p), "shots": int(shots),
                       "cycles": int(cycles), "entries": []}
            runs.append(current)
        elif current is not None and (match := ENTRY.fullmatch(line)):
            name, fx, fz, shots, rate, se, mean, p99 = match.groups()
            if name in DECODERS and int(shots) == current["shots"] and name not in {e["decoder"] for e in current["entries"]}:
                current["entries"].append({"decoder": name, "status": "completed",
                                           "failures_X": int(fx), "failures_Z": int(fz),
                                           "reported_per_cycle_rate": float(rate),
                                           "reported_standard_error": float(se),
                                           "reported_mean_ms": float(mean), "reported_p99_ms": float(p99)})
    for run in runs:
        present = {entry["decoder"] for entry in run["entries"]}
        for name in DECODERS:
            if name not in present:
                run["entries"].append({"decoder": name, "status": "unfinished"})
        run["status"] = "completed" if len(present) == len(DECODERS) else "unfinished"
    return runs


def export(log_path, out_prefix):
    log_path = Path(log_path)
    prefix = Path(out_prefix)
    if log_path.resolve() == prefix.resolve():
        raise ValueError("output path cannot be the active log")
    runs = parse(log_path.read_text(errors="replace"))
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_suffix(".json").open("w") as file:
        json.dump({"source_log": str(log_path.resolve()), "runs": runs}, file, indent=2)
    with prefix.with_suffix(".csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for run in runs:
            for entry in run["entries"]:
                writer.writerow({key: (run | entry).get(key) for key in FIELDS})
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()
    runs = export(args.log, args.out_prefix)
    print(f"parsed {len(runs)} runs; {sum(r['status'] == 'completed' for r in runs)} complete")


if __name__ == "__main__":
    main()
