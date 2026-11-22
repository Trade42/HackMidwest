"""Check that the plugin's side-by-side mode gives the answers of its sequential mode, shot for shot.

Run from challenge-1-ml-decoder-kit/ (PRIORTILT_REPO set):

  python -m hierarchy.parallel_check --x data/p0.004/validation_X.npz --z data/p0.004/validation_Z.npz --shots 300

For each basis it decodes the first --shots shots twice through the evaluator's create_session:
with the options as given and "parallel" false, and with "parallel" true. It compares, per
shot, the 12 predicted bits, the flags agreed, stage and winner, and the candidates' weights
and iteration counts, and it prints the mean time per shot of both modes on this machine. It
exits non-zero if anything differs. Held-out files are refused.
"""
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from evaluation.adapters import create_session, validate_shot_result
from hierarchy.stages import kit_problem, load_shots

SAME = ("agreed", "stage", "winner", "converged", "iterations", "weight_P1", "weight_P2", "weight_R", "iterations_P1",
        "iterations_P2", "iterations_R", "predicted_R")


def decode(problem, options, data, shots):
    """(predictions, metrics, seconds) of the first shots through one session."""
    session = create_session("hierarchy.plugin:make_decoder", problem, options)
    predictions, metrics, seconds = [], [], []
    for full, part in zip(data.detectors[:shots], data.basis_input[:shots], strict=True):
        start = perf_counter()
        result = session.decode_shot(full, part)
        seconds.append(perf_counter() - start)
        predictions.append(validate_shot_result(result))
        metrics.append(result.metrics)
    close = getattr(session.hierarchy, "close", None)
    if close:
        close()
    return np.stack(predictions), metrics, np.array(seconds)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--x", required=True, help="a shot file of X memory (not held out)")
    parser.add_argument("--z", required=True, help="a shot file of Z memory (not held out)")
    parser.add_argument("--shots", type=int, default=300)
    parser.add_argument("--options", help="the plugin options JSON (default: the weight rule, which needs no checkpoint)")
    args = parser.parse_args(argv)
    options = json.loads(Path(args.options).read_text(encoding="utf-8")) if args.options else {"rule": "weight"}
    different = 0
    for path in (args.x, args.z):
        data = load_shots(path)                        # refuses held-out files
        problem = kit_problem(data.basis, data.p)
        one, one_metrics, one_seconds = decode(problem, options | {"parallel": False}, data, args.shots)
        two, two_metrics, two_seconds = decode(problem, options | {"parallel": True}, data, args.shots)
        bad = [shot for shot in range(len(one)) if not np.array_equal(one[shot], two[shot])
               or any(one_metrics[shot].get(key) != two_metrics[shot].get(key) for key in SAME)]
        different += len(bad)
        escalated = sum(not shot["agreed"] for shot in one_metrics)
        print(f"{Path(path).name}: {data.basis} memory, p = {data.p:g}, {len(one)} shots, relay-BP and the rule ran on {escalated}; "
              f"shots where the two modes differ: {len(bad)}" + (f" {bad[:20]}" if bad else ""))
        print(f"  mean time per shot on this machine: sequential {one_seconds.mean() * 1e3:.1f} ms, side by side "
              f"{two_seconds.mean() * 1e3:.1f} ms (wall time, two cores)", flush=True)
    print("RESULT: " + ("the side-by-side mode gives the sequential answers on every shot" if not different else "THE TWO MODES DIFFER"))
    return different


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
