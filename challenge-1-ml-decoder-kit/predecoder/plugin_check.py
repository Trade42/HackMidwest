"""Does the plugin give the answers compare.py measured? The first shots of a file through the team's evaluator loop.

Run from challenge-1-ml-decoder-kit/:

  python -m predecoder.plugin_check --data data/p0.004/validation_X.npz --per-shot <compare folder>/per_shot.npz \\
      --head <head.pt> --trunk <the trunk checkpoint> --settings predecoder/logs/fit_X_p0.004.json --shots 200

It builds the plugin with evaluation.adapters.create_session (as evaluation.evaluate does),
decodes the first --shots shots with evaluation.evaluate.run_basis, one shot at a time, and
compares every prediction with the head arm's in compare.py's per-shot file. They agree exactly
when the network runs on the same kind of device as the logits compare.py read (both on a CPU
here); on a GPU, batch-one and batched logits differ in their last digits.
"""
import argparse
import json

import numpy as np

from evaluation.adapters import create_session
from evaluation.data import load_dataset
from evaluation.evaluate import run_basis
from predecoder.decode import kit_problem


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--per-shot", required=True, help="per_shot.npz of predecoder.compare with the head arm")
    parser.add_argument("--head", required=True)
    parser.add_argument("--trunk", required=True)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--shots", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", help="a JSON file for the numbers")
    args = parser.parse_args(argv)
    full = load_dataset(args.data)
    if full.path.name.startswith(("test_", "go_")):
        raise SystemExit("not on held-out files")
    shots = min(args.shots, len(full.detectors))
    import dataclasses
    data = dataclasses.replace(full, detectors=full.detectors[:shots], observables=full.observables[:shots])
    problem = kit_problem(data.basis, data.p)
    options = {"heads": {data.basis: args.head}, "trunks": {data.basis: args.trunk}, "settings": args.settings,
               "device": args.device, "torch_threads": args.threads}
    session = create_session("predecoder.plugin:make_decoder", problem, options)
    result = run_basis(session, data)
    with np.load(args.per_shot) as file:
        measured = file["predicted_head"][:shots]
        truth = file["truth"][:shots]
    if not np.array_equal(truth, data.observables):
        raise SystemExit("the per-shot file was not made from this shot file")
    same = (result["predictions"] == measured).all(axis=1)
    seconds = result["total_seconds"]
    summary = {"data": full.identity(), "shots": shots, "same_prediction": int(same.sum()),
               "failures_plugin": int(result["failed"].sum()), "failures_compare": int((measured != truth).any(axis=1).sum()),
               "seconds_mean": float(seconds.mean()), "seconds_median": float(np.median(seconds)),
               "seconds_p99": float(np.percentile(seconds, 99)), "device": args.device, "threads": args.threads,
               "settings": session.settings}
    print(f"{full.path.name}: the first {shots} shots through the plugin and evaluation.evaluate.run_basis: the same "
          f"12 bits as compare.py's head arm on {int(same.sum())} of {shots}; failures {summary['failures_plugin']} "
          f"(compare.py: {summary['failures_compare']}); time per shot, everything inside decode_shot, on {args.device} "
          f"with {args.threads} threads: mean {summary['seconds_mean'] * 1e3:.0f} ms, median "
          f"{summary['seconds_median'] * 1e3:.0f} ms, p99 {summary['seconds_p99'] * 1e3:.0f} ms")
    if args.out:
        with open(args.out, "w") as file:
            json.dump(summary, file, indent=1, default=str)
    if not same.all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
