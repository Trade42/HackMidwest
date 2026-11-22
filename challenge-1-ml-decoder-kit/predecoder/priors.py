"""One shot's priors for BP, from the head's logits; and the fit of the two knobs on TRAIN shots.

The head gives every column of H a probability q that it fired in this shot. BP takes priors.
Two knobs decide how much BP trusts the head:

  alpha    how far to move from the kit's prior towards the head:
             mode "linear":  alpha * q + (1 - alpha) * kit
             mode "logodds": the same mixture of the two log-odds
           alpha = 1 hands BP the head's probabilities as they are; alpha = 0 is the kit.
  eps      the priors are clipped to [eps, 1 - eps], so that no column is ever ruled out or
           forced in, however sure the head is.

Why a knob at all: the head has already read the syndrome, and BP reads it again. Giving BP
the head's numbers as they are counts the same evidence twice, which can make BP overconfident.
How much to hold back is an empirical question, so it is fitted, on train shots only.

The logits are rounded to float16 before use, everywhere (the saved files hold float16), so a
shot gets the same priors from a saved file and from the plugin.

Fit, from challenge-1-ml-decoder-kit/ (needs the logits of predecoder.evaluate on a train file):

  python -m predecoder.priors --data data/p0.004/train_X.npz --logits <folder of predecoder.evaluate> \\
      --shots 1000 --workers 4 --out <fit.json>

It decodes the first --shots shots under every setting of the grid with relay-BP at the kit's
settings, a decoder built per shot, and reports for each: failures, "hard" shots (the first
80-iteration leg did not converge) and mean iterations. Failures are rare (about 1 shot in 200
at p = 0.004), too rare to rank a dozen settings on 1,000 shots, while hard shots (about 1 in
7 with the kit's priors) hold almost all failures. So the choice is: among the settings that
fail no more than the kit's priors do on the same shots plus 2, the fewest hard shots, then the
fewest failures, then the fewest iterations (if none qualifies: the fewest failures). The
chosen setting goes into the JSON file with every row.
"""
import argparse
import itertools
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np

from decoders import RELAY_SETTINGS
from predecoder.decode import decode_with_priors, kit_problem, settings

MODES = ("linear", "logodds")
DEFAULT = {"mode": "linear", "alpha": 1.0, "eps": 1e-6}


def head_probabilities(logits):
    """float64 probabilities from logits, after rounding them to float16."""
    rounded = np.asarray(logits, np.float16).astype(np.float64)
    return 1.0 / (1.0 + np.exp(-rounded))


def shot_priors(logits, kit_priors, alpha=1.0, eps=1e-6, mode="linear"):
    """The priors of one shot (or of [shots, columns]) for BP."""
    kit = np.asarray(kit_priors, np.float64)
    if mode == "linear":
        mixed = alpha * head_probabilities(logits) + (1 - alpha) * kit
    elif mode == "logodds":                  # the head's logit is its log-odds
        log_odds = alpha * np.asarray(logits, np.float16).astype(np.float64) + (1 - alpha) * np.log(kit / (1 - kit))
        mixed = 1.0 / (1.0 + np.exp(-log_odds))
    else:
        raise ValueError(f"mode is one of {MODES}")
    return np.clip(mixed, eps, 1 - eps)


def load_settings(path_or_dict):
    """{mode, alpha, eps} from a fit file, a dict, or None (the default)."""
    if path_or_dict is None:
        return dict(DEFAULT)
    if isinstance(path_or_dict, dict):
        chosen = path_or_dict
    else:
        with open(path_or_dict) as file:
            chosen = json.load(file)["chosen"]
    return {key: chosen[key] for key in DEFAULT}


# ---------- the fit ----------

_JOB = {}


def _start(H, L, logits_path, kit_priors):
    _JOB.update(H=H, L=L, logits=np.load(logits_path, mmap_mode="r"), kit=kit_priors)


def _decode(task):
    shot, setting, syndrome, truth = task
    priors = shot_priors(_JOB["logits"][shot], _JOB["kit"], setting["alpha"], setting["eps"], setting["mode"])
    result = decode_with_priors(_JOB["H"], _JOB["L"], priors, syndrome)
    hard = (not result.converged) or result.iterations > RELAY_SETTINGS["pre_iter"]
    return json.dumps(setting, sort_keys=True), shot, bool((result.predicted != truth).any()), hard, result.iterations


RULE = ("among the settings with at most the kit's failures + 2: fewest hard shots, then failures, then iterations; "
        "if none: fewest failures")


def choose(control, rows):
    """The chosen row: see RULE."""
    allowed = [row for row in rows if row["failures"] <= control["failures"] + 2]
    if not allowed:
        return min(rows, key=lambda row: (row["failures"], row["hard"], row["mean_iterations"]))
    return min(allowed, key=lambda row: (row["hard"], row["failures"], row["mean_iterations"]))


def grid(modes, alphas, epsilons):
    rows = [{"mode": mode, "alpha": float(alpha), "eps": float(eps)} for mode, alpha, eps in itertools.product(modes, alphas, epsilons)]
    return [{"mode": "linear", "alpha": 0.0, "eps": 1e-6}] + rows        # the kit's priors, as the control


def main(argv=None):
    from evaluation.data import load_dataset
    from predecoder.evaluate import read_logits

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="a TRAIN shot file of make_dataset.py (its name starts with train_)")
    parser.add_argument("--logits", required=True, help="the folder predecoder.evaluate wrote for that file")
    parser.add_argument("--shots", type=int, default=1000, help="the first this many shots of the file")
    parser.add_argument("--modes", nargs="+", default=["linear", "logodds"], choices=MODES)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.8, 1.0])
    parser.add_argument("--eps", type=float, nargs="+", default=[1e-4, 1e-2])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", required=True, help="the JSON file of the fit")
    args = parser.parse_args(argv)
    if not Path(args.data).name.startswith("train_"):
        raise SystemExit("the knobs are fitted on train shots only: give a train_* file")

    start = time.time()
    data = load_dataset(args.data)
    meta, logits_path = read_logits(args.logits, data)
    problem = kit_problem(data.basis, data.p)
    shots = min(args.shots, len(data.detectors))
    syndromes, truth = data.basis_input[:shots], data.observables[:shots]
    settings_grid = grid(args.modes, args.alphas, args.eps)
    tasks = [(k, setting, syndromes[k], truth[k]) for setting in settings_grid for k in range(shots)]
    results = {json.dumps(s, sort_keys=True): np.zeros((shots, 3)) for s in settings_grid}
    with multiprocessing.get_context("spawn").Pool(args.workers, _start,
                                                   (problem.H.tocsr(), problem.L, str(logits_path), problem.priors)) as pool:
        for key, shot, failed, hard, iterations in pool.imap_unordered(_decode, tasks, chunksize=8):
            results[key][shot] = failed, hard, iterations
    rows = []
    print(f"fit on {data.path.name} (train), the first {shots} shots, p={data.p:g}; relay-BP at the kit's settings, a decoder "
          f"built per shot; {args.workers} processes, {time.time() - start:.0f} s")
    print(f"{'mode':8s} {'alpha':>6s} {'eps':>8s} {'failures':>9s} {'hard shots':>11s} {'mean iterations':>16s}")
    for setting in settings_grid:
        values = results[json.dumps(setting, sort_keys=True)]
        row = setting | {"failures": int(values[:, 0].sum()), "hard": int(values[:, 1].sum()),
                         "mean_iterations": float(values[:, 2].mean())}
        rows.append(row)
        print(f"{row['mode']:8s} {row['alpha']:6g} {row['eps']:8g} {row['failures']:9d} {row['hard']:11d} {row['mean_iterations']:16.1f}")
    chosen = choose(rows[0], rows[1:])
    print(f"chosen: mode {chosen['mode']}, alpha {chosen['alpha']:g}, eps {chosen['eps']:g} ({RULE}; the kit's priors "
          f"are row 1)")
    with open(args.out, "w") as file:
        json.dump({"data": data.identity(), "shots": shots, "logits": meta, "relay": settings(), "rows": rows,
                   "chosen": {key: chosen[key] for key in DEFAULT}, "rule": RULE},
                  file, indent=1)


if __name__ == "__main__":
    main()
