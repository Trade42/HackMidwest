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

Fit, from challenge-1-ml-decoder-kit/. Failures are rare (about 1 shot in 200 at p = 0.004),
far too rare to rank a dozen settings on a plain slice of train shots. So the fit decodes two
train files that subset.py writes from a run of compare.py's "fresh" arm on train_X.npz:

  train_X_fresh_failed.npz   every train shot relay-BP with the kit's priors gets wrong
  train_X_fresh_right.npz    a random sample of the shots it gets right

  python -m predecoder.priors --data <failed file> <right file> --weights 1 <right shots in all / sample size> \
      --logits <folder of predecoder.evaluate for each file, in the same order> --workers 4 --out <fit.json>

For every setting of the grid each file is decoded by relay-BP at the kit's settings, a decoder
built per shot. A setting's score is its weighted failures: the failed shots it leaves wrong
plus the right shots it breaks, scaled to the whole train file. The chosen setting has the
lowest score; ties go to fewer weighted hard shots (relay-BP's first leg alone, 80 iterations
with the same priors, does not converge), then fewer mean iterations. Row 1 is the kit's
priors, the control: it must fail every shot of the failed file and none of the right file.
The rule was fixed before any trained head was decoded. Every row goes into the JSON file.
Each (file, setting) is saved in <out>.parts/ as soon as it is done, so a stopped fit resumes.
"""
import argparse
import hashlib
import itertools
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np

from predecoder.decode import decode_with_priors, first_leg_converges, kit_problem, settings

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
    hard = not first_leg_converges(_JOB["H"], priors, syndrome)
    return json.dumps(setting, sort_keys=True), shot, bool((result.predicted != truth).any()), hard, result.iterations


RULE = "lowest weighted failures (failed shots left wrong + right shots broken, scaled); then weighted hard shots; then iterations"


def choose(rows):
    """The chosen row among the candidates (the control excluded by the caller): see RULE."""
    return min(rows, key=lambda row: (row["score"], row["hard_weighted"], row["mean_iterations"]))


def grid(modes, alphas, epsilons):
    rows = [{"mode": mode, "alpha": float(alpha), "eps": float(eps)} for mode, alpha, eps in itertools.product(modes, alphas, epsilons)]
    return [{"mode": "linear", "alpha": 0.0, "eps": 1e-6}] + rows        # the kit's priors, as the control


def main(argv=None):
    from evaluation.data import load_dataset
    from predecoder.evaluate import read_logits

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="+", required=True, help="TRAIN shot files (names start with train_)")
    parser.add_argument("--weights", type=float, nargs="+", help="one per file: how many train shots one of its shots stands for")
    parser.add_argument("--logits", nargs="+", required=True, help="the folder predecoder.evaluate wrote for each file")
    parser.add_argument("--shots", type=int, help="only the first this many shots of each file")
    # The default grid was set from the smoke head on 300 train shots (logs/fit_smoke_train300_grid.log): log-odds
    # mixing beat linear mixing there, and alpha = 1 failed most.
    parser.add_argument("--modes", nargs="+", default=["logodds"], choices=MODES)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--eps", type=float, nargs="+", default=[1e-4])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--control-only", action="store_true",
                        help="decode the control row (the kit's priors) only, into the cache; it needs no head")
    parser.add_argument("--out", required=True, help="the JSON file of the fit")
    args = parser.parse_args(argv)
    weights = args.weights or [1.0] * len(args.data)
    if not len(args.data) == len(args.logits) == len(weights):
        raise SystemExit("give one --logits folder and one weight per --data file")
    if not all(Path(path).name.startswith("train_") for path in args.data):
        raise SystemExit("the knobs are fitted on train shots only: give train_* files")

    start = time.time()
    files = []
    for path, folder in zip(args.data, args.logits, strict=True):
        data = load_dataset(path)
        meta, logits_path = read_logits(folder, data)
        files.append((data, meta, logits_path))
    problem = kit_problem(files[0][0].basis, files[0][0].p)
    if any(data.p != problem.p or data.basis != problem.basis for data, _, _ in files):
        raise SystemExit("all files must have the same basis and p")
    settings_grid = grid(args.modes, args.alphas, args.eps)[:1 if args.control_only else None]
    keys = [json.dumps(setting, sort_keys=True) for setting in settings_grid]
    cache = Path(args.out).with_suffix(".parts")
    cache.mkdir(parents=True, exist_ok=True)
    results = {}
    for index, (data, meta, logits_path) in enumerate(files):
        shots = len(data.detectors) if args.shots is None else min(args.shots, len(data.detectors))
        results[index] = {}
        with multiprocessing.get_context("spawn").Pool(args.workers, _start,
                                                       (problem.H.tocsr(), problem.L, str(logits_path), problem.priors)) as pool:
            for setting, key in zip(settings_grid, keys):
                # One file per (shot file, head, setting): a stopped fit continues where it stopped.
                head = None if setting["alpha"] == 0 else meta["model"]["head_sha256"]     # alpha 0: the kit's priors, no head
                what = json.dumps({"data": data.sha256, "head": head, "shots": shots, "setting": setting}, sort_keys=True)
                path = cache / f"{hashlib.sha256(what.encode()).hexdigest()[:16]}.npy"
                if not path.exists():
                    tasks = [(k, setting, data.basis_input[k], data.observables[k]) for k in range(shots)]
                    values = np.zeros((shots, 3))
                    for _, shot, failed, hard, iterations in pool.imap_unordered(_decode, tasks, chunksize=4):
                        values[shot] = failed, hard, iterations
                    np.save(path, values)
                results[index][key] = np.load(path)
                print(f"  {data.path.name}: {shots} shots, {setting}, {time.time() - start:.0f} s", flush=True)
    rows = []
    print(f"fit on {', '.join(f'{d.path.name} (weight {w:g})' for (d, _, _), w in zip(files, weights))}, p={problem.p:g}; "
          f"relay-BP at the kit's settings, a decoder built per shot; {args.workers} processes, {time.time() - start:.0f} s")
    names = [data.path.stem for data, _, _ in files]
    print(f"{'mode':8s} {'alpha':>6s} {'eps':>8s} " + " ".join(f"{'fail ' + n[:18]:>24s}" for n in names)
          + f" {'score':>8s} {'hard (weighted)':>16s} {'mean iter':>10s}")
    for setting, key in zip(settings_grid, keys):
        per_file = [results[i][key] for i in range(len(files))]
        failures = [int(values[:, 0].sum()) for values in per_file]
        hard = [int(values[:, 1].sum()) for values in per_file]
        total = sum(len(values) * w for values, w in zip(per_file, weights))
        row = setting | {"failures": failures, "hard": hard,
                         "score": float(sum(f * w for f, w in zip(failures, weights))),
                         "hard_weighted": float(sum(h * w for h, w in zip(hard, weights))),
                         "mean_iterations": float(sum(values[:, 2].sum() * w for values, w in zip(per_file, weights)) / total)}
        rows.append(row)
        print(f"{row['mode']:8s} {row['alpha']:6g} {row['eps']:8g} " + " ".join(f"{f:24d}" for f in failures)
              + f" {row['score']:8.1f} {row['hard_weighted']:16.1f} {row['mean_iterations']:10.1f}")
    if args.control_only:
        print("control only: nothing chosen")
        return
    chosen = choose(rows[1:])
    print(f"chosen: mode {chosen['mode']}, alpha {chosen['alpha']:g}, eps {chosen['eps']:g} ({RULE}); row 1 is the kit's "
          f"priors")
    edge = (chosen["alpha"] in (min(args.alphas), max(args.alphas)) and len(args.alphas) > 1)
    if edge:
        print("note: the chosen alpha is at an edge of the grid; extend the grid (on train shots) before freezing it")
    with open(args.out, "w") as file:
        json.dump({"data": [data.identity() for data, _, _ in files], "weights": weights, "logits": [m for _, m, _ in files],
                   "relay": settings(), "rows": rows, "chosen": {key: chosen[key] for key in DEFAULT}, "rule": RULE,
                   "chosen_at_grid_edge": edge}, file, indent=1)


if __name__ == "__main__":
    main()
