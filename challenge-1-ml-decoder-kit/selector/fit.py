"""Fit the selector on TRAINING shots, and save it as a small JSON model file.

Run from challenge-1-ml-decoder-kit/:

  python -m selector.fit --pool T P1 P2 R \\
      --train <member files of train_X at one p> \\
      --train <member files of train_Z at one p> \\
      --out selector/models/<name>.json

Each --train takes the member files of ONE data file (hierarchy.candidates or the laptop's
selector file for P1, P2, R; hierarchy.tcbp_member for T; hierarchy.joint_member for J). Give
--train once per data file. --pool names the members whose answers may be chosen; a member
outside the pool is ignored.

It refuses any file made from held-out shots: a path or a recorded data name that starts with
`test_` or `go_` (also after a `selector_` prefix, as the laptop names its files), or a path
under `clean/` (where the run on the test shots writes). Validation files are refused too
unless --allow-validation is given: the model is fitted on training files only.

It prints the training shots, the cross-validated choice of lambda (model.py), the weights, and
the sha256 of the model file.
"""
import argparse
import re
from pathlib import Path

import numpy as np

from selector import features as F
from selector import model as M

HELD_OUT = re.compile(r"(^|[/\\_.-])(test|go)_")
CLEAN = re.compile(r"(^|[/\\])clean[/\\]")


def refuse_held_out(name, what):
    """Stop if a file to fit on was made from held-out shots (by its path or by the data name it records)."""
    text = str(name)
    base = text.replace("\\", "/").rsplit("/", 1)[-1]
    if HELD_OUT.search("/" + base) or CLEAN.search(text):
        raise SystemExit(f"refused: {what} ({text}) is made from held-out shots; the selector is never fitted on test_* or go_* files")


def is_validation(name):
    base = str(name).replace("\\", "/").rsplit("/", 1)[-1]
    return re.search(r"(^|[_.-])validation_", base) is not None


def training_table(groups, pool, allow_validation=False, out=print):
    """The choice tables of the training files, stacked, with their labels and a record of each file."""
    tables, records = [], []
    for paths in groups:
        for path in paths:
            refuse_held_out(path, "a member file")
            if is_validation(path) and not allow_validation:
                raise SystemExit(f"refused: {path} is a validation file; fit on train files (or pass --allow-validation for a stated exception)")
        view, info = F.load(paths)
        refuse_held_out(info.data_name, f"the data file recorded in {paths[0]}")
        if info.data_name and is_validation(info.data_name) and not allow_validation:
            raise SystemExit(f"refused: the member files of {info.data_name} are validation files")
        choices = F.table(view, pool=pool)
        right = F.labels(choices, info.truth)
        tables.append((choices, right))
        any_right = np.array([right[choices.starts[i]:choices.starts[i + 1]].any() for i in range(len(choices.shots))], bool)
        record = {"data_name": info.data_name or "(not recorded)", "data_sha256": info.data_sha256, "basis": info.basis, "p": info.p,
                  "shots": int(len(info.truth)), "choice_shots": int(len(choices.shots)), "choice_shots_with_a_right_answer": int(any_right.sum()),
                  "member_files": {str(path): M.sha256_of(path) for path in paths}}
        records.append(record)
        out(f"  {record['data_name']:18s} {info.basis} p = {info.p}  shots {record['shots']:6d}  choice shots {record['choice_shots']:5d}"
            f"  with a right answer {record['choice_shots_with_a_right_answer']:5d}   data sha256 {info.data_sha256[:16]}...")
    hashes = [record["data_sha256"] for record in records]
    if len(set(hashes)) != len(hashes):
        raise SystemExit("two --train groups are the same data file")
    features = np.concatenate([t.features for t, _ in tables])
    gives = np.concatenate([t.gives for t, _ in tables])
    right = np.concatenate([r for _, r in tables])
    starts = [0]
    for t, _ in tables:
        starts.extend(starts[-1] + t.starts[1:])
    return features, np.array(starts, int), right, gives, tables[0][0].names, records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", nargs="+", action="append", required=True, help="the member files of one training data file")
    parser.add_argument("--pool", nargs="+", default=list(F.POOL), choices=F.POOL, help="the members whose answers may be chosen")
    parser.add_argument("--out", required=True, help="the model file to write (.json)")
    parser.add_argument("--allow-validation", action="store_true", help="allow validation files in --train (never used for a frozen model)")
    args = parser.parse_args(argv)

    pool = tuple(name for name in F.POOL if name in args.pool)
    for name in F.ALWAYS_ON:
        if name not in pool:
            raise SystemExit(f"--pool must hold the always-on members {F.ALWAYS_ON}")
    print(f"fitting the selector on {len(args.train)} training file(s); pool {', '.join(pool)}")
    features, starts, right, gives, names, records = training_table(args.train, pool, args.allow_validation)
    baseline = gives[:, F.POOL.index("T")]
    cv = M.cross_validate(features, starts, right, baseline=baseline)
    print(f"\n{M.FOLDS}-fold cross-validation over the training choice shots that have a right answer (folds by shot, seed {M.FOLD_SEED}):")
    print(f"  {'lambda':>8s} {'right':>7s} {'of':>5s}   held out: both wrong, only the selector wrong, only TCBP-JOBS wrong")
    for lam, counts in cv.items():
        print(f"  {lam:8g} {counts['right']:7d} {counts['shots']:5d}   {counts['both_wrong']:5d} {counts['only_selector_wrong']:5d} {counts['only_baseline_wrong']:5d}")
    lam = M.choose_lambda(cv)
    mean, scale = M.standardize(features)
    weights, used = M.fit_weights((features - mean) / scale, starts, right, lam)
    print(f"chosen lambda {lam:g}; final fit on all {used} training choice shots with a right answer")
    chosen, _ = M.pick(features, starts, {"mean": mean, "scale": scale, "weights": weights})
    train_right = int(right[chosen].sum())
    t_right = int(sum(right[starts[i]:starts[i + 1]][np.flatnonzero(baseline[starts[i]:starts[i + 1]])[0]] for i in range(len(starts) - 1)))
    print(f"on the training choice shots themselves (in-sample, not a result): the selector right on {train_right}, TCBP-JOBS on {t_right}")
    print("\nweights (on standardized features; a positive weight favours an answer with more of the feature):")
    for name, w, mu, sd in zip(names, weights, mean, scale):
        print(f"  {name:20s} {w:+8.3f}   (mean {mu:7.3f}, spread {sd:7.3f})")
    model = {"kind": "conditional logit over the distinct answers of a choice shot (selector/model.py)",
             "features": list(names), "pool": list(pool), "mean": mean, "scale": scale, "weights": weights, "lambda": lam,
             "cross_validation": {str(k): v for k, v in cv.items()}, "folds": M.FOLDS, "fold_seed": M.FOLD_SEED,
             "trained_on": records, "training_choice_shots_used": used}
    sha = M.save(model, args.out)
    print(f"\nwrote {args.out}  sha256 {sha}")
    return model


if __name__ == "__main__":
    main()
