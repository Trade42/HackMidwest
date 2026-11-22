"""Leakage check on real files: move the truth between shots and show that no choice of the selector changes.

Run from challenge-1-ml-decoder-kit/:

  python -m selector.shuffle_check --model selector/models/<name>.json --members <member files of one data file> --work <scratch folder>

It writes copies of the member files into --work in which the truth, and every array made from
it (each `failed_*`; J's rows by their `shot`), are permuted between shots with a fixed seed. It
then applies the model to the originals and to the copies, and stops with status 1 if any
feature, choice or answer differs. The truth is read only to permute it.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from selector import evaluate as E
from selector import features as F
from selector import model as M

SEED = 20261122


def shuffled_copies(paths, work, seed=SEED):
    with np.load(paths[0], allow_pickle=False) as file:
        count = int(file["shots_in_file"]) if "shots_in_file" in file.files else len(file["truth"])
    full = None
    for path in paths:
        with np.load(path, allow_pickle=False) as file:
            if "shot" not in file.files:
                full = file["truth"]
                break
    permutation = np.random.default_rng(seed).permutation(count)
    new_truth = full[permutation]
    out = []
    Path(work).mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as file:
            arrays = {key: file[key] for key in file.files}
        rows = arrays["shot"] if "shot" in arrays else np.arange(count)
        arrays["truth"] = new_truth[rows]
        for key in [key for key in arrays if key.startswith("failed_")]:
            arrays[key] = (arrays[f"predicted_{key[len('failed_'):]}"] != new_truth[rows]).any(axis=1)
        target = Path(work) / f"{index}_{Path(path).name}"
        np.savez(target, **arrays)
        out.append(target)
    moved = int((new_truth != full).any(axis=1).sum())
    return out, moved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--members", nargs="+", required=True)
    parser.add_argument("--work", required=True, help="a scratch folder for the shuffled copies")
    args = parser.parse_args(argv)
    model = M.load(args.model)
    copies, moved = shuffled_copies(args.members, args.work)
    first = E.selector_row(F.load(args.members)[0], model)
    second = E.selector_row(F.load(copies)[0], model)
    same = {"features": np.array_equal(first[3].features, second[3].features), "choices": np.array_equal(first[4], second[4]),
            "answers": np.array_equal(first[0], second[0]), "choice shots": np.array_equal(first[1], second[1])}
    view, info = F.load(args.members)
    shuffled_truth = F.load(copies)[1].truth
    print(f"{info.data_name or args.members[0]}: the truth moved on {moved} of {len(info.truth)} shots (seed {SEED});"
          f" failures of the selector against the true truth {int((first[0] != info.truth).any(1).sum())},"
          f" against the moved truth {int((second[0] != shuffled_truth).any(1).sum())}")
    print("unchanged under the shuffle: " + ", ".join(f"{key} {'yes' if value else 'NO'}" for key, value in same.items()))
    return 0 if all(same.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
