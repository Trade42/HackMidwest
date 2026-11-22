"""Two small train files for the fit of priors.py: the shots one arm of compare.py gets wrong, and a sample of the rest.

Run from challenge-1-ml-decoder-kit/, after `python -m predecoder.compare --data data/p0.004/train_X.npz --arms fresh ...`:

  python -m predecoder.subset --data data/p0.004/train_X.npz --per-shot <compare folder>/per_shot.npz --arm fresh \\
      --right 2000 --seed 5 --out <folder>

It writes <folder>/train_X_fresh_failed.npz and <folder>/train_X_fresh_right.npz in the format of
make_dataset.py (plus `shot`: each shot's row in the source file), and prints the weight of the
right file for priors.py: the right shots in all, divided by the sample size. Only train files
are accepted. The shot files are data: keep them out of git.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.data import load_dataset


def from_parts(folder, data_sha256, arm):
    """(shot indices, failed) from the part files of compare.py for this shot file and arm (no head arm)."""
    from predecoder.compare import part_tag

    tag = part_tag(data_sha256, arm)
    shots, failed = [], []
    for path in sorted(folder.glob(f"{arm}_{tag}_*.npy")):
        first = int(path.stem.split("_")[2])
        rows = np.load(path)
        shots.append(first + np.arange(len(rows)))
        failed.append(rows[:, 0] > 0)
    if not shots:
        raise SystemExit(f"{folder} holds no part of the {arm} arm for this shot file")
    shots, failed = np.concatenate(shots), np.concatenate(failed)
    if len(np.unique(shots)) != len(shots):
        raise SystemExit(f"{folder} holds parts of two runs with different chunk sizes")
    return shots, failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--per-shot", help="per_shot.npz of predecoder.compare on that file")
    parser.add_argument("--parts", help="or: the parts/ folder of a compare run that did not finish (its finished chunks)")
    parser.add_argument("--arm", default="fresh")
    parser.add_argument("--right", type=int, default=2000, help="the size of the sample of right shots")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    source = Path(args.data)
    if not source.name.startswith("train_"):
        raise SystemExit("subsets are made of train shots only")
    data = load_dataset(source)
    if (args.per_shot is None) == (args.parts is None):
        raise SystemExit("give --per-shot or --parts")
    if args.per_shot:
        with np.load(args.per_shot) as file:
            failed = file[f"failed_{args.arm}"].astype(bool)
            truth = file["truth"]
        if not np.array_equal(truth, data.observables[:len(truth)]):
            raise SystemExit("the per-shot file was not made from this shot file")
        decoded = np.arange(len(failed))
    else:
        decoded, failed = from_parts(Path(args.parts), data.sha256, args.arm)
    wrong = decoded[failed]
    right_all = decoded[~failed]
    sample = np.sort(np.random.default_rng(args.seed).choice(right_all, size=min(args.right, len(right_all)), replace=False))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    written = {}
    for name, shots in ((f"{stem}_{args.arm}_failed", wrong), (f"{stem}_{args.arm}_right", sample)):
        path = out / f"{name}.npz"
        np.savez_compressed(path, p=np.array(data.p), seed=np.array(data.seed), basis=np.array(data.basis),
                            detectors=data.detectors[shots], observables=data.observables[shots],
                            basis_detectors=data.basis_detectors, shot=shots)
        written[name] = {"path": str(path), "shots": int(len(shots))}
    weight = len(right_all) / max(1, len(sample))
    summary = {"source": data.identity(), "decoded_shots": int(len(failed)), "from": args.per_shot or args.parts, "arm": args.arm, "failed": int(len(wrong)),
               "right": int(len(right_all)), "sample": int(len(sample)), "seed": args.seed, "right_weight": weight,
               "files": written}
    with open(out / f"{stem}_{args.arm}_subsets.json", "w") as file:
        json.dump(summary, file, indent=1)
    print(f"{source.name}: of {len(failed)} decoded shots the {args.arm} arm fails {len(wrong)}; wrote them and a sample of "
          f"{len(sample)} of the {len(right_all)} right ones (seed {args.seed}) to {out}")
    print(f"weights for priors.py: failed file 1, right file {weight:.4f}")


if __name__ == "__main__":
    main()
