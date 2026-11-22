"""Are two candidate files the same, shot for shot?

Run from challenge-1-ml-decoder-kit/:

  python -m hierarchy.compare a.npz b.npz

Takes two files of hierarchy.candidates (an output file, or a part file of a run that is not
finished), or the laptop's selector_*.npz. For each of R, P1 and P2 it counts the shots where
the two files hold different predicted bits, convergence, iteration counts or fault counts, and
the largest difference in weight. The times are not compared: they depend on the machine.

Two runs with the same mode, seeds and settings must agree on every shot, on any machine. A
difference means another build of a decoder, another platform's arithmetic, or another mode.
"""
import argparse

import numpy as np

from hierarchy.rules import ORDER

EXACT = ("predicted", "converged", "iterations", "n_faults")


def load(path):
    with np.load(path, allow_pickle=False) as file:
        rows = {key: file[key] for key in file.files}
    shots = len(rows["truth"])
    rows.setdefault("shot", np.arange(shots))
    rows.setdefault("ran_R", np.ones(shots, bool))
    return rows


def compare(a, b):
    """{decoder: counts} over the shots both files hold. Raises if they are not the same shots."""
    shared, rows_a, rows_b = np.intersect1d(a["shot"], b["shot"], return_indices=True)
    if not len(shared):
        raise SystemExit("the two files share no shot")
    if not np.array_equal(a["truth"][rows_a], b["truth"][rows_b]):
        raise SystemExit("the two files were not made from the same shots: the truth differs")
    result = {"shots": len(shared)}
    for name in ORDER:
        both = np.ones(len(shared), bool)
        if name == "R":
            both = a["ran_R"][rows_a] & b["ran_R"][rows_b]
        pick_a, pick_b = rows_a[both], rows_b[both]
        counts = {"shots": int(both.sum())}
        for label, rows, pick in (("a", a, pick_a), ("b", b, pick_b)):
            counts[f"failures_{label}"] = int((rows[f"predicted_{name}"][pick] != rows["truth"][pick]).any(axis=1).sum())
        for field in EXACT:
            differ = a[f"{field}_{name}"][pick_a] != b[f"{field}_{name}"][pick_b]
            counts[field] = int((differ.any(axis=1) if differ.ndim == 2 else differ).sum())
        gap = np.abs(a[f"weight_{name}"][pick_a] - b[f"weight_{name}"][pick_b])
        counts["weight"] = float(gap.max()) if len(gap) else 0.0
        result[name] = counts
    return result


def same(result, tolerance=1e-9):
    return all(result[name][field] == 0 for name in ORDER for field in EXACT) and all(result[name]["weight"] <= tolerance for name in ORDER)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a")
    parser.add_argument("b")
    args = parser.parse_args(argv)
    a, b = load(args.a), load(args.b)
    result = compare(a, b)
    print(f"a = {args.a}\nb = {args.b}\n{result['shots']} shots in both files, the same truth in both")
    for key in ("data_sha256", "problem_sha256"):
        if key in a and key in b:
            print(f"{key}: {'the same' if str(a[key]) == str(b[key]) else 'DIFFERENT'} ({a[key]})")
    print(f"{'decoder':8s} {'shots':>7s} {'failures a':>10s} {'failures b':>10s}   shots that differ in: "
          f"{'predicted':>9s} {'converged':>9s} {'iterations':>10s} {'n_faults':>8s}   largest weight gap")
    for name in ORDER:
        c = result[name]
        print(f"{name:8s} {c['shots']:7d} {c['failures_a']:10d} {c['failures_b']:10d}   {'':21s} "
              f"{c['predicted']:9d} {c['converged']:9d} {c['iterations']:10d} {c['n_faults']:8d}   {c['weight']:.3g}")
    print("RESULT: " + ("the two files agree on every shot" if same(result) else "the two files differ"))
    return result


if __name__ == "__main__":
    main()
