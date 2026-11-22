"""How the two pictures of the noise differ, in the decoder's own terms.

  python -m neutral_atom_noise.explain --factors 0.7,0.9

A decoder's priors say how likely each column of H is: each column is one kind of fault, and it flips 1 to 6
detectors. This prints, for the matched and the generic priors of each factor, how the expected number of faults
per shot splits over the kinds of column:

- "readout-like": a column that flips the same check in two consecutive rounds and no logical bit. These are the
  faults a readout or reset flip makes (and a phase error on a check atom, which only flips its own readout);
- columns that flip 3 or 4 detectors, and 5 or 6: faults on data atoms, and "hook" faults that one check atom
  spreads onto several data atoms;
- columns that flip a logical bit.

It also prints how many columns the matched picture thinks are at least 20 times rarer than the generic one does.
"""
import argparse

import numpy as np

from neutral_atom_noise.model import neutral_atom
from neutral_atom_noise.priors import effects, kit_problem, matched_priors, uniform_priors

CHECKS = 72      # X checks; the decoded detectors come in rounds of 72


def column_kinds(basis="X"):
    """A label for every column of the kit's H: readout-like, 3-4, 5-6 detectors, or the rest."""
    H, L, _ = kit_problem(basis)
    kinds = []
    for rows, observables in effects(H, L):
        check, rnd = [r % CHECKS for r in rows], [r // CHECKS for r in rows]
        if not observables and len(rows) == 2 and check[0] == check[1] and abs(rnd[0] - rnd[1]) == 1:
            kinds.append("readout-like (one check, two rounds)")
        elif len(rows) <= 4:
            kinds.append("3 or 4 detectors")
        else:
            kinds.append("5 or 6 detectors")
    flips_logical = np.asarray(L.sum(axis=0)).ravel() > 0
    return np.array(kinds), flips_logical


def explain(factors, basis="X"):
    kinds, flips_logical = column_kinds(basis)
    names = sorted(set(kinds))
    lines = [f"| Factor | Priors | Expected faults per shot | " + " | ".join(f"share: {n}" for n in names)
             + " | share: flips a logical bit |", "|---|---|---|" + "---|" * (len(names) + 1)]
    notes = []
    for factor in factors:
        noise = neutral_atom(factor)
        matched, _ = matched_priors(basis, noise)
        generic = uniform_priors(basis, noise.two_qubit_error())
        for label, priors in (("matched", matched), (f"generic (p = {noise.two_qubit_error():.4f})", generic)):
            shares = [priors[kinds == n].sum() / priors.sum() for n in names]
            lines.append(f"| {factor:g} | {label} | {priors.sum():.1f} | " + " | ".join(f"{s:.1%}" for s in shares)
                         + f" | {priors[flips_logical].sum() / priors.sum():.1%} |")
        rarer = int((matched < generic / 20).sum())
        notes.append(f"factor {factor:g}: {rarer} of {len(matched)} columns are at least 20 times rarer in the matched "
                     f"picture than in the generic one; the largest matched ÷ generic ratio is {(matched / generic).max():.2f}")
    return "\n".join(lines) + "\n\n" + "\n".join(notes)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--factors", default="0.7,0.9")
    parser.add_argument("--basis", choices=["X", "Z"], default="X")
    args = parser.parse_args()
    print(explain([float(f) for f in args.factors.split(",")], args.basis))


if __name__ == "__main__":
    main()
