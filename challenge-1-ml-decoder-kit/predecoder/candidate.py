"""The predecoder's answer as one more candidate beside the hierarchy's members, on the same shots.

Run from challenge-1-ml-decoder-kit/:

  python -m predecoder.candidate --ours <compare folder>/per_shot.npz \\
      --members <selector_validation_X.npz of the laptop> --tcbp <validation_X.tcbp.npz of the server>

The members' files are read, never written: the laptop's `priortilt/results_8b/selector_*.npz`
(R = relay-BP, P1 and P2 = PriorTilt under two seeds) and the server's
`files/tcbp_p0.004/*.tcbp.npz` (T = TCBP-JOBS), from `origin/laptop-artifacts` and
`origin/server-logs`. Ours is `compare.py`'s per-shot file: H = relay-BP with the head's priors.
All must hold the same shots in the same order (their truth is compared).

It prints the failures of every candidate; on the shots where P1, P2 and T do not all agree,
how often each candidate is right; how many shots H gets right where R, P1, P2 and T all fail;
and the best possible choice (an oracle) over the members with and without H. Nothing here
changes hierarchy/: this only reads its members' answers.
"""
import argparse
import json

import numpy as np

from predecoder.decode import paired


def load(path, names):
    with np.load(path, allow_pickle=False) as file:
        return file["truth"], {name: file[f"predicted_{name}"] for name in names if f"predicted_{name}" in file.files}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ours", required=True, help="per_shot.npz of predecoder.compare (with the head arm)")
    parser.add_argument("--members", required=True, help="a selector_*.npz of the laptop (R, P1, P2)")
    parser.add_argument("--tcbp", required=True, help="a *.tcbp.npz of the server (T)")
    parser.add_argument("--out", help="a JSON file for the numbers")
    args = parser.parse_args(argv)
    truth, ours = load(args.ours, ["head", "fresh", "kit"])
    member_truth, members = load(args.members, ["R", "P1", "P2"])
    tcbp_truth, tcbp = load(args.tcbp, ["T"])
    if not (np.array_equal(truth, member_truth) and np.array_equal(truth, tcbp_truth)):
        raise SystemExit("the files do not hold the same shots in the same order: their truth differs")
    if "head" not in ours:
        raise SystemExit(f"{args.ours} has no predicted bits of the head arm")
    candidates = members | tcbp | {"H": ours["head"]}
    right = {name: (bits == truth).all(axis=1) for name, bits in candidates.items()}
    shots = len(truth)
    print(f"{shots} shots; failures: " + ", ".join(f"{name} {int((~ok).sum())}" for name, ok in right.items()))
    disagree = ~((candidates["P1"] == candidates["P2"]).all(axis=1) & (candidates["P1"] == candidates["T"]).all(axis=1))
    print(f"P1, P2 and T do not all agree on {int(disagree.sum())} shots; right there: "
          + ", ".join(f"{name} {int(ok[disagree].sum())}" for name, ok in right.items()))
    others = ["R", "P1", "P2", "T"]
    none_right = ~np.any([right[name] for name in others], axis=0)
    rescued = none_right & right["H"]
    print(f"R, P1, P2 and T all fail {int(none_right.sum())} shots; H is right on {int(rescued.sum())} of them")
    print(f"the best possible choice: over R, P1, P2, T fails {int(none_right.sum())}; with H as well "
          f"{int((none_right & ~right['H']).sum())}")
    paired_counts = {}
    for name in others:
        both, only_h, only_other, p_value = paired(~right["H"], ~right[name])
        paired_counts[name] = {"both": both, "only_H": only_h, f"only_{name}": only_other, "p": p_value}
        print(f"paired H against {name}: both fail {both}, only H {only_h}, only {name} {only_other}, exact two-sided p = {p_value:.3g}")
    agree_wrong = ~disagree & ~right["P1"]
    print(f"P1 = P2 = T but wrong on {int(agree_wrong.sum())} shots; H is right on {int((agree_wrong & right['H']).sum())} of them")
    summary = {"shots": shots, "failures": {name: int((~ok).sum()) for name, ok in right.items()},
               "disagree": int(disagree.sum()), "right_where_disagree": {name: int(ok[disagree].sum()) for name, ok in right.items()},
               "all_members_fail": int(none_right.sum()), "head_right_where_all_members_fail": int(rescued.sum()),
               "oracle_without_head": int(none_right.sum()), "oracle_with_head": int((none_right & ~right["H"]).sum()),
               "members_agree_but_wrong": int(agree_wrong.sum()), "head_right_there": int((agree_wrong & right["H"]).sum()),
               "paired": paired_counts}
    if args.out:
        with open(args.out, "w") as file:
            json.dump(summary, file, indent=1)


if __name__ == "__main__":
    main()
