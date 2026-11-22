"""Apply a frozen selector model to the members' files of an X and a Z data file, and score it.

Run from challenge-1-ml-decoder-kit/:

  python -m selector.evaluate --model selector/models/<name>.json \\
      --x <member files of the X data file> --z <member files of the Z data file> \\
      --save <out folder>/<name>

This only applies a model; it never fits one. It works on any shots, the test shots included.

For each basis it prints, on the same shots: the selector, TCBP-JOBS alone, the ensemble's rule
f and its vote gw (scored by hierarchy.ensemble), the hand-written rules ac and bw, each member
alone, and the ORACLE rows (a perfect choice, which needs the truth: a ceiling, not a decoder).
Then the paired comparison of the selector with T, f and gw: both fail, only the selector
fails, only the other fails, and the exact two-sided binomial test on the shots only one of the
two gets wrong. Then the kit's pair score (shot k of X with shot k of Z; a pair fails when
either fails), with the same paired comparisons on pairs.

Where joint relay-BP was not run (no J file, as at p = 0.005), rule f is TCBP-JOBS's answer
wherever P1, P2, T are not unanimous, which is TCBP-JOBS alone, as in `ensemble.py --no-joint`.

The time of the selector's row is "combined offline": the sum of the members' decode calls it
needs (P1, P2 and T on every shot; R, and J where in the pool, on the choice shots), each timed on
its own machine as recorded in its file. A member file without a time per shot (TCBP-JOBS in
batch mode) makes the row's time unmeasured; the known parts are printed with their machines.

--save writes <prefix>_X.npz and <prefix>_Z.npz (per shot: failed, predicted, choice; and the
failed mask of every compared row) and <prefix>.json (the numbers printed).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from hierarchy.ensemble import Shots, score
from hierarchy.rules import paired_test, wilson_interval
from selector import features as F
from selector import model as M

COMPARED = ("T", "f", "gw")
SHOWN = ("T", "f", "gw", "ac", "bw", "R", "P1", "P2", "J")


def selector_row(view, model):
    """(answer [shots, 12], choice [shots] bool, seconds spent by the selector itself) for one View."""
    started = time.perf_counter()
    choices = F.table(view, pool=tuple(model["pool"]))
    if list(choices.names) != list(model["features"]):
        raise SystemExit(f"the model's features {model['features']} are not the table's {list(choices.names)}")
    picked, _ = M.pick(choices.features, choices.starts, model) if len(choices.shots) else (np.array([], int), None)
    answer = F.answers(view, choices, picked)
    spent = time.perf_counter() - started
    choice = np.zeros(view.shots, bool)
    choice[choices.shots] = True
    return answer, choice, spent, choices, picked


def one_basis(paths, model, label, out=print):
    view, info = F.load(paths)
    joint = "J" in view.members
    if not joint:
        view.members["J"] = F.empty_member("J", view.shots)
    shots = Shots(info.truth, info.data_sha256, info.basis, info.p, info.data_name)
    rows = {row["key"]: row for row in score(view.members, shots, None) if "failed" in row}
    answer, choice, spent, choices, picked = selector_row(view, model)
    failed = (answer != info.truth).any(axis=1)
    out(f"\n{info.basis} memory, p = {info.p}, {len(info.truth)} shots of {info.data_name or '(name not recorded)'}"
        f" (data sha256 {info.data_sha256}); {label}")
    for name, member in view.members.items():
        out(f"  {name:3s} {int(member.covers.sum()):6d} shots  from {member.source}")
    if not joint:
        out("  no J file: joint relay-BP was not run on these shots, so rule f is TCBP-JOBS's answer on every shot")
    out(f"  choice shots (P1, P2, T not unanimous): {int(choice.sum())} ({choice.mean():.2%}); the selector chose among the pool "
        f"{', '.join(model['pool'])}: {len(choices.shot)} distinct answers on them")
    table = [("selector", "the learned selector", failed)]
    names = {"T": "T: TCBP-JOBS alone", "f": "f: J where it converged, else T (the frozen ensemble rule)",
             "gw": "gw: the vote of J, T, P1, P2, R; a tie: that order", "ac": "ac: part 1's best rule without learning",
             "bw": "bw: the lowest weight of P1, P2, T, R", "R": "R: relay-BP alone", "P1": "P1: PriorTilt, seed 1",
             "P2": "P2: PriorTilt, seed 1001", "J": "J: joint relay-BP (on its listed shots only)"}
    for key in SHOWN:
        if key in rows and rows[key].get("whole", True) and (key != "J"):
            table.append((key, names[key], rows[key]["failed"]))
    oracles = [key for key in rows if key.startswith("oracle_") and (joint or not key.endswith("_J"))]
    for key in oracles:
        table.append((key, "ORACLE, a ceiling (needs the truth): a perfect choice among " + ", ".join(key.split("_")[1:]), rows[key]["failed"]))
    out(f"  {'row':66s} {'failures':>8s}  {'rate':>7s}  95% Wilson interval")
    for key, name, mask in table:
        low, high = wilson_interval(int(mask.sum()), len(mask))
        out(f"  {name:66s} {int(mask.sum()):8d}  {mask.mean():7.3%}  [{low:.3%}, {high:.3%}]")
    out("  paired with the selector on the same shots: both fail, only the selector fails, only the other fails, exact two-sided p")
    paired = {}
    for key in COMPARED:
        if key in rows:
            both, only_sel, only_other, p_value = paired_test(failed, rows[key]["failed"])
            paired[key] = {"both": both, "only_selector": only_sel, "only_other": only_other, "p": p_value}
            out(f"    against {key:3s}: {both:4d} {only_sel:4d} {only_other:4d}   p = {p_value:.3g}")
    times = time_of_row(view, model, choice)
    for line in times["lines"]:
        out("  " + line)
    out(f"  the selector's own work (features and scores of {int(choice.sum())} choice shots) took {spent * 1e3:.1f} ms on this machine in all")
    result = {"basis": info.basis, "p": info.p, "data_name": info.data_name, "data_sha256": info.data_sha256, "shots": int(len(info.truth)),
              "member_files": info.paths, "joint": joint, "choice_shots": int(choice.sum()),
              "failures": {key: int(mask.sum()) for key, _, mask in table}, "paired_with_selector": paired,
              "time": {key: value for key, value in times.items() if key != "lines"}}
    arrays = {"failed": failed, "predicted": answer, "choice": choice, "data_sha256": info.data_sha256, "basis": info.basis, "p": info.p}
    arrays |= {f"failed_{key}": mask for key, _, mask in table}
    return result, arrays


def time_of_row(view, model, choice):
    """The selector's row time per shot as the sum of its members' decode calls (combined offline)."""
    lines, parts, unmeasured = ["time per shot, combined offline (each member's decode calls, timed on its own machine):"], {}, []
    for name in ("P1", "P2", "T", "R", "J"):
        if name not in view.members or name not in model["pool"]:
            continue
        member = view.members[name]
        where = member.covers if name in F.ALWAYS_ON else member.covers & choice
        if not where.any():
            continue
        seconds = member.seconds[where]
        per_shot = float(np.nansum(seconds) / view.shots) if not np.isnan(seconds).any() else None
        run_on = "every shot" if name in F.ALWAYS_ON else f"{int(where.sum())} choice shots"
        if per_shot is None:
            unmeasured.append(name)
            lines.append(f"  {name:3s} on {run_on}: no time per shot in its file ({member.times_from})")
        else:
            parts[name] = per_shot
            lines.append(f"  {name:3s} on {run_on}: {per_shot * 1e3:8.1f} ms per shot of the file on average ({member.times_from})")
    total = None if unmeasured else sum(parts.values())
    lines.append("  the row: " + (f"{total * 1e3:.1f} ms per shot, combined offline" if total is not None else
                                   f"unmeasured (no time per shot for {', '.join(unmeasured)}); the known parts sum to "
                                   f"{sum(parts.values()) * 1e3:.1f} ms per shot"))
    return {"lines": lines, "ms_per_shot_by_member": {k: v * 1e3 for k, v in parts.items()},
            "ms_per_shot": None if total is None else total * 1e3, "unmeasured_members": unmeasured}


def pairs(x, z, out=print):
    """The kit's pair score from the two bases' per-shot arrays."""
    from decoders import per_cycle_rate

    if str(x["basis"]) != "X" or str(z["basis"]) != "Z" or float(x["p"]) != float(z["p"]) or len(x["failed"]) != len(z["failed"]):
        raise SystemExit("--x must be X memory and --z Z memory at the same p, with as many shots")
    keys = [key[len("failed_"):] for key in x if key.startswith("failed_") and key in z]
    out(f"\npairs at p = {float(x['p']):g}: shot k of X with shot k of Z, {len(x['failed'])} pairs; a pair fails when either shot fails")
    out(f"  {'row':34s} {'failures [X Z]':>15s} {'failed pairs':>13s}  rate per cycle (the kit's)")
    table = {}
    for key in keys:
        pair = x[f"failed_{key}"] | z[f"failed_{key}"]
        rate, error = per_cycle_rate(pair, 12)
        table[key] = {"failures_X": int(x[f"failed_{key}"].sum()), "failures_Z": int(z[f"failed_{key}"].sum()),
                      "failed_pairs": int(pair.sum()), "per_cycle": float(rate), "error": float(error), "pair_failed": pair}
        name = key + ("  (a ceiling)" if key.startswith("oracle") else "")
        out(f"  {name:34s} {'[%d %d]' % (table[key]['failures_X'], table[key]['failures_Z']):>15s} {int(pair.sum()):13d}  {rate:.2e} +- {error:.1e}")
    out("  paired on pairs with the selector: both fail, only the selector fails, only the other fails, exact two-sided p")
    paired = {}
    for key in COMPARED:
        if key in table:
            both, only_sel, only_other, p_value = paired_test(table["selector"]["pair_failed"], table[key]["pair_failed"])
            paired[key] = {"both": both, "only_selector": only_sel, "only_other": only_other, "p": p_value}
            out(f"    against {key:3s}: {both:4d} {only_sel:4d} {only_other:4d}   p = {p_value:.3g}")
    if "T" in paired:
        win = table["selector"]["failed_pairs"] < table["T"]["failed_pairs"] and paired["T"]["p"] < 0.05
        out(f"  fewer failed pairs than TCBP-JOBS alone, paired p < 0.05: {'YES' if win else 'NO'}"
            f" ({table['selector']['failed_pairs']} against {table['T']['failed_pairs']}, p = {paired['T']['p']:.3g})")
    return {key: {k: v for k, v in value.items() if k != "pair_failed"} for key, value in table.items()}, paired


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="the frozen model file (.json) of selector.fit")
    parser.add_argument("--expect-sha256", help="stop unless the model file has this sha256")
    parser.add_argument("--x", nargs="+", help="the member files of the X data file")
    parser.add_argument("--z", nargs="+", help="the member files of the Z data file")
    parser.add_argument("--label", default="", help="what these shots are, for the printout (validation, test)")
    parser.add_argument("--save", help="prefix of the output files")
    args = parser.parse_args(argv)
    if not args.x and not args.z:
        raise SystemExit("give --x and/or --z")
    sha = M.sha256_of(args.model)
    if args.expect_sha256 and sha != args.expect_sha256:
        raise SystemExit(f"the model file {args.model} has sha256 {sha}, not {args.expect_sha256}")
    model = M.load(args.model)
    print(f"selector model {args.model}  sha256 {sha}")
    print(f"  fitted on: " + "; ".join(f"{r['data_name']} p = {r['p']} ({r['data_sha256'][:12]}...)" for r in model["trained_on"])
          + f"; lambda {model['lambda']:g}; pool {', '.join(model['pool'])}")
    trained = {r["data_sha256"] for r in model["trained_on"]}
    results, arrays = {"model": args.model, "model_sha256": sha, "label": args.label}, {}
    for basis, paths in (("X", args.x), ("Z", args.z)):
        if not paths:
            continue
        result, per_shot = one_basis(paths, model, args.label)
        if result["data_sha256"] in trained:
            print(f"  WARNING: the model was fitted on these very shots ({result['data_name']}): in-sample, not a result")
            result["in_sample"] = True
        if result["basis"] != basis:
            raise SystemExit(f"--{basis.lower()} holds {result['basis']} memory")
        results[basis], arrays[basis] = result, per_shot
    if "X" in arrays and "Z" in arrays:
        results["pairs"], results["pairs_paired"] = pairs(arrays["X"], arrays["Z"])
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        for basis, per_shot in arrays.items():
            np.savez_compressed(f"{args.save}_{basis}.npz", **per_shot)
        Path(f"{args.save}.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
        print(f"\nwrote {args.save}.json and the per-shot {', '.join(f'{args.save}_{b}.npz' for b in arrays)}")
    return results


if __name__ == "__main__":
    main()
