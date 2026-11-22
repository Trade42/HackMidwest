"""The hierarchy's rules, scored offline on the candidates of saved shots.

Run from challenge-1-ml-decoder-kit/, always as a module:

  python -m hierarchy.select --candidates out/validation_X.chunks.npz
  python -m hierarchy.select --candidates out/validation_X.chunks.npz --logits net/validation_X.npz

--candidates takes one or more files of hierarchy.candidates (or the laptop's selector_*.npz,
which have the same arrays); several files are scored as one set, in the order given.
--logits takes, per candidates file and in the same order, the .npz that
`python -m neural.evaluate_stim --save` wrote for the same shots. If the network was trained
with `--logicals light`, its logits belong to other logical operators than the kit's; the
candidates' bits are then turned into those operators first (rules.bits_for_network).

The rules, with R = relay-BP, P1 and P2 = PriorTilt under two seeds:

  R, P1, P2 alone
  weight     P1; where P2 differs, the lowest weight of R, P1, P2       (the plugin's "weight" rule)
  weight_converged   P1; where P2 differs, the lowest weight among the converged   (the plugin's "converged" rule)
  converged  the lowest weight among the converged of R, P1, P2          (runs all three on every shot)
  network    P1; where P2 differs, the candidate the network finds most likely   (the plugin's "network" rule)
  network C  as network, with the score "network minus weight / C"
  perfect    a perfect choice, which needs the truth: the floor for any rule over these candidates

For each rule: the failures, the rate with a 95% Wilson interval, how often it chose a right
candidate when one existed, and the mean time per shot of the stages it runs. The truth is
used for scoring only.

A constant C is fitted, so it is an input here. Either give it (--constant), or let the tool
choose it on other shots (--fit-candidates, --fit-logits, --constants) and score it on these.
If the fit shots and the scored shots are the same file, the output says so.
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hierarchy.rules import (ORDER, P1, agree, bits_for_network, lowest_weight, most_likely, paired_test, tie_break,
                             wilson_interval)

R = ORDER.index("R")
FIELDS = ("predicted", "converged", "iterations", "weight", "seconds")


# ---------- reading ----------

def load_candidates(paths):
    """The candidates of one or more files as one set: arrays [candidates, shots, ...] and a description per file."""
    parts, sources = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as file:
            shots = len(file["truth"])
            part = {field: np.stack([file[f"{field}_{name}"] for name in ORDER]) for field in FIELDS}
            part["truth"] = file["truth"].astype(np.uint8)
            part["ran"] = np.ones((len(ORDER), shots), bool)
            if "ran_R" in file.files:          # in session mode relay-BP runs only where P1 and P2 differ
                part["ran"][R] = file["ran_R"]
            sources.append({"path": str(path), "shots": shots, "data_sha256": str(file["data_sha256"]),
                            "basis": str(file["basis"]) if "basis" in file.files else None, "about": _about(file)})
        parts.append(part)
    pool = {key: np.concatenate([part[key] for part in parts], axis=0 if key == "truth" else 1) for key in parts[0]}
    pool["weight"] = np.where(pool["ran"], pool["weight"], np.inf)
    return pool, sources


def _about(file):
    if "mode" not in file.files:
        return "no record of the run in the file (the laptop's format)"
    machine = json.loads(str(file["machine"]))
    return (f"{file['basis']} memory, p = {float(file['p']):g}, {file['seed_scheme']}; times: "
            f"{'/'.join(str(n) for n in np.unique(file['processes']))} process(es) side by side on {machine['processor']}")


@dataclass
class NetworkLogits:
    """What the network said about the scored shots."""
    logits: np.ndarray        # [shots, 12], of the network's own logical operators
    candidates: np.ndarray    # [candidates, shots, 12]: the candidates' bits in those operators
    ms: float = None          # the network's time per call, if known
    where: str = "not given"  # where that time comes from


def load_logits(paths, pool, sources):
    """The network's logits for the same shots, the candidates' bits in the network's operators, and its time per call."""
    if len(paths) != len(sources):
        raise SystemExit("give one logits file per candidates file, in the same order")
    logits_parts, candidate_parts, times, start = [], [], [], 0
    for path, source in zip(paths, sources, strict=True):
        with np.load(path, allow_pickle=False) as file:
            logits, observables, answer = file["logits"].astype(np.float64), file["observables"], file["predictions"]
            operators = str(file["logits_are_for"]).split()[0] if "logits_are_for" in file.files else "kit"
        rows = slice(start, start + source["shots"])
        if observables.shape != pool["truth"][rows].shape or not np.array_equal(observables, pool["truth"][rows]):
            raise SystemExit(f"{path} does not hold the shots of {source['path']}: the observables differ")
        logits_parts.append(logits)
        candidate_parts.append(bits_for_network(pool["predicted"][:, rows], logits > 0, answer,
                                                _model_from_kit(source["basis"], operators, source["path"])))
        start += source["shots"]
        numbers = Path(path).with_suffix(".json")
        if numbers.is_file():
            record = json.loads(numbers.read_text(encoding="utf-8"))
            times.append((record["latency_ms_mean"], record["latency_shots"], record.get("device", "unknown device")))
    network = NetworkLogits(np.concatenate(logits_parts), np.concatenate(candidate_parts, axis=1))
    if len(times) == len(paths):
        network.ms = sum(ms * shots for ms, shots, _ in times) / sum(shots for _, shots, _ in times)
        network.where = f"batch-one mean on {', '.join(sorted({device for _, _, device in times}))}, from the .json beside the logits"
    return network


def _model_from_kit(basis, operators, candidates_path):
    """The matrix that turns kit bits into the network's bits; None for a network trained on the kit's observables."""
    if operators == "kit":
        return None
    if basis is None:
        raise SystemExit(f"the logits belong to the {operators!r} logical operators, and {candidates_path} does not say its basis")
    from neural.graph import build_graph, gf2_inverse    # the operators are defined in neural/graph.py

    return gf2_inverse(build_graph("gross", basis, None, operators).kit_from_model)


# ---------- scoring ----------

@dataclass
class Row:
    key: str
    name: str
    failed: np.ndarray = None      # [shots] bool. None: not measured on these files
    ms: np.ndarray = None          # [shots] time of the stages the rule runs, in ms
    predicted: np.ndarray = None   # [shots, 12]
    note: str = ""


def score(pool, network=None, constants=()):
    """One Row per rule. network: a NetworkLogits. constants: (value, where it comes from) pairs for the "network minus weight / C" rows."""
    predicted, weight, ran, truth = pool["predicted"], pool["weight"], pool["ran"], pool["truth"]
    shots = np.arange(len(truth))
    ms = pool["seconds"] * 1e3
    differ = ~agree(predicted)
    right = (predicted == truth[None]).all(axis=-1) & ran
    relay_everywhere = bool(ran[R].all())
    not_measured = f"not measured: relay-BP ran on {int(ran[R].sum())} of {len(truth)} shots"
    hierarchy_ms = ms[P1] + ms[ORDER.index("P2")] + np.where(differ, ms[R], 0.0)

    def chosen(key, name, choice, time, note=""):
        answer = predicted[choice, shots]
        return Row(key, name, (answer != truth).any(axis=1), time, answer, note)

    rows = []
    if relay_everywhere:
        rows.append(chosen("R", "R: relay-BP, kit settings", np.full(len(truth), R), ms[R]))
    else:
        rows.append(Row("R", "R: relay-BP, kit settings", note=not_measured))
    rows.append(chosen("P1", "P1: PriorTilt, first seed", np.full(len(truth), P1), ms[P1]))
    rows.append(chosen("P2", "P2: PriorTilt, second seed", np.full(len(truth), ORDER.index("P2")), ms[ORDER.index("P2")]))
    rows.append(chosen("weight", "P1; where P2 differs, the lowest weight of R, P1, P2",
                       tie_break(predicted, lowest_weight(weight, ran)), hierarchy_ms))
    rows.append(chosen("weight_converged", "P1; where P2 differs, the lowest weight among the converged",
                       tie_break(predicted, lowest_weight(weight, pool["converged"] & ran)), hierarchy_ms))
    if relay_everywhere:
        rows.append(chosen("converged", "the lowest weight among the converged of R, P1, P2",
                           lowest_weight(weight, pool["converged"] & ran), ms.sum(axis=0)))
    else:
        rows.append(Row("converged", "the lowest weight among the converged of R, P1, P2", note=not_measured))
    if network is not None:
        network_time = hierarchy_ms + np.where(differ, network.ms or 0.0, 0.0)
        note = "" if network.ms is not None else "time without the network's"
        rows.append(chosen("network", "P1; where P2 differs, the network's most likely",
                           tie_break(predicted, most_likely(network.candidates, network.logits, allowed=ran)), network_time, note))
        for constant, origin in constants:
            rows.append(chosen(f"network_C{constant:g}", f"  the same with network minus weight / {constant:g}",
                               tie_break(predicted, most_likely(network.candidates, network.logits, weight, constant, allowed=ran)),
                               network_time, "; ".join(filter(None, [f"C {origin}", note]))))
    if relay_everywhere:
        rows.append(Row("perfect", "a perfect choice among R, P1, P2 (needs the truth)", ~right.any(axis=0)))
    else:
        rows.append(Row("perfect", "a perfect choice among R, P1, P2 (needs the truth)", note=not_measured))
    rows.append(Row("perfect_tie_break", "P1; where P2 differs, a perfect choice (needs the truth)",
                    np.where(differ, ~right.any(axis=0), ~right[P1])))
    return rows


def facts(pool):
    """Counts that describe the candidates, whatever the rule."""
    predicted, ran, truth = pool["predicted"], pool["ran"], pool["truth"]
    right = (predicted == truth[None]).all(axis=-1) & ran
    differ = ~agree(predicted)
    same_as_p1 = (predicted == predicted[P1][None]).all(axis=-1) | ~ran
    not_all_equal = ~same_as_p1.all(axis=0)
    none_converged = ~(pool["converged"] & ran).any(axis=0)
    return {"shots": len(truth), "p1_p2_differ": int(differ.sum()), "relay_ran": int(ran[R].sum()),
            "not_all_equal": int(not_all_equal.sum()), "pickable": not_all_equal & right.any(axis=0),
            "unconverged": {name: int((~pool["converged"][index] & ran[index]).sum()) for index, name in enumerate(ORDER)},
            "none_converged": none_converged}


def fit_constant(pool, network, constants):
    """The constant with the fewest failures of the "network minus weight / C" rule on these shots.

    Equal failures: the largest constant, the one that leans least on the weight. Returns
    (constant, {constant: failures}).
    """
    failures = {}
    for constant in constants:
        choice = tie_break(pool["predicted"], most_likely(network.candidates, network.logits, pool["weight"], constant,
                                                          allowed=pool["ran"]))
        answer = pool["predicted"][choice, np.arange(len(choice))]
        failures[constant] = int((answer != pool["truth"]).any(axis=1).sum())
    return max(constants, key=lambda constant: (-failures[constant], constant)), failures


# ---------- printing ----------

def report(pool, sources, rows, network_note=None, out=print):
    """Print the description of the candidates, the table of the rules and the paired comparison with relay-BP."""
    count = facts(pool)
    shots, pickable = count["shots"], count["pickable"]
    out(f"candidates: {shots} shots from {len(sources)} file(s)")
    for source in sources:
        out(f"  {source['path']}: {source['shots']} shots of the data file with sha256 {source['data_sha256']}")
        out(f"    {source['about']}")
    if network_note:
        out(f"network: {network_note}")
    out(f"P1 and P2 differ on {count['p1_p2_differ']} shots ({count['p1_p2_differ'] / shots:.3%}): only there the hierarchy "
        "runs relay-BP and asks the rule")
    out(f"relay-BP ran on {count['relay_ran']} shots; the candidates that ran are not all equal on {count['not_all_equal']} shots "
        f"({count['not_all_equal'] / shots:.3%}), and on {int(pickable.sum())} of those one of them is right")
    out("unconverged: " + ", ".join(f"{name} {number}" for name, number in count["unconverged"].items())
        + f"; no candidate converged on {int(count['none_converged'].sum())} shots")
    out("")
    out(f"{'rule':56s} {'failures':>8s}  {'rate':>7s}  {'95% Wilson interval':21s}  {'right pick when one existed':27s}  {'ms/shot':>8s}")
    for row in rows:
        if row.failed is None:
            out(f"{row.name:56s} {row.note}")
            continue
        failures = int(row.failed.sum())
        low, high = wilson_interval(failures, shots)
        picks = f"{int((~row.failed & pickable).sum())} of {int(pickable.sum())}"
        time = f"{row.ms.mean():8.1f}" if row.ms is not None else f"{'':8s}"
        out(f"{row.name:56s} {failures:8d}  {failures / shots:7.3%}  [{low:.3%}, {high:.3%}]{'':5s}  {picks:27s}  {time}"
            + (f"  ({row.note})" if row.note else ""))
    out('"right pick when one existed": among the shots where the candidates that ran are not all equal and one is right.')
    out("ms/shot: the mean over all shots of the decode calls the rule makes (and the network's batch-one time where it is asked).")
    reference = rows[0]
    if reference.failed is not None:
        out("")
        out("against relay-BP on the same shots: both fail, the rule only, relay-BP only, exact two-sided p (binomial on the discordant shots)")
        for row in rows[1:]:
            if row.failed is not None and row.predicted is not None:
                both, only_rule, only_relay, p_value = paired_test(row.failed, reference.failed)
                out(f"{row.name:56s} {both:5d} {only_rule:5d} {only_relay:5d}   p = {p_value:.3g}")
    stuck = count["none_converged"]
    by_key = {row.key: row for row in rows}
    if "network" in by_key:
        out("")
        for key, label in (("weight", "the weight rule"), ("weight_converged", "the converged-weight rule")):
            both, only_network, only_other, p_value = paired_test(by_key["network"].failed, by_key[key].failed)
            out(f"the network rule against {label} on the same shots: both fail {both}, the network rule only {only_network}, "
                f"{label} only {only_other}, exact two-sided p = {p_value:.3g}")
    out("")
    out(f"shots where no candidate converged: {int(stuck.sum())}; the weight rule fails on {int(by_key['weight'].failed[stuck].sum())} of them"
        + (f", the network rule on {int(by_key['network'].failed[stuck].sum())}" if "network" in by_key else ""))


def numbers(pool, sources, rows):
    """The report as a dict for JSON."""
    count = facts(pool)
    table = []
    for row in rows:
        entry = {"key": row.key, "rule": row.name.strip(), "note": row.note}
        if row.failed is not None:
            failures = int(row.failed.sum())
            entry |= {"failures": failures, "rate": failures / count["shots"],
                      "wilson_95": list(wilson_interval(failures, count["shots"])),
                      "right_picks": int((~row.failed & count["pickable"]).sum()),
                      "ms_per_shot": None if row.ms is None else float(row.ms.mean())}
        table.append(entry)
    return {"sources": sources, "shots": count["shots"], "p1_p2_differ": count["p1_p2_differ"],
            "relay_ran": count["relay_ran"], "not_all_equal": count["not_all_equal"],
            "right_pick_possible": int(count["pickable"].sum()), "unconverged": count["unconverged"],
            "none_converged": int(count["none_converged"].sum()), "rules": table}


def check_evaluator(path, rows, key, sources, out=print):
    """Compare a rule's answers with the predictions the team's evaluator saved for the plugin on the same shots."""
    basis = sources[0]["basis"]
    if len(sources) != 1 or basis is None:
        raise SystemExit("--evaluator-shots compares one candidates file of hierarchy.candidates with one basis of one evaluator run")
    row = {row.key: row for row in rows}.get({"converged": "weight_converged"}.get(key, key))
    if row is None or row.predicted is None:
        raise SystemExit(f"the rule {key!r} was not scored on these files")
    with np.load(path, allow_pickle=False) as file:
        predictions, failed = file[f"predictions_{basis}"], file[f"failed_{basis}"]
    if predictions.shape != row.predicted.shape:
        raise SystemExit(f"{path} holds {len(predictions)} {basis} shots, the candidates {len(row.predicted)}")
    different = np.flatnonzero((predictions != row.predicted).any(axis=1))
    out("")
    out(f"the evaluator's {basis} predictions in {path}: {int(failed.sum())} failures; the rule {key!r} here: {int(row.failed.sum())} failures")
    out(f"  shots where the two answers differ: {len(different)}" + (f" (shots {different[:20].tolist()})" if len(different) else ""))
    return len(different)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", nargs="+", required=True, help="files of hierarchy.candidates, scored as one set")
    parser.add_argument("--logits", nargs="+", help="per candidates file: the .npz of neural.evaluate_stim --save on the same shots")
    parser.add_argument("--network-ms", type=float, help="the network's time per call, where the .json beside the logits is missing")
    parser.add_argument("--constant", type=float, action="append", default=[],
                        help="score 'network minus weight / C' with this C; it must come from other shots than these")
    parser.add_argument("--fit-candidates", nargs="+", help="candidates of OTHER shots, to choose C on")
    parser.add_argument("--fit-logits", nargs="+", help="the logits of the fit shots")
    parser.add_argument("--constants", type=float, nargs="+", help="the values of C to try on the fit shots")
    parser.add_argument("--evaluator-shots", help="a *_shots.npz of evaluation.evaluate for the plugin on the same shots, to compare with")
    parser.add_argument("--evaluator-rule", default=None, help="the rule the plugin used: network, weight or converged (default: network if --logits)")
    parser.add_argument("--save", help="write the numbers of the report to this .json file")
    args = parser.parse_args(argv)

    pool, sources = load_candidates(args.candidates)
    network, network_note = None, None
    if args.logits:
        network = load_logits(args.logits, pool, sources)
        if args.network_ms is not None:
            network.ms, network.where = args.network_ms, "given with --network-ms"
        network_note = f"logits from {', '.join(args.logits)}; time per call " + (
            f"{network.ms:.1f} ms ({network.where})" if network.ms is not None else network.where)
    elif args.constant or args.fit_candidates or args.constants:
        raise SystemExit("a constant belongs to the network rule: give --logits")

    constants, warnings = [(constant, "given as an input") for constant in args.constant], []
    if args.fit_candidates or args.fit_logits or args.constants:
        if not (args.fit_candidates and args.fit_logits and args.constants):
            raise SystemExit("fitting C needs --fit-candidates, --fit-logits and --constants together")
        fit_pool, fit_sources = load_candidates(args.fit_candidates)
        best, failures = fit_constant(fit_pool, load_logits(args.fit_logits, fit_pool, fit_sources), args.constants)
        fit_hashes = [source["data_sha256"] for source in fit_sources]
        print(f"C fitted on {len(fit_pool['truth'])} shots (data sha256 {', '.join(fit_hashes)}): failures "
              + ", ".join(f"C = {constant:g}: {number}" for constant, number in failures.items()) + f"; chosen C = {best:g}")
        origin = "fitted on other shots"
        if set(fit_hashes) & {source["data_sha256"] for source in sources}:
            origin = "FITTED ON THESE SAME SHOTS"
            warnings.append("WARNING: C was fitted on the shots it is scored on. Its row is not a result: fit on one file and score on another.")
        constants.append((best, origin))
    if args.constant:
        warnings.append("A constant given with --constant counts as a result only if it was chosen on other shots than these.")

    rows = score(pool, network, constants)
    report(pool, sources, rows, network_note)
    for line in warnings:
        print(line)
    if args.evaluator_shots:
        check_evaluator(args.evaluator_shots, rows, args.evaluator_rule or ("network" if network is not None else "weight"), sources)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(numbers(pool, sources, rows), indent=2) + "\n", encoding="utf-8")
    return rows


if __name__ == "__main__":
    main()
