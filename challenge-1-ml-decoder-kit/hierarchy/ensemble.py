"""The ensemble: several decoders' answers to the same saved shots, combined offline by one rule.

Run from challenge-1-ml-decoder-kit/:

  python -m hierarchy.ensemble --members cand/validation_X.chunks.npz tcbp/validation_X.tcbp.npz \\
      joint/validation_X.joint.npz --logits net/validation_X.npz --save out/validation_X.json

A member is one decoder's 12 predicted bits per shot, read from the files of this folder's
tools by the name in the file:

  P1, P2   PriorTilt under two seeds       hierarchy.candidates (or the laptop's selector files)
  R        relay-BP, the kit's settings    hierarchy.candidates
  T        TCBP-JOBS                       hierarchy.tcbp_member
  J        joint relay-BP                  hierarchy.joint_member: only the shots it was given

--logits is the file `python -m neural.evaluate_stim --save` wrote for the same shots. Every
file must come from the same data file (sha256), or the tool stops.

The rules. P1, P2 and T are "always on": they decode every shot. R, J and the network are
the later stage: a rule runs them only on the shots it does not accept at once.

  a    P1 where P1 = P2; else the network picks among R, P1, P2            (part 1's rule)
  aw   the same with the lowest weight in place of the network             (part 1's rule without learning)
  b    P1 where P1 = P2 = T; else the network picks among P1, P2, T, R
  bw   the same with the lowest weight in place of the network             (no learning)
  c    as b, with J among the candidates on the shots where J exists
  d    the majority of P1, P2, T; if all three differ, the network picks among P1, P2, T, R
  e    as d, with J among the candidates on the shots where J exists
  f    P1 where P1 = P2 = T; else J's answer where J converged, and T's answer where it did not   (no learning)
  g    P1 where P1 = P2 = T; else the answer that most of J, T, P1, P2, R give; a tie goes to the network
  gw   the same, with a tie going to the first of J, T, P1, P2, R that is in it                    (no learning)

"The network picks" is part 1's score: the candidate whose 12 bits the network finds most
likely (rules.network_score), among the candidates that decoded the shot. No rule has a fitted
number. The truth is used to score, and for the rows marked ORACLE (a perfect choice among
some members: the floor of any rule over them).

For every rule: the failures with a 95% Wilson interval, on how many shots the later stage
runs, the mean time per shot as the sum of the stages it runs (if the files hold times),
and the paired comparison with relay-BP alone and with TCBP-JOBS alone on the same shots.
These rows are combined offline from per-shot files. They are not one decoder timed in one
process, as part 1's row from the team's evaluator is.

Choosing a rule: score the training file with --save, then score the validation file with
--chosen-on <the training .json>. The rule with the fewest failures on the training shots
(ties: the earlier rule in the list above) is marked in the validation table. A .json of the
same data file is refused: no rule is chosen on the shots it is reported on.

--write-selection <file> lists the shots where the always-on members are not unanimous: the
shots for hierarchy.joint_member.
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hierarchy.rules import bits_for_network, network_score, paired_test, wilson_interval
from hierarchy.select import _model_from_kit

ALWAYS_ON = ("P1", "P2", "T")
# key, description, the members that must agree to accept, a majority vote first?, the candidates of the later stage, the scorer
RULES = [
    ("a", "P1 where P1 = P2; else the network among R, P1, P2 (part 1)", ("P1", "P2"), False, ("R", "P1", "P2"), "network"),
    ("aw", "P1 where P1 = P2; else the lowest weight of R, P1, P2", ("P1", "P2"), False, ("R", "P1", "P2"), "weight"),
    ("b", "P1 where P1 = P2 = T; else the network among P1, P2, T, R", ALWAYS_ON, False, ("P1", "P2", "T", "R"), "network"),
    ("bw", "P1 where P1 = P2 = T; else the lowest weight of P1, P2, T, R", ALWAYS_ON, False, ("P1", "P2", "T", "R"), "weight"),
    ("c", "as b, with J among the candidates where it exists", ALWAYS_ON, False, ("P1", "P2", "T", "R", "J"), "network"),
    ("d", "majority of P1, P2, T; all differ: the network among P1, P2, T, R", ALWAYS_ON, True, ("P1", "P2", "T", "R"), "network"),
    ("e", "as d, with J among the candidates where it exists", ALWAYS_ON, True, ("P1", "P2", "T", "R", "J"), "network"),
    ("f", "P1 where P1 = P2 = T; else J where it converged, else T", ALWAYS_ON, False, ("J", "T"), "joint"),
    ("g", "P1 where P1 = P2 = T; else most of J, T, P1, P2, R; a tie: the network", ALWAYS_ON, False, ("J", "T", "P1", "P2", "R"), "votes+network"),
    ("gw", "P1 where P1 = P2 = T; else most of J, T, P1, P2, R; a tie: that order", ALWAYS_ON, False, ("J", "T", "P1", "P2", "R"), "votes"),
]
ORACLES = [("R", "P1", "P2"), ("P1", "P2", "T"), ("P1", "P2", "T", "R"), ("P1", "P2", "T", "R", "J")]


@dataclass
class Member:
    name: str
    predicted: np.ndarray   # [shots, 12]; zeros where the member did not decode the shot
    covers: np.ndarray      # [shots] bool: the member decoded this shot
    seconds: np.ndarray     # [shots]; NaN where the file holds no time
    weight: np.ndarray      # [shots]; NaN where the file holds no weight
    converged: np.ndarray   # [shots] bool: the correction reproduces the syndrome (True where the file does not say)
    source: str
    times_from: str


@dataclass
class Shots:
    truth: np.ndarray       # [shots, 12]
    data_sha256: str
    basis: str = None
    p: float = None


@dataclass
class Network:
    logits: np.ndarray      # [shots, 12], of the network's own logical operators
    answer: np.ndarray      # [shots, 12]: the kit's bits of the network's own answer
    model_from_kit: np.ndarray = None
    ms: float = None
    where: str = "not given"


# ---------- reading ----------

def load_members(paths):
    """Every member of the given files, as {name: Member}, and what the files say about the shots."""
    files = []
    for path in paths:
        with np.load(path, allow_pickle=False) as file:
            files.append((str(path), {key: file[key] for key in file.files}))
    hashes = {str(arrays["data_sha256"]) for _, arrays in files}
    if len(hashes) != 1:
        raise SystemExit("the member files were made from different data files (sha256): "
                         + "; ".join(f"{path}: {arrays['data_sha256']}" for path, arrays in files))
    whole = [arrays for _, arrays in files if "shot" not in arrays]
    if not whole:
        raise SystemExit("no member file holds every shot of the data file")
    shots = Shots(whole[0]["truth"].astype(np.uint8), hashes.pop())
    count, members = len(shots.truth), {}
    for path, arrays in files:
        rows = arrays["shot"].astype(int) if "shot" in arrays else np.arange(len(arrays["truth"]))
        size = int(arrays["shots_in_file"]) if "shots_in_file" in arrays else len(arrays["truth"])
        if size != count or (len(rows) and rows.max() >= count) or not np.array_equal(arrays["truth"], shots.truth[rows]):
            raise SystemExit(f"{path} does not line up with the other member files: other shots or another truth")
        if "basis" in arrays:
            shots.basis, shots.p = str(arrays["basis"]), float(arrays["p"])
        times_from = "no record in the file"
        if "machine" in arrays:
            processes = "/".join(str(n) for n in np.unique(arrays["processes"])) + " process(es) on " if "processes" in arrays else ""
            times_from = processes + json.loads(str(arrays["machine"]))["processor"]
        for name in [key[len("predicted_"):] for key in arrays if key.startswith("predicted_")]:
            if name in members:
                raise SystemExit(f"member {name} is in two files: {members[name].source} and {path}")
            covers = np.zeros(count, bool)
            covers[rows] = arrays[f"ran_{name}"] if f"ran_{name}" in arrays else True
            member = Member(name, np.zeros((count, 12), np.uint8), covers, np.full(count, np.nan), np.full(count, np.nan),
                            np.ones(count, bool), path, times_from)
            member.predicted[rows] = arrays[f"predicted_{name}"]
            for field in ("seconds", "weight", "converged"):
                if f"{field}_{name}" in arrays:
                    getattr(member, field)[rows] = arrays[f"{field}_{name}"]
            members[name] = member
    return members, shots


def load_network(path, shots, network_ms=None):
    """The network's logits for the same shots."""
    with np.load(path, allow_pickle=False) as file:
        logits, observables, answer = file["logits"].astype(np.float64), file["observables"], file["predictions"]
        operators = str(file["logits_are_for"]).split()[0] if "logits_are_for" in file.files else "kit"
    if observables.shape != shots.truth.shape or not np.array_equal(observables, shots.truth):
        raise SystemExit(f"{path} does not hold the shots of the member files: the observables differ")
    network = Network(logits, answer, _model_from_kit(shots.basis, operators, "the member files"))
    numbers = Path(path).with_suffix(".json")
    if network_ms is not None:
        network.ms, network.where = network_ms, "given with --network-ms"
    elif numbers.is_file():
        record = json.loads(numbers.read_text(encoding="utf-8"))
        network.ms, network.where = record["latency_ms_mean"], f"batch-one mean on {record.get('device', 'an unknown device')}"
    return network


# ---------- the rules ----------

def same(first, second):
    return first.covers & second.covers & (first.predicted == second.predicted).all(axis=1)


def later_stage_pick(pool, scorer, network):
    """Index [shots] into pool of the candidate the scorer prefers among those that decoded the shot (the first on a tie)."""
    predicted = np.stack([member.predicted for member in pool])
    covers = np.stack([member.covers for member in pool])
    order = -np.arange(len(pool), dtype=float)[:, None] * np.ones(covers.shape[1])      # earlier in the pool is better
    by_network = None
    if "network" in scorer:
        own = bits_for_network(predicted, network.logits > 0, network.answer, network.model_from_kit)
        by_network = network_score(own, network.logits)
    if scorer == "network":
        score = by_network
    elif scorer == "weight":
        score = -np.stack([member.weight for member in pool])
    elif scorer == "joint":               # J where it decoded the shot and converged; otherwise T
        score = np.stack([member.converged * 2.0 if member.name == "J" else np.full(covers.shape[1], float(member.name == "T"))
                          for member in pool])
    else:                                 # votes: how many candidates that decoded the shot give this same answer
        votes = sum((covers[other] & (predicted == predicted[other][None]).all(axis=-1)) for other in range(len(pool))).astype(float)
        score = votes * 1e6 + (by_network if scorer == "votes+network" else order)
    return np.where(covers, score, -np.inf).argmax(axis=0)


def apply(rule, members, network):
    """One rule on every shot: (answer [shots, 12], later [shots] bool, time [shots] in seconds or None, note)."""
    _, _, accept, majority, pool_names, scorer = rule
    missing = [name for name in accept if name not in members] + ([] if any(name in members for name in pool_names) else ["a candidate"])
    if "network" in scorer and network is None:
        return None, None, None, "not measured: no logits"
    if missing or ("J" in pool_names and "J" not in members):
        return None, None, None, "not measured: no file for " + ", ".join(missing or ["J"])
    first, others = members[accept[0]], [members[name] for name in accept[1:]]
    if not all(member.covers.all() for member in (first, *others)):
        return None, None, None, "not measured: an always-on member does not hold every shot"
    answer = first.predicted.copy()
    later = ~np.logical_and.reduce([same(first, other) for other in others])
    if majority:                                   # three members: an answer two of them share
        second, third = others
        pair = same(second, third) & ~same(first, second) & ~same(first, third)
        answer[pair] = second.predicted[pair]
        later = ~(same(first, second) | same(first, third) | same(second, third))
    pool = [members[name] for name in pool_names if name in members]
    if scorer == "weight" and any(np.isnan(member.weight[member.covers]).any() for member in pool):
        return None, None, None, "not measured: a candidate has no weight"
    pick = later_stage_pick(pool, scorer, network)
    chosen = np.stack([member.predicted for member in pool])[pick, np.arange(len(pick))]
    answer[later] = chosen[later]
    # Time: the always-on members on every shot; on the later-stage shots also the other candidates that decoded the shot.
    seconds = sum(members[name].seconds for name in accept)
    extra = sum(np.where(member.covers, member.seconds, 0.0) for member in pool if member.name not in accept)
    if "network" in scorer:
        extra = extra + (np.nan if network.ms is None else network.ms / 1e3)
    seconds = seconds + np.where(later, extra, 0.0)
    return answer, later, (None if np.isnan(seconds).any() else seconds), ""


def score(members, shots, network):
    """One dict per row of the report: the members alone, the rules, the ORACLE rows."""
    truth, rows = shots.truth, []
    for name, member in members.items():
        failed = (member.predicted != truth).any(axis=1) & member.covers
        seconds = member.seconds[member.covers]
        rows.append({"key": name, "kind": "member", "name": f"{name} alone" + ("" if member.covers.all() else " (on the shots it decoded)"),
                     "failed": failed, "shots": int(member.covers.sum()), "whole": bool(member.covers.all()),
                     "ms": None if np.isnan(seconds).any() or not len(seconds) else float(seconds.mean() * 1e3)})
    for rule in RULES:
        answer, later, seconds, note = apply(rule, members, network)
        row = {"key": rule[0], "kind": "rule", "name": f"{rule[0]}: {rule[1]}", "note": note, "shots": len(truth), "whole": True}
        if answer is not None:
            row |= {"failed": (answer != truth).any(axis=1), "later": later, "ms": None if seconds is None else float(seconds.mean() * 1e3)}
            if "J" in rule[4]:
                row["note"] = f"J exists on {int((later & members['J'].covers).sum())} of the {int(later.sum())} later-stage shots"
        rows.append(row)
    for names in ORACLES:
        if all(name in members for name in names):
            right = [((members[name].predicted == truth).all(axis=1) & members[name].covers) for name in names]
            rows.append({"key": "oracle_" + "_".join(names), "kind": "oracle", "shots": len(truth), "whole": True,
                         "name": "ORACLE (needs the truth): a perfect choice among " + ", ".join(names),
                         "failed": ~np.logical_or.reduce(right)})
    return rows


def unanimous(members):
    """[shots] bool: the always-on members that are present all predict the same bits."""
    present = [members[name] for name in ALWAYS_ON if name in members]
    return np.logical_and.reduce([same(present[0], other) for other in present[1:]]) if len(present) > 1 else np.ones(len(present[0].covers), bool)


# ---------- printing ----------

def report(members, shots, network, rows, chosen=None, out=print):
    count = len(shots.truth)
    out(f"{shots.basis or '?'} memory, p = {shots.p if shots.p is not None else '?'}, {count} shots of the data file with sha256 {shots.data_sha256}")
    for member in members.values():
        out(f"  {member.name:3s} from {member.source}: {int(member.covers.sum())} shots; times: {member.times_from}"
            + ("" if not np.isnan(member.seconds[member.covers]).any() else " (no time per shot in the file)"))
    if network is not None:
        out(f"  network: time per call {'%.1f ms' % network.ms if network.ms is not None else 'unknown'} ({network.where})")
    split = ~unanimous(members)
    out(f"the always-on members present ({', '.join(name for name in ALWAYS_ON if name in members)}) are not unanimous on "
        f"{int(split.sum())} shots ({split.mean():.3%})")
    out("")
    out(f"{'row':78s} {'failures':>8s} {'of':>6s}  {'rate':>7s}  {'95% Wilson interval':20s} {'later stage on':>16s}  {'ms/shot':>8s}")
    for row in rows:
        if "failed" not in row:
            out(f"{row['name']:78s} {row['note']}")
            continue
        failures = int(row["failed"].sum())
        low, high = wilson_interval(failures, row["shots"])
        later = f"{int(row['later'].sum())} ({row['later'].mean():.2%})" if "later" in row else ""
        time = f"{row['ms']:8.1f}" if row.get("ms") is not None else f"{'':8s}"
        mark = "   <- chosen on the training shots" if chosen and row["key"] == chosen else ""
        out(f"{row['name']:78s} {failures:8d} {row['shots']:6d}  {failures / max(row['shots'], 1):7.3%}  [{low:.3%}, {high:.3%}]{'':3s} "
            f"{later:>16s}  {time}{mark}" + (f"   ({row['note']})" if row.get("note") else ""))
    out("ms/shot: the sum of the decode calls a row makes, averaged over all shots; empty where a file holds no time per shot.")
    out("Each time is from the machine named above for its member: times from different machines are added here, so read them as a rough cost.")
    by_key = {row["key"]: row for row in rows}
    for reference, label in (("R", "relay-BP alone"), ("T", "TCBP-JOBS alone")):
        if reference in by_key and by_key[reference]["whole"]:
            out("")
            out(f"against {label} on the same shots: both fail, the row only, {label} only, exact two-sided p (binomial on the discordant shots)")
            for row in rows:
                if "failed" in row and row["whole"] and row["key"] != reference:
                    both, only_row, only_reference, p_value = paired_test(row["failed"], by_key[reference]["failed"])
                    out(f"{row['name']:78s} {both:5d} {only_row:5d} {only_reference:5d}   p = {p_value:.3g}")


def numbers(members, shots, network, rows):
    """The report as a dict for JSON."""
    by_key = {row["key"]: row for row in rows}
    table = []
    for row in rows:
        entry = {"key": row["key"], "kind": row["kind"], "row": row["name"], "note": row.get("note", ""), "shots": row["shots"]}
        if "failed" in row:
            failures = int(row["failed"].sum())
            entry |= {"failures": failures, "wilson_95": list(wilson_interval(failures, row["shots"])), "ms_per_shot": row.get("ms"),
                      "later_stage_shots": int(row["later"].sum()) if "later" in row else None}
            for reference in ("R", "T"):
                if reference in by_key and by_key[reference]["whole"] and row["whole"] and row["key"] != reference:
                    both, only_row, only_reference, p_value = paired_test(row["failed"], by_key[reference]["failed"])
                    entry[f"against_{reference}"] = {"both": both, "row_only": only_row, "reference_only": only_reference, "p": p_value}
        table.append(entry)
    return {"data_sha256": shots.data_sha256, "basis": shots.basis, "p": shots.p, "shots": len(shots.truth),
            "members": {member.name: {"file": member.source, "shots": int(member.covers.sum()), "times_from": member.times_from}
                        for member in members.values()},
            "network_ms": None if network is None else network.ms, "not_unanimous": int((~unanimous(members)).sum()), "rows": table}


def choose_rule(paths, shots):
    """The rule with the fewest failures on the training shots of the given .json files (ties: the earlier rule)."""
    records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    if any(record["data_sha256"] == shots.data_sha256 for record in records):
        raise SystemExit("--chosen-on names a .json of the data file that is being scored: choose the rule on other shots")
    totals = {}
    for key, *_ in RULES:
        found = [next((row.get("failures") for row in record["rows"] if row["key"] == key), None) for record in records]
        if all(value is not None for value in found):
            totals[key] = sum(found)
    if not totals:
        raise SystemExit("no rule was measured in every --chosen-on file")
    chosen = min(totals, key=lambda key: (totals[key], [rule[0] for rule in RULES].index(key)))
    shots_used = sum(record["shots"] for record in records)
    return chosen, (f"rule chosen on {shots_used} training shots (data sha256 {', '.join(record['data_sha256'] for record in records)}): "
                    + ", ".join(f"{key} {value}" for key, value in totals.items()) + f" failures; chosen: {chosen}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--members", nargs="+", required=True, help="per-shot files of candidates, tcbp_member, joint_member for ONE data file")
    parser.add_argument("--logits", help="the .npz of neural.evaluate_stim --save on the same shots")
    parser.add_argument("--network-ms", type=float, help="the network's time per call, where the .json beside the logits is missing")
    parser.add_argument("--chosen-on", nargs="+", help="the --save .json of OTHER shots (training files): marks the rule they choose")
    parser.add_argument("--save", help="write the numbers of the report to this .json file")
    parser.add_argument("--write-selection", help="write the shots where the always-on members are not unanimous, for joint_member")
    args = parser.parse_args(argv)

    members, shots = load_members(args.members)
    if "P1" not in members or "P2" not in members:
        raise SystemExit("the ensemble needs P1 and P2 (a file of hierarchy.candidates) among --members")
    network = load_network(args.logits, shots, args.network_ms) if args.logits else None
    chosen = None
    if args.chosen_on:
        chosen, line = choose_rule(args.chosen_on, shots)
        print(line)
    rows = score(members, shots, network)
    report(members, shots, network, rows, chosen)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(numbers(members, shots, network, rows) | {"chosen_rule": chosen}, indent=2) + "\n", encoding="utf-8")
    if args.write_selection:
        listed = np.flatnonzero(~unanimous(members))
        Path(args.write_selection).parent.mkdir(parents=True, exist_ok=True)
        with open(args.write_selection, "wb") as file:
            np.savez(file, shot=listed, data_sha256=shots.data_sha256, shots_in_file=len(shots.truth),
                     members=np.array([name for name in ALWAYS_ON if name in members]))
        print(f"\nwrote {args.write_selection}: {len(listed)} shots where {', '.join(name for name in ALWAYS_ON if name in members)} are not unanimous")
    return rows


if __name__ == "__main__":
    main()
