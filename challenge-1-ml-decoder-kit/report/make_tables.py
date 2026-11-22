"""Make every table of the report from the files under report/data/.

Run from challenge-1-ml-decoder-kit/:

  python -m report.make_tables

It writes report/tables/<name>.md and <name>.csv, and copies each table into REPORT.md and
SLIDES.md between the markers `<!-- BEGIN table:<name> -->` and `<!-- END table:<name> -->`.
A row whose result file is not in report/data/ is printed as "not measured": no number is
taken from anywhere else. If a file contradicts itself or the other rows (other test shots,
counts that do not add up), the run stops with the reason.

Conventions (the kit's): shot k of X memory is paired with shot k of Z memory; a pair fails
when either fails; with failure fraction P over the pairs, the rate per cycle is
1 - (1 - P)^(1/12). Times are per pair (one X decode plus one Z decode), in milliseconds.
"""
import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from report import loaders, stats
from report.loaders import DATA, Mismatch
from report.rows import (COMPARISONS, DECODERS, EARLIER_ROWS, LADDER, LOSS_CONDITIONS, OPTIONAL_RUNGS, P_VALUES, REFERENCE, TCBP_OWN, TEST_ROWS,
                         TEST_SHA256)

HERE = Path(__file__).resolve().parent
# REPORT_OUT sends the tables elsewhere and leaves the documents alone: used by the tests only.
TABLES = Path(os.environ.get("REPORT_OUT") or HERE) / "tables"
DOCUMENTS = [] if os.environ.get("REPORT_OUT") else [HERE / "REPORT.md", HERE / "APPENDIX.md", HERE / "SLIDES.md", HERE / "REPRODUCE.md"]
NOT_MEASURED = "not measured"
# A timed row's time is "not clean" when the notebook's CPU quota throttled it for more than this share of the row's
# own wall time (the job's START and END lines give the throttled seconds; the counts are not affected, only the time).
THROTTLE_LIMIT = 0.05

# Single numbers that the prose of REPORT.md and SLIDES.md quotes. A document writes
# <!--f:name-->value<!--/f--> and this script puts the current value between the two comments,
# so a sentence can never keep a number that the files no longer support. tables/facts.md lists them all.
FACTS = {}


def fact(name, value, source):
    FACTS[name] = (NOT_MEASURED if value is None else str(value), source)


# ---------- small helpers ----------

def ms(value):
    if value is None:
        return NOT_MEASURED
    return f"{value:,.0f}" if value >= 100 else f"{value:.1f}"


def rate_text(rate, error):
    return f"{stats.sci(rate)} ± {stats.sci(error, 1)}"


def interval_text(low, high):
    return f"{stats.sci(low, 1)} to {stats.sci(high, 1)}"


def percent(failures, shots, digits=2):
    low, high = stats.wilson(failures, shots)
    return f"{failures / shots:.{digits}%} ({low:.{digits}%} to {high:.{digits}%})"


def markdown(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def write(name, text, records=None):
    """One table: the Markdown text as <name>.md, and the raw values as <name>.csv."""
    TABLES.mkdir(exist_ok=True)
    (TABLES / f"{name}.md").write_text(text.rstrip() + "\n", encoding="utf-8")
    if records:
        fields = list(dict.fromkeys(key for record in records for key in record))
        with (TABLES / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    return text.rstrip()


# ---------- the rows on the test shots ----------

RETIME = "_retime"   # eval_p<p>_<row>_retime: the same row run again in a quiet window, for its time only


def attach_run(row, run, monitor, jobs):
    """What the server's own records say about a timed row's run: what ran beside it, the throttling, whether its time is clean."""
    row["during"] = loaders.machine_during(monitor, run["start"], run["end"]) if run and run["start"] and run["end"] else None
    if row["during"] and {"START", "END"} <= set(run["state"]):     # the job's own lines give the seconds throttled during the row exactly
        row["during"]["throttled_s"] = run["state"]["END"]["throttled_s"] - run["state"]["START"]["throttled_s"]
    row["throttle_share"] = None
    if row["during"] and run["end"] and run["start"]:
        wall = (run["end"] - run["start"]).total_seconds()
        row["during"]["wall_s"] = wall
        row["throttle_share"] = row["during"]["throttled_s"] / wall if wall else None
    row["time_clean"] = not (row["throttle_share"] is not None and row["throttle_share"] > THROTTLE_LIMIT)
    row["beside"] = run["beside"] if run else None
    row["run"] = run
    row["jobs_beside"] = (sorted(name for name, job in jobs.items() if job["start"] and run["start"] <= job["start"] < (run["end"] or datetime.max))
                          if run and run["start"] else None)


def take_retime(row, folder, log, monitor, jobs, p):
    """The same row decoded again in a quiet window: its answers must equal the first run's pair for pair; then its time replaces the first."""
    found = loaders.evaluator_rows(folder)
    again = found.get(row["name"]) if row["name"] in found else (next(iter(found.values())) if len(found) == 1 else None)
    if again is None:
        raise Mismatch(f"{folder}: no decoder named {row['name']} in the re-timed run")
    if again["test_sha256"] != row["test_sha256"]:
        raise Mismatch(f"{folder}: the re-timed run decoded other shot files than the first run")
    if row["failed_pair"] is None or again["failed_pair"] is None:
        raise Mismatch(f"{folder}: a per-shot file is missing, so the re-timed run cannot be shown to give the first run's answers")
    differ = int((row["failed_pair"] != again["failed_pair"]).sum())
    if differ or (row["failures_X"], row["failures_Z"]) != (again["failures_X"], again["failures_Z"]):
        raise Mismatch(f"{folder}: the re-timed run's answers differ from the first run's on {differ} pairs; a re-timing must change only the time")
    first = {"mean_ms": row["mean_ms"], "throttle_share": row.get("throttle_share"), "start": row["run"]["start"] if row.get("run") else None}
    for key in ("mean_ms", "median_ms", "p99_ms", "seconds_pair", "wall_seconds", "peak_rss_mb", "setup_seconds", "site", "workers", "per_shot_file"):
        row[key] = again[key]
    attach_run(row, log.get((p, folder.name.removeprefix(f"eval_p{p}_"))), monitor, jobs)
    row["retimed"] = {"first": first, "folder": str(folder.relative_to(DATA)), "source": again["source"]}
    row["source"] = row["source"] + " (counts) and " + again["source"] + " (time)"


def load_test_rows(data=DATA):
    """({p: {decoder key: row or None}}, unregistered folders, {p: the ensemble tool's offline output or None}).

    Every row is checked against the test files' identity. A row without an evaluator run can be
    stood in for by the offline combination of per-shot files (rows.py, `member`); it is then
    marked as such, and its time is a sum of decode calls.
    """
    log = loaders.clean_rows_log(data / "test" / "clean_rows.out", data / "test" / "test_rows.out")
    jobs = loaders.runner_logs(sorted((data / "test").glob("runner_*.out"))) if (data / "test").is_dir() else {}
    monitor = loaders.cpu_monitor(data / "test" / "cpu_monitor.out")
    measured, combined = {}, {}
    for p in P_VALUES:
        measured[p] = {}
        combined[p] = loaders.ensemble(data / "test" / f"ensemble_p{p}", p)
        if combined[p] and next(iter(combined[p]["rows"].values()))["test_sha256"] != TEST_SHA256[p]:
            raise Mismatch(f"test/ensemble_p{p}: these are not the team's test shots for p = {p}")
        for spec in TEST_ROWS:
            folder = data / "test" / spec.folder.format(p=p)
            row = None
            if spec.kind == "evaluator" and (folder / "summary.json").is_file():
                found = loaders.evaluator_rows(folder)
                if spec.name:
                    row = found.get(spec.name)
                elif len(found) == 1:
                    row = next(iter(found.values()))
                else:
                    raise Mismatch(f"{folder}: {len(found)} decoders in one folder; name the one meant in rows.py")
            elif spec.kind == "tcbp":
                row = loaders.tcbp_member(folder, p)
            elif spec.kind == "ensemble" and combined[p] and combined[p]["rule"] in combined[p]["rows"]:
                row = dict(combined[p]["rows"][combined[p]["rule"]])
                row["stand_in_note"] = f" (rule {row['description']}; frozen on the training shots)"
            if row is None and spec.member and combined[p] and spec.member in combined[p]["rows"]:
                row = dict(combined[p]["rows"][spec.member])           # no evaluator row: the offline combination stands in
                row["stand_in_note"] = " (from per-shot files, not a timed evaluator row)"
            if row is not None:
                if abs(float(row["p"]) - float(p)) > 1e-12:
                    raise Mismatch(f"{folder}: the file says p = {row['p']}, the folder says {p}")
                if row["test_sha256"] != TEST_SHA256[p]:
                    raise Mismatch(f"{folder}: these are not the team's test shots for p = {p} (sha256 {row['test_sha256']})")
                row["label_note"] = row.pop("stand_in_note", "")
                row["cores"] = spec.cores
                if spec.rule and not row.get("offline"):        # our plugin: the tie-break its own summary records, per basis
                    used = row.get("rule")
                    if not used:
                        raise Mismatch(f"{folder}: the summary records no tie-break rule, so the row cannot be labelled")
                    if spec.rule == "weight" and set(used.values()) != {"weight"}:
                        raise Mismatch(f"{folder}: registered as the row without a network, but its summary records {used}")
                    if spec.rule == "network" and "network" not in used.values():
                        raise Mismatch(f"{folder}: registered as the row with the network, but its summary records {used}")
                    if spec.rule == "network" and set(used.values()) != {"network"}:
                        row["label_note"] = " (" + ", ".join(f"{basis} memory: {'the network' if used[basis] == 'network' else 'lowest weight'}" for basis in "XZ") + ")"
                run = None if row.get("offline") else log.get((p, spec.folder.format(p=p).removeprefix(f"eval_p{p}_")))
                attach_run(row, run, monitor, jobs)
                retime = folder.with_name(folder.name + RETIME)
                if spec.kind == "evaluator" and not row.get("offline") and (retime / "summary.json").is_file():
                    take_retime(row, retime, log, monitor, jobs, p)
            measured[p][spec.decoder] = row
        reference = measured[p].get(REFERENCE)
        for key, row in measured[p].items():
            if row is not None:
                row["paired"] = (stats.paired(reference["failed_pair"], row["failed_pair"])
                                 if key != REFERENCE and reference is not None and reference["failed_pair"] is not None
                                 and row["failed_pair"] is not None else None)
                row["interval"] = stats.per_cycle_interval(row["failures_pair"], row["shots"])
    known = {spec.folder.format(p=p) + suffix for spec in TEST_ROWS for p in P_VALUES for suffix in ("", RETIME)}
    known |= {f"ensemble_p{p}" for p in P_VALUES}
    extra = sorted(path.name for path in (data / "test").glob("*") if path.is_dir() and path.name not in known) if (data / "test").is_dir() else []
    return measured, extra, combined


def where_text(row):
    if row.get("offline"):
        return f"{row['site']} ({row['device']} decoders, ICRN server); its time is {row['timer']}"
    text = f"{row['device']}, {row['workers']} process" + (f" on {row['cores']} cores" if row.get("cores", 1) > 1 else "") + f", {row['site']}"
    if row.get("retimed"):
        first = row["retimed"]["first"]
        text += (f"; **the time is from a second run in a quiet window** (`{row['retimed']['folder']}`), whose answers equal the first run's on every pair; "
                 f"the first run's mean was {ms(first['mean_ms'])} ms"
                 + (f", throttled for {first['throttle_share']:.1%} of its wall time" if first.get("throttle_share") is not None else ""))
    if row.get("during"):
        during = row["during"]
        others = ", ".join(f"{kind.replace('_', ' ')} {count}" for kind, count in during["other_max"].items() if count)
        text += (f"; while it ran: {during['cpus_mean']:.1f} CPUs busy on average (at most {during['cpus_max']:.1f}), up to {during['evaluators_max']} "
                 f"evaluator rows at once" + (f", other work of ours at most: {others}" if others else "")
                 + f", the CPU quota held the notebook back for {during['throttled_s']:,.0f} s"
                 + (f" of the row's {during['wall_s']:,.0f} s ({during['throttled_s'] / during['wall_s']:.1%})" if during.get("wall_s") else " in all"))
        if not row.get("time_clean", True):
            text += f"; **time not clean** (throttled more than {THROTTLE_LIMIT:.0%} of its wall time): the counts stand, the time does not"
    elif row.get("beside") is not None:
        text += f"; {len(row['beside'])} other evaluator rows running beside it" if row["beside"] else "; no other evaluator row beside it"
    if row.get("jobs_beside"):
        text += "; server jobs started in that time: " + ", ".join(row["jobs_beside"])
    if row["device"] == "GPU":
        text += "; its time is " + row["timer"]
    return text


def paired_text(row, key, reference=None):
    if key == REFERENCE:
        return "the reference"
    if row.get("paired") is None:
        return "relay-BP not measured yet" if reference is None else "no per-shot file"
    paired = row["paired"]
    return f"{paired['both']} / {paired['row_only']} / {paired['reference_only']} (p = {stats.p_text(paired['p'])})"


def test_tables(measured):
    texts, records = {}, []
    header = ["Decoder", "Built by", "Failures X, Z", "Failed pairs", "Rate per cycle ± error", "95% interval",
              "Time per pair, ms: mean / median / p99", "Mean time ÷ relay-BP's", "Against relay-BP, same pairs: both fail / only this row / only relay-BP (exact p)",
              "Where it ran"]
    time_lines = []
    for p in P_VALUES:
        lines = []
        base = measured[p].get(REFERENCE)
        base_ms = base["mean_ms"] if base and not base.get("offline") and base.get("time_clean", True) else None
        for key, row in measured[p].items():
            decoder = DECODERS[key]
            record = {"p": p, "decoder": key, "label": decoder.label, "built_by": decoder.whose, "measured": row is not None}
            if row is None:
                lines.append([decoder.label, decoder.whose] + [NOT_MEASURED] * 8)
                time_lines.append([p, decoder.label] + [NOT_MEASURED] * 6)
            else:
                timed = row["mean_ms"] is not None and row["median_ms"] is not None and not row.get("offline")
                clean = row.get("time_clean", True)
                ratio = row["mean_ms"] / base_ms if timed and clean and base_ms else None
                ratio_text = ("time not clean" if timed and not clean
                              else "the reference" if key == REFERENCE and timed else f"{ratio:.2f}" if ratio is not None
                              else "relay-BP not measured yet" if timed and base is None
                              else "relay-BP's time not clean" if timed and base is not None and not base.get("time_clean", True)
                              else NOT_MEASURED if timed else "not timed in one process")
                times = (NOT_MEASURED if row["mean_ms"] is None else f"{ms(row['mean_ms'])} / not timed per pair" if row["median_ms"] is None
                         else " / ".join(ms(row[name]) for name in ("mean_ms", "median_ms", "p99_ms")) + ("" if clean else " (not clean)"))
                lines.append([decoder.label + row["label_note"], decoder.whose, f"{row['failures_X']}, {row['failures_Z']}",
                              f"{row['failures_pair']} of {row['shots']:,}", rate_text(row["rate"], row["error"]),
                              interval_text(*row["interval"]), times, ratio_text, paired_text(row, key, base), where_text(row)])
                time_lines.append([p, decoder.label + row["label_note"], f"{row['failures_pair']} of {row['shots']:,}",
                                   "not timed in one process" if not timed else ms(row["mean_ms"]) + ("" if clean else " (not clean)"), ratio_text if timed else "",
                                   ms(row["median_ms"]) if timed else "", ms(row["p99_ms"]) if timed else "",
                                   "" if not timed else f"{row['device']}, {row.get('cores', 1)} core" + ("s" if row.get("cores", 1) > 1 else "")])
                paired = row.get("paired") or {}
                record |= {"pairs": row["shots"], "failures_X": row["failures_X"], "failures_Z": row["failures_Z"],
                           "failures_pair": row["failures_pair"], "rate_per_cycle": row["rate"], "kit_error": row["error"],
                           "wilson95_low": row["interval"][0], "wilson95_high": row["interval"][1],
                           "mean_ms_per_pair": row["mean_ms"], "median_ms_per_pair": row["median_ms"], "p99_ms_per_pair": row["p99_ms"],
                           "mean_time_over_relay_bp": ratio, "time_clean": clean if row is not None else None,
                           "throttled_share_of_wall_time": row.get("throttle_share"), "both_fail": paired.get("both"), "only_this_row_fails": paired.get("row_only"),
                           "only_relay_bp_fails": paired.get("reference_only"), "exact_p": paired.get("p"),
                           "device": row["device"], "processes": row["workers"], "machine": row["site"],
                           "rows_beside": "; ".join(row["beside"]) if row.get("beside") else "",
                           "peak_rss_mb_parent": row.get("peak_rss_mb"), "source": row["source"],
                           "test_X_sha256": row["test_sha256"]["X"], "test_Z_sha256": row["test_sha256"]["Z"]}
            records.append(record)
            short = f"t{p[-1]}_{key.replace('-', '_')}"          # t3, t4, t5 for p = 0.003, 0.004, 0.005
            source = f"tables/test_p{p}.md"
            paired = (row or {}).get("paired") or {}
            fact(f"{short}_pairs", row and row["failures_pair"], source)
            fact(f"{short}_xz", row and f"{row['failures_X']}, {row['failures_Z']}", source)
            fact(f"{short}_rate", row and stats.sci(row["rate"]), source)
            fact(f"{short}_error", row and stats.sci(row["error"], 1), source)
            fact(f"{short}_mean_ms", row and row["mean_ms"] is not None and (ms(row["mean_ms"]) + ("" if row.get("time_clean", True) else " (not clean)")) or None, source)
            fact(f"{short}_median_ms", row and row["median_ms"] is not None and ms(row["median_ms"]) or None, source)
            fact(f"{short}_p99_ms", row and row["p99_ms"] is not None and ms(row["p99_ms"]) or None, source)
            fact(f"{short}_both", paired.get("both"), source)
            fact(f"{short}_only_this", paired.get("row_only"), source)
            fact(f"{short}_only_relay", paired.get("reference_only"), source)
            fact(f"{short}_paired_p", stats.p_text(paired["p"]) if paired else None, source)
            fact(f"{short}_wall_hours", row and row.get("wall_seconds") and f"{row['wall_seconds'] / 3600:.1f}", source)
            fact(f"{short}_time_ratio", f"{row['mean_ms'] / base_ms:.2f}" if row and base_ms and row["mean_ms"] is not None and not row.get("offline")
                 and row.get("time_clean", True) else None, source)
        note = (f"Test shots, p = {p}: `data/p{p}/test_X.npz` and `test_Z.npz` (10,000 shots per basis, Stim seeds 301 and 302; "
                f"sha256 `{TEST_SHA256[p]['X'][:12]}…` and `{TEST_SHA256[p]['Z'][:12]}…`). Every measured row decoded these two files.")
        texts[p] = write(f"test_p{p}", note + "\n\n" + markdown(header, lines), [r for r in records if r["p"] == p])
    write("test_all", "\n\n".join(texts[p] for p in P_VALUES), records)
    texts["time"] = write("test_time", "Times on the test shots, each beside the row's failures, so that a time is never read without its accuracy. "
                          "\"Mean time ÷ relay-BP's\" below 1 means faster than relay-BP on average. One process per row on the team's server; "
                          "a row on two cores or on a GPU says so. A time is marked \"not clean\" when the server's CPU quota throttled it for more than "
                          f"{THROTTLE_LIMIT:.0%} of the row's wall time; it is then given no ratio and is not drawn in the time figures.\n\n"
                          + markdown(["p", "Decoder", "Failed pairs", "Mean time per pair, ms", "Mean time ÷ relay-BP's", "Median, ms", "99th percentile, ms", "Runs on"],
                                     time_lines))

    # the sweep in one table: a decoder per row, a noise level per column
    own = {p: loaders.tcbp_log(DATA / file) for p, file, _usable, _why in TCBP_OWN if (DATA / file).is_file()}
    lines = []
    for key in DECODERS:
        cells = []
        for p in P_VALUES:
            row = measured[p][key]
            if row is None and key == "tcbp" and p in own:
                cells.append(f"not measured on these shots; on its own shots: {stats.sci(own[p]['rate_printed'])} ± {stats.sci(own[p]['error_printed'], 1)}")
            elif row is None:
                cells.append(NOT_MEASURED)
            else:
                time = f"{ms(row['mean_ms'])} ms" if row["mean_ms"] is not None else "time " + NOT_MEASURED
                cells.append(f"{rate_text(row['rate'], row['error'])} · {row['failures_pair']} pairs · {time}")
        note = next((measured[p][key]["label_note"] for p in P_VALUES if measured[p][key]), "")
        lines.append([DECODERS[key].label + note] + cells)
    sweep = write("test_sweep", markdown(["Decoder"] + [f"p = {p}" for p in P_VALUES], lines)
                  + "\n\nEach cell: logical error rate per cycle ± the kit's error · failed pairs of 10,000 · mean time per pair. "
                    "Test shots; `report/tables/test_p<p>.md` has the intervals, the paired counts and where each row ran.")
    return texts, sweep


def pair_text(paired):
    return f"{paired['both']} / {paired['row_only']} / {paired['reference_only']} (p = {stats.p_text(paired['p'])})"


def ladder_tables(combined):
    """Whose gain is whose, and the ablation of the learned tie-break: both from the offline combination on the test files."""
    from scipy.stats import binomtest
    lines, ablation, records = [], [], []
    pooled = {"R": 0, "aw": 0, "a": 0, "floor": 0, "only_network": 0, "only_weight": 0, "levels": []}
    for p in P_VALUES:
        rows = dict(combined[p]["rows"]) if combined.get(p) else {}
        if not all(key in rows for key in ("R", "P1", "aw")):
            lines.append([p, "every rung"] + [NOT_MEASURED] * 6)
            ablation.append([p, "pairs"] + [NOT_MEASURED] * 5)
            for name in ("aw_pairs", "a_pairs", "only_network", "only_weight", "p"):
                fact(f"abl{p[-1]}_{name}", None, "tables/test_ablation.md")
            continue
        if rows["aw"]["later"] and "oracle_R_P1_P2" in rows:   # a perfect tie-break: P1 where the two runs agree, else right if any of the three is
            failed = {basis: np.where(rows["aw"]["later"][basis].astype(bool), rows["oracle_R_P1_P2"]["failed"][basis], rows["P1"]["failed"][basis]) for basis in "XZ"}
            pair = failed["X"] | failed["Z"]
            rate, error = stats.per_cycle_rate(int(pair.sum()), len(pair))
            rows["floor"] = rows["aw"] | {"failed": failed, "failed_pair": pair, "failures_X": int(failed["X"].sum()), "failures_Z": int(failed["Z"].sum()),
                                          "failures_pair": int(pair.sum()), "rate": rate, "error": error}
        previous = None
        for key, label, whose in LADDER:
            row = rows.get(key)
            if row is None and key in OPTIONAL_RUNGS:
                continue
            if row is None:
                lines.append([p, label, whose] + [NOT_MEASURED] * 5)
                continue
            lines.append([p, label, whose, f"{row['failures_X']}, {row['failures_Z']}", f"{row['failures_pair']} of {row['shots']:,}",
                          rate_text(row["rate"], row["error"]),
                          "the first rung" if previous is None else pair_text(stats.paired(previous["failed_pair"], row["failed_pair"])),
                          "the reference" if key == "R" else pair_text(stats.paired(rows["R"]["failed_pair"], row["failed_pair"]))])
            records.append({"p": p, "rung": key, "label": label, "built_by": whose, "failures_X": row["failures_X"], "failures_Z": row["failures_Z"],
                            "failures_pair": row["failures_pair"], "pairs": row["shots"], "rate_per_cycle": row["rate"], "kit_error": row["error"]})
            fact(f"lad{p[-1]}_{key}_pairs", row["failures_pair"], "tables/test_ladder.md")
            fact(f"lad{p[-1]}_{key}_rate", stats.sci(row["rate"]), "tables/test_ladder.md")
            previous = row
        for key, label in (("T", "TCBP-JOBS (NVIDIA, GPU), for reference"),):
            if key in rows:
                row = rows[key]
                lines.append([p, label, "NVIDIA; run by teammates", f"{row['failures_X']}, {row['failures_Z']}", f"{row['failures_pair']} of {row['shots']:,}",
                              rate_text(row["rate"], row["error"]), "not a rung", pair_text(stats.paired(rows["R"]["failed_pair"], row["failed_pair"]))])
                fact(f"lad{p[-1]}_{key}_pairs", row["failures_pair"], "tables/test_ladder.md")
        classical = "ac" if "ac" in rows else "aw"          # the best tie-break without learning that the files hold
        pooled["rule"] = "lowest weight among the converged candidates" if classical == "ac" else "lowest weight"
        if "a" in rows:
            hard = {basis: int(rows["aw"]["later"][basis].sum()) for basis in "XZ"} if rows["aw"]["later"] else {"X": None, "Z": None}
            for scope, mask in (("X memory, shots", lambda row: row["failed"]["X"]), ("Z memory, shots", lambda row: row["failed"]["Z"]),
                                ("pairs (the kit's score)", lambda row: row["failed_pair"])):
                paired = stats.paired(mask(rows[classical]), mask(rows["a"]))
                theirs = combined[p]["ablation"].get(scope[0], {}).get(f"a_against_{classical}") if scope[0] in "XZ" else None
                if theirs and (theirs["only_the_network_rule_fails"], theirs["only_the_rule_without_fails"]) != (paired["row_only"], paired["reference_only"]):
                    raise Mismatch(f"test/ensemble_p{p}: {scope[0]} memory, the ensemble tool's ablation says {theirs['only_the_network_rule_fails']} / "
                                   f"{theirs['only_the_rule_without_fails']} discordant shots, the per-shot files give {paired['row_only']} / {paired['reference_only']}")
                count = hard.get(scope[0]) if scope[0] in "XZ" and scope.startswith(("X", "Z")) else None
                ablation.append([p, scope, "" if count is None else count, int(mask(rows[classical]).sum()), int(mask(rows["a"]).sum()),
                                 f"{paired['row_only']} / {paired['reference_only']}", stats.p_text(paired["p"])])
            pooled["levels"].append(p)
            pooled["only_network"] += paired["row_only"]
            pooled["only_weight"] += paired["reference_only"]
            for key, source_key in (("R", "R"), ("aw", classical), ("a", "a")):
                pooled[key] += rows[source_key]["failures_pair"]
            pooled["floor"] += rows["floor"]["failures_pair"] if "floor" in rows else 0
            for name, value in (("aw_pairs", rows[classical]["failures_pair"]), ("a_pairs", rows["a"]["failures_pair"]),
                                ("only_network", paired["row_only"]), ("only_weight", paired["reference_only"]), ("p", stats.p_text(paired["p"]))):
                fact(f"abl{p[-1]}_{name}", value, "tables/test_ablation.md")
        else:
            ablation.append([p, "pairs (the kit's score)"] + [NOT_MEASURED] * 5)
            for name in ("aw_pairs", "a_pairs", "only_network", "only_weight", "p"):
                fact(f"abl{p[-1]}_{name}", None, "tables/test_ablation.md")
    source = "tables/test_ablation.md"
    if pooled["levels"]:
        discordant = pooled["only_network"] + pooled["only_weight"]
        p_value = float(binomtest(pooled["only_network"], discordant).pvalue) if discordant else 1.0
        levels = ", ".join(pooled["levels"])
        counts = (f"The classical tie-break ({pooled['rule']}) fails {pooled['aw']} pairs and the network {pooled['a']}; only the network rule fails "
                  f"{pooled['only_network']} pairs, only the classical rule {pooled['only_weight']}, exact p = {stats.p_text(p_value)}.")
        ablation.append([levels + " together", "pairs (the kit's score)", "", pooled["aw"], pooled["a"],
                         f"{pooled['only_network']} / {pooled['only_weight']}", stats.p_text(p_value)])
        further = 1 - pooled["a"] / pooled["aw"] if pooled["aw"] else 0.0
        gap = (pooled["aw"] - pooled["a"]) / (pooled["aw"] - pooled["floor"]) if pooled["aw"] > pooled["floor"] else 0.0
        if pooled["a"] < pooled["aw"] and p_value < 0.05:
            framing = (f"A classical hierarchy gets most of the improvement over relay-BP ({pooled['R']} to {pooled['aw']} failed pairs over p = {levels}); "
                       f"the structure-aware network adds a further {further:.1%}, closing {gap:.0%} of the gap to a perfect tie-breaker "
                       f"({pooled['floor']} failed pairs). " + counts)
        elif pooled["a"] < pooled["aw"]:
            framing = (f"The hierarchy is the result ({pooled['R']} failed pairs for relay-BP, {pooled['aw']} for the hierarchy, over p = {levels}); "
                       f"the network is a promising tie-breaker whose gain is not yet significant. " + counts)
        else:
            framing = (f"The hierarchy is the result ({pooled['R']} failed pairs for relay-BP, {pooled['aw']} for the hierarchy, over p = {levels}); "
                       f"the neural tie-break did not lower its failures on the test shots. " + counts)
        for name, value in (("levels", levels), ("R", pooled["R"]), ("aw", pooled["aw"]), ("a", pooled["a"]), ("floor", pooled["floor"]),
                            ("only_network", pooled["only_network"]), ("only_weight", pooled["only_weight"]), ("p", stats.p_text(p_value)),
                            ("further", f"{further:.1%}"), ("gap_closed", f"{gap:.0%}")):
            fact(f"abl_pooled_{name}", value, source)
        fact("framing", framing, source)
    else:
        for name in ("levels", "R", "aw", "a", "floor", "only_network", "only_weight", "p", "further", "gap_closed"):
            fact(f"abl_pooled_{name}", None, source)
        fact("framing", None, source)
    ladder = write("test_ladder",
                   "**Whose gain is whose, on the test shots.** Every rung is scored on the same per-shot answers (relay-BP and PriorTilt under two seeds "
                   "on every shot, a fresh decoder per 500 shots), so each step is a paired comparison with the rung before it. These rows are combined "
                   "offline from per-shot files; they are not the timed rows of the table above, and relay-BP's count here differs from its timed row by "
                   "a few pairs because its random draws differ.\n\n"
                   + markdown(["p", "Rung", "Whose work", "Failures X, Z", "Failed pairs", "Rate per cycle ± error",
                               "Against the rung before: both fail / only this / only the one before (exact p)",
                               "Against relay-BP: both fail / only this / only relay-BP (exact p)"], lines), records)
    ablated = write("test_ablation",
                    "**The ablation: what the learned part adds.** The same hierarchy on the same candidates, with the best tie-break that needs no "
                    f"learning ({pooled.get('rule', 'lowest weight')}) and with the neural network. Only the shots where the two fast decodes disagree can differ.\n\n"
                    + markdown(["p", "Counted over", "Shots where the two fast decodes disagree", "Failures, classical tie-break", "Failures, neural tie-break",
                                "Only the network rule fails / only the classical rule fails", "Exact p"], ablation))
    return ladder, ablated


def comparisons_table(measured):
    lines, records = [], []
    for p in P_VALUES:
        for first, second in COMPARISONS:
            a, b = measured[p].get(first), measured[p].get(second)
            name = f"c{p[-1]}_{first.replace('-', '_')}_vs_{second.replace('-', '_')}"
            if a is None or b is None or a["failed_pair"] is None or b["failed_pair"] is None:
                lines.append([p, DECODERS[first].label, DECODERS[second].label] + [NOT_MEASURED] * 4)
                for part in ("both", "only_first", "only_second", "p"):
                    fact(f"{name}_{part}", None, "tables/test_comparisons.md")
                continue
            paired = stats.paired(b["failed_pair"], a["failed_pair"])
            lines.append([p, DECODERS[first].label + a["label_note"], DECODERS[second].label + b["label_note"],
                          f"{a['failures_pair']} against {b['failures_pair']}", paired["both"],
                          f"{paired['row_only']} / {paired['reference_only']}", stats.p_text(paired["p"])])
            records.append({"p": p, "first": first, "second": second, "first_failed_pairs": a["failures_pair"], "second_failed_pairs": b["failures_pair"],
                            "both_fail": paired["both"], "only_first_fails": paired["row_only"], "only_second_fails": paired["reference_only"],
                            "exact_p": paired["p"]})
            for part, value in (("both", paired["both"]), ("only_first", paired["row_only"]), ("only_second", paired["reference_only"]),
                                ("p", stats.p_text(paired["p"]))):
                fact(f"{name}_{part}", value, "tables/test_comparisons.md")
    text = ("Paired comparisons on the test shots: two decoders on the same 10,000 pairs, and how many pairs only one of the two fails. "
            "The exact p is the two-sided binomial test on those pairs (under \"no difference\" each is a fair coin). "
            "This is where each gain is attributed: PriorTilt is IonQ's, the hierarchy and its tie-break are ours.\n\n"
            + markdown(["p", "First decoder", "Second decoder", "Failed pairs: first against second", "Both fail",
                        "Only the first fails / only the second fails", "Exact p"], lines))
    return write("test_comparisons", text, records)


# ---------- the teammates' earlier rows, and the cross-check against the timed rows ----------

def earlier_table(measured):
    lines, records, checks = [], [], []
    for p, where, evaluation, router, gain in EARLIER_ROWS:
        rows = loaders.evaluator_rows(DATA / evaluation)
        rows |= {"router": next(iter(loaders.evaluator_rows(DATA / router).values()))}
        comparison = json.loads((DATA / gain / "comparison.json").read_text(encoding="utf-8"))["paired_failures"]
        for name, label in (("bposd", "BP-OSD"), ("relay-bp", "relay-BP"), ("relay-first-solution", "relay-BP, first solution only (a control)"),
                            ("relay-first-leg-or-full", "relay-BP, first leg or full (a control)"), ("router", "relay-BP with the learned router")):
            row = rows[name]
            if row["test_sha256"] != TEST_SHA256[p]:
                raise Mismatch(f"{evaluation}: not the team's test shots for p = {p}")
            paired = (f"{comparison['both_fail']} / {comparison['candidate_only_fail']} / {comparison['reference_only_fail']} "
                      f"(p = {stats.p_text(comparison['exact_discordant_sign_test_p'])})") if name == "router" else ""
            lines.append([p, label, f"{row['failures_X']}, {row['failures_Z']}", f"{row['failures_pair']} of {row['shots']:,}",
                          rate_text(row["rate"], row["error"]), " / ".join(ms(row[k]) for k in ("mean_ms", "median_ms", "p99_ms")), paired, where])
            records.append({"p": p, "decoder": name, "failures_X": row["failures_X"], "failures_Z": row["failures_Z"],
                            "failures_pair": row["failures_pair"], "pairs": row["shots"], "rate_per_cycle": row["rate"], "kit_error": row["error"],
                            "mean_ms_per_pair": row["mean_ms"], "median_ms_per_pair": row["median_ms"], "p99_ms_per_pair": row["p99_ms"],
                            "machine": where, "source": row["source"]})
        timed = measured[p].get(REFERENCE)
        earlier = rows["relay-bp"]
        if timed is None or timed["failed_pair"] is None:
            for name in ("differ", "timed_pairs", "earlier_pairs"):
                fact(f"cross{p[-1]}_{name}", None, "tables/checks.md")
            checks.append(f"p = {p}: relay-BP's timed row has not landed, so it cannot be compared with the teammates' earlier run ({where}).")
        else:
            differ = int((timed["failed_pair"] != earlier["failed_pair"]).sum())
            fact(f"cross{p[-1]}_differ", differ, "tables/checks.md")
            fact(f"cross{p[-1]}_timed_pairs", timed["failures_pair"], "tables/checks.md")
            fact(f"cross{p[-1]}_earlier_pairs", earlier["failures_pair"], "tables/checks.md")
            if differ and timed["site"] == earlier["site"]:
                raise Mismatch(f"p = {p}: relay-BP's timed row differs from the teammates' earlier run on the same machine on {differ} pairs; "
                               "relay-BP in one process on the same file should give the same answers")
            checks.append(f"p = {p}: relay-BP's timed row and the teammates' earlier run ({where}) give "
                          + ("the same outcome on every one of the 10,000 pairs." if differ == 0 else
                             f"a different outcome on {differ} of the 10,000 pairs ({timed['failures_pair']} against {earlier['failures_pair']} failed pairs)."))
        router, earlier_router = measured[p].get("router"), rows["router"]
        if router is not None and router["failed_pair"] is not None and earlier_router["failed_pair"] is not None:
            differ = int((router["failed_pair"] != earlier_router["failed_pair"]).sum())
            checks.append(f"p = {p}: the teammates' router, timed row against their earlier run ({where}): "
                          + ("the same outcome on every one of the 10,000 pairs." if differ == 0 else
                             f"a different outcome on {differ} of the 10,000 pairs ({router['failures_pair']} against {earlier_router['failures_pair']} failed pairs)."))
    text = ("The teammates' own evaluator runs on the same test files, made earlier on other machines or under other load. "
            "They are a cross-check of the counts. **Their times are not comparable with the timed table** and are shown only with their machine.\n\n"
            + markdown(["p", "Decoder", "Failures X, Z", "Failed pairs", "Rate per cycle ± error", "Time per pair, ms: mean / median / p99",
                        "Router against relay-BP: both / only router / only relay-BP (exact p)", "Machine"], lines))
    return write("earlier_runs", text, records), checks


# ---------- TCBP-JOBS on its own sampled shots ----------

def tcbp_own_table():
    lines, records = [], []
    for p, file, usable, why in TCBP_OWN:
        if not (DATA / file).is_file():
            lines.append([p] + [NOT_MEASURED] * 5)
            continue
        row = loaders.tcbp_log(DATA / file)
        low, high = (stats.per_cycle(v) for v in stats.wilson(row["failures_X"] + row["failures_Z"], row["shots"]))
        time = f"{ms(row['mean_ms'])} / {ms(row['p99_ms'])}" if usable else f"not usable: {why} (logged: {ms(row['mean_ms'])} / {ms(row['p99_ms'])})"
        lines.append([p, f"{row['failures_X']}, {row['failures_Z']}", f"{row['shots']:,} per basis, its own sample (seed {row['seed']})",
                      f"{stats.sci(row['rate_printed'])} ± {stats.sci(row['error_printed'], 1)}", time, f"`{row['source']}`"])
        fact(f"tcbp_own{p[-1]}_rate", stats.sci(row["rate_printed"]), "tables/tcbp_own_shots.md")
        fact(f"tcbp_own{p[-1]}_xz", f"{row['failures_X']}, {row['failures_Z']}", "tables/tcbp_own_shots.md")
        records.append({"p": p, **{k: row[k] for k in ("failures_X", "failures_Z", "shots", "rate_printed", "error_printed", "mean_ms", "p99_ms", "started", "host")},
                        "time_usable": usable, "why_not": why, "source": row["source"]})
    text = ("TCBP-JOBS as the teammates first ran it (`tcbp_jobs_baseline.py`, an NVIDIA A100 slice on the ICRN server). "
            "**These are its own freshly sampled shots, not the saved test files**, so there is no paired count against the other rows, "
            "and the log prints the rate per cycle but not the number of failed pairs.\n\n"
            + markdown(["p", "Failures X, Z", "Shots", "Rate per cycle ± error (as logged)", "Time per pair, ms: mean / p99", "File"], lines))
    return write("tcbp_own_shots", text, records)


# ---------- exploration shots: the hierarchy's parts at p = 0.004 ----------

def exploration_table():
    folder = DATA / "exploration" / "priortilt"
    sets = {"X": ["selector_validation_X.npz", "selector_train_X.npz"], "Z": ["selector_validation_Z.npz", "selector_train_Z.npz"]}
    shots = {basis: loaders.selector_files(folder, names) for basis, names in sets.items()}
    if any(value is None for value in shots.values()):
        return write("exploration_p0.004", NOT_MEASURED), {}
    facts, lines, records = {}, [], []
    tie_break_note, tcbp = "", {}
    rule = {basis: loaders.weight_rule(shots[basis]) for basis in "XZ"}
    rule_c = {basis: loaders.weight_rule(shots[basis], converged_only=True) for basis in "XZ"}
    network = {int(re.search(r"step(\d+)", file.name).group(1)): loaders.network_rule_table(file)
               for file in sorted(folder.glob("results_network_select_step*.txt"), key=lambda file: int(re.search(r"step(\d+)", file.name).group(1)))}

    def add(scope, label, failed, reference, seconds, whose):
        count, total = int(failed.sum()), len(failed)
        paired = stats.paired(reference, failed) if reference is not None else None
        lines.append([scope, label, whose, f"{count} of {total:,}", percent(count, total, 3),
                      "the reference" if paired is None else f"{paired['both']} / {paired['row_only']} / {paired['reference_only']} (p = {stats.p_text(paired['p'])})",
                      "" if seconds is None else ms(float(np.mean(seconds)) * 1e3)])
        records.append({"shots": scope, "row": label, "built_by": whose, "failures": count, "of": total,
                        "both_fail": paired and paired["both"], "only_this_row": paired and paired["row_only"],
                        "only_relay_bp": paired and paired["reference_only"], "exact_p": paired and paired["p"],
                        "mean_ms_per_shot": None if seconds is None else float(np.mean(seconds)) * 1e3})
        return count

    for basis in "XZ":
        s, (failed, agreed, seconds) = shots[basis], rule[basis]
        scope = f"{basis} memory, 25,000 shots"
        reference = s["failed_R"].astype(bool)
        facts[f"R_{basis}"] = add(scope, "relay-BP (kit baseline)", reference, None, s["seconds_R"], "the kit")
        facts[f"P1_{basis}"] = add(scope, "PriorTilt, seed A", s["failed_P1"].astype(bool), reference, s["seconds_P1"], "IonQ")
        facts[f"P2_{basis}"] = add(scope, "PriorTilt, seed B", s["failed_P2"].astype(bool), reference, s["seconds_P2"], "IonQ")
        facts[f"W_{basis}"] = add(scope, "our hierarchy, lowest-weight tie-break", failed, reference, seconds, "ours")
        facts[f"C_{basis}"] = add(scope, "our hierarchy, lowest weight among the converged candidates", rule_c[basis][0], reference, rule_c[basis][2], "ours")
        tcbp[basis] = loaders.tcbp_files(DATA / "exploration" / "tcbp_p0.004", [f"validation_{basis}.tcbp.npz", f"train_{basis}.tcbp.npz"])
        if tcbp[basis] and tcbp[basis]["data_sha256"] != s["data_sha256"]:
            raise Mismatch("exploration/tcbp_p0.004: TCBP-JOBS decoded other shot files than the candidates")
        if tcbp[basis]:
            facts[f"T_{basis}"] = add(scope, "TCBP-JOBS (NVIDIA, GPU; answers only, no time)", tcbp[basis]["failed"], reference, None, "NVIDIA; run by us")
        facts[f"disagree_{basis}"] = int((~agreed).sum())
        if basis == "X":
            recomputed = {}
            for step in network:
                logits = loaders.network_logits(DATA / "exploration" / "cascade_X_s1",
                                                [f"step{step}_validation_X_p0.004.npz", f"step{step}_train_X_p0.004.npz"], s["truth"])
                if logits is not None:
                    recomputed[step] = loaders.network_rule(s, logits)
            facts["recomputed_steps"] = sorted(recomputed)
            for step, table in network.items():
                if table and "b. network breaks ties" in table["rows"]:
                    count, time = table["rows"]["b. network breaks ties"]
                    if table["rows"]["R alone"][0] != facts["R_X"]:
                        raise Mismatch(f"{table['source']}: relay-BP's failures differ from the per-shot files")
                    if step in recomputed:
                        mine = recomputed[step]
                        if int(mine.sum()) != count:
                            raise Mismatch(f"{table['source']}: the network rule recomputed from the logits fails {int(mine.sum())} shots, the laptop's table says {count}")
                        versus, weight, best = stats.paired(reference, mine), stats.paired(failed, mine), stats.paired(rule_c[basis][0], mine)
                        fact("expl_X_N_vs_C_only_network", best["row_only"], "tables/exploration_p0.004.md")
                        fact("expl_X_N_vs_C_only_classical", best["reference_only"], "tables/exploration_p0.004.md")
                        fact("expl_X_N_vs_C_p", stats.p_text(best["p"]), "tables/exploration_p0.004.md")
                        cell = f"{versus['both']} / {versus['row_only']} / {versus['reference_only']} (p = {stats.p_text(versus['p'])})"
                        fact("expl_X_N_only_relay", versus["reference_only"], "tables/exploration_p0.004.md")
                        fact("expl_X_N_only_ours", versus["row_only"], "tables/exploration_p0.004.md")
                        fact("expl_X_N_vs_W_only_network", weight["row_only"], "tables/exploration_p0.004.md")
                        fact("expl_X_N_vs_W_only_weight", weight["reference_only"], "tables/exploration_p0.004.md")
                        fact("expl_X_N_vs_W_p", stats.p_text(weight["p"]), "tables/exploration_p0.004.md")
                        tie_break_note = (f"The neural tie-break against the lowest-weight tie-break on the same 25,000 X shots (network at training step {step:,}, "
                                          f"recomputed here from its per-shot logits): both fail {weight['both']}, only the network rule fails {weight['row_only']}, "
                                          f"only the weight rule fails {weight['reference_only']} (exact p = {stats.p_text(weight['p'])}). Against the better classical "
                                          f"tie-break, lowest weight among the converged candidates: both fail {best['both']}, only the network rule fails "
                                          f"{best['row_only']}, only the classical rule fails {best['reference_only']} (exact p = {stats.p_text(best['p'])}).")
                    else:
                        cell = "the laptop's table gives the count only"
                    lines.append([scope, f"our hierarchy, neural tie-break (network at training step {step:,})", "ours",
                                  f"{count} of 25,000", percent(count, 25000, 3), cell, ms(time)])
                    records.append({"shots": scope, "row": f"hierarchy, network step {step}", "built_by": "ours", "failures": count, "of": 25000,
                                    "mean_ms_per_shot": time})
                    facts[f"N{step}_X"] = count
    # pairs: shot k of X with shot k of Z, in the same file order (validation, then train)
    pair = {name: shots["X"][f"failed_{name}"].astype(bool) | shots["Z"][f"failed_{name}"].astype(bool) for name in ("R", "P1", "P2")}
    pair["W"] = rule["X"][0] | rule["Z"][0]
    pair["C"] = rule_c["X"][0] | rule_c["Z"][0]
    rungs = [("R", "relay-BP (kit baseline)", "the kit"), ("P1", "PriorTilt, seed A", "IonQ"), ("P2", "PriorTilt, seed B", "IonQ"),
             ("W", "our hierarchy, lowest-weight tie-break", "ours"), ("C", "our hierarchy, lowest weight among the converged candidates", "ours")]
    if all(tcbp.get(basis) for basis in "XZ"):
        pair["T"] = tcbp["X"]["failed"] | tcbp["Z"]["failed"]
        rungs.append(("T", "TCBP-JOBS (NVIDIA, GPU)", "NVIDIA; run by us"))
        versus = stats.paired(pair["T"], pair["C"])
        tcbp_note = (f"Our hierarchy (lowest weight among the converged candidates) against TCBP-JOBS on the same 25,000 pairs: both fail {versus['both']}, "
                     f"only ours fails {versus['row_only']}, only TCBP-JOBS fails {versus['reference_only']} (exact p = {stats.p_text(versus['p'])}). "
                     "**TCBP-JOBS fails fewer pairs than our hierarchy here.**" if int(pair["T"].sum()) < int(pair["C"].sum()) else "")
    else:
        tcbp_note = ""
    pair_lines = []
    for name, label, whose in rungs:
        count, total = int(pair[name].sum()), len(pair[name])
        rate, error = stats.per_cycle_rate(count, total)
        paired = stats.paired(pair["R"], pair[name]) if name != "R" else None
        seconds = (None if name == "T" else rule["X"][2] + rule["Z"][2] if name == "W" else rule_c["X"][2] + rule_c["Z"][2] if name == "C"
                   else shots["X"][f"seconds_{name}"] + shots["Z"][f"seconds_{name}"])
        pair_lines.append([label, whose, f"{count} of {total:,}", rate_text(rate, error), interval_text(*stats.per_cycle_interval(count, total)),
                           "the reference" if paired is None else f"{paired['both']} / {paired['row_only']} / {paired['reference_only']} (p = {stats.p_text(paired['p'])})",
                           "no time: batch mode on a shared GPU" if seconds is None else ms(float(seconds.mean()) * 1e3)])
        facts[f"pairs_{name}"] = count
        facts[f"rate_{name}"] = rate
        records.append({"shots": "25,000 pairs", "row": label, "built_by": whose, "failures": count, "of": total, "rate_per_cycle": rate, "kit_error": error,
                        "both_fail": paired and paired["both"], "only_this_row": paired and paired["row_only"],
                        "only_relay_bp": paired and paired["reference_only"], "exact_p": paired and paired["p"],
                        "mean_ms_per_pair": None if seconds is None else float(seconds.mean()) * 1e3})
    # where the failures are: the shots on which the two PriorTilt runs agree, and those on which they differ
    agreed = np.concatenate([rule[basis][1] for basis in "XZ"])
    first = np.concatenate([shots[basis]["failed_P1"].astype(bool) for basis in "XZ"])
    left = np.concatenate([rule[basis][0] for basis in "XZ"])
    split_lines = [[label, f"{int(mask.sum()):,}", f"{int((mask & agreed).sum()):,}", f"{int((mask & ~agreed).sum()):,}",
                    f"{(mask & ~agreed).sum() / mask.sum():.1%}"]
                   for label, mask in (("all shots (X and Z memory)", np.ones(len(agreed), bool)), ("shots PriorTilt, seed A, gets wrong", first),
                                       ("shots still wrong after the lowest-weight tie-break", left))]
    source = "tables/exploration_p0.004.md"
    fact("expl_shots", f"{len(agreed):,}", source)
    fact("expl_disagree", int((~agreed).sum()), source)
    fact("expl_disagree_share", f"{(~agreed).mean():.1%}", source)
    fact("expl_one_run_wrong", int(first.sum()), source)
    fact("expl_one_run_wrong_disagree", int((first & ~agreed).sum()), source)
    fact("expl_left", int(left.sum()), source)
    fact("expl_left_agree", int((left & agreed).sum()), source)
    fact("expl_pairs", f"{len(pair['R']):,}", source)
    for name in ("R", "P1", "P2", "W", "C", "T"):
        fact(f"expl_pairs_{name}", facts.get(f"pairs_{name}"), source)
        fact(f"expl_rate_{name}", f"pairs_{name}" in facts and stats.sci(facts[f"rate_{name}"]) or None, source)
    fact("expl_X_T", facts.get("T_X"), source)
    for name in ("W", "C"):
        versus = stats.paired(pair["R"], pair[name])
        fact(f"expl_pairs_{name}_only_relay", versus["reference_only"], source)
        fact(f"expl_pairs_{name}_only_ours", versus["row_only"], source)
        fact(f"expl_pairs_{name}_p", stats.p_text(versus["p"]), source)
    fact("expl_pairs_W_fewer", f"{1 - facts['pairs_W'] / facts['pairs_R']:.1%}", source)
    for basis in "XZ":
        for name in ("R", "P1", "P2", "W", "C"):
            fact(f"expl_{basis}_{name}", facts[f"{name}_{basis}"], source)
    # the network's count is quoted for the snapshot whose logits are copied, so that its paired counts are of the same snapshot
    steps = [step for step in facts.get("recomputed_steps", []) if f"N{step}_X" in facts]
    if steps:
        fact("expl_X_N", facts[f"N{steps[-1]}_X"], source)
        fact("expl_X_N_step", f"{steps[-1]:,}", source)
    text = ("**Exploration shots, not test shots**: the team's train and validation files at p = 0.004 (`data/p0.004/train_*.npz`, "
            "`validation_*.npz`; 25,000 shots per basis). The rules were chosen on these same shots, which flatters them. "
            "Decoded on a laptop (Ryzen 9 8945HS) beside a GPU training: compare the times with each other only. "
            "In this first table a time is per shot of one basis.\n\n"
            + markdown(["Shots", "Decoder", "Built by", "Failures", "Share of shots (95% interval)",
                        "Against relay-BP, same shots: both fail / only this row / only relay-BP (exact p)", "Mean time per shot, ms"], lines)
            + ("\n\n" + tie_break_note if tie_break_note else "")
            + "\n\nThe same decodes as pairs (shot k of X with shot k of Z), the kit's score; times per pair:\n\n"
            + markdown(["Decoder", "Built by", "Failed pairs", "Rate per cycle ± error", "95% interval",
                        "Against relay-BP: both fail / only this row / only relay-BP (exact p)", "Mean time per pair, ms"], pair_lines)
            + ("\n\n" + tcbp_note if tcbp_note else "")
            + "\n\nWhere the failures are: the shots on which the two PriorTilt runs predict the same 12 bits (\"agree\"), and the others:\n\n"
            + markdown(["Set", "Shots", "The two runs agree", "They disagree", "Share in \"disagree\""], split_lines)
            + "\n\nSources: `report/data/exploration/priortilt/selector_*.npz` (per shot: relay-BP and PriorTilt under two seeds), and for the "
              "rows with the network `results_network_select_step*.txt` (the laptop's own table).")
    return write("exploration_p0.004", text, records), facts


def failure_split(combined):
    """Where the failures are: {set of shots: [(what, shots where the two fast decodes agree, where they disagree)]}.

    One entry for the exploration shots at p = 0.004 (the laptop's per-shot files) and one per noise
    level of the test shots (the offline combination's files), X and Z memory shots together.
    """
    sets = {}
    names = ["selector_validation_X.npz", "selector_train_X.npz", "selector_validation_Z.npz", "selector_train_Z.npz"]
    shots = loaders.selector_files(DATA / "exploration" / "priortilt", names)
    if shots is not None:
        failed, agreed, _ = loaders.weight_rule(shots)
        sets["exploration shots, p = 0.004"] = (agreed, [("one fast decode (PriorTilt, seed A)", shots["failed_P1"].astype(bool)),
                                                        ("the hierarchy, lowest-weight tie-break", failed)])
    for p in P_VALUES:
        rows = combined[p]["rows"] if combined.get(p) else {}
        if "aw" in rows and rows["aw"]["later"] and "P1" in rows:
            join = lambda row: np.concatenate([row["failed"]["X"], row["failed"]["Z"]])
            agreed = ~np.concatenate([rows["aw"]["later"]["X"], rows["aw"]["later"]["Z"]]).astype(bool)
            sets[f"test shots, p = {p}"] = (agreed, [("one fast decode (PriorTilt, seed A)", join(rows["P1"])), ("the hierarchy, lowest-weight tie-break", join(rows["aw"]))]
                                            + ([("the hierarchy, neural tie-break", join(rows["a"]))] if "a" in rows else []))
    return {name: [("all shots", int(agreed.sum()), int((~agreed).sum()))]
                  + [(f"shots wrong under {label}", int((mask & agreed).sum()), int((mask & ~agreed).sum())) for label, mask in masks]
            for name, (agreed, masks) in sets.items()}


def failure_split_table(combined):
    lines = [[name, what, f"{same + differ:,}", f"{same:,}", f"{differ:,}", f"{differ / (same + differ):.1%}" if same + differ else ""]
             for name, bars in failure_split(combined).items() for what, same, differ in bars]
    return write("where_failures", "Where the failures are. \"Agree\" means the two runs of PriorTilt (two random seeds) predict the same 12 bits; the hierarchy "
                 "then takes that answer, and a wrong one cannot be repaired by any tie-break. X and Z memory shots are counted together.\n\n"
                 + markdown(["Shots", "Set", "Shots in the set", "The two runs agree", "They disagree", "Share in \"disagree\""], lines))


def exploration_ensemble_table():
    """The ensemble on exploration shots: the members' saved answers on the team's validation files, combined offline by hierarchy.ensemble."""
    lines, records = [], []
    for folder in sorted((DATA / "exploration" / "ensemble").glob("ens_p*")):
        p = folder.name.removeprefix("ens_p")
        found = loaders.ensemble(folder, p, split="validation")
        if not found:
            continue
        rows = found["rows"]
        order = [key for key in ("R", "P1", "P2", "T", "J") if key in rows] + sorted(key for key in rows if key not in ("R", "P1", "P2", "T", "J") and not key.startswith("oracle"))
        order += sorted(key for key in rows if key.startswith("oracle"))
        for key in order:
            row = rows[key]
            text = row["description"] if row["description"] != key else {"R": "R alone", "P1": "P1 alone", "P2": "P2 alone", "T": "T alone"}.get(key, key)
            marked = " **(the rule chosen on the training shots)**" if key == found["rule"] else ""
            cells = [pair_text(stats.paired(rows[other]["failed_pair"], row["failed_pair"])) if other in rows and other != key else "" for other in ("R", "T")]
            lines.append([p, text + marked, f"{row['failures_X']}, {row['failures_Z']}", f"{row['failures_pair']} of {row['shots']:,}",
                          rate_text(row["rate"], row["error"]), cells[0], cells[1]])
            records.append({"p": p, "key": key, "row": text, "failures_X": row["failures_X"], "failures_Z": row["failures_Z"],
                            "failures_pair": row["failures_pair"], "pairs": row["shots"], "rate_per_cycle": row["rate"]})
            fact(f"ens{p[-1]}_{key}_pairs", row["failures_pair"], "tables/exploration_ensemble.md")
        fact(f"ens{p[-1]}_shots", f"{next(iter(rows.values()))['shots']:,}", "tables/exploration_ensemble.md")
    if not lines:
        return write("exploration_ensemble", NOT_MEASURED)
    return write("exploration_ensemble",
                 "**Exploration shots, not test shots**: the team's validation files (5,000 shots per basis), each member's saved answers combined offline by "
                 "`hierarchy.ensemble`. Members: R relay-BP, P1 and P2 PriorTilt under two seeds, T TCBP-JOBS (GPU), J joint relay-BP on the shots where "
                 "P1, P2 and T are not unanimous. A rule that names \"the network\" uses the learned tie-break; the others need no learning of ours. "
                 "An ORACLE is given the truth: a ceiling, not a decoder. A pair is shot k of X with shot k of Z.\n\n"
                 + markdown(["p", "Row", "Failures X, Z", "Failed pairs", "Rate per cycle ± error",
                             "Against relay-BP alone: both fail / only this / only relay-BP (exact p)",
                             "Against TCBP-JOBS alone: both fail / only this / only TCBP-JOBS (exact p)"], lines)
                 + "\n\nSource: `report/data/exploration/ensemble/ens_p<p>/validation_{X,Z}.npz` and `.json`.", records)


def exploration_p5_table():
    """p = 0.005 on exploration shots: the team's validation_X file, the first look at the hierarchy at that noise."""
    shots = loaders.selector_files(DATA / "exploration" / "priortilt_p0.005", ["selector_validation_X.npz"])
    if shots is None:
        return write("exploration_p0.005", NOT_MEASURED)
    failed, agreed, seconds = loaders.weight_rule(shots)
    reference, lines = shots["failed_R"].astype(bool), []
    for label, whose, mask, time in (("relay-BP (kit baseline)", "the kit", reference, shots["seconds_R"]),
                                     ("PriorTilt, seed A", "IonQ", shots["failed_P1"].astype(bool), shots["seconds_P1"]),
                                     ("PriorTilt, seed B", "IonQ", shots["failed_P2"].astype(bool), shots["seconds_P2"]),
                                     ("our hierarchy, lowest-weight tie-break", "ours", failed, seconds),
                                     ("our hierarchy, lowest weight among the converged candidates", "ours", *loaders.weight_rule(shots, converged_only=True)[::2])):
        count, total = int(mask.sum()), len(mask)
        lines.append([label, whose, f"{count} of {total:,}", percent(count, total), "the reference" if mask is reference else pair_text(stats.paired(reference, mask)),
                      ms(float(np.mean(time)) * 1e3)])
        name = {"relay-BP (kit baseline)": "R", "PriorTilt, seed A": "P1", "PriorTilt, seed B": "P2", "our hierarchy, lowest-weight tie-break": "W"}.get(label, "C")
        fact(f"expl5_X_{name}", count, "tables/exploration_p0.005.md")
    fact("expl5_disagree", int((~agreed).sum()), "tables/exploration_p0.005.md")
    fact("expl5_shots", f"{len(agreed):,}", "tables/exploration_p0.005.md")
    return write("exploration_p0.005",
                 f"**Exploration shots, not test shots**: X memory at p = 0.005, the team's `data/p0.005/validation_X.npz` ({len(agreed):,} shots), decoded on the "
                 f"laptop beside a GPU training (compare the times with each other only; a time is per shot of one basis). The two PriorTilt runs disagree on "
                 f"{int((~agreed).sum())} shots ({(~agreed).mean():.1%}).\n\n"
                 + markdown(["Decoder", "Built by", "Failures", "Share of shots (95% interval)",
                             "Against relay-BP, same shots: both fail / only this row / only relay-BP (exact p)", "Mean time per shot, ms"], lines)
                 + "\n\nSource: `report/data/exploration/priortilt_p0.005/selector_validation_X.npz`.")


# ---------- loss ----------

def loss_table():
    pc, server = loaders.loss_pc(DATA / "loss" / "pc"), loaders.loss_server(DATA / "loss" / "server")
    lines, records = [], []
    labels = {condition.key: condition.label for condition in LOSS_CONDITIONS}
    datasets = sorted({(row["s"], row["machine"], row["seed"]) for row in list(pc.values()) + list(server.values())})
    for s, machine, seed in datasets:
        source = {key[-1]: row for key, row in (pc if machine == "this PC" else server).items() if row["s"] == s and row["seed"] == seed}
        for condition in LOSS_CONDITIONS:
            row = source.get(condition.key)
            if row is None:
                if condition.key in ("blind", "starter"):
                    lines.append([f"{s:g}", f"{machine}, seed {seed}", condition.label] + [NOT_MEASURED] * 3)
                continue
            versus = row.get("paired_by_reference", {}).get("blind" if condition.key == "starter" else "starter")
            paired = "" if not versus else (f"{versus['reference']} only {versus['reference_only']}, this row only {versus['row_only']} "
                                           f"(p = {stats.p_text(versus['p'])}); {versus['reference']} ÷ this = {versus['ratio']:.2f} "
                                           f"({versus['ratio95'][0]:.2f} to {versus['ratio95'][1]:.2f}), on {versus['shots']:,} shots")
            lines.append([f"{s:g}", f"{machine}, seed {seed}", condition.label, f"{row['failures']} of {row['shots']:,}",
                          percent(row["failures"], row["shots"]), paired.replace("starter", "flag-aware").replace("blind", "loss-blind")])
            records.append({"s": s, "machine": machine, "seed": seed, "condition": condition.key, "label": labels[condition.key],
                            "failures": row["failures"], "shots": row["shots"], "source": row["source"],
                            "paired_reference": versus and versus["reference"], "reference_only": versus and versus["reference_only"],
                            "row_only": versus and versus["row_only"], "ratio": versus and versus["ratio"]})
    for key, name in (("blind", "blind"), ("starter", "aware"), ("oracle-round", "oracle")):
        row = pc.get((0.001, key))
        fact(f"loss_{name}", row and f"{row['failures']} of {row['shots']:,}", "tables/loss.md")
        fact(f"loss_{name}_share", row and f"{row['failures'] / row['shots']:.1%}", "tables/loss.md")
    fact("loss_lost_atoms", pc.get((0.001, "starter")) and f"{pc[(0.001, 'starter')]['lost_data_atoms']:.1f}", "data/loss/pc/L1_X_p0.004_s1e-3.json")
    versus = (pc.get((0.001, "oracle-round")) or {}).get("paired_by_reference", {}).get("starter")
    fact("loss_oracle_ratio", versus and f"{versus['ratio']:.2f}", "tables/loss.md")
    fact("loss_oracle_ratio_interval", versus and f"{versus['ratio95'][0]:.2f} to {versus['ratio95'][1]:.2f}", "tables/loss.md")
    versus = (pc.get((0.001, "starter")) or {}).get("paired_by_reference", {}).get("blind")
    fact("loss_aware_ratio", versus and f"{versus['ratio']:.1f}", "tables/loss.md")
    text = ("X memory, p = 0.004, relay-BP at the kit's settings on shots simulated with Clifft under **our own simplified loss model** "
            "(after every CX each of its two atoms is lost with probability s; flags are perfect). It is not Infleqtion's hardware model. "
            "These are exploration numbers on shots made for this study; a shot here is one X-memory shot, not a pair. "
            "Each loss rate and machine has its own dataset, so rows of different datasets are not paired with each other.\n\n"
            + markdown(["s per atom per CX", "Dataset", "Decoder", "Failures", "Share of shots (95% interval)", "Paired on the same shots"], lines)
            + "\n\nAn ORACLE row is given the true loss round on purpose: it is a ceiling for a tracker of the loss time, not a decoder result. "
              "Sources: `report/data/loss/pc/*.json`, `report/data/loss/server/*/tables.md`.")
    return write("loss", text, records)


# ---------- the measurements behind the design, and what did not work ----------

def background_table():
    folder = DATA / "background"
    lines = []

    def add(what, result, file):
        lines.append([what, result, f"`{file}`"])

    file = folder / "g2_server_p0.004_summary.json"
    if file.is_file():
        found = json.loads(file.read_text(encoding="utf-8"))["candidates"]
        wrong = [found[f"p0.004_{basis}"]["converged_wrong"] for basis in "XZ"]
        shots, present, lowest = (sum(part[key] for part in wrong) for key in ("shots", "any_right", "lowest_right"))
        for name, value in (("cand_shots", shots), ("cand_present", present), ("cand_lowest", lowest)):
            fact(name, value, "tables/background.md")
        add("Is the right answer among 16 relay-BP candidates (other random draws), on the shots where relay-BP settled on a wrong answer? p = 0.004",
            f"it is among them on {present} of {shots} shots ({present / shots:.0%}); the lowest weight picks it on {lowest} ({lowest / shots:.0%}). "
            "The information is in the candidates and weight is a poor score: room for a learned one", "background/g2_server_p0.004_summary.json")
    file = folder / "g2_summary.json"
    if file.is_file():
        joint = json.loads(file.read_text(encoding="utf-8"))["tables"]["joint"]
        for p in ("0.003", "0.004"):
            parts = [joint[f"p{p}_{basis}"] for basis in "XZ"]
            fixes, failures, breaks, controls = (sum(part[key] for part in parts) for key in ("fixes", "F", "breaks", "C"))
            slow = sum(part["ms_C"]["mean"] for part in parts) / len(parts) / 1000
            add(f"Joint decoding (X and Z checks together, relay-BP on all 1,728 detectors) on relay-BP's failures, p = {p}",
                f"fixes {fixes} of {failures} failures and breaks {breaks} of {controls} shots relay-BP had right; about {slow:.0f} s per ordinary shot "
                "on our PC (8 processes), far too slow to run on every shot", "background/g2_summary.json")
    file = folder / "g3_frontier_p0.004.csv"
    if file.is_file():
        with file.open(encoding="utf-8") as stream:
            schedules = {row["schedule"]: row for row in csv.DictReader(stream)}
        kit = schedules["80/60/5"]
        for name, words in (("80/60/3", "stopping relay-BP after 3 solutions instead of 5"), ("80/60/1", "stopping relay-BP at its first solution"),
                            ("40/30/1", "a fast relay-BP schedule (first leg 40, later legs 30, stop at the first solution)")):
            row = schedules[name]
            add(f"A cheaper relay-BP: {words}; p = 0.004, the 5,000 validation pairs",
                f"{row['pair_failures']} failed pairs against the kit's {kit['pair_failures']} (only the kit's fails {row['pair_only_kit_wrong']}, "
                f"only this one {row['pair_only_this_wrong']}, p = {stats.p_text(float(row['pair_paired_p']))}), "
                f"{float(row['speedup_vs_kit_mean_iterations']):.2f} times fewer iterations", "background/g3_frontier_p0.004.csv")
    file = folder / "relay_schedule_paired_X_p0.004_5000_seed1.log"
    if file.is_file():
        text = file.read_text(encoding="utf-8")
        counts = re.findall(r"kit    scoring:\s+(\d+) of (\d+) failures", text)
        paired = re.search(r"kit    scoring: both fail (\d+), only A (\d+), only B (\d+), neither \d+\. Exact McNemar p = (\S+)", text)
        add("The same fast schedule (40 / 30 / 1) as a fallback, X memory, p = 0.004, 5,000 shots sampled for that comparison",
            f"{counts[1][0]} failures against the kit's {counts[0][0]} of {counts[0][1]} (only the kit's fails {paired.group(2)}, only the fast one "
            f"{paired.group(3)}, p = {stats.p_text(float(paired.group(4)))})", "background/relay_schedule_paired_X_p0.004_5000_seed1.log")
    file = folder / "predecoder_real2_p0.004_300_300_300_20.log"
    if file.is_file():
        rows = dict(re.findall(r"^  (no predecoder|bp picks, LLR >= 12|model, 99\.9% precise)\s+commits.*?failures (\d+/\d+)", file.read_text(encoding="utf-8"), re.M))
        hard = dict(re.findall(r"^  (no predecoder|bp picks, LLR >= 12|model, 99\.0% precise)\s+commits.*?hard shots\s+(\S+)%", file.read_text(encoding="utf-8"), re.M))
        add("A classic predecoder (commit to sure faults first, pass the rest to relay-BP), p = 0.004, 300 shots sampled for that study",
            f"failures {rows['no predecoder']} without it, {rows['bp picks, LLR >= 12']} with BP's own confident picks, {rows['model, 99.9% precise']} with a "
            f"trained model; hard shots {hard['no predecoder']}% without, {hard['bp picks, LLR >= 12']}% and {hard['model, 99.0% precise']}% with: no gain",
            "background/predecoder_real2_p0.004_300_300_300_20.log")
    file = folder / "train_X_p0.0001_seed1.log"
    if file.is_file():
        last = re.search(r"done: (\d+) shots.*?p=(\S+), \d+ DEM shots: (\S+)% of shots wrong \(always-0 guess (\S+)%\)", file.read_text(encoding="utf-8"))
        add("A plain convolutional network that predicts the 12 logical bits (no structure of the code in it), 0.7 M parameters",
            f"learns at very low noise only: at p = {last.group(2)} it has {last.group(3)}% of shots wrong after {int(last.group(1)):,} training shots "
            f"(answering \"nothing flipped\" gets {last.group(4)}%); that is 30 to 50 times below the challenge's noise levels",
            "background/train_X_p0.0001_seed1.log")
    file = folder / "plain_cnn_runs" / "README.md"
    if file.is_file():
        for line in file.read_text(encoding="utf-8").splitlines():
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) == 4 and cells[0].startswith(("`x_4h`", "`x_1h_w32`")):
                add(f"The same kind of plain network at the challenge's noise, on the laptop GPU: run {cells[0]}, {cells[2]} parameters",
                    cells[3], "background/plain_cnn_runs/README.md")
    pc = loaders.loss_pc(DATA / "loss" / "pc")
    base, exact, stop = (pc.get((0.001, key)) for key in ("starter", "exact", "starter@0.5/stop1"))
    if base and exact and stop:
        for row, words, verdict in ((exact, "exact weights for where a lost atom can have been", "no gain"),
                                    (stop, "stopping relay-BP at its first solution", "more failures")):
            versus = row["paired_by_reference"]["starter"]
            add(f"Under loss (X memory, p = 0.004, s = 0.001, 4,000 shots): {words}",
                f"{row['failures']} failures against the flag-aware decoder's {base['failures']} (only flag-aware fails {versus['reference_only']}, "
                f"only this one {versus['row_only']}, p = {stats.p_text(versus['p'])}): {verdict}", row["source"])
    server = loaders.loss_server(DATA / "loss" / "server")
    for key, words in (("starter-nofinal", "without the final-readout flip of a flagged data atom"),
                       ("starter-nohook", "without the columns for a flagged check atom of the other basis"),
                       ("starter-ratebase", "without trusting \"no flag\" (every unflagged atom also gets loss priors)")):
        row, reference = server.get((0.001, 5300, key)), server.get((0.001, 5300, "starter"))
        versus = row and row["paired_by_reference"].get("starter")
        if row and versus:
            add(f"Under loss, a piece of the flag-aware rule left out: {words} (X memory, p = 0.004, s = 0.001, {versus['shots']:,} shots)",
                f"{row['failures']} of {row['shots']:,} failures; on the same shots only flag-aware fails {versus['reference_only']}, only this one "
                f"{versus['row_only']} (p = {stats.p_text(versus['p'])}); flag-aware ÷ this = {versus['ratio']:.2f} ({versus['ratio95'][0]:.2f} to {versus['ratio95'][1]:.2f})",
                row["source"])
    tables = {int(re.search(r"step(\d+)", path.name).group(1)): loaders.network_rule_table(path)
              for path in (DATA / "exploration" / "priortilt").glob("results_network_select_step*.txt")}
    if tables:
        steps = sorted(tables)
        tie = ", ".join(str(tables[step]["rows"]["b. network breaks ties"][0]) for step in steps)
        alone = ", ".join(f"{tables[step]['rows']['d. network alone'][0] / 25000:.1%}" for step in steps)
        add("Longer training of the X network, as a tie-breaker (X memory, p = 0.004, 25,000 exploration shots)",
            f"at training steps {', '.join(f'{step:,}' for step in steps)} the hierarchy fails {tie} shots, while the network alone goes from "
            f"{alone.split(', ')[0]} to {alone.split(', ')[-1]} of shots wrong: the hard shots are hard for the network too",
            "exploration/priortilt/results_network_select_step*.txt")
    text = ("Measurements that decided the design, and what we tried that did not work. **All on exploration shots or on small studies made for "
            "the purpose: none of this is a result on the test shots.** Each line names the copied file under `report/data/` it is read from.\n\n"
            + markdown(["Question or attempt", "What was measured", "File"], lines))
    return write("background", text)


# ---------- the networks ----------

def networks_table():
    lines, records = [], []
    scores = loaders.scores_md(DATA / "exploration" / "cascade_X_s1" / "SCORES.md")
    for name, where in (("cascade_X_s1", "a laptop GPU (RTX 4060, 8 GB)"), ("cascade_Z_s1", "the ICRN server (A100 slice, 20 GB)")):
        log = loaders.train_log(DATA / "networks" / f"{name}.train.log")
        if log is None or not log["reports"]:
            lines.append([name] + [NOT_MEASURED] * 8)
            continue
        last = log["reports"][-1]
        wrong = " / ".join(f"{last['wrong'][p]:.1%}" for p in P_VALUES)
        speed = sorted(log["shots_per_second"])[len(log["shots_per_second"]) // 2]
        start = "the X network's weights, mirrored" if log["warm_start"] else "random weights"
        lines.append([f"{log['basis']} memory (`{name}`)", where, f"width {log['width']}, depth {log['depth']}, {log['parameters']:,} parameters",
                      start, f"{last['step']:,} of {last['of']:,}", f"{last['step'] * log['batch']:,}", f"{speed:,.0f}",
                      f"{last['step'] * log['batch'] / speed / 3600:.1f}", wrong])
        fact(f"net_{log['basis']}_step", f"{last['step']:,}", "tables/networks.md")
        fact("net_parameters", f"{log['parameters']:,}", "tables/networks.md")
        for p in P_VALUES:
            fact(f"net_{log['basis']}_wrong{p[-1]}", f"{last['wrong'][p]:.1%}", "tables/networks.md")
        records.append({"network": name, "basis": log["basis"], "machine": where, "parameters": log["parameters"], "last_step_in_log": last["step"],
                        "planned_steps": last["of"], "shots_seen": last["step"] * log["batch"], "median_shots_per_second": speed,
                        **{f"wrong_p{p}": last["wrong"][p] for p in P_VALUES}, "source": log["source"]})
    text = ("The two networks, from their training logs (`report/data/networks/*.train.log`). \"Shots wrong\" is the share of shots with at least "
            "one of the 12 logical bits wrong, on error-model shots drawn for the training report: a training diagnostic, not a result on Stim shots.\n\n"
            + markdown(["Network", "Trained on", "Size", "Started from", "Training step (last in the copied log)", "Shots seen",
                        "Shots per second (median of the log)", "GPU hours so far (shots seen ÷ that speed)", "Shots wrong at p = 0.003 / 0.004 / 0.005"], lines))
    if scores:
        steps = sorted({step for step, _ in scores})
        files = [name for name in ("validation_X_p0.003", "train_X_p0.003", "validation_X_p0.004", "train_X_p0.004", "validation_X_p0.005", "train_X_p0.005")
                 if any((step, name) in scores for step in steps)]
        score_lines = [[f"{step:,}"] + [f"{scores[(step, name)][0]:.2%} of {scores[(step, name)][3]:,}" if (step, name) in scores else "" for name in files]
                       for step in steps]
        text += ("\n\nThe X network alone, on the team's Stim shot files (exploration shots; `report/data/exploration/cascade_X_s1/SCORES.md`), "
                 "share of shots wrong:\n\n" + markdown(["Training step"] + [f"`{name}`" for name in files], score_lines))
    return write("networks", text, records)


# ---------- resources ----------

def resources_table(measured):
    lines = []
    for key, decoder in DECODERS.items():
        rows = [measured[p][key] for p in P_VALUES]
        if not any(rows):
            lines.append([decoder.label, NOT_MEASURED, NOT_MEASURED, NOT_MEASURED, decoder.learned])
            continue
        first = next(row for row in rows if row)
        memory = " / ".join(f"{row['peak_rss_mb']:,.0f}" if row and row.get("peak_rss_mb") else NOT_MEASURED for row in rows)
        setup = " / ".join(f"{row['setup_seconds']:.1f}" if row and row.get("setup_seconds") is not None else NOT_MEASURED for row in rows)
        lines.append([decoder.label, f"{first['device']}, {first['workers']} process, 1 thread", memory, setup, decoder.learned])
    text = ("What each decoder needed for its row on the test shots. Memory is the peak resident memory of the evaluator's process (the decoder, "
            "the shot file and Python together), as the evaluator records it; start-up is building the decoder for both bases, outside the per-shot time. "
            "Values are for p = 0.003 / 0.004 / 0.005.\n\n"
            + markdown(["Decoder", "Runs on", "Peak memory, MB", "Start-up, s", "What is learned in it"], lines))
    machine = DATA / "machines" / "server_job_000.out"
    if machine.is_file():
        printed = machine.read_text(encoding="utf-8", errors="replace")
        cpus = re.search(r"cgroup cpu\.max: (\d+) (\d+)", printed)
        memory = re.search(r"^Mem:\s+(\d+)", printed, re.M)
        gpu = re.search(r"^(NVIDIA [^,]+),", printed, re.M)
        slice_ = re.search(r"/ (\d+)MiB", (DATA / "machines" / "server_nvidia-smi.txt").read_text(encoding="utf-8", errors="replace")
                           .split("MIG devices")[-1]) if (DATA / "machines" / "server_nvidia-smi.txt").is_file() else None
        if cpus and memory:
            fact("server_cpus", int(cpus.group(1)) // int(cpus.group(2)), "data/machines/server_job_000.out")
            fact("server_ram_gb", memory.group(1), "data/machines/server_job_000.out")
            fact("server_gpu", gpu.group(1) if gpu else None, "data/machines/server_job_000.out")
            fact("server_gpu_slice_mib", f"{int(slice_.group(1)):,}" if slice_ else None, "data/machines/server_nvidia-smi.txt")
            text += (f"\n\nThe server (the team's ICRN notebook): {FACTS['server_cpus'][0]} CPUs by quota, {FACTS['server_ram_gb'][0]} GB of memory, "
                     f"a slice of {FACTS['server_gpu_slice_mib'][0]} MiB of an {FACTS['server_gpu'][0]} "
                     "(`report/data/machines/server_job_000.out`, `server_nvidia-smi.txt`).")
    for p, file, _usable, _why in TCBP_OWN:
        if (DATA / file).is_file():
            fitted = re.findall(r"TCBP model fitted in (\S+) s", (DATA / file).read_text(encoding="utf-8", errors="replace"))
            fact(f"tcbp_fit{p[-1]}_s", " and ".join(fitted) if fitted else None, f"data/{file}")
    return write("resources", text)


# ---------- the checks ----------

def checks_text(measured, extra, cross_checks):
    lines = ["Checks made by `python -m report.make_tables` on this run:", ""]
    count = sum(row is not None for p in P_VALUES for row in measured[p].values())
    lines.append(f"- **Rows on the test shots:** {count} of {len(TEST_ROWS) * len(P_VALUES)} registered rows have a file under `report/data/test/`; the others print \"{NOT_MEASURED}\".")
    lines.append("- **Same shots for every row:** each measured row names the sha256 of the test files it decoded, and the run stops unless they are "
                 "the team's `data/p<p>/test_X.npz` and `test_Z.npz` (the values are in `report/rows.py`). A paired count is only made between such rows.")
    lines.append("- **The rate uses pairs and 12 cycles:** for each evaluator row the per-shot file is re-counted (X failures, Z failures, pair = X or Z) "
                 "and the rate per cycle is recomputed as 1 - (1 - P)^(1/12); the run stops if this differs from the evaluator's `summary.json`.")
    lines.append("- **Times are per pair:** mean, median and p99 are recomputed from `seconds_total_pair` (the whole decode of one X shot plus one Z shot, "
                 "every stage and rerun included) and must equal the evaluator's summary.")
    lines.append("- **The paired count agrees with the team's `evaluation/compare_gain.py`:** `report/test_report.py` recomputes the two committed "
                 "comparisons (`results/icrn/gain_p0.003_portable`, `results/mac/gain_p0.004`) from the per-shot files and gets the same counts.")
    lines += [f"- **Cross-check with the teammates' earlier runs.** {text}" for text in cross_checks]
    if extra:
        lines.append(f"- **Folders under `report/data/test/` that no table reads:** {', '.join('`' + name + '`' for name in extra)}.")
    return write("checks", "\n".join(lines))


def tcbp_statement(measured):
    """Where TCBP-JOBS stands against our decoder, said the same way wherever the comparison appears."""
    value = lambda name: FACTS[name][0]
    both = [p for p in P_VALUES if measured[p]["tcbp"] and measured[p]["hierarchy"]]
    ahead = [p for p in both if measured[p]["tcbp"]["failures_pair"] < measured[p]["hierarchy"]["failures_pair"]]
    behind = [p for p in both if measured[p]["tcbp"]["failures_pair"] > measured[p]["hierarchy"]["failures_pair"]]
    counts = lambda levels: ", ".join(f"{measured[p]['tcbp']['failures_pair']} against {measured[p]['hierarchy']['failures_pair']} at p = {p}" for p in levels)
    if both:
        parts = []
        if ahead:
            parts.append(f"TCBP-JOBS, NVIDIA's GPU decoder, fails fewer pairs than our decoder on the test shots ({counts(ahead)})")
        if behind:
            parts.append(("and more" if ahead else "TCBP-JOBS, NVIDIA's GPU decoder, fails more pairs than our decoder on the test shots") + f" ({counts(behind)})")
        same = [p for p in both if p not in ahead and p not in behind]
        if same:
            parts.append(f"the two fail equally many pairs at p = {', '.join(same)}")
        text = ("; ".join(parts) if not (ahead and behind) else " ".join(parts)) + "."
        text = text[0].upper() + text[1:]
    elif FACTS.get("expl_pairs_T", (NOT_MEASURED,))[0] != NOT_MEASURED:
        text = (f"TCBP-JOBS, NVIDIA's GPU decoder, is not measured on the test shots yet; on the exploration shots at p = 0.004 it fails fewer pairs than "
                f"our hierarchy ({value('expl_pairs_T')} against {value('expl_pairs_C')} of {value('expl_pairs')}).")
    else:
        text = "TCBP-JOBS, NVIDIA's GPU decoder, is not measured against our decoder on the same shots."
    fact("tcbp_statement", text, "tables/test_all.md, tables/exploration_p0.004.md")


def time_words(row):
    """A row's mean time per pair in words, with its state: measured and clean, not clean, or not timed in one process."""
    if row is None or row["mean_ms"] is None or row.get("offline"):
        return None
    words = f"{ms(row['mean_ms'])} ms per pair"
    return words if row.get("time_clean", True) else words + " (a time not clean: the server throttled the row)"


def findings(measured):
    """The paragraph "what the numbers say" of REPORT.md: the framing, our decoder against relay-BP at each p, and TCBP-JOBS."""
    parts = [] if FACTS.get("framing", (NOT_MEASURED,))[0] == NOT_MEASURED else [FACTS["framing"][0]]
    sentences = []
    for p in P_VALUES:
        ours, relay = measured[p].get("hierarchy"), measured[p].get(REFERENCE)
        if ours is None:
            continue
        text = f"at p = {p} our decoder fails {ours['failures_pair']} of {ours['shots']:,} pairs"
        if relay is not None and ours.get("paired"):
            paired = ours["paired"]
            text += (f", relay-BP {relay['failures_pair']} (only ours fails {paired['row_only']}, only relay-BP {paired['reference_only']}, "
                     f"exact p = {stats.p_text(paired['p'])})")
        mine, theirs = time_words(ours), time_words(relay)
        if mine:
            text += f", in {mine}" + (f" against relay-BP's {theirs}" if theirs else "")
        parallel = measured[p].get("hierarchy-parallel")
        if time_words(parallel):
            text += f"; with its two fast decodes side by side on two cores, {time_words(parallel)}"
        sentences.append(text)
    if sentences:
        parts.append("On the test shots, " + "; ".join(sentences) + ".")
    else:
        parts.append("Our decoder's rows on the test shots are not measured yet.")
    parts.append(FACTS["tcbp_statement"][0])
    fact("findings", " ".join(parts), "tables/test_p0.003.md, test_p0.004.md, test_p0.005.md, test_ablation.md")


# ---------- the section for the top-level README ----------

README_HEADING = "## Our part: a hierarchical decoder with a learned tie-breaker, and a study of atom loss"


def readme_section(measured):
    """The one section appended to the top-level README.md: at most 25 lines, every number from the tables above."""
    value = lambda name: FACTS[name][0]
    tcbp = value("tcbp_statement")
    lines = [README_HEADING, "",
             "Two runs of a fast decoder (IonQ's PriorTilt) settle the shots they agree on; where they disagree, relay-BP joins and a neural",
             "network picks the most likely answer (`challenge-1-ml-decoder-kit/hierarchy/`, `neural/`). " + value("framing"),
             f"Under atom loss (our own simplified model), relay-BP with the loss flags fails {value('loss_aware_share')} of shots where a loss-blind one",
             f"fails {value('loss_blind_share')}, and an ORACLE told when each atom was lost fails {value('loss_oracle_ratio')} times fewer still: the ceiling for a loss-time tracker (`loss/`).",
             ""]
    header = ["Decoder"] + [f"p = {p}" for p in P_VALUES]
    table = []
    for key in DECODERS:
        cells = []
        for p in P_VALUES:
            row = measured[p][key]
            cells.append(NOT_MEASURED if row is None else f"{stats.sci(row['rate'])} ± {stats.sci(row['error'], 1)}, "
                         + (f"{ms(row['mean_ms'])} ms" if row["mean_ms"] is not None and not row.get("offline") else "not timed"))
        table.append([DECODERS[key].label] + cells)
    lines += markdown(header, table).split("\n")
    lines += ["",
              "Test shots (10,000 pairs per p): logical error rate per cycle ± the kit's error, mean time per pair. " + tcbp,
              "", "![Logical error rate per cycle against p](challenge-1-ml-decoder-kit/report/figures/ler_vs_p.png)", "",
              "Write-up, with every table made by a script from committed files: [report/REPORT.md](challenge-1-ml-decoder-kit/report/REPORT.md). "
              "Everything we built and why: [information.md](information.md)."]
    if len(lines) > 25:
        raise Mismatch(f"the README section has {len(lines)} lines; the limit is 25")
    return write("readme_section", "\n".join(lines))


# ---------- copying the tables into the documents ----------

def inject(tables):
    for document in DOCUMENTS:
        if not document.is_file():
            continue
        text = document.read_text(encoding="utf-8")
        for name, table in tables.items():
            pattern = re.compile(rf"(<!-- BEGIN table:{re.escape(name)} -->\n).*?(\n<!-- END table:{re.escape(name)} -->)", re.S)
            text = pattern.sub(lambda found: found.group(1) + table + found.group(2), text)
        missing = sorted(set(re.findall(r"<!-- BEGIN table:(\S+) -->", text)) - set(tables))
        if missing:
            raise Mismatch(f"{document.name} asks for tables that do not exist: {missing}")
        unknown = sorted(set(re.findall(r"<!--f:(\w+)-->", text)) - set(FACTS))
        if unknown:
            raise Mismatch(f"{document.name} quotes facts that do not exist: {unknown}")
        text = re.sub(r"<!--f:(\w+)-->.*?<!--/f-->", lambda found: f"<!--f:{found.group(1)}-->{FACTS[found.group(1)][0]}<!--/f-->", text, flags=re.S)
        document.write_text(text, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)
    try:
        measured, extra, combined = load_test_rows()
        per_p, sweep = test_tables(measured)
        ladder, ablated = ladder_tables(combined)
        earlier, cross_checks = earlier_table(measured)
        exploration, _ = exploration_table()
        tables = {f"test_p{p}": per_p[p] for p in P_VALUES} | {"test_time": per_p["time"]} | {
            "test_sweep": sweep, "earlier_runs": earlier, "tcbp_own_shots": tcbp_own_table(), "exploration_p0.004": exploration,
            "test_comparisons": comparisons_table(measured), "test_ladder": ladder, "test_ablation": ablated,
            "where_failures": failure_split_table(combined), "exploration_p0.005": exploration_p5_table(),
            "exploration_ensemble": exploration_ensemble_table(),
            "loss": loss_table(), "networks": networks_table(), "background": background_table(),
            "resources": resources_table(measured)}
        tables["checks"] = checks_text(measured, extra, cross_checks)
        tcbp_statement(measured)
        findings(measured)
        readme_section(measured)
        write("facts", "Single numbers quoted in the prose of `REPORT.md` and `SLIDES.md` (written there as `<!--f:name-->value<!--/f-->`), "
                       "each with the generated table it is taken from.\n\n"
              + markdown(["Name", "Value", "From"], [[f"`{name}`", value, f"`{source}`"] for name, (value, source) in sorted(FACTS.items())]))
        inject(tables)
    except Mismatch as problem:
        print(f"STOPPED: {problem}", file=sys.stderr)
        return 1
    count = sum(row is not None for p in P_VALUES for row in measured[p].values())
    print(f"wrote {len(tables) + 1} tables to {TABLES}/; {count} of {len(TEST_ROWS) * len(P_VALUES)} test rows measured")
    for name in extra:
        print(f"  note: report/data/test/{name} is not a registered row")
    return 0


if __name__ == "__main__":
    sys.exit(main())
