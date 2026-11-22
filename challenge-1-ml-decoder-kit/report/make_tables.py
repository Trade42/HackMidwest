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
from report.rows import (COMPARISONS, DECODERS, EARLIER_ROWS, EXTENSION, PTCBP_LABEL, PTCBP_TEST, LADDER, LOSS_CONDITIONS, OPTIONAL_RUNGS, OPERATING_PAIRED, P_VALUES, PREDECODER_FALLBACK, PREDECODER_FALLBACK_LABEL, PREDECODER_LABEL, PREDECODER_TEST,
                         REFERENCE, TCBP_OWN, TEST_ROWS, TEST_SHA256)

HERE = Path(__file__).resolve().parent
# REPORT_OUT sends the tables elsewhere and leaves the documents alone: used by the tests only.
TABLES = Path(os.environ.get("REPORT_OUT") or HERE) / "tables"
DOCUMENTS = [] if os.environ.get("REPORT_OUT") else [HERE / "REPORT.md", HERE / "APPENDIX.md", HERE / "SLIDES.md", HERE / "REPRODUCE.md"]
NOT_MEASURED = "not measured"
# A timed row's time is "not clean" when the notebook's CPU quota throttled it for more than this share of the row's
# own wall time (the counts are not affected, only the time). The kernel's throttled seconds are summed over the
# notebook's per-CPU run queues, and the quota (12 CPUs) throttles only when at least that many CPUs' worth is busy,
# so one process's throttled share is estimated as throttled seconds / (quota x wall time).
THROTTLE_LIMIT = 0.05

# Rows run or read after the test results were seen, at the team's request. They are ablations: none of them is our decoder,
# and none may change its headline, its framing or the frozen choice of its rules (choosing on the test shots now would fit them).
AFTER_TEST = "run after the test results were seen, at the team's request; an ablation, not our decoder"
AFTER_SHORT = "run after the test results were seen (an ablation)"
SWAP_RULE = "d"   # the ensemble's pre-registered rule closest to TCBP-JOBS in place of relay-BP in our hierarchy

# Single numbers that the prose of REPORT.md and SLIDES.md quotes. A document writes
# <!--f:name-->value<!--/f--> and this script puts the current value between the two comments,
# so a sentence can never keep a number that the files no longer support. tables/facts.md lists them all.
FACTS = {}
LEFT_OUT = []   # rows registered after the test results that failed a check and are left out of every table


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


def attach_run(row, run, monitor, jobs, quota=None):
    """What the server's own records say about a timed row's run: what ran beside it, the throttling, whether its time is clean."""
    row["during"] = loaders.machine_during(monitor, run["start"], run["end"]) if run and run["start"] and run["end"] else None
    if row["during"] and {"START", "END"} <= set(run["state"]):     # the job's own lines give the seconds throttled during the row exactly
        row["during"]["throttled_s"] = run["state"]["END"]["throttled_s"] - run["state"]["START"]["throttled_s"]
    row["throttle_share"] = None
    if row["during"] and run["end"] and run["start"]:
        wall = (run["end"] - run["start"]).total_seconds()
        row["during"]["wall_s"] = wall
        row["during"]["quota"] = quota
        row["throttle_share"] = row["during"]["throttled_s"] / ((quota or 1) * wall) if wall else None
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
    attach_run(row, log.get((p, folder.name.removeprefix(f"eval_p{p}_"))), monitor, jobs, loaders.cpu_quota(DATA / "test" / "cpu_monitor.out"))
    row["retimed"] = {"first": first, "folder": str(folder.relative_to(DATA)), "source": again["source"]}
    row["source"] = row["source"] + " (counts) and " + again["source"] + " (time)"


def load_test_rows(data=DATA):
    """({p: {decoder key: row or None}}, unregistered folders, {p: the ensemble tool's offline output or None}).

    Every row is checked against the test files' identity. A row without an evaluator run can be
    stood in for by the offline combination of per-shot files (rows.py, `member`); it is then
    marked as such, and its time is a sum of decode calls.
    """
    log = loaders.clean_rows_log(data / "test" / "clean_rows.out", data / "test" / "test_rows.out", data / "test" / "clean_retime.out")
    jobs = loaders.runner_logs(sorted((data / "test").glob("runner_*.out"))) if (data / "test").is_dir() else {}
    monitor = loaders.cpu_monitor(data / "test" / "cpu_monitor.out")
    quota = loaders.cpu_quota(data / "test" / "cpu_monitor.out")
    test_end = loaders.test_run_end(data / "test" / "test_rows.out")
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
                found = [tuple(map(int, hit)) for hit in re.findall(r"J exists on (\d+) of the (\d+)", row.get("note", ""))]
                ran, disputed = sum(hit[0] for hit in found), sum(hit[1] for hit in found)
                tcbp = combined[p]["rows"].get("T")
                same = tcbp is not None and all(np.array_equal(row["failed"][basis], tcbp["failed"][basis]) for basis in "XZ")
                row["same_as_tcbp"] = same
                row["stand_in_note"] = (" (rule f, frozen on the training shots: where PriorTilt twice and TCBP-JOBS agree, that answer; elsewhere "
                                        "joint decoding where it converged, else TCBP-JOBS"
                                        + (f"; joint decoding ran on {ran} of the {disputed} disputed shots" if found else "")
                                        + ("; so it fails exactly the shots TCBP-JOBS fails" if same else "") + ")")
                fact(f"ens_is_tcbp{p[-1]}", "yes" if tcbp and all(np.array_equal(row["failed"][b], tcbp["failed"][b]) for b in "XZ") else "no",
                     f"tables/test_p{p}.md")
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
                    if spec.rule in ("weight", "converged") and set(used.values()) != {spec.rule}:
                        raise Mismatch(f"{folder}: registered as the row with the {spec.rule} rule, but its summary records {used}")
                    if spec.rule == "network" and "network" not in used.values():
                        raise Mismatch(f"{folder}: registered as the row with the network, but its summary records {used}")
                    if spec.rule == "network" and set(used.values()) != {"network"}:
                        row["label_note"] = " (" + ", ".join(f"{basis} memory: " + {"network": "the network", "weight": "lowest weight",
                                                                                     "converged": "lowest weight among the converged"}[used[basis]] for basis in "XZ") + ")"
                run = None if row.get("offline") else log.get((p, spec.folder.format(p=p).removeprefix(f"eval_p{p}_")))
                attach_run(row, run, monitor, jobs, quota)
                row["after_test"] = bool(run and run["start"] and test_end and run["start"] > test_end)
                if row["after_test"]:            # ctl 120 and later: labelled wherever the row appears
                    if run["end"] is None or run["exit"] != 0:
                        raise Mismatch(f"{folder}: run after the test results, but test_rows.out has no END line with exit 0 for it")
                    row["label_note"] += f" ({AFTER_TEST})"
                    if spec.member and combined[p] and spec.member in combined[p]["rows"]:   # the pre-registered offline decode it replaces in the tables
                        row["offline_before"] = combined[p]["rows"][spec.member]["failures_pair"]
                retime = folder.with_name(folder.name + RETIME)
                if spec.kind == "evaluator" and not row.get("offline") and (retime / "summary.json").is_file():
                    take_retime(row, retime, log, monitor, jobs, p)
            measured[p][spec.decoder] = row
        for single, double in (("hierarchy", "hierarchy-parallel"), ("hierarchy-converged", "hierarchy-converged-parallel")):
            one, two = measured[p].get(single), measured[p].get(double)
            if one and two and not one.get("offline") and one.get("per_shot_file") and two.get("per_shot_file"):   # the side-by-side row must be the same decoder, only faster
                with np.load(DATA / one["per_shot_file"], allow_pickle=False) as first, np.load(DATA / two["per_shot_file"], allow_pickle=False) as second:
                    same = all(np.array_equal(first[key], second[key]) for key in ("predictions_X", "predictions_Z"))
                if not same and double == "hierarchy-converged-parallel":     # an ablation added after the test: left out, said so in checks.md
                    LEFT_OUT.append(f"{DECODERS[double].short} at p = {p}: its predictions differ from the one-core row's, so it is left out")
                    measured[p][double] = None
                    continue
                if not same:
                    raise Mismatch(f"p = {p}: the two-core row {double} predicts other bits than the one-core row; it cannot be labelled the same decoder")
                two["same_answers"] = True
            elif two is not None and double == "hierarchy-converged-parallel":
                two["label_note"] += "; its answers are not yet checked against a timed one-core row"
        reference = measured[p].get(REFERENCE)
        for key, row in measured[p].items():
            if row is not None:
                row["paired"] = (stats.paired(reference["failed_pair"], row["failed_pair"])
                                 if key != REFERENCE and reference is not None and reference["failed_pair"] is not None
                                 and row["failed_pair"] is not None else None)
                row["interval"] = stats.per_cycle_interval(row["failures_pair"], row["shots"])
    known = {spec.folder.format(p=p) + suffix for spec in TEST_ROWS for p in P_VALUES for suffix in ("", RETIME)}
    known |= {f"ensemble_p{p}" for p in P_VALUES} | {f"net_p{p}" for p in P_VALUES} | {"predecoder", "predecoder_tcbp"}
    extra = sorted(path.name for path in (data / "test").glob("*") if path.is_dir() and path.name not in known) if (data / "test").is_dir() else []
    return measured, extra, combined


def where_text(row):
    if row.get("offline"):
        return f"{row['site']} ({row['device']} decoders, ICRN server); its time is {row['timer']}"
    text = f"{row['device']}, {row['workers']} process" + (f" on {row['cores']} cores" if row.get("cores", 1) > 1 else "") + f", {row['site']}"
    if row.get("retimed"):
        first = row["retimed"]["first"]
        text += (f"; **re-timed: a second run in a quiet window confirms the time** (`{row['retimed']['folder']}`); its answers equal the first "
                 f"run's on every pair, the time shown is the second run's, and the first run's mean was {ms(first['mean_ms'])} ms"
                 + (f", with the CPU quota throttling at most {first['throttle_share']:.1%} of one process's time" if first.get("throttle_share") is not None else ""))
    if row.get("during"):
        during = row["during"]
        others = ", ".join(f"{kind.replace('_', ' ')} {count}" for kind, count in during["other_max"].items() if count)
        text += (f"; while it ran: {during['cpus_mean']:.1f} CPUs busy on average (at most {during['cpus_max']:.1f}), up to {during['evaluators_max']} "
                 f"evaluator rows at once" + (f", other work of ours at most: {others}" if others else "")
                 + f", the CPU quota throttled the notebook for {during['throttled_s']:,.0f} s summed over its CPUs"
                 + (f" during the row's {during['wall_s']:,.0f} s, at most {row['throttle_share']:.1%} of one process's time"
                    if during.get("wall_s") and row.get("throttle_share") is not None else ""))
        if not row.get("time_clean", True):
            text += f"; **time not clean** (throttled more than {THROTTLE_LIMIT:.0%} of its wall time): the counts stand, the time does not"
    elif row.get("beside") is not None:
        text += f"; {len(row['beside'])} other evaluator rows running beside it" if row["beside"] else "; no other evaluator row beside it"
    if row.get("jobs_beside"):
        text += "; server jobs started in that time: " + ", ".join(row["jobs_beside"])
    if row["device"] == "GPU":
        idle = (row.get("run") or {}).get("gpu_idle")
        text += ("; no other process on the GPU at its start" if idle else "; other processes on the GPU at its start" if idle is False else "")
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
                same_timer = row["device"] == "CPU" or row["device"].startswith("CPU,")    # a GPU row times the decode call alone: no ratio with a CPU row
                ratio = row["mean_ms"] / base_ms if timed and clean and base_ms and same_timer else None
                ratio_text = ("time not clean" if timed and not clean
                              else "not comparable: another device and timer" if timed and not same_timer
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
                                   "" if not timed else "GPU" if row["device"] == "GPU" else f"{row['device']}, {row.get('cores', 1)} core" + ("s" if row.get("cores", 1) > 1 else "")])
                paired = row.get("paired") or {}
                record |= {"label_note": row["label_note"].strip(" ()"), "after_test": bool(row.get("after_test")),
                           "pairs": row["shots"], "failures_X": row["failures_X"], "failures_Z": row["failures_Z"],
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
                 and row.get("time_clean", True) and row["device"] != "GPU" else None, source)
        note = (f"Test shots, p = {p}: `data/p{p}/test_X.npz` and `test_Z.npz` (10,000 shots per basis, Stim seeds 301 and 302; "
                f"sha256 `{TEST_SHA256[p]['X'][:12]}…` and `{TEST_SHA256[p]['Z'][:12]}…`). Every measured row decoded these two files.")
        texts[p] = write(f"test_p{p}", note + "\n\n" + markdown(header, lines), [r for r in records if r["p"] == p])
    write("test_all", "\n\n".join(texts[p] for p in P_VALUES), records)
    shares = [row["throttle_share"] for p in P_VALUES for row in measured[p].values()
              if row and not row.get("offline") and row.get("throttle_share") is not None]
    fact("max_throttle", shares and f"{max(shares):.1%}", "tables/test_p<p>.md")
    beside = [row["during"]["evaluators_max"] for p in P_VALUES for row in measured[p].values()
              if row and not row.get("offline") and row.get("during")]
    fact("max_beside", beside and max(beside), "tables/test_p<p>.md")
    texts["time"] = write("test_time", "Times on the test shots, each beside the row's failures, so that a time is never read without its accuracy. "
                          "\"Mean time ÷ relay-BP's\" below 1 means faster than relay-BP on average. One process per row on the team's server; "
                          "a row on two cores or on a GPU says so. A time is marked \"not clean\" when the server's CPU quota throttled it for more than "
                          f"{THROTTLE_LIMIT:.0%} of its wall time, estimated as the kernel's throttled seconds (summed over the server's CPUs) divided by "
                          "the quota of 12 CPUs times the row's wall time; it is then given no ratio and is not drawn in the time figures.\n\n"
                          + markdown(["p", "Decoder", "Failed pairs", "Mean time per pair, ms", "Mean time ÷ relay-BP's", "Median, ms", "99th percentile, ms", "Runs on"],
                                     time_lines))

    # the sweep in one table: a decoder per row, a noise level per column
    own = {p: loaders.tcbp_log(DATA / file) for p, file, _usable, _why in TCBP_OWN if (DATA / file).is_file()}
    lines = []
    for key in DECODERS:
        cells = []
        for p in P_VALUES:
            row = measured[p][key]
            if row is not None and row.get("offline"):
                mark = "offline, no time" + ("; fails exactly TCBP-JOBS's shots" if row.get("same_as_tcbp") else "")
                cells.append(f"{rate_text(row['rate'], row['error'])} · {row['failures_pair']} pairs · {mark}")
            elif row is None and key == "tcbp" and p in own:
                cells.append(f"not measured on these shots; on its own shots: {stats.sci(own[p]['rate_printed'])} ± {stats.sci(own[p]['error_printed'], 1)}")
            elif row is None:
                cells.append(NOT_MEASURED)
            else:
                time = f"{ms(row['mean_ms'])} ms" if row["mean_ms"] is not None else "time " + NOT_MEASURED
                cells.append(f"{rate_text(row['rate'], row['error'])} · {row['failures_pair']} pairs · {time}"
                             + (f" · **{AFTER_SHORT}**" if row.get("after_test") else "")
                             + (f" · pre-registered offline decode: {row['offline_before']} pairs" if row.get("offline_before") is not None else ""))
        lines.append([DECODERS[key].label] + cells)
    slide_lines = []
    for key in ("bposd", "relay-bp", "router", "priortilt", "hierarchy", "hierarchy-parallel", "tcbp", "ensemble"):
        cells = []
        for p in P_VALUES:
            row = measured[p][key]
            if row is None:
                cells.append(NOT_MEASURED)
            elif row.get("offline"):
                cells.append(f"{row['failures_pair']:,} · offline")
            else:
                cells.append(f"{row['failures_pair']:,} · {ms(row['mean_ms'])} ms" if row["mean_ms"] is not None else f"{row['failures_pair']:,}")
        slide_lines.append([DECODERS[key].short + (", 1 core" if key == "hierarchy" else "")] + cells)
    sweep_slide = write("test_sweep_slide", markdown(["Decoder"] + [f"p = {p}" for p in P_VALUES], slide_lines)
          + "\n\nFailed pairs of 10,000 test pairs · mean time per pair. \"Offline\": combined from saved answers, no time. TCBP-JOBS runs on a GPU.")
    sweep = write("test_sweep", markdown(["Decoder"] + [f"p = {p}" for p in P_VALUES], lines)
                  + "\n\nEach cell: logical error rate per cycle ± the kit's error · failed pairs of 10,000 · mean time per pair. \"Offline\" marks a "
                    "row combined from saved per-shot answers, with no time measured in one process. "
                  + ("A cell marked \"" + AFTER_SHORT + "\" is a row the team asked for after the test results were seen: an ablation, not our decoder. "
                     if any(row and row.get("after_test") for p in P_VALUES for row in measured[p].values()) else "")
                  + "Test shots; `report/tables/test_p<p>.md` has the intervals, the paired counts and where each row ran.")
    return texts, (sweep, sweep_slide)


def pair_text(paired):
    return f"{paired['both']} / {paired['row_only']} / {paired['reference_only']} (p = {stats.p_text(paired['p'])})"


def ladder_tables(combined):
    """Whose gain is whose, and the ablation of the learned tie-break: both from the offline combination on the test files."""
    from scipy.stats import binomtest
    lines, ablation, records = [], [], []
    short = {}                                               # (key, label) -> {p: failed pairs}, for the compact table of REPORT.md
    pooled = {"R": 0, "P1": 0, "aw": 0, "a": 0, "floor": 0, "only_network": 0, "only_weight": 0, "levels": []}
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
            short.setdefault((label, whose), {})[p] = row["failures_pair"]
            previous = row
        for key, label in (("T", "TCBP-JOBS (NVIDIA, GPU), for reference"),):
            if key in rows:
                row = rows[key]
                lines.append([p, label, "NVIDIA; run by teammates", f"{row['failures_X']}, {row['failures_Z']}", f"{row['failures_pair']} of {row['shots']:,}",
                              rate_text(row["rate"], row["error"]), "not a rung", pair_text(stats.paired(rows["R"]["failed_pair"], row["failed_pair"]))])
                fact(f"lad{p[-1]}_{key}_pairs", row["failures_pair"], "tables/test_ladder.md")
                short.setdefault((label, "NVIDIA"), {})[p] = row["failures_pair"]
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
            for key, source_key in (("R", "R"), ("P1", "P1"), ("aw", classical), ("a", "a")):
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
        plural = lambda count: f"{count} pair" + ("" if count == 1 else "s")
        counts = (f"The classical tie-break ({pooled['rule']}) fails {plural(pooled['aw'])} and the network {pooled['a']}; only the network rule fails "
                  f"{plural(pooled['only_network'])}, only the classical rule {pooled['only_weight']}, exact p = {stats.p_text(p_value)}.")
        ablation.append([levels + " together", "pairs (the kit's score)", "", pooled["aw"], pooled["a"],
                         f"{pooled['only_network']} / {pooled['only_weight']}", stats.p_text(p_value)])
        further = 1 - pooled["a"] / pooled["aw"] if pooled["aw"] else 0.0
        gap = (pooled["aw"] - pooled["a"]) / (pooled["aw"] - pooled["floor"]) if pooled["aw"] > pooled["floor"] else 0.0
        drop = pooled["R"] - pooled["a"]
        priortilt_share = (pooled["R"] - pooled["P1"]) / drop if drop > 0 else None
        verdict = "better" if pooled["a"] < pooled["aw"] and p_value < 0.05 else "worse" if pooled["a"] > pooled["aw"] and p_value < 0.05 else "level"
        head = (f"On the test shots (one offline decode of each shot, the same candidates for every rung; p = {levels} together), relay-BP fails "
                f"{pooled['R']} pairs, PriorTilt run once (IonQ's decoder) {pooled['P1']}, and our decoder as frozen (two PriorTilt runs, relay-BP "
                f"where they disagree, the neural tie-break) {pooled['a']}; a perfect tie-break would fail {pooled['floor']}. ")
        if priortilt_share is not None:
            head += (f"Of the drop from relay-BP to our decoder, {priortilt_share:.0%} is already there with PriorTilt run once; the hierarchy around it "
                     f"gives the rest. ")
        arm = (f"As an ablation, with the best hand-written tie-break ({pooled['rule']}) in place of the network, the hierarchy fails {pooled['aw']} "
               f"(only the network rule fails {plural(pooled['only_network'])}, only the hand-written rule {pooled['only_weight']}, "
               f"exact p = {stats.p_text(p_value)}): ")
        if verdict == "better":
            framing = head + arm + f"the network adds a further {further:.1%}, closing {gap:.0%} of the gap to a perfect tie-break."
        elif verdict == "worse":
            framing = head + arm + "the network does worse than the hand-written rule on the test shots."
        else:
            framing = head + arm + "the network is no better than the hand-written rule."
        for name, value in (("levels", levels), ("R", pooled["R"]), ("P1", pooled["P1"]), ("aw", pooled["aw"]), ("a", pooled["a"]),
                            ("floor", pooled["floor"]), ("only_network", pooled["only_network"]), ("only_weight", pooled["only_weight"]),
                            ("p", stats.p_text(p_value)), ("further", f"{further:.1%}"), ("gap_closed", f"{gap:.0%}"),
                            ("priortilt_share", None if priortilt_share is None else f"{priortilt_share:.0%}")):
            fact(f"abl_pooled_{name}", value, source)
        fact("framing", framing, source)
        fact("framing_slide", framing, source)
        words = {"better": "better than", "worse": "worse than", "level": "no better than"}[verdict]
        fact("abl_verdict", words, source)
        fact("slide3_title", f"The network is a poor decoder alone, and as a tie-break it is {words} the best hand-written rule", source)
        fact("slide3_close", {"better": f"So most of the gain over relay-BP comes from PriorTilt and the hierarchy, and the network adds a further {further:.1%}.",
                              "worse": "So the gain over relay-BP comes from PriorTilt and the hierarchy; on the test shots the network costs a little.",
                              "level": "So the gain over relay-BP comes from PriorTilt and the hierarchy; on the test shots the network does not add to it."}[verdict],
             source)
    else:
        fact("framing_slide", "pending: the ablation on the test shots has not landed.", source)
        for name in ("levels", "R", "P1", "aw", "a", "floor", "only_network", "only_weight", "p", "further", "gap_closed", "priortilt_share"):
            fact(f"abl_pooled_{name}", None, source)
        fact("framing", None, source)
        fact("slide3_title", "The network is a poor decoder alone; its value as a tie-break is decided on the test shots", source)
        fact("slide3_close", "The test shots decide whether the network adds to the hierarchy.", source)
        fact("abl_verdict", "not yet compared with", source)
    ladder = write("test_ladder",
                   "**Whose gain is whose, on the test shots.** Every rung is scored on the same per-shot answers (relay-BP and PriorTilt under two seeds "
                   "on every shot, a fresh decoder per 500 shots), so each step is a paired comparison with the rung before it. These rows are combined "
                   "offline from per-shot files; they are not the timed rows of the table above. Relay-BP keeps random state from shot to shot, so this "
                   "decode (a fresh decoder every 500 shots) and the timed row (one decoder through all 10,000 shots) differ: `checks.md` gives both "
                   "counts and the paired test between them. Compare rungs with rungs, and timed rows with timed rows.\n\n"
                   + markdown(["p", "Rung", "Whose work", "Failures X, Z", "Failed pairs", "Rate per cycle ± error",
                               "Against the rung before: both fail / only this / only the one before (exact p)",
                               "Against relay-BP: both fail / only this / only relay-BP (exact p)"], lines), records)
    levels_all = [p for p in P_VALUES if all(p in counts for counts in short.values())]
    write("test_ladder_short",
          "**Whose gain is whose, on the test shots**: failed pairs of 10,000 per noise level, every rung scored on the same per-shot answers "
          "(one offline decode: relay-BP and PriorTilt under two seeds on every shot, a fresh decoder per 500 shots; not the timed rows above). "
          "`APPENDIX.md` B has each rung's paired test against the rung before it and against relay-BP.\n\n"
          + markdown(["Rung", "Whose work"] + [f"p = {p}" for p in P_VALUES] + ["All three p"],
                     [[label, whose] + [counts.get(p, NOT_MEASURED) for p in P_VALUES]
                      + [sum(counts[p] for p in levels_all) if levels_all == list(P_VALUES) else ""]
                      for (label, whose), counts in short.items()]))
    ablated = write("test_ablation",
                    "**The ablation: what the learned part adds** (the offline decode of the ladder above, not the timed rows). The same hierarchy on the same candidates, with the best tie-break that needs no "
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
                lines.append([p, "", DECODERS[first].label, DECODERS[second].label] + [NOT_MEASURED] * 4)
                for part in ("both", "only_first", "only_second", "p"):
                    fact(f"{name}_{part}", None, "tables/test_comparisons.md")
                continue
            paired = stats.paired(b["failed_pair"], a["failed_pair"])
            mode = ("both timed" if not a.get("offline") and not b.get("offline") else "both offline" if a.get("offline") and b.get("offline")
                    else "**mixed**: a timed row against an offline combination, with other random draws")
            lines.append([p, mode, DECODERS[first].label + a["label_note"], DECODERS[second].label + b["label_note"],
                          f"{a['failures_pair']} against {b['failures_pair']}", paired["both"],
                          f"{paired['row_only']} / {paired['reference_only']}", stats.p_text(paired["p"])])
            records.append({"p": p, "first": first, "second": second, "first_after_test": bool(a.get("after_test")),
                            "second_after_test": bool(b.get("after_test")), "first_failed_pairs": a["failures_pair"], "second_failed_pairs": b["failures_pair"],
                            "both_fail": paired["both"], "only_first_fails": paired["row_only"], "only_second_fails": paired["reference_only"],
                            "exact_p": paired["p"]})
            for part, value in (("both", paired["both"]), ("only_first", paired["row_only"]), ("only_second", paired["reference_only"]),
                                ("p", stats.p_text(paired["p"]))):
                fact(f"{name}_{part}", value, "tables/test_comparisons.md")
    text = ("Paired comparisons on the test shots: two decoders on the same 10,000 pairs, and how many pairs only one of the two fails. "
            "The exact p is the two-sided binomial test on those pairs (under \"no difference\" each is a fair coin). "
            "This is where each gain is attributed: PriorTilt is IonQ's, the hierarchy and its tie-break are ours. A pair marked \"mixed\" compares a "
            "timed row with an offline combination; relay-BP's random state makes those two decodes differ (`checks.md`), so read it with care.\n\n"
            + markdown(["p", "Decodes", "First decoder", "Second decoder", "Failed pairs: first against second", "Both fail",
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


def time_split_table(measured):
    """Where our timed decoder's time goes, per noise level, from its per-shot stage times."""
    lines = []
    for p in P_VALUES:
        for key in ("hierarchy", "hierarchy-converged"):
            row = measured[p].get(key)
            if row is None or row.get("offline"):
                continue
            shots = loaders.plugin_shots(DATA / "test" / f"eval_p{p}_{key}")
            if shots is None:
                continue
            per_pair = lambda stage: sum(np.nan_to_num(shots[f"seconds_{stage}_{basis}"]) for basis in "XZ").mean() * 1e3
            disagree = sum(int((~shots[f"agreed_{basis}"]).sum()) for basis in "XZ")
            lines.append([p, DECODERS[key].label, f"{disagree} of {2 * len(shots['agreed_X']):,}", ms(per_pair("P1") + per_pair("P2")),
                          ms(per_pair("R")), ms(per_pair("network")), ms(row["mean_ms"]) + ("" if row.get("time_clean", True) else " (not clean)")])
            fact(f"split{p[-1]}_{key.replace('-', '_')}_disagree", disagree, "tables/time_split.md")
    return write("time_split", "Where our decoder's time goes on the test shots, per pair (an X shot and a Z shot), averaged over all 10,000 pairs. "
                 "The two PriorTilt runs decode every shot; relay-BP and the tie-break run only on the shots where the two runs disagree. "
                 "From the plugin's own per-shot timers; the last column is the evaluator's timer, which wraps everything.\n\n"
                 + markdown(["p", "Row", "Shots where the two runs disagree", "Two PriorTilt runs, ms", "relay-BP, ms", "the network, ms",
                             "Whole decode, ms (the evaluator)"], lines) if lines else NOT_MEASURED)


def failure_split(combined, measured=None):
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
        timed = (measured or {}).get(p, {}).get("hierarchy")
        shots = loaders.plugin_shots(DATA / "test" / f"eval_p{p}_hierarchy") if timed is not None else None
        if shots is not None and timed["failed_pair"] is not None:
            with np.load(DATA / timed["per_shot_file"], allow_pickle=False) as file:
                failed = np.concatenate([file["failed_X"].astype(bool), file["failed_Z"].astype(bool)])
            agreed = np.concatenate([shots["agreed_X"], shots["agreed_Z"]])
            sets[f"test shots, p = {p}, the timed decoder"] = (agreed, [("our decoder, neural tie-break", failed)])
            continue
        rows = combined[p]["rows"] if combined.get(p) else {}
        if "aw" in rows and rows["aw"]["later"] and "P1" in rows:
            join = lambda row: np.concatenate([row["failed"]["X"], row["failed"]["Z"]])
            agreed = ~np.concatenate([rows["aw"]["later"]["X"], rows["aw"]["later"]["Z"]]).astype(bool)
            sets[f"test shots, p = {p}"] = (agreed, [("one fast decode (PriorTilt, seed A)", join(rows["P1"])), ("the hierarchy, lowest-weight tie-break", join(rows["aw"]))]
                                            + ([("the hierarchy, neural tie-break", join(rows["a"]))] if "a" in rows else []))
    return {name: [("all shots", int(agreed.sum()), int((~agreed).sum()))]
                  + [(f"shots wrong under {label}", int((mask & agreed).sum()), int((mask & ~agreed).sum())) for label, mask in masks]
            for name, (agreed, masks) in sets.items()}


def failure_split_table(combined, measured=None):
    sets = failure_split(combined, measured)
    lines = [[name, what, f"{same + differ:,}", f"{same:,}", f"{differ:,}", f"{differ / (same + differ):.1%}" if same + differ else ""]
             for name, bars in sets.items() for what, same, differ in bars]
    timed = sets.get("test shots, p = 0.004, the timed decoder")
    source = "tables/where_failures.md"
    if timed:
        (_, same, differ), (_, fail_agree, fail_differ) = timed[0], timed[1]
        for name, value in (("tsplit_shots", f"{same + differ:,}"), ("tsplit_disagree", differ), ("tsplit_share", f"{differ / (same + differ):.2%}"),
                            ("tsplit_fail", fail_agree + fail_differ), ("tsplit_fail_disagree", fail_differ), ("tsplit_fail_agree", fail_agree)):
            fact(name, value, source)
    else:
        for name in ("tsplit_shots", "tsplit_disagree", "tsplit_share", "tsplit_fail", "tsplit_fail_disagree", "tsplit_fail_agree"):
            fact(name, None, source)
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


def ablation_80k_table():
    """The ablation on 80,000 fresh X shots at p = 0.004, never used for a fit: recomputed from the per-shot file, checked against its summary."""
    folder = DATA / "exploration" / "ablation_80k"
    if not (folder / "per_shot.npz").is_file():
        return write("ablation_80k", NOT_MEASURED)
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    with np.load(folder / "per_shot.npz", allow_pickle=False) as file:
        shots = {key: file[key] for key in file.files}
    total = len(shots["failed_a"])
    rows = [("P1", "PriorTilt, one run (seed A)", "IonQ"), ("P2", "PriorTilt, one run (seed B)", "IonQ"),
            ("aw", "our hierarchy, lowest-weight tie-break", "ours, nothing learned"),
            ("ac", "our hierarchy, lowest weight among the converged candidates", "ours, nothing learned"),
            ("a", "our hierarchy, neural tie-break (final X network)", "ours, learned"),
            ("perfect_tie_break", "ORACLE: a perfect tie-break on the same candidates (a ceiling, not a decoder)", "needs the truth")]
    listed = {key.split(":")[0].removeprefix("failed_"): value["failures"] for key, value in summary["rows"].items()}
    differ = shots["p1_p2_differ"].astype(bool)
    floor = shots["failed_perfect_tie_break"].astype(bool)
    chance = differ & ~floor                       # the shots where the two runs differ and some candidate is right
    lines = []
    for key, label, whose in rows:
        failed = shots[f"failed_{key}"].astype(bool)
        count = int(failed.sum())
        if key in listed and listed[key] != count:
            raise Mismatch(f"exploration/ablation_80k: {key} fails {count} shots in per_shot.npz and {listed[key]} in summary.json")
        right = "" if key in ("P1", "P2", "perfect_tie_break") else f"{int((chance & ~failed).sum())} of {int(chance.sum())}"
        lines.append([label, whose, f"{count} of {total:,}", percent(count, total, 3), right])
        fact(f"a80_{key}", count, "tables/ablation_80k.md")
    paired_lines = []
    for first, second in (("a", "aw"), ("a", "ac"), ("ac", "aw")):
        paired = stats.paired(shots[f"failed_{second}"], shots[f"failed_{first}"])
        name = f"{first} against {second}"
        theirs = summary["paired"].get(name)
        if theirs and (theirs["first_only"], theirs["second_only"]) != (paired["row_only"], paired["reference_only"]):
            raise Mismatch(f"exploration/ablation_80k: {name} discordant counts differ between per_shot.npz and summary.json")
        names = {"a": "the neural tie-break", "aw": "lowest weight", "ac": "lowest weight among the converged"}
        paired_lines.append([f"{names[first]} against {names[second]}", paired["both"], paired["row_only"], paired["reference_only"], stats.p_text(paired["p"])])
        for part, value in (("only_first", paired["row_only"]), ("only_second", paired["reference_only"]), ("p", stats.p_text(paired["p"]))):
            fact(f"a80_{first}_vs_{second}_{part}", value, "tables/ablation_80k.md")
    fact("a80_shots", f"{total:,}", "tables/ablation_80k.md")
    fact("a80_differ", int(differ.sum()), "tables/ablation_80k.md")
    fact("a80_chance", int(chance.sum()), "tables/ablation_80k.md")
    for key in ("a", "ac", "aw"):
        fact(f"a80_{key}_right", int((chance & ~shots[f"failed_{key}"].astype(bool)).sum()), "tables/ablation_80k.md")
    fact("a80_agree_wrong", int((~differ & floor).sum()), "tables/ablation_80k.md")
    fact("a80_network_alone", f"{shots['network_alone_failed'].mean():.1%}", "tables/ablation_80k.md")
    return write("ablation_80k",
                 f"**The ablation at scale, on exploration shots never used for a fit**: {total:,} fresh X-memory shots at p = 0.004 (Stim seeds "
                 f"{int(shots['file_seed'].min())} to {int(shots['file_seed'].max())}), decoded in session mode as the plugin decodes, the final X network "
                 f"(step 120,000). One shot is one X-memory shot, not a pair. The two PriorTilt runs differ on {int(differ.sum())} shots; on "
                 f"{int(chance.sum())} of them at least one candidate is right, and the last column counts how often each tie-break picks a right one there.\n\n"
                 + markdown(["Decoder", "Whose work", "Failures", "Share of shots (95% interval)", "Right pick where one existed"], lines)
                 + "\n\nPaired, on the same shots:\n\n"
                 + markdown(["Comparison", "Both fail", "Only the first fails", "Only the second fails", "Exact p"], paired_lines)
                 + "\n\nSource: `report/data/exploration/ablation_80k/per_shot.npz` (recomputed here) and `summary.json` (checked against it), "
                   "from branch `hierarchy` at commit `534d428`. Relay-BP alone was not decoded on these shots.")


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
    tracked, shares = loaders.loss_tracker(DATA / "loss" / "tracker")
    tracker_parts_check(tracked)
    tracker_facts(tracked, shares)
    server = server | tracked
    lines, records = [], []
    labels = {condition.key: condition.label for condition in LOSS_CONDITIONS}
    datasets = sorted({(row["s"], row["machine"], row["seed"]) for row in list(pc.values()) + list(server.values())})
    for s, machine, seed in datasets:
        source = {key[-1]: row for key, row in (pc if (machine == "this PC" and seed < 7000) else server).items() if row["s"] == s and row["seed"] == seed}
        for condition in LOSS_CONDITIONS:
            row = source.get(condition.key)
            if row is None:
                if condition.key == "starter" or (condition.key == "blind" and seed < 7000):     # the tracker's datasets had no loss-blind decoder
                    lines.append([f"{s:g}", f"{machine}, seed {seed}", condition.label] + [NOT_MEASURED] * 3)
                continue
            listed = row.get("paired_by_reference", {})
            versus = listed.get("blind") if condition.key == "starter" else (listed.get("starter") or listed.get("exact"))
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
    for file in sorted(folder.glob("results_hierarchy_heldout_step*.txt")):
        text = file.read_text(encoding="utf-8-sig")
        step = int(re.search(r"step(\d+)", file.name).group(1))
        found = re.findall(r"threshold fitted on train_X only \((strict|loose)[^)]*\): c = (\S+)\n\s+validation_X \(held out\): (\d+) shots \| coverage (\S+)% "
                           r"\((\d+) answered by the network\) \| failures hierarchy (\d+) \(\S+\) against base (\d+) \(\S+\) \| discordant: hierarchy-only wrong "
                           r"(\d+), base-only wrong (\d+), exact two-sided p = (\S+) \| mean ms/shot (\S+) \(base alone (\S+);", text)
        for kind, c, shots, coverage, _, ours, base, only_ours, only_base, p_value, time, base_time in found:
            add(f"\"Network first\": the network answers where its confidence is at least c, else the hierarchy (X memory, p = 0.005, network at step {step:,}; "
                f"c fitted on train_X with the {kind} rule, scored on validation_X, {int(shots):,} shots)",
                f"the network answers {coverage}% of the shots; {ours} failures against the hierarchy's {base} (only network-first fails {only_ours}, only the "
                f"hierarchy {only_base}, p = {p_value}); {time} against {base_time} ms per shot on the laptop beside a GPU training: no gain in accuracy",
                f"background/{file.name}")
    file = DATA / "exploration" / "ablation_80k" / "summary.json"
    if file.is_file():
        table = json.loads(file.read_text(encoding="utf-8"))
        base = next(value["failures"] for key, value in table["rows"].items() if key.startswith("failed_a:"))
        for key, words in (("c_procedure", "c fitted on train_X and validation_X"), ("c_train", "c fitted on train_X only")):
            entry = table.get("network_first", {}).get(key)
            if entry:
                paired = entry["paired_against_a"]
                add(f"\"Network first\" at p = 0.004 ({words}), on the {table['shots']:,} fresh X shots of the ablation at scale, never used for a fit",
                    f"the network answers {entry['answered']:.0%} of the shots; {entry['failures']} failures against the hierarchy's {base} with the neural "
                    f"tie-break (only network-first fails {paired['first_only']}, only the hierarchy {paired['second_only']}, p = {stats.p_text(paired['p_exact_two_sided'])}): "
                    "level in accuracy", "exploration/ablation_80k/summary.json")
    file = DATA / "loss" / "tracker" / "accuracy_val_learned.json"
    if file.is_file():
        versions = json.loads(file.read_text(encoding="utf-8"))["versions"]
        learned, model = versions.get("learned", {}).get("all"), versions.get("order1_explain1", {}).get("all")
        if learned and model:
            add("A learned loss-time tracker (a small recurrent network, a GRU, over the 13 detector rounds of a flagged atom) in place of the "
                f"tracker's statistical model (X memory, p = 0.004, s = 0.001, validation shots, {learned['atoms']:,} lost data atoms)",
                f"right round {learned['top1']:.1%} and within one round {learned['within1']:.1%} of the time, against {model['top1']:.1%} and "
                f"{model['within1']:.1%} for the model; mean log-probability of the true round {learned['mean_log_p_true']:.3f} against "
                f"{model['mean_log_p_true']:.3f}: it does not beat the model, so it was not decoded", "loss/tracker/accuracy_val_learned.json")
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


def tracker_parts_check(tracked):
    """The tracker's headline recomputed from its per-shot test parts (seed 7300): each condition's failures and the paired counts."""
    folder = DATA / "loss" / "tracker" / "parts" / "test"
    parts = sorted(folder.glob("*.npz")) if folder.is_dir() else []
    if not parts:
        fact("trk_parts", None, "data/loss/tracker/parts/test/")
        return None
    failed, conditions, shots = [], None, []
    for file in parts:
        with np.load(file, allow_pickle=False) as data:
            names = [str(name) for name in data["conditions"]]
            if conditions is not None and names != conditions:
                raise Mismatch(f"{file}: another list of conditions than the other parts")
            conditions = names
            failed.append(data["failed"].astype(bool))
            shots.append(data["shots"])
    failed, shots = np.concatenate(failed), np.concatenate(shots)
    if len(np.unique(shots)) != len(shots):
        raise Mismatch("loss/tracker/parts/test: a shot appears twice")
    counts = {name: int(failed[:, index].sum()) for index, name in enumerate(conditions)}
    for name, count in counts.items():
        row = tracked.get((0.001, 7300, name))
        if row is not None and (row["failures"], row["shots"]) != (count, len(shots)):
            raise Mismatch(f"loss/tracker/parts/test: {name} fails {count} of {len(shots)} in the parts and {row['failures']} of {row['shots']} in test.json")
    versus = stats.paired(failed[:, conditions.index("exact")], failed[:, conditions.index("tracker")])
    row = tracked.get((0.001, 7300, "tracker"))
    theirs = row and row["paired_by_reference"].get("exact")
    if theirs and (theirs["reference_only"], theirs["row_only"]) != (versus["reference_only"], versus["row_only"]):
        raise Mismatch("loss/tracker/parts/test: the paired counts of the tracker against exact differ from test.json")
    fact("trk_parts", f"recomputed from its per-shot test results ({len(parts)} files in `report/data/loss/tracker/parts/test/`)", "data/loss/tracker/parts/test/")
    return counts


def tracker_facts(tracked, shares):
    """The loss-time tracker's numbers for the prose: its test split first, the replication and the pooled share beside it."""
    source = "data/loss/tracker/test.json"
    test = {key[-1]: row for key, row in tracked.items() if key[1] == 7300}
    for name in ("starter", "exact", "tracker", "oracle-round"):
        row = test.get(name)
        fact(f"trk_{name.replace('-', '_')}", row and f"{row['failures']} of {row['shots']:,}", source)
        fact(f"trk_{name.replace('-', '_')}_share", row and f"{row['failures'] / row['shots']:.2%}", source)
    for reference in ("exact", "starter"):
        versus = (test.get("tracker") or {}).get("paired_by_reference", {}).get(reference)
        fact(f"trk_vs_{reference}", versus and (f"only the flag-aware decoder fails {versus['reference_only']} shots, only the tracker {versus['row_only']}, "
                                                f"sign test p = {stats.p_text(versus['p'])}; {versus['ratio']:.2f} times fewer failures with the tracker "
                                                f"({versus['ratio95'][0]:.2f} to {versus['ratio95'][1]:.2f})"), source)
    for name, label in (("test", "trk_gain"), ("replication_seed7310", "trk_gain_replication")):
        entry = shares.get(name, {}).get(("tracker", "exact"))
        fact(label, entry and f"{entry['share']:.0%} ({entry['share95'][0]:.0%} to {entry['share95'][1]:.0%})", f"data/loss/tracker/{name}.json")
    pooled = DATA / "loss" / "tracker" / "pooled_7300_7310.json"
    entry = None
    if pooled.is_file():
        table = json.loads(pooled.read_text(encoding="utf-8"))
        entry = next((e for e in table.get("gain_share", []) if e["condition"] == "tracker" and e["reference"] == "exact"), None)
    fact("trk_gain_pooled", entry and f"{entry['share']:.0%} ({entry['share95'][0]:.0%} to {entry['share95'][1]:.0%})", "data/loss/tracker/pooled_7300_7310.json")
    accuracy = DATA / "loss" / "tracker" / "accuracy_test.json"
    every = None
    if accuracy.is_file():
        table = json.loads(accuracy.read_text(encoding="utf-8"))
        every = table["versions"][table["main"]["version"]]["all"]
    fact("trk_top1", every and f"{every['top1']:.1%}", "data/loss/tracker/accuracy_test.json")
    fact("trk_within1", every and f"{every['within1']:.1%}", "data/loss/tracker/accuracy_test.json")
    fact("trk_atoms", every and f"{every['atoms']:,}", "data/loss/tracker/accuracy_test.json")
    timing = DATA / "loss" / "tracker" / "timing.json"
    frozen = None
    if timing.is_file():
        table = json.loads(timing.read_text(encoding="utf-8"))
        frozen = next((row for row in table["rows"] if row["version"].startswith("order1_explain1")), None)
    fact("trk_ms", frozen and f"{frozen['one_shot_mean_ms']:.1f}", "data/loss/tracker/timing.json")
    fact("trk_ms_p99", frozen and f"{frozen['one_shot_p99_ms']:.1f}", "data/loss/tracker/timing.json")
    secondary = DATA / "loss" / "tracker" / "test_secondary.json"
    pair = None
    if secondary.is_file():
        pair = next((entry for entry in json.loads(secondary.read_text(encoding="utf-8")).get("pairs", []) if entry["a"] == "tracker"), None)
    fact("trk_secondary", pair and (f"an earlier version of the tracker fails {pair['b_failures']} of the same shots against the frozen one's {pair['a_failures']} "
                                    f"(only the frozen one fails {pair['a_only']}, only the earlier one {pair['b_only']}, p = {stats.p_text(pair['sign_test_p'])})"),
         "data/loss/tracker/test_secondary.json")
    value = lambda name: FACTS[name][0]
    if test.get("tracker") and test.get("exact"):
        sentence = (f"On the tracker's own test shots (a new set of {test['tracker']['shots']:,} shots at s = 0.001, the tracker frozen on validation shots "
                    f"beforehand), relay-BP given the tracker's estimate fails {value('trk_tracker')}, against {value('trk_exact')} for the flag-aware "
                    f"decoder that does not know the loss time, with exact location weights ({value('trk_vs_exact')}), and {value('trk_oracle_round')} for the ORACLE. "
                    f"So the tracker recovers {value('trk_gain')} of the oracle's gain.")
        if value("trk_parts") != NOT_MEASURED:
            sentence += f" Each of these counts is {value('trk_parts')}."
        if value("trk_gain_replication") != NOT_MEASURED:
            sentence += f" A replication on the team's server, on other shots, gives {value('trk_gain_replication')}"
            sentence += f"; both together, {value('trk_gain_pooled')}." if value("trk_gain_pooled") != NOT_MEASURED else "."
        if value("trk_top1") != NOT_MEASURED:
            sentence += (f" Scored against the simulator's record of {value('trk_atoms')} lost data atoms, the tracker names the right round "
                         f"{value('trk_top1')} of the time and is within one round {value('trk_within1')} of the time.")
        if value("trk_secondary") != NOT_MEASURED:
            sentence += f" The result does not hang on which version was frozen: {value('trk_secondary')}."
    else:
        sentence = "Its rows on its own test shots are not measured yet."
    fact("trk_sentence", sentence, source)


def test_job_facts():
    """What the test job printed before it decoded the test shots: the rule chosen per basis on exploration shots."""
    file = DATA / "test" / "test_run.out"
    text = file.read_text(encoding="utf-8", errors="replace") if file.is_file() else ""
    source = "data/test/test_run.out"
    for basis in "XZ":
        found = re.search(rf"PLUGIN_RULE {basis} (\S+)\s+\(network (\d+) failures, converged (\d+), on (\d+) shots; logits of checkpoint step (\d+)\)", text)
        fact(f"rule_{basis}", found and found.group(1), source)
        fact(f"rule_{basis}_network", found and found.group(2), source)
        fact(f"rule_{basis}_converged", found and found.group(3), source)
        fact(f"rule_{basis}_shots", found and f"{int(found.group(4)):,}", source)
        fact(f"rule_{basis}_step", found and f"{int(found.group(5)):,}", source)
    frozen = re.search(r"TEST RUN DRIVER START: frozen (\w+)", text)
    fact("test_frozen_commit", frozen and frozen.group(1)[:12], source)


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
    alone = []
    for p in P_VALUES:
        cells = []
        for basis in "XZ":
            found = None
            for file in sorted((DATA / "test" / f"net_p{p}").glob("*.json")) if (DATA / "test" / f"net_p{p}").is_dir() else []:
                record = json.loads(file.read_text(encoding="utf-8"))
                if record.get("basis") == basis and record.get("shots") == 10000:
                    found = record
            cells.append(NOT_MEASURED if found is None else f"{found['failures']:,} of {found['shots']:,} ({found['failure_fraction']:.1%})")
            fact(f"netalone{p[-1]}_{basis}", found and f"{found['failure_fraction']:.1%}", f"data/test/net_p{p}/")
        alone.append([p] + cells)
    alone_text = ("\n\nThe final networks alone on the **test shots** (the team's test files, `neural.evaluate_stim`; `report/data/test/net_p<p>/`), "
                  "share of shots with any logical bit wrong. Alone they are far worse than relay-BP; they are only ever asked to choose among candidates.\n\n"
                  + markdown(["p", "X network (step 120,000)", "Z network (step 100,000)"], alone))
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
    text += alone_text
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
        after = [p for p, row in zip(P_VALUES, rows) if row and row.get("after_test")]
        lines.append([decoder.label + (f" (at p = {', '.join(after)}: {AFTER_TEST})" if after else ""),
                      f"{first['device']}, {first['workers']} process, 1 thread", memory, setup, decoder.learned])
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

def mode_checks(measured, combined):
    """The timed rows against the offline decode of the same shots: relay-BP against R, our decoder against the network rule a."""
    lines = []
    for p in P_VALUES:
        rows = combined[p]["rows"] if combined.get(p) else {}
        for key, member, words in ((REFERENCE, "R", "relay-BP"), ("hierarchy", "a", "our decoder (neural tie-break)")):
            timed = measured[p].get(key)
            if key == "hierarchy" and timed is None:
                timed = measured[p].get("hierarchy-parallel")
            if timed is None or timed.get("offline") or timed["failed_pair"] is None or member not in rows:
                continue
            paired = stats.paired(rows[member]["failed_pair"], timed["failed_pair"])
            lines.append(f"p = {p}: {words}, timed row (one decoder through all 10,000 shots) {timed['failures_pair']} failed pairs, offline decode "
                         f"(a fresh decoder every 500 shots) {rows[member]['failures_pair']}; only the timed one fails {paired['row_only']}, only the offline one "
                         f"{paired['reference_only']}, exact p = {stats.p_text(paired['p'])}"
                         + (" (below 0.05; we read it as the run-to-run spread of a randomized decoder: relay-BP draws random strengths from a "
                            "generator seeded once, so the two modes give each shot other draws, and the offline mode replays the same seed in every "
                            "500-shot chunk, which the paired test does not account for; a comparison that mixes the two modes is marked \"mixed\")"
                            if paired["p"] < 0.05 else ""))
        timed = measured[p].get("hierarchy")
        shots = loaders.plugin_shots(DATA / "test" / f"eval_p{p}_hierarchy") if timed is not None and not timed.get("offline") else None
        if shots is not None and "aw" in rows and rows["aw"]["later"]:
            online = np.concatenate([~shots["agreed_X"], ~shots["agreed_Z"]]).astype(bool)
            offline = np.concatenate([rows["aw"]["later"]["X"], rows["aw"]["later"]["Z"]]).astype(bool)
            lines.append(f"p = {p}: the two PriorTilt runs disagree on {int(online.sum())} of {len(online):,} shots in our timed row and on "
                         f"{int(offline.sum())} in the offline decode; {int((online & offline).sum())} of those shots are in both. Both use the same seeds and the "
                         "same test of agreement (the 12 predicted bits); PriorTilt draws random numbers from shot to shot, so the two modes give "
                         "other draws and other disputed shots")
    return lines


def checks_text(measured, extra, cross_checks, modes=()):
    lines = ["Checks made by `python -m report.make_tables` on this run:", ""]
    timed = [(p, key) for p in P_VALUES for key, row in measured[p].items() if row is not None and not row.get("offline")]
    offline = [(p, key) for p in P_VALUES for key, row in measured[p].items() if row is not None and row.get("offline")]
    missing = [(p, key) for p in P_VALUES for key, row in measured[p].items() if row is None]
    name = lambda pairs: "; ".join(f"{DECODERS[key].short} at p = {p}" for p, key in pairs) or "none"
    lines.append(f"- **Rows on the test shots:** of {len(TEST_ROWS) * len(P_VALUES)} registered rows, {len(timed)} are timed evaluator rows, "
                 f"{len(offline)} are combined offline from saved answers with no time in one process ({name(offline)}), and {len(missing)} "
                 f"print \"{NOT_MEASURED}\" ({name(missing)}).")
    lines.append("- **Same shots for every row:** each measured row names the sha256 of the test files it decoded, and the run stops unless they are "
                 "the team's `data/p<p>/test_X.npz` and `test_Z.npz` (the values are in `report/rows.py`). A paired count is only made between such rows.")
    lines.append("- **The rate uses pairs and 12 cycles:** for each evaluator row the per-shot file is re-counted (X failures, Z failures, pair = X or Z) "
                 "and the rate per cycle is recomputed as 1 - (1 - P)^(1/12); the run stops if this differs from the evaluator's `summary.json`.")
    lines.append("- **Times are per pair:** mean, median and p99 are recomputed from `seconds_total_pair` (the whole decode of one X shot plus one Z shot, "
                 "every stage and rerun included) and must equal the evaluator's summary.")
    lines.append("- **The paired count agrees with the team's `evaluation/compare_gain.py`:** `report/test_report.py` recomputes the two committed "
                 "comparisons (`results/icrn/gain_p0.003_portable`, `results/mac/gain_p0.004`) from the per-shot files and gets the same counts.")
    lines += [f"- **Cross-check with the teammates' earlier runs.** {text}" for text in cross_checks]
    lines += [f"- **Timed row against the offline decode of the same shots.** {text}." for text in modes]
    lines += [f"- **Left out:** {text}." for text in LEFT_OUT]
    after = [(p, key) for p in P_VALUES for key, row in measured[p].items() if row is not None and row.get("after_test")]
    if after:
        lines.append(f"- **Rows run after the test results were seen** (their START in `test_rows.out` is after its TEST RUN END line; each is labelled "
                     f"\"{AFTER_TEST}\" wherever it appears, and none is our decoder): {name(after)}.")
    if extra:
        lines.append(f"- **Folders under `report/data/test/` that no table reads:** {', '.join('`' + name + '`' for name in extra)}.")
    return write("checks", "\n".join(lines))


def tcbp_statement(measured):
    """Where TCBP-JOBS stands against our decoder, said the same way wherever the comparison appears; every clause names both."""
    value = lambda name: FACTS[name][0]
    both = [p for p in P_VALUES if measured[p]["tcbp"] and measured[p]["hierarchy"]]
    if both:
        clauses = []
        for p in both:
            tcbp, ours = measured[p]["tcbp"]["failures_pair"], measured[p]["hierarchy"]["failures_pair"]
            verb = "fewer" if tcbp < ours else "more" if tcbp > ours else "as many"
            text = (f"at p = {p} TCBP-JOBS fails {verb} pairs than our decoder ({tcbp} against {ours}" if verb != "as many"
                    else f"at p = {p} TCBP-JOBS and our decoder fail as many pairs ({tcbp} against {ours}")
            first, second = measured[p]["tcbp"]["failed_pair"], measured[p]["hierarchy"]["failed_pair"]
            if first is not None and second is not None:
                paired = stats.paired(second, first)          # reference: ours; row: TCBP-JOBS
                significant = "" if paired["p"] < 0.05 else ", not a significant difference"
                text += (f"; only TCBP-JOBS fails {paired['row_only']}, only ours {paired['reference_only']}, "
                         f"exact p = {stats.p_text(paired['p'])}{significant}")
            clauses.append(text + ")")
        text = "Against TCBP-JOBS, NVIDIA's GPU decoder, on the same test pairs: " + "; ".join(clauses) + "."
        against_relay = []
        for p in P_VALUES:
            tcbp, relay = measured[p]["tcbp"], measured[p].get(REFERENCE)
            if tcbp is None or relay is None or tcbp["failed_pair"] is None:
                continue
            paired = stats.paired(relay["failed_pair"], tcbp["failed_pair"])
            against_relay.append(f"{tcbp['failures_pair']} against {relay['failures_pair']} at p = {p} (exact p = {stats.p_text(paired['p'])})")
        if against_relay:
            text += " TCBP-JOBS against relay-BP's timed row: " + ", ".join(against_relay) + "."
    elif FACTS.get("expl_pairs_T", (NOT_MEASURED,))[0] != NOT_MEASURED:
        text = (f"TCBP-JOBS, NVIDIA's GPU decoder, is not measured against our decoder on the test shots yet; on the exploration shots at p = 0.004 "
                f"TCBP-JOBS fails fewer pairs than our hierarchy ({value('expl_pairs_T')} against {value('expl_pairs_C')} of {value('expl_pairs')}).")
    else:
        text = "TCBP-JOBS, NVIDIA's GPU decoder, is not measured against our decoder on the same shots."
    fact("tcbp_statement", text, "tables/test_all.md, tables/exploration_p0.004.md")


def tcbp_fit_caveat():
    """TCBP-JOBS fits its model at start-up and the fit varies from run to run: three fits on the same validation X shots at p = 0.005."""
    file = DATA / "exploration" / "tcbp_fits" / "validation_X_p0.005_fits.npz"
    if not file.is_file():
        fact("tcbp_fit_caveat", "", "data/exploration/tcbp_fits/validation_X_p0.005_fits.npz")
        return
    with np.load(file, allow_pickle=False) as data:
        counts = [int(data[f"failed_{key}"].sum()) for key in ("stored", "job1", "job3")]
        batch = bool(data["job3_batch_equals_per_shot"])
    spread = (max(counts) - min(counts)) / (sum(counts) / len(counts)) / 2
    fact("tcbp_fit_caveat", f"A caveat on every comparison with TCBP-JOBS: it fits its model when it starts, and the fit varies from run to run. On the same "
                            f"5,000 validation X shots at p = 0.005 (exploration shots), three fits with the kit's priors fail {counts[0]}, {counts[1]} and "
                            f"{counts[2]} shots" + (" (batch and per-shot decoding give the same answers, so the spread is the fit's)" if batch else "")
                            + f", about ±{spread:.0%} around their mean. A comparison with TCBP-JOBS's stored test file carries that spread; a fresh "
                            "arm in the same run is the like-for-like control.", "data/exploration/tcbp_fits/validation_X_p0.005_fits.npz")


def time_words(row):
    """A row's mean time per pair in words, with its state: measured and clean, not clean, or not timed in one process."""
    if row is None or row["mean_ms"] is None or row.get("offline"):
        return None
    words = f"{ms(row['mean_ms'])} ms per pair"
    return words if row.get("time_clean", True) else words + " (a time not clean: see the table)"


def findings(measured):
    """The paragraph "what the numbers say" of REPORT.md: the framing, our decoder against relay-BP at each p, and TCBP-JOBS."""
    parts = []
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
            text += f", or {time_words(parallel)} with its two fast decodes side by side on two cores"
        priortilt = measured[p].get("priortilt")
        if priortilt is not None and not priortilt.get("offline") and time_words(priortilt):
            text += f"; PriorTilt run once fails {priortilt['failures_pair']} in {time_words(priortilt)}"
        sentences.append(text)
    by_p = {sentence.split(" our decoder")[0].removeprefix("at p = "): sentence for sentence in sentences}
    for p in P_VALUES:
        text = by_p.get(p)
        fact(f"findings{p[-1]}", text and (text[0].upper() + text[1:] + "."), f"tables/test_p{p}.md")


NO_NETWORK = {"hierarchy-converged": "the best hand-written rule (lowest weight among the converged candidates)",
              "hierarchy-weight": "plain lowest weight"}


def network_timed_table(measured):
    """How much the network does, timed against timed: our decoder's timed row against the same hierarchy with a hand-written
    tie-break in its place, each through the team's evaluator on the same test pairs, in one process (and the same on two cores)."""
    from scipy.stats import binomtest
    source = "tables/test_network_timed.md"
    lines, records, pooled = [], [], {arm: {"levels": [], "after": [], "net": 0, "arm": 0, "only_net": 0, "only_arm": 0} for arm in NO_NETWORK}
    for p in P_VALUES:
        ours = measured[p].get("hierarchy")
        for arm_key, words in NO_NETWORK.items():
            arm = measured[p].get(arm_key)
            n = f"nt{p[-1]}_{arm_key.removeprefix('hierarchy-')}"
            if (ours is None or arm is None or ours.get("offline") or arm.get("offline")
                    or ours["failed_pair"] is None or arm["failed_pair"] is None):
                lines.append([p, "1 core", words, NOT_MEASURED + (" (only offline)" if arm is not None and arm.get("offline") else "")] + [""] * 4)
                for name in ("net", "arm", "only_net", "only_arm", "p", "ms_net", "ms_arm"):
                    fact(f"{n}_{name}", None, source)
                continue
            paired = stats.paired(arm["failed_pair"], ours["failed_pair"])
            mark = f" [{AFTER_SHORT}]" if arm.get("after_test") else ""
            lines.append([p, "1 core", words + mark, f"{ours['failures_pair']} against {arm['failures_pair']}", paired["both"],
                          f"{paired['row_only']} / {paired['reference_only']}", stats.p_text(paired["p"]),
                          f"{ms(ours['mean_ms'])} against {ms(arm['mean_ms'])}"])
            records.append({"p": p, "cores": 1, "without_network": arm_key, "after_test": arm.get("after_test"), "network_failed_pairs": ours["failures_pair"],
                            "without_failed_pairs": arm["failures_pair"], "both_fail": paired["both"], "only_network_fails": paired["row_only"],
                            "only_without_fails": paired["reference_only"], "exact_p": paired["p"], "network_mean_ms": ours["mean_ms"], "without_mean_ms": arm["mean_ms"]})
            for name, value in (("net", ours["failures_pair"]), ("arm", arm["failures_pair"]), ("only_net", paired["row_only"]),
                                ("only_arm", paired["reference_only"]), ("p", stats.p_text(paired["p"])), ("ms_net", ms(ours["mean_ms"])), ("ms_arm", ms(arm["mean_ms"]))):
                fact(f"{n}_{name}", value, source)
            bucket = pooled[arm_key]
            bucket["levels"].append(p)
            bucket["after"] += [p] if arm.get("after_test") else []
            bucket["net"] += ours["failures_pair"]; bucket["arm"] += arm["failures_pair"]
            bucket["only_net"] += paired["row_only"]; bucket["only_arm"] += paired["reference_only"]
        two, arm = measured[p].get("hierarchy-parallel"), measured[p].get("hierarchy-converged-parallel")
        if two is not None and arm is not None and two["failed_pair"] is not None and arm["failed_pair"] is not None:
            paired = stats.paired(arm["failed_pair"], two["failed_pair"])
            same = " (each gives its one-core row's answers on every shot)" if two.get("same_answers") and arm.get("same_answers") else ""
            lines.append([p, "2 cores", NO_NETWORK["hierarchy-converged"] + (f" [{AFTER_SHORT}]" if arm.get("after_test") else "") + same,
                          f"{two['failures_pair']} against {arm['failures_pair']}", paired["both"], f"{paired['row_only']} / {paired['reference_only']}",
                          stats.p_text(paired["p"]), f"{ms(two['mean_ms'])} against {ms(arm['mean_ms'])}"])
            fact(f"nt{p[-1]}_two_ms_net", ms(two["mean_ms"]), source)
            fact(f"nt{p[-1]}_two_ms_arm", ms(arm["mean_ms"]), source)
        else:
            lines.append([p, "2 cores", NO_NETWORK["hierarchy-converged"], NOT_MEASURED] + [""] * 4)
            fact(f"nt{p[-1]}_two_ms_net", None, source)
            fact(f"nt{p[-1]}_two_ms_arm", None, source)
    verdicts = {}
    for arm_key, bucket in pooled.items():
        n = f"ntall_{arm_key.removeprefix('hierarchy-')}"
        if not bucket["levels"]:
            for name in ("levels", "net", "arm", "only_net", "only_arm", "p", "verdict"):
                fact(f"{n}_{name}", None, source)
            continue
        discordant = bucket["only_net"] + bucket["only_arm"]
        p_value = float(binomtest(bucket["only_net"], discordant).pvalue) if discordant else 1.0
        verdict = ("better than" if bucket["net"] < bucket["arm"] and p_value < 0.05 else "worse than" if bucket["net"] > bucket["arm"] and p_value < 0.05
                   else "not significantly different from")
        verdicts[arm_key] = (verdict, bucket, p_value)
        lines.append([", ".join(bucket["levels"]) + (" together" if len(bucket["levels"]) > 1 else ""), "1 core", NO_NETWORK[arm_key], f"{bucket['net']} against {bucket['arm']}", "",
                      f"{bucket['only_net']} / {bucket['only_arm']}", stats.p_text(p_value), ""])
        for name, value in (("levels", ", ".join(bucket["levels"])), ("net", bucket["net"]), ("arm", bucket["arm"]), ("only_net", bucket["only_net"]),
                            ("only_arm", bucket["only_arm"]), ("p", stats.p_text(p_value)), ("verdict", verdict)):
            fact(f"{n}_{name}", value, source)
    per_p = []
    for p in P_VALUES:
        clauses = []
        for arm_key, short_words in (("hierarchy-converged", "the best hand-written rule"), ("hierarchy-weight", "plain lowest weight")):
            n = f"nt{p[-1]}_{arm_key.removeprefix('hierarchy-')}"
            if FACTS[f"{n}_net"][0] == NOT_MEASURED:
                continue
            after = measured[p][arm_key].get("after_test")
            clauses.append(f"against {short_words}{' [' + AFTER_SHORT + ']' if after else ''} {FACTS[n + '_net'][0]} against {FACTS[n + '_arm'][0]} failed pairs "
                           f"(only with the network {FACTS[n + '_only_net'][0]}, only without {FACTS[n + '_only_arm'][0]}, exact p = {FACTS[n + '_p'][0]}; "
                           f"{FACTS[n + '_ms_net'][0]} against {FACTS[n + '_ms_arm'][0]} ms per pair)")
        if clauses:
            per_p.append(f"at p = {p}, " + " and ".join(clauses))
    fact("net_timed_per_p", per_p and ("With the network against without it, timed rows on one core: " + "; ".join(per_p) + "."), source)
    if verdicts:
        parts = []
        for arm_key, (verdict, bucket, p_value) in verdicts.items():
            after = f"; its rows at p = {', '.join(bucket['after'])} were {AFTER_TEST}" if bucket["after"] else ""
            parts.append(f"against {NO_NETWORK[arm_key]} at p = {', '.join(bucket['levels'])}" + (" together" if len(bucket["levels"]) > 1 else "") + f", the network fails {bucket['net']} pairs and the rule "
                         f"{bucket['arm']} (only the network fails {bucket['only_net']}, only the rule {bucket['only_arm']}, exact p = {stats.p_text(p_value)}: "
                         f"{verdict} it{after})")
        text = ("How much the network does, timed against timed (each row through the team's evaluator, one process on the team's server, the same "
                "test pairs): " + "; ".join(parts) + ".")
    else:
        text = "The hierarchy without the network has no timed row on the test shots."
    fact("net_timed_statement", text, source)
    best = verdicts.get("hierarchy-converged")
    close = [abs(float(FACTS[f"nt{p[-1]}_converged_ms_net"][0].replace(",", "")) / float(FACTS[f"nt{p[-1]}_converged_ms_arm"][0].replace(",", "")) - 1)
             for p in (best[1]["levels"] if best else [])]
    timing = ("and on one core its time is within 4% of the rule's at every p, the size of the run-to-run spread of these rows, so it costs no time the "
              "timer can see" if close and max(close) < 0.04 else "and the times differ by more than the rows' spread at some p: see the table")
    fact("net_timed_plain", best and {
        "not significantly different from": f"In plain words: on the timed rows the network does not do measurably more than a good hand-written rule, {timing}. "
                                            "The gain over relay-BP is PriorTilt's and the hierarchy's, as in the offline ablation above.",
        "better than": f"In plain words: on the timed rows the network fails fewer pairs than the best hand-written rule, a significant difference, {timing}.",
        "worse than": f"In plain words: on the timed rows the network fails more pairs than the best hand-written rule, a significant difference, {timing}.",
    }[best[0]], source)
    fact("net_timed_slide_title", best and (f"Timed against timed, the network is {best[0]} the best hand-written tie-break "
                                            f"({best[1]['net']} against {best[1]['arm']} failed pairs at p = {', '.join(best[1]['levels'])}, exact p = {stats.p_text(best[2])})"),
         source)
    return write("test_network_timed",
                 "**How much the network does, timed against timed.** Our decoder's timed row (the hierarchy with the neural tie-break) against the same "
                 "hierarchy with a hand-written tie-break in place of the network, each a timed row of the team's evaluator on the same 10,000 test pairs, "
                 "in one process (\"1 core\"; \"2 cores\": the two fast decodes side by side). Only the shots where the two fast decodes disagree can differ. "
                 f"A row marked \"{AFTER_SHORT}\" is an ablation the team asked for after the test results were seen, not our decoder; the rows at p = 0.003 and 0.004 of the best hand-written rule on one core ran in the "
                 "test run itself, among its pre-registered arms. Times are comparable to a few percent (`APPENDIX.md` A).\n\n"
                 + markdown(["p", "Cores", "Without the network: the tie-break", "Failed pairs: with the network against without", "Both fail",
                             "Only with the network fails / only without", "Exact p", "Mean time per pair, ms: with against without"], lines), records)


def timed_ablation(measured):
    """The ablation again on the timed rows where both arms were timed: the network against the best hand-written tie-break."""
    levels, after, network, classical, only_network, only_classical = [], [], 0, 0, 0, 0
    for p in P_VALUES:
        ours, arm = measured[p].get("hierarchy"), measured[p].get("hierarchy-converged")
        if ours is None or arm is None or ours.get("offline") or arm.get("offline") or ours["failed_pair"] is None:
            continue
        paired = stats.paired(arm["failed_pair"], ours["failed_pair"])
        levels.append(p)
        after += [p] if arm.get("after_test") else []
        network += ours["failures_pair"]; classical += arm["failures_pair"]
        only_network += paired["row_only"]; only_classical += paired["reference_only"]
    if levels:
        from scipy.stats import binomtest
        discordant = only_network + only_classical
        p_value = float(binomtest(only_network, discordant).pvalue) if discordant else 1.0
        missing = [p for p in P_VALUES if p not in levels]
        text = (f"In the timed rows, where both tie-breaks have one (p = {', '.join(levels)}), the network fails {network} pairs and the best "
                f"hand-written rule {classical} (only the network rule fails {only_network}, only the hand-written rule {only_classical}, exact p = "
                f"{stats.p_text(p_value)}, " + ("a significant difference" if p_value < 0.05 else "not a significant difference") + ")"
                + (f"; at p = {', '.join(missing)} the hand-written rule has no timed row" if missing else "")
                + (f"; the hand-written rule's row at p = {', '.join(after)} was {AFTER_TEST}" if after else "")
                + ". The timed and the offline decodes draw other random numbers, so they need not agree on which rule is ahead.")
    else:
        text = "The hand-written tie-break has no timed row on the test shots."
    fact("abl_timed", text, "tables/test_comparisons.md")
    direction = "ahead" if levels and network < classical else "behind" if levels and network > classical else "level"
    joined = ", ".join(levels[:-1]) + " and " + levels[-1] if len(levels) > 1 else "".join(levels)
    fact("abl_timed_slide", levels and (f"In the timed rows at p = {joined} the network is {direction} ({network} against {classical} failed "
                                        f"pairs, exact p = {stats.p_text(p_value)}): " + ("a significant difference" if p_value < 0.05
                                        else "not significant either way")
                                        + (f"; the hand-written rule's row at p = {', '.join(after)} was {AFTER_SHORT}" if after else "") + "."),
         "tables/test_comparisons.md")


def ensemble_statement(measured, combined):
    """Our ensemble against TCBP-JOBS on the same test pairs, with what its gain rests on (joint decoding of the disputed shots)."""
    clauses, short = [], []
    for p in P_VALUES:
        ensemble, tcbp = measured[p].get("ensemble"), measured[p].get("tcbp")
        if ensemble is None or tcbp is None or tcbp["failed_pair"] is None:
            continue
        joint = [entry for entry in (combined[p] or {}).get("joint", {}).values() if entry]
        ran = sum(entry["shots"] for entry in joint)
        seconds = [entry["ms_per_shot"] / 1e3 for entry in joint if entry["ms_per_shot"]]
        if ensemble.get("same_as_tcbp") and ran == 0:
            clauses.append(f"at p = {p} joint decoding was not run, so the ensemble fails exactly TCBP-JOBS's {tcbp['failures_pair']} pairs")
            short.append(f"equals TCBP-JOBS at p = {p} (joint decoding not run)")
            continue
        paired = stats.paired(tcbp["failed_pair"], ensemble["failed_pair"])
        short.append(f"fails {ensemble['failures_pair']} pairs against TCBP-JOBS's {tcbp['failures_pair']} at p = {p} (only the ensemble "
                     f"{paired['row_only']}, only TCBP-JOBS {paired['reference_only']}, exact p = {stats.p_text(paired['p'])}"
                     + (f"; joint decoding on {ran} disputed shots, about {sum(seconds) / len(seconds):.0f} s each" if seconds else "") + ")")
        clauses.append(f"at p = {p} {ensemble['failures_pair']} against TCBP-JOBS's {tcbp['failures_pair']} failed pairs (only the ensemble fails "
                       f"{paired['row_only']}, only TCBP-JOBS {paired['reference_only']}, exact p = {stats.p_text(paired['p'])}; joint decoding ran on "
                       f"{ran} disputed shots" + (f", about {sum(seconds) / len(seconds):.0f} s each" if seconds else "") + ")")
    if clauses:
        text = ("Our ensemble, combined offline from saved answers with its rule frozen beforehand (where two PriorTilt runs and TCBP-JOBS agree, "
                "that answer; otherwise joint decoding of both check types, the slow decoder of the measurements before building): "
                + "; ".join(clauses) + ". Its members are NVIDIA's, IonQ's and the kit's; the rule is ours, and it has no time measured as one decoder.")
    else:
        text = "Our ensemble is not measured on the test shots yet."
    fact("ensemble_statement", text, "tables/test_p<p>.md (the ensemble and TCBP-JOBS rows)")
    fact("ensemble_short", short and ("our ensemble rule, frozen beforehand and combined offline from the saved answers of NVIDIA's, IonQ's "
                                      "and the kit's decoders, with no time measured as one decoder, " + ", ".join(short[:-1])
                                      + (", and " if len(short) > 1 else "") + short[-1]), "tables/test_p<p>.md (the ensemble and TCBP-JOBS rows)")


def swap_table(combined):
    """TCBP-JOBS in place of relay-BP in our hierarchy, read from the rules the ensemble already scored on the test shots.

    No rule is computed here. Rule d (majority of P1, P2, T; where all three differ, the network among P1, P2, T, R) is paired
    against TCBP-JOBS alone (T) and against our frozen decoder's offline decode (rule a), all three from the same files, so the
    comparison stays in one decode mode. Where its failures differ from T's, the shots are split by what rule d did there.
    """
    lines, records, split_lines = [], [], []
    compact = {"d": [], "T": [], "a": [], "f": []}
    source = "tables/test_after_swap.md"
    names = {SWAP_RULE: "rule d: TCBP-JOBS in place of relay-BP (where at least two of the three, the two PriorTilt runs and TCBP-JOBS, agree, "
                        "that answer; where all three differ, the network among them and relay-BP)",
             "f": "rule f, the ensemble's frozen rule, for reference (where the two PriorTilt runs and TCBP-JOBS all agree, that answer; "
                  "elsewhere joint decoding where it converged, else TCBP-JOBS)"}
    for p in P_VALUES:
        rows = combined[p]["rows"] if combined.get(p) else {}
        short = f"swap{p[-1]}"
        if not all(key in rows for key in (SWAP_RULE, "T", "a", "P1", "aw")):
            lines.append([p, names[SWAP_RULE]] + [NOT_MEASURED] * 5)
            for key in compact:
                compact[key].append(NOT_MEASURED)
            for name in ("d_pairs", "d_xz", "T_pairs", "a_pairs", "f_pairs", "vsT_both", "vsT_only_d", "vsT_only_T", "vsT_p",
                         "vsA_only_d", "vsA_only_a", "vsA_p"):
                fact(f"{short}_{name}", None, source)
            continue
        for key in (SWAP_RULE, "f"):
            if key not in rows:
                continue
            row = rows[key]
            versus_t = stats.paired(rows["T"]["failed_pair"], row["failed_pair"])
            versus_a = stats.paired(rows["a"]["failed_pair"], row["failed_pair"])
            lines.append([p, names[key], f"{row['failures_X']}, {row['failures_Z']}", f"{row['failures_pair']} of {row['shots']:,}",
                          rate_text(row["rate"], row["error"]),
                          f"{row['failures_pair']} against {rows['T']['failures_pair']}; " + pair_text(versus_t),
                          f"{row['failures_pair']} against {rows['a']['failures_pair']}; " + pair_text(versus_a)])
            records.append({"p": p, "rule": key, "failures_X": row["failures_X"], "failures_Z": row["failures_Z"], "failures_pair": row["failures_pair"],
                            "pairs": row["shots"], "rate_per_cycle": row["rate"], "kit_error": row["error"],
                            "tcbp_failed_pairs": rows["T"]["failures_pair"], "both_fail_with_tcbp": versus_t["both"],
                            "only_this_rule_fails_vs_tcbp": versus_t["row_only"], "only_tcbp_fails": versus_t["reference_only"], "exact_p_vs_tcbp": versus_t["p"],
                            "rule_a_failed_pairs": rows["a"]["failures_pair"], "both_fail_with_rule_a": versus_a["both"],
                            "only_this_rule_fails_vs_rule_a": versus_a["row_only"], "only_rule_a_fails": versus_a["reference_only"],
                            "exact_p_vs_rule_a": versus_a["p"], "source": row["source"]})
            if key == SWAP_RULE:
                for name, value in (("d_pairs", row["failures_pair"]), ("d_xz", f"{row['failures_X']}, {row['failures_Z']}"),
                                    ("T_pairs", rows["T"]["failures_pair"]), ("a_pairs", rows["a"]["failures_pair"]),
                                    ("vsT_both", versus_t["both"]), ("vsT_only_d", versus_t["row_only"]), ("vsT_only_T", versus_t["reference_only"]),
                                    ("vsT_p", stats.p_text(versus_t["p"])), ("vsA_only_d", versus_a["row_only"]),
                                    ("vsA_only_a", versus_a["reference_only"]), ("vsA_p", stats.p_text(versus_a["p"]))):
                    fact(f"{short}_{name}", value, source)
        fact(f"{short}_f_pairs", rows["f"]["failures_pair"] if "f" in rows else None, source)
        versus_t, versus_a = (stats.paired(rows[key]["failed_pair"], rows[SWAP_RULE]["failed_pair"]) for key in ("T", "a"))
        compact["d"].append(f"**{rows[SWAP_RULE]['failures_pair']}** · against T: only d {versus_t['row_only']}, only T {versus_t['reference_only']} "
                            f"(p = {stats.p_text(versus_t['p'])}) · against a: only d {versus_a['row_only']}, only a {versus_a['reference_only']} "
                            f"(p = {stats.p_text(versus_a['p'])})")
        compact["T"].append(str(rows["T"]["failures_pair"]))
        compact["a"].append(str(rows["a"]["failures_pair"]))
        compact["f"].append(str(rows["f"]["failures_pair"]) if "f" in rows else NOT_MEASURED)
        # where rule d and TCBP-JOBS part, per basis, in shots: rule d's answer is P1 where the two runs agree, T's where exactly two
        # of the three agree and they do not, and the network's pick where all three differ (later_d)
        cells = {}
        for basis in "XZ":
            with np.load(DATA / "test" / f"ensemble_p{p}" / f"test_{basis}.npz", allow_pickle=False) as file:
                failed_d, failed_t, failed_p1 = (file[f"failed_{key}"].astype(bool) for key in (SWAP_RULE, "T", "P1"))
                disputed, network = file["later_aw"].astype(bool), file[f"later_{SWAP_RULE}"].astype(bool)
            if (network & ~disputed).any():
                raise Mismatch(f"test/ensemble_p{p}: rule d sent a shot to the network where the two PriorTilt runs agree")
            if (failed_d[~disputed] != failed_p1[~disputed]).any():
                raise Mismatch(f"test/ensemble_p{p}: rule d does not take the two PriorTilt runs' answer where they agree")
            two = disputed & ~network
            if (failed_d[two] != failed_t[two]).any():
                raise Mismatch(f"test/ensemble_p{p}: rule d differs from TCBP-JOBS where exactly two of the three agree; read as a majority it cannot")
            only_d, only_t = failed_d & ~failed_t, failed_t & ~failed_d
            cells[basis] = {"agree": (int((only_d & ~disputed).sum()), int((only_t & ~disputed).sum())),
                            "network": (int((only_d & network).sum()), int((only_t & network).sum())),
                            "disputed": int(disputed.sum()), "sent": int(network.sum())}
        for where, words in (("agree", "the two PriorTilt runs agree: rule d keeps their answer"),
                             ("network", "all three differ: rule d takes the network's pick among P1, P2, T and R")):
            split_lines.append([p, words] + [f"{cells[basis][where][0]} / {cells[basis][where][1]}" for basis in "XZ"])
            for basis in "XZ":
                fact(f"{short}_{where}_{basis}_only_d", cells[basis][where][0], source)
                fact(f"{short}_{where}_{basis}_only_T", cells[basis][where][1], source)
        split_lines.append([p, "the two PriorTilt runs disagree and TCBP-JOBS sides with one of them: rule d's answer is TCBP-JOBS's (checked on every shot)",
                            "0 / 0", "0 / 0"])
        for basis in "XZ":
            fact(f"{short}_sent_{basis}", cells[basis]["sent"], source)
    text = (f"**{AFTER_TEST[0].upper() + AFTER_TEST[1:]}.** TCBP-JOBS in place of relay-BP in our hierarchy, as the team asked after the test "
            "results: no new rule is computed, the rows are rules the ensemble scored on the test shots among its pre-registered rules "
            "(`hierarchy.ensemble`; P1 and P2 PriorTilt under two seeds, T TCBP-JOBS, R relay-BP, all from the same saved answers). Rule d is the "
            "closest to that swap. Both comparisons stay in one decode mode: TCBP-JOBS alone (T) and our frozen decoder's offline decode "
            "(rule a: P1 where the two runs agree, else the network among R, P1, P2) come from the same files. Offline rows: no time.\n\n"
            + markdown(["p", "Rule (scored offline by the ensemble on the test shots)", "Failures X, Z", "Failed pairs", "Rate per cycle ± error",
                        "Against TCBP-JOBS alone: failed pairs; both fail / only this rule / only TCBP-JOBS (exact p)",
                        "Against our frozen decoder's offline decode (rule a): failed pairs; both fail / only this rule / only rule a (exact p)"], lines)
            + ("\n\n**Where rule d and TCBP-JOBS part**, in shots (not pairs): per basis, shots only rule d fails / shots only TCBP-JOBS fails, split by "
               "what rule d did on the shot.\n\n" + markdown(["p", "On the shots where", "X memory: only rule d / only TCBP-JOBS",
                                                               "Z memory: only rule d / only TCBP-JOBS"], split_lines) if split_lines else "")
            + "\n\nSource: `report/data/test/ensemble_p<p>/test_{X,Z}.npz` and `.json`.")
    words = {"d": f"TCBP-JOBS in place of relay-BP in our hierarchy (the ensemble's rule d; {AFTER_TEST})",
             "T": "TCBP-JOBS alone (T; NVIDIA, GPU)",
             "a": "our frozen decoder, the offline decode of the same shots (rule a)",
             "f": "our ensemble's frozen rule f (TCBP-JOBS plus joint decoding)"}
    write("test_after_swap_short",
          f"**{AFTER_TEST[0].upper() + AFTER_TEST[1:]}.** Failed pairs of 10,000 test pairs, every row scored offline by the ensemble on the same "
          "saved answers (no time); against T and a: the pairs only one of the two fails, exact p. `APPENDIX.md` J has the full table.\n\n"
          + markdown(["Row"] + [f"p = {p}" for p in P_VALUES], [[words[key]] + cells for key, cells in compact.items()]))
    return write("test_after_swap", text, records)


def swap_statement():
    """The one or two plain sentences on the swap, from the facts of swap_table."""
    value = lambda name: FACTS[name][0]
    if value("swap4_d_pairs") == NOT_MEASURED or value("swap5_d_pairs") == NOT_MEASURED:
        fact("swap_statement", "TCBP-JOBS in place of relay-BP in our hierarchy is not measured on the test shots.", "tables/test_after_swap.md")
        return

    def against_t(p):
        n = p[-1]
        d, t, pv = int(value(f"swap{n}_d_pairs")), int(value(f"swap{n}_T_pairs")), value(f"swap{n}_vsT_p")
        words = "level with" if float(pv) >= 0.05 else "behind" if d > t else "ahead of"
        return words, (f"{words} TCBP-JOBS alone at p = {p} ({d} against {t} failed pairs; only the swap fails {value(f'swap{n}_vsT_only_d')}, "
                       f"only TCBP-JOBS {value(f'swap{n}_vsT_only_T')}, exact p = {pv})")
    four, five = against_t("0.004"), against_t("0.005")
    text = (f"TCBP-JOBS in place of relay-BP in our hierarchy ({AFTER_TEST}; the ensemble's rule d, scored offline on the test shots) is "
            f"{four[1]} and {five[1]}. Against our frozen decoder's offline decode of the same shots (rule a) it fails "
            f"{value('swap4_d_pairs')} against {value('swap4_a_pairs')} pairs at p = 0.004 (exact p = {value('swap4_vsA_p')}) and "
            f"{value('swap5_d_pairs')} against {value('swap5_a_pairs')} at p = 0.005 (exact p = {value('swap5_vsA_p')}): TCBP-JOBS is the "
            "stronger third candidate, but where the two PriorTilt runs agree the swap keeps their answer. At p = 0.005 they agree on a wrong "
            f"answer that TCBP-JOBS gets right on {value('swap5_agree_X_only_d')} X and {value('swap5_agree_Z_only_d')} Z shots (and on a right "
            f"answer where TCBP-JOBS is wrong on {value('swap5_agree_X_only_T')} and {value('swap5_agree_Z_only_T')}); where all three differ, the "
            f"network's pick fails {value('swap5_network_X_only_d')} X and {value('swap5_network_Z_only_d')} Z shots that TCBP-JOBS gets right "
            f"({value('swap5_network_X_only_T')} and {value('swap5_network_Z_only_T')} the other way). What beats TCBP-JOBS is the ensemble's "
            f"rule f, which adds joint decoding: {value('swap4_f_pairs')} against {value('swap4_T_pairs')} failed pairs at p = 0.004.")
    fact("swap_statement", text, "tables/test_after_swap.md")


def predecoder_table(measured):
    """The learned predecoder on the saved test shots: relay-BP with the head's per-shot priors (H) against relay-BP with the kit's
    priors built the same way, per shot (fresh: the declared pair), TCBP-JOBS (T) and the evaluator's relay-BP row (R), one basis.

    The counts are the run's own (predecoder.compare and predecoder.secondary); its per-shot file is not in git. T's and R's totals
    are checked against the rows of this report on the same shots, and the run must name the team's test file.
    """
    source = "tables/test_predecoder.md"
    lines, records = [], []
    for p, basis, mark in PREDECODER_TEST:
        n = f"pre{p[-1]}{basis}"
        mark = f" [{mark}]" if mark else ""
        files = [DATA / "test" / "predecoder" / f"test_run_p{p}_{basis}_{kind}.json" for kind in ("summary", "secondary")]
        head_only = DATA / "test" / "predecoder" / f"test_run_p{p}_{basis}_head_arm_summary.json"
        if not files[0].is_file() and head_only.is_file():     # the head arm lands first; the full summary and the secondary rows follow
            files[0] = head_only
        if not files[0].is_file():
            lines.append([p, basis, "relay-BP with the head's priors (H)" + mark] + [NOT_MEASURED] * 4)
            for name in ("H", "fresh", "T", "R", "vs_fresh_only_H", "vs_fresh_only_other", "vs_fresh_p", "vs_T_only_H", "vs_T_only_other", "vs_T_p",
                         "vs_R_only_H", "vs_R_only_other", "vs_R_p", "it_H", "it_fresh", "ratio", "it_ratio"):
                fact(f"{n}_{name}", None, source)
            continue
        summary = json.loads(files[0].read_text(encoding="utf-8"))
        secondary = (json.loads(files[1].read_text(encoding="utf-8")) if files[1].is_file()
                     else {"data": summary["data"], "failures": {"H (relay-BP, the head's priors)": {"failures": summary["arms"]["head"]["failures"]}}, "pairs": {}})
        for record in (summary, secondary):
            if record["data"]["sha256"] != TEST_SHA256[p][basis] or record["data"]["basis"] != basis or abs(record["data"]["p"] - float(p)) > 1e-12:
                raise Mismatch(f"test/predecoder, p = {p}, {basis}: not the team's test file (sha256 {record['data']['sha256'][:12]})")
        arms, failures, pairs = summary["arms"], secondary["failures"], secondary["pairs"]
        key = lambda table, start: next((name for name in table if name.startswith(start)), None)
        has_fresh = "fresh" in arms and "head_vs_fresh" in summary.get("pairs", {}) and key(pairs, "H against fresh") is not None
        declared, theirs = (summary["pairs"]["head_vs_fresh"], pairs[key(pairs, "H against fresh")]) if has_fresh else (None, None)
        if has_fresh and (declared["both"], declared["only_head"], declared["only_fresh"]) != (theirs["both"], theirs["only_H"], theirs["only_other"]):
            raise Mismatch(f"test/predecoder, p = {p}, {basis}: the summary and the secondary file give other counts for head against fresh")
        if arms["head"]["failures"] != failures[key(failures, "H ")]["failures"] or (
                has_fresh and arms["fresh"]["failures"] != failures[key(failures, "fresh")]["failures"]):
            raise Mismatch(f"test/predecoder, p = {p}, {basis}: the summary and the secondary file give other failures")
        for start, report_key, words in (("T ", "tcbp", "TCBP-JOBS"), ("R ", REFERENCE, "relay-BP's timed row")):
            row = measured[p].get(report_key)
            if key(failures, start) is None:
                continue
            if row is not None and row[f"failures_{basis}"] != failures[key(failures, start)]["failures"]:
                raise Mismatch(f"test/predecoder, p = {p}, {basis}: {words} fails {failures[key(failures, start)]['failures']} shots there and "
                               f"{row[f'failures_{basis}']} in this report's row on the same shots")
        shots = summary["shots"]
        rows = [("H", "relay-BP with the head's priors, a decoder built per shot (H; ours)" + mark, arms["head"]["failures"], None, arms["head"]),
                ("fresh", "relay-BP with the kit's priors, a decoder built per shot (the declared comparison; the kit)",
                 arms["fresh"]["failures"] if has_fresh else None, theirs, arms["fresh"] if has_fresh else None),
                ("T", "TCBP-JOBS (NVIDIA, GPU; its row of this report)", failures[key(failures, "T ")]["failures"] if key(failures, "T ") else None,
                 pairs[key(pairs, "H against T")] if key(pairs, "H against T") else None, None),
                ("R", "relay-BP, the evaluator's timed row (one decoder through all shots; its row of this report)",
                 failures[key(failures, "R ")]["failures"] if key(failures, "R ") else None, pairs[key(pairs, "H against R")] if key(pairs, "H against R") else None, None)]
        for short, label, count, paired, arm in rows:
            if count is None:
                lines.append([p, basis, label] + ["not measured yet"] * 4)
                for name in (f"{n}_{short}", f"{n}_vs_{short}_only_H", f"{n}_vs_{short}_only_other", f"{n}_vs_{short}_p"):
                    fact(name, None, source)
                continue
            low, high = stats.wilson(count, shots)
            lines.append([p, basis, label, f"{count:,} of {shots:,}", f"{count / shots:.2%} ({low:.2%} to {high:.2%})",
                          "" if paired is None else f"{paired['both']} / {paired['only_H']} / {paired['only_other']} (p = {stats.p_text(paired['p'])})",
                          "" if arm is None else f"{arm['mean_iterations']:,.0f}; {arm['unconverged']} not converged"])
            records.append({"p": p, "basis": basis, "row": short, "failures": count, "shots": shots, "wilson95_low": low, "wilson95_high": high,
                            "both_fail": paired and paired["both"], "only_H_fails": paired and paired["only_H"], "only_this_row_fails": paired and paired["only_other"],
                            "exact_p": paired and paired["p"], "mean_iterations": arm and arm["mean_iterations"], "unconverged": arm and arm["unconverged"]})
            fact(f"{n}_{short}", count, source)
            if paired is not None:
                tag = {"fresh": "vs_fresh", "T": "vs_T", "R": "vs_R"}[short]
                fact(f"{n}_{tag}_only_H", paired["only_H"], source)
                fact(f"{n}_{tag}_only_other", paired["only_other"], source)
                fact(f"{n}_{tag}_p", stats.p_text(paired["p"]), source)
        fact("pre_columns", f"{summary['logits']['columns']:,}", source)
        fact(f"{n}_it_H", f"{arms['head']['mean_iterations']:,.0f}", source)
        fact(f"{n}_it_fresh", f"{arms['fresh']['mean_iterations']:,.0f}" if has_fresh else None, source)
        fact(f"{n}_ratio", f"{arms['fresh']['failures'] / arms['head']['failures']:.1f}" if has_fresh and arms["head"]["failures"] else None, source)
        fact(f"{n}_it_ratio", f"{arms['fresh']['mean_iterations'] / arms['head']['mean_iterations']:.1f}" if has_fresh else None, source)
    landed = [(p, basis) for p, basis, mark in PREDECODER_TEST if mark and FACTS[f"pre{p[-1]}{basis}_H"][0] != NOT_MEASURED]
    fact("pre_extension", " ".join(
        f"On the {basis} test shots at p = {p}, relay-BP with the head's priors fails {FACTS[f'pre{p[-1]}{basis}_H'][0]} "
        + (f"against {FACTS[f'pre{p[-1]}{basis}_fresh'][0]} with the kit's priors (only the head's "
           f"{FACTS[f'pre{p[-1]}{basis}_vs_fresh_only_H'][0]} against {FACTS[f'pre{p[-1]}{basis}_vs_fresh_only_other'][0]}, exact p = "
           f"{FACTS[f'pre{p[-1]}{basis}_vs_fresh_p'][0]}"
           + (": not significant, so the declared line, a paired exact p below 0.05, is not passed at this p)"
              if float(FACTS[f"pre{p[-1]}{basis}_vs_fresh_p"][0]) >= 0.05 else ": a significant gain)")
           if FACTS[f"pre{p[-1]}{basis}_fresh"][0] != NOT_MEASURED
           else "(the kit's priors, the declared comparison, not measured yet)")
        + (f", and TCBP-JOBS fails {FACTS[f'pre{p[-1]}{basis}_T'][0]} (only the head's "
           f"{FACTS[f'pre{p[-1]}{basis}_vs_T_only_H'][0]} against {FACTS[f'pre{p[-1]}{basis}_vs_T_only_other'][0]}, exact p = {FACTS[f'pre{p[-1]}{basis}_vs_T_p'][0]})."
           if FACTS[f"pre{p[-1]}{basis}_T"][0] != NOT_MEASURED else "; its rows beside TCBP-JOBS and relay-BP are not measured yet.")
        for p, basis in landed) or "The registered extension (X at p = 0.004 and 0.003, Z at p = 0.005) has not landed yet.", source)
    fact("pre_z_reason", "the Z run of the registered extension is a later run, shown below" if ("0.005", "Z") in landed
         else "Z memory has no result yet", source)
    fallback_lines = []
    for p, basis, what in PREDECODER_FALLBACK:
        n = f"fb{p[-1]}{basis}"
        file = DATA / "test" / "predecoder" / f"fallback_test_{basis}_p{p}.json"
        log = file.with_suffix(".log")
        if not file.is_file():
            fallback_lines.append([p, basis, what] + ["not measured yet"] * 6)
            for name in ("rule", "H", "T", "back", "vs_T_only_rule", "vs_T_only_T", "vs_T_p", "vs_H_only_rule", "vs_H_only_H", "vs_H_p",
                         "conv_shots", "conv_H", "conv_T", "back_H", "back_T"):
                fact(f"{n}_{name}", None, source)
            continue
        record = json.loads(file.read_text(encoding="utf-8"))
        if record["data"]["sha256"] != TEST_SHA256[p][basis] or record["data"]["basis"] != basis:
            raise Mismatch(f"test/predecoder/{file.name}: not the team's test file")
        failures, pairs, back, shots = record["failures"], record["pairs"], record["fallback_shots"], record["data"]["shots"]
        held = DATA / "test" / "predecoder" / f"test_run_p{p}_{basis}_summary.json"
        if held.is_file() and json.loads(held.read_text(encoding="utf-8"))["arms"]["head"]["failures"] != failures["H"]["failures"]:
            raise Mismatch(f"test/predecoder/{file.name}: H fails {failures['H']['failures']} shots here and otherwise in the held-out run")
        if measured[p].get("tcbp") is not None and measured[p]["tcbp"][f"failures_{basis}"] != failures["T"]["failures"]:
            raise Mismatch(f"test/predecoder/{file.name}: TCBP-JOBS fails {failures['T']['failures']} shots here and otherwise in its row")
        split = re.search(r"H fails (\d+) of its unconverged shots and (\d+) converged ones; T fails (\d+) of the shots that fall back",
                          log.read_text(encoding="utf-8")) if log.is_file() else None
        if split is None:
            raise Mismatch(f"test/predecoder/{log.name}: no line splitting the failures by convergence")
        back_h, conv_h, back_t = map(int, split.groups())
        if conv_h + back_h != failures["H"]["failures"] or conv_h + back_t != failures["rule"]["failures"]:
            raise Mismatch(f"test/predecoder/{log.name}: the split by convergence does not add up to the failures")
        versus_t, versus_h = pairs["rule against T"], pairs["rule against H"]
        low, high = stats.wilson(failures["rule"]["failures"], shots)
        fallback_lines.append([p, basis, what, f"{back:,} of {shots:,}",
                               f"{failures['rule']['failures']:,} ({failures['rule']['failures'] / shots:.2%}, {low:.2%} to {high:.2%})",
                               f"{failures['H']['failures']:,}", f"{failures['T']['failures']:,}",
                               f"{versus_t['both']} / {versus_t['only_rule']} / {versus_t['only_T']} (p = {stats.p_text(versus_t['p'])})",
                               f"{versus_h['both']} / {versus_h['only_rule']} / {versus_h['only_H']} (p = {stats.p_text(versus_h['p'])})"])
        records.append({"p": p, "basis": basis, "row": "fallback rule", "failures": failures["rule"]["failures"], "shots": shots,
                        "wilson95_low": low, "wilson95_high": high, "fallback_shots": back, "post_hoc": basis == "X"})
        for name, value in (("rule", failures["rule"]["failures"]), ("H", failures["H"]["failures"]), ("T", failures["T"]["failures"]), ("back", back),
                            ("vs_T_only_rule", versus_t["only_rule"]), ("vs_T_only_T", versus_t["only_T"]), ("vs_T_p", stats.p_text(versus_t["p"])),
                            ("vs_H_only_rule", versus_h["only_rule"]), ("vs_H_only_H", versus_h["only_H"]), ("vs_H_p", stats.p_text(versus_h["p"])),
                            ("conv_shots", f"{shots - back:,}"), ("conv_H", conv_h), ("conv_T", failures["T"]["failures"] - back_t),
                            ("back_H", back_h), ("back_T", back_t)):
            fact(f"{n}_{name}", value, source)
    if FACTS["fb5Z_rule"][0] == NOT_MEASURED:
        fact("fb5Z_clause", "Its clean test will be the Z test file at p = 0.005, registered to be opened only after the rule was frozen "
                            "(`predecoder/logs/fallback_rule_frozen.md`); that run has not landed yet.", source)
    else:
        fact("fb5Z_clause", f"Its clean test, the Z test file at p = 0.005, registered to be opened only after the rule was frozen, "
                            f"{PREDECODER_TEST[-1][2]}: the rule fails {FACTS['fb5Z_rule'][0]} against TCBP-JOBS's {FACTS['fb5Z_T'][0]} "
                            f"(only the rule {FACTS['fb5Z_vs_T_only_rule'][0]}, only TCBP-JOBS {FACTS['fb5Z_vs_T_only_T'][0]}, exact p = "
                            f"{FACTS['fb5Z_vs_T_p'][0]}) and the head's priors alone {FACTS['fb5Z_H'][0]}.", source)
    pair_lines, pair_file = [], DATA / "test" / "predecoder" / "pair_test_p0.005.json"
    pair_words = {"H": "relay-BP with the head's priors in both bases, H: " + EXTENSION,
                  "rule": "the fallback rule in both bases (designed after the first test result: post hoc on its X half)",
                  "T": "TCBP-JOBS (its row of this report)", "R": "relay-BP, the evaluator's timed row (its row of this report)"}
    if pair_file.is_file():
        record = json.loads(pair_file.read_text(encoding="utf-8"))
        decoders, paired, pairs_n = record["decoders"], record["paired"], record["pairs"]
        for short, report_key in (("T", "tcbp"), ("R", REFERENCE)):
            row = measured["0.005"].get(report_key)
            if row is not None and (decoders[short]["failed_pairs"], decoders[short]["failures_X"], decoders[short]["failures_Z"]) != (
                    row["failures_pair"], row["failures_X"], row["failures_Z"]):
                raise Mismatch(f"test/predecoder/pair_test_p0.005.json: {short} differs from this report's row on the same pairs")
        for short in ("H", "rule", "T", "R"):
            if short not in decoders:
                pair_lines.append([pair_words[short]] + ["not measured yet"] * 5)
                continue
            entry = decoders[short]
            rate, error = stats.per_cycle_rate(entry["failed_pairs"], pairs_n)
            if abs(rate - entry["per_cycle"]) > 1e-12:
                raise Mismatch(f"test/predecoder/pair_test_p0.005.json: {short}'s rate per cycle is not 1 - (1 - P)^(1/12) of its failed pairs")
            cells = []
            for other in ("T", "R"):
                name = next((k for k in paired if k.startswith(f"{short} against {other}")), None)
                entry_p = paired.get(name) if name else None
                only_mine = entry_p and next(v for k, v in entry_p.items() if k.startswith("only_") and not k.endswith(other))
                cells.append("" if short in ("T", "R") or entry_p is None else
                             f"{entry_p['both']} / {only_mine} / {entry_p['only_' + other]} (p = {stats.p_text(entry_p['p'])})")
                if entry_p is not None and short in ("H", "rule"):
                    fact(f"pair_{short}_vs_{other}_only_mine", only_mine, source)
                    fact(f"pair_{short}_vs_{other}_only_other", entry_p["only_" + other], source)
                    fact(f"pair_{short}_vs_{other}_p", stats.p_text(entry_p["p"]), source)
            pair_lines.append([pair_words[short], f"{entry['failures_X']}, {entry['failures_Z']}", f"{entry['failed_pairs']:,} of {pairs_n:,}",
                               rate_text(rate, error)] + cells)
            fact(f"pair_{short}", entry["failed_pairs"], source)
    else:
        pair_lines.append([pair_words["H"]] + ["not measured yet"] * 5)
    for short in ("H", "rule", "T", "R"):
        if f"pair_{short}" not in FACTS:
            fact(f"pair_{short}", None, source)
        for other in ("T", "R"):
            for part in ("only_mine", "only_other", "p"):
                if f"pair_{short}_vs_{other}_{part}" not in FACTS and short in ("H", "rule"):
                    fact(f"pair_{short}_vs_{other}_{part}", None, source)
    value = lambda name: FACTS[name][0]
    fact("pre_pair", "The pair score at p = 0.005 has not landed yet." if value("pair_H") == NOT_MEASURED else
         (f"Scored as the kit scores (a pair of an X and a Z shot fails when either fails), at p = 0.005: relay-BP with the head's priors fails "
          f"{value('pair_H')} pairs against TCBP-JOBS's {value('pair_T')} (only H {value('pair_H_vs_T_only_mine')}, only TCBP-JOBS "
          f"{value('pair_H_vs_T_only_other')}, exact p = {value('pair_H_vs_T_p')}) and relay-BP's {value('pair_R')} (exact p = {value('pair_H_vs_R_p')})"
          + (f"; the fallback rule, post hoc on its X half, fails {value('pair_rule')} (against TCBP-JOBS: only the rule "
             f"{value('pair_rule_vs_T_only_mine')}, only TCBP-JOBS {value('pair_rule_vs_T_only_other')}, exact p = {value('pair_rule_vs_T_p')})"
             if value("pair_rule") != NOT_MEASURED else "") + ". This pair score is part of the registered extension, not a held-out result, "
          "and it is not in the pair-scored tables of our decoder; accuracy only (its times come from a shared pool on the team's PC); TCBP-JOBS's "
          "counts come from its stored test file and carry its fit's spread (above)."), source)
    validation = DATA / "exploration" / "predecoder" / "fallback_validation_X_p0.005.json"
    record = json.loads(validation.read_text(encoding="utf-8")) if validation.is_file() else None
    for name, value in (("rule", record and record["failures"]["rule"]["failures"]), ("T", record and record["failures"]["T"]["failures"]),
                        ("H", record and record["failures"]["H"]["failures"]),
                        ("only_rule", record and record["pairs"]["rule against T"]["only_rule"]),
                        ("only_T", record and record["pairs"]["rule against T"]["only_T"]),
                        ("p", record and stats.p_text(record["pairs"]["rule against T"]["p"])), ("back", record and record["fallback_shots"])):
        fact(f"fbval_{name}", value, "data/exploration/predecoder/fallback_validation_X_p0.005.json")
    gate = DATA / "exploration" / "predecoder" / "compare_validation_X_summary.json"
    if gate.is_file():
        record = json.loads(gate.read_text(encoding="utf-8"))
        pair = record["pairs"]["head_vs_fresh"]
        for name, value in (("H", record["arms"]["head"]["failures"]), ("fresh", record["arms"]["fresh"]["failures"]), ("shots", f"{record['shots']:,}"),
                            ("only_H", pair["only_head"]), ("only_fresh", pair["only_fresh"]), ("p", stats.p_text(pair["p"]))):
            fact(f"pregate_{name}", value, "data/exploration/predecoder/compare_validation_X_summary.json")
    else:
        for name in ("H", "fresh", "shots", "only_H", "only_fresh", "p"):
            fact(f"pregate_{name}", None, "data/exploration/predecoder/compare_validation_X_summary.json")
    return write("test_predecoder",
                 f"**The learned predecoder on the test shots: {PREDECODER_LABEL}.** Relay-BP at the kit's settings, given for each shot the priors "
                 "that a second output (a \"head\") of our trained network computes from that shot's detectors; truth enters only the scoring. "
                 "Per noise level and basis: the saved test file of that basis alone, so a row counts shots, not pairs, and cannot sit in the "
                 "pair-scored tables. H against the others: both fail / only H fails / only the other fails (exact p), on the same shots. "
                 "Counts from the run's own files (`predecoder.compare`, `predecoder.secondary`); TCBP-JOBS's and relay-BP's totals are checked "
                 "against their rows in this report.\n\n"
                 + markdown(["p", "Basis", "Decoder", "Failed shots", "Share (95% interval)", "H against it: both / only H / only it (exact p)",
                             "Mean BP iterations per shot; shots not converged"], lines)
                 + "\n\nSource: `report/data/test/predecoder/test_run_p<p>_<basis>_{summary,secondary}.json` (copied from `predecoder/logs/`)."
                 + f"\n\n**Not held out: the fallback rule, {PREDECODER_FALLBACK_LABEL}.** The rule: for every shot, the answer of relay-BP with "
                 "the head's priors (H) where that relay-BP converged (its own success flag, no truth), else TCBP-JOBS's answer (T). No knob, no "
                 "threshold (`predecoder/logs/fallback_rule_frozen.md`). It is not an equal of the rows above.\n\n"
                 + markdown(["p", "Basis", "What this number is", "Shots that fall back to TCBP-JOBS", "The rule fails (share, 95% interval)",
                             "H fails", "TCBP-JOBS fails", "Rule against TCBP-JOBS: both / only the rule / only TCBP-JOBS (exact p)",
                             "Rule against H: both / only the rule / only H (exact p)"], fallback_lines)
                 + "\n\nSource: `report/data/test/predecoder/fallback_test_<basis>_p<p>.{json,log}` (`predecoder.fallback`)."
                 + f"\n\n**The pair score at p = 0.005: {EXTENSION}; the fallback rule's half on X is post hoc (designed after the first test result), "
                 "its half on Z clean; accuracy only.** The kit's score: a pair of an X shot "
                 "and a Z shot fails when either fails. Not a held-out result, and not in the pair-scored tables of our decoder. TCBP-JOBS's and "
                 "relay-BP's counts are checked against their rows in this report.\n\n"
                 + markdown(["Decoder", "Failures X, Z", "Failed pairs", "Rate per cycle ± error",
                             "Against TCBP-JOBS: both / only this / only TCBP-JOBS (exact p)", "Against relay-BP's timed row: both / only this / only relay-BP (exact p)"],
                            pair_lines)
                 + "\n\nSource: `report/data/test/predecoder/pair_test_p0.005.{json,log}` (`predecoder.pair`).",
                 records)


def per_x_shot(row):
    """A timed evaluator row's mean time per X shot, in seconds, from its own per-shot file (the X half of each pair), or None."""
    if row is None or row.get("offline") or not row.get("per_shot_file"):
        return None
    with np.load(DATA / row["per_shot_file"], allow_pickle=False) as file:
        return float(file["seconds_total_X"].mean()) if "seconds_total_X" in file.files else None


def predecoder_latency_table(measured):
    """The predecoder's time per X shot, measured cleanly on exploration shots (the first 1,000 of validation_X at each p, one process,
    one thread on the team's server), with the test rows' times per X shot beside it for scale, labelled: other shots, so not paired."""
    source = "tables/predecoder_latency.md"
    names = ("net_mean", "net_median", "net_p99", "bp_mean", "bp_median", "bp_p99", "conv", "shots", "conv_ms", "unconv", "unconv_ms", "total",
             "relay_x", "tcbp_x", "relay_pair", "unconv_share")
    rows, notes, jobs = [], [], []
    for p in P_VALUES:
        folder = DATA / "exploration" / "predecoder" / f"latency_validation_X_p{p}"
        n = f"lat{p[-1]}"
        if not (folder / "summary.json").is_file() or not (folder / "per_shot.npz").is_file():
            rows.append([p] + [NOT_MEASURED] * 4)
            for name in names:
                fact(f"{n}_{name}", None, source)
            continue
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        if summary["data"]["basis"] != "X" or abs(summary["data"]["p"] - float(p)) > 1e-12 or summary["data"]["seed"] != 201:
            raise Mismatch(f"exploration/predecoder/latency_validation_X_p{p}: not the validation X file at p = {p}")
        with np.load(folder / "per_shot.npz", allow_pickle=False) as file:
            seconds = file["seconds_build_head"] + file["seconds_decode_head"]
            converged = file["converged_head"].astype(bool)
        head = summary["arms"]["head"]
        if abs(seconds.mean() - (head["build_seconds_mean"] + head["decode_seconds_mean"])) > 1e-9 or len(seconds) != summary["shots"]:
            raise Mismatch(f"exploration/predecoder/latency_validation_X_p{p}: the per-shot times do not give the summary's mean")
        network = summary["logits"]["per_shot_seconds"]
        if summary["logits"]["model"]["torch_threads"] != 1 or summary["workers"] != 1:
            raise Mismatch(f"exploration/predecoder/latency_validation_X_p{p}: not one process and one thread")
        ms_ = lambda value: ms(value * 1e3)
        relay, tcbp = measured[p].get(REFERENCE), measured[p].get("tcbp")
        relay_x = per_x_shot(relay)
        tcbp_x = None
        if tcbp is not None and (DATA / "test" / f"tcbp_p{p}" / "test_X.tcbp.npz").is_file():
            with np.load(DATA / "test" / f"tcbp_p{p}" / "test_X.tcbp.npz", allow_pickle=False) as file:
                tcbp_x = float(file["seconds_T"].astype(float).mean())
        values = {"net_mean": ms_(network["mean"]), "net_median": ms_(network["median"]), "net_p99": ms_(network["p99"]),
                  "bp_mean": ms_(seconds.mean()), "bp_median": ms_(float(np.median(seconds))), "bp_p99": ms_(float(np.percentile(seconds, 99))),
                  "conv": f"{int(converged.sum()):,}", "shots": f"{len(seconds):,}", "conv_ms": ms_(seconds[converged].mean()),
                  "unconv": int((~converged).sum()), "unconv_ms": ms_(seconds[~converged].mean()) if (~converged).any() else None,
                  "total": ms_(network["mean"] + seconds.mean()), "relay_x": relay_x and ms_(relay_x), "tcbp_x": tcbp_x and ms_(tcbp_x),
                  "relay_pair": relay and ms(relay["mean_ms"]), "unconv_share": f"{(~converged).mean():.1%}",
                  "total_s": network["mean"] + seconds.mean(), "unconv_frac": float((~converged).mean()), "tcbp_x_s": tcbp_x}
        for name in names:
            fact(f"{n}_{name}", values[name], source)
        LATENCY[p] = values
        rows.append([p, f"{values['net_mean']} / {values['net_median']} / {values['net_p99']}", f"{values['bp_mean']} / {values['bp_median']} / {values['bp_p99']}",
                     f"{values['conv']} of {values['shots']} ({values['conv_ms']} ms); not converged {values['unconv']}"
                     + (f" ({values['unconv_ms']} ms)" if values["unconv_ms"] else ""), values["total"]])
        notes.append(f"p = {p}: relay-BP's timed row {values['relay_x']} ms per X shot (its {values['relay_pair']} ms per pair is an X shot plus a Z shot), "
                     f"TCBP-JOBS {values['tcbp_x']} ms per X shot")
    for name in names:                  # the names the prose quotes for p = 0.005
        fact(f"lat_{name}", FACTS[f"lat5_{name}"][0] if FACTS[f"lat5_{name}"][0] != NOT_MEASURED else None, source)
    return write("predecoder_latency",
                 "**Exploration shots, not test shots: the predecoder's time per X shot.** The team's server, one process and one thread "
                 "(server jobs cpu/050 and cpu/060), the first 1,000 X shots of `data/p<p>/validation_X.npz` (so no test file was decoded twice); "
                 "the head and the final X network pinned by sha256 as in the test run. Per X shot, not per pair; ms: mean / median / 99th percentile.\n\n"
                 + markdown(["p", "The network and the head, on the CPU", "Relay-BP with the head's priors, built and run per shot", "Relay-BP converged (mean ms)",
                             "The two together, mean ms per X shot"], rows)
                 + "\n\nFor scale only, other shots and so no paired comparison, on the test X shots: " + "; ".join(notes)
                 + " (TCBP-JOBS: a GPU decode call alone, not comparable with a CPU time).\n\n"
                 "Source: `report/data/exploration/predecoder/latency_validation_X_p<p>/` (`summary.json`, `per_shot.npz`, `latency.txt`) and the jobs' "
                 "printouts `job_cpu_050-predecoder-latency.out`, `job_cpu_060-predecoder-latency-p0.003-p0.004.out`.")


OPERATING = "an operating rule chosen after the test results were seen, from the measured rows; not a frozen decoder"
LATENCY = {}   # p -> the predecoder's latency values, raw (seconds and fractions), from predecoder_latency_table


def operating_rule_table(measured):
    """Sal's request (15:15 UTC): which method at which noise level, per X shot. X test failures from the test rows; times per X shot from the
    timed rows' own per-shot files (our pipeline) and from the predecoder's clean sample on exploration shots. Unpaired counts."""
    source = "tables/operating_rule.md"
    lines, choice = [], []
    for p in P_VALUES:
        n = p[-1]
        ours, two, tcbp = measured[p].get("hierarchy"), measured[p].get("hierarchy-parallel"), measured[p].get("tcbp")
        lat, lat_ok = (lambda name: LATENCY[p][name]), p in LATENCY
        options = []
        if ours is not None and per_x_shot(ours) is not None:
            options.append(("our pipeline (the hierarchy with the network), one core", ours["failures_X"], per_x_shot(ours), "test shots, its timed row"))
        if two is not None and per_x_shot(two) is not None:
            options.append(("our pipeline, its two fast decodes on two cores", two["failures_X"], per_x_shot(two), "test shots, its timed row"))
        if FACTS[f"pre{n}X_H"][0] != NOT_MEASURED and lat_ok:
            options.append(("the predecoder (network and head, then relay-BP with the head's priors)", int(FACTS[f"pre{n}X_H"][0]), lat("total_s"),
                            "validation sample (exploration shots)"))
        if FACTS[f"fb{n}X_rule"][0] != NOT_MEASURED and lat_ok:
            extra = lat("unconv_frac") * (lat("tcbp_x_s") or 0)
            options.append(("the predecoder with the fallback (TCBP-JOBS where its relay-BP did not converge; needs a GPU; the rule was designed after "
                             "the first test result: post hoc at p = 0.005 and 0.004)", int(FACTS[f"fb{n}X_rule"][0]),
                            lat("total_s") + extra, f"validation sample, plus TCBP-JOBS's GPU decode on the {lat('unconv_share')} of shots that fall back"))
        for label, failures, seconds, where in options:
            lines.append([p, label, f"{failures} of 10,000", ms(seconds * 1e3), where])
        if tcbp is not None:
            lines.append([p, "TCBP-JOBS alone (NVIDIA, GPU), for reference", f"{tcbp['failures_X']} of 10,000",
                          ms(lat("tcbp_x_s") * 1e3) if lat_ok and lat("tcbp_x_s") is not None else "", "test shots; a GPU decode call alone"])
        one_core = [option for option in options if "two cores" not in option[0]]
        if one_core:
            fewest = min(option[1] for option in one_core)
            best = min((option for option in one_core if option[1] == fewest), key=lambda option: option[2])
            tied = [option[0] for option in one_core if option[1] == fewest and option is not best]
            name = best[0].split(" (")[0]
            pipeline = next((option for option in one_core if option[0].startswith("our pipeline")), None)
            if pipeline is not None and best is not pipeline and p in OPERATING_PAIRED:
                choice.append(f"at p = {p}, {name} ({best[1]} X failures against our pipeline's {pipeline[1]}, {OPERATING_PAIRED[p]}, at "
                              f"{ms(best[2] * 1e3)} against {ms(pipeline[2] * 1e3)} ms per X shot)")
            else:
                choice.append(f"at p = {p}, {name} ({best[1]} X failure{'' if best[1] == 1 else 's'}, {ms(best[2] * 1e3)} ms per X shot"
                              + (f"; level on failures with {', '.join(t.split(' (')[0] for t in tied)}, and faster" if tied else "") + ")")
            fact(f"op{n}_choice", name, source)
        else:
            fact(f"op{n}_choice", None, source)
    statement = ("Which method at which noise level (" + OPERATING + "): " + "; ".join(choice) + ". The counts are of 10,000 X test shots and are not "
                 "paired here (the predecoder's per-shot answers are not in the repository; the two verdicts against our pipeline are the review's paired "
                 "count on that file), so a difference of a few shots is within the noise; the times "
                 "come from other runs on other shots and are a guide, not a paired measurement; the fallback needs a GPU for TCBP-JOBS; the fallback rule "
                 "was designed after the first test result, so its counts at p = 0.005 and 0.004 are post hoc, not held out.") if choice else None
    fact("op_statement", statement, source)
    return write("operating_rule",
                 f"**Which method at which noise level: {OPERATING}.** X memory only: failures of the 10,000 X test shots at each p, and the mean time "
                 "per X shot. Our pipeline's time is the X half of its timed rows' own per-shot times on the test shots; the predecoder's is its clean "
                 "one-thread sample on 1,000 validation X shots (exploration shots); each row says which. Unpaired, so a difference of a few shots is "
                 "within the noise.\n\n"
                 + markdown(["p", "Method", "X test failures", "Mean time per X shot, ms", "Where the time comes from"], lines))


FOLDERS = {"bposd": "`decoders.py` (the kit)", "relay-bp": "`decoders.py` (the kit)", "router": "`evaluation/router.py` (teammates)",
           "priortilt": "`hierarchy/priortilt_plugin.py` (IonQ's decoder)", "hierarchy-weight": "`hierarchy/`", "hierarchy-converged": "`hierarchy/`",
           "hierarchy": "`hierarchy/`, `neural/`", "hierarchy-parallel": "`hierarchy/`, `neural/`", "ensemble": "`hierarchy/ensemble.py`",
           "tcbp": "`hierarchy/tcbp_member.py` (NVIDIA's decoder)", "hierarchy-converged-parallel": "`hierarchy/`"}
LABELS = {"pre": "pre-registered (the test run)", "pre-offline": "pre-registered (the test run), offline decode, no time",
          "after": "after-test ablation", "after-offline": "after-test ablation, offline from saved answers, no time",
          "secondary": "secondary: chosen after the predecoder's p = 0.004 gate failed",
          "extension": "registered extension, run after the X p = 0.005 test result",
          "post hoc": "post hoc: designed after the first test result",
          "after-design": "designed after the first test result; frozen before this file was opened",
          "ptcbp": "designed after the test results were seen; formerly a separate branch",
          "selector": "frozen on validation shots, run once on the test shots; its declared line (fewer failed pairs than TCBP-JOBS, paired) not met"}


def master_table(measured, combined):
    """Every method at each p, in one table, with the folder it lives in and the label of each number. No pick across label classes:
    rows of different labels are not equals (the PR #7 review)."""
    groups = {}

    def put(method, folder, label, p, cell):
        groups.setdefault((method, folder, LABELS[label]), {})[p] = cell

    for spec in TEST_ROWS:
        key = spec.decoder
        for p in P_VALUES:
            row = measured[p].get(key)
            if row is None:
                continue
            label = ("after-offline" if row.get("offline") else "after") if row.get("after_test") else ("pre-offline" if row.get("offline") else "pre")
            time = "offline" if row.get("offline") else (f"{ms(row['mean_ms'])} ms per pair" + (" (GPU call)" if row["device"] == "GPU" else ""))
            put(DECODERS[key].label, FOLDERS.get(key, ""), label, p, f"{row['failures_pair']:,} pairs (X {row['failures_X']:,}) · {time}")
        if spec.member in ("aw", "ac"):         # the pre-registered offline decode of the hand-written rules, kept beside their timed rows
            for p in P_VALUES:
                member = (combined.get(p) or {}).get("rows", {}).get(spec.member)
                if member is not None:
                    put(DECODERS[key].label, FOLDERS.get(key, ""), "pre-offline", p,
                        f"{member['failures_pair']:,} pairs (X {member['failures_X']:,}) · offline")
    for p in P_VALUES:
        rows = combined[p]["rows"] if combined.get(p) else {}
        if SWAP_RULE in rows:
            row = rows[SWAP_RULE]
            put("TCBP-JOBS in place of relay-BP in our hierarchy (the ensemble's rule d)", "`hierarchy/ensemble.py`", "after-offline", p,
                f"{row['failures_pair']} pairs (X {row['failures_X']}) · offline")
    for p, basis, mark in PREDECODER_TEST:
        n = f"pre{p[-1]}{basis}"
        if FACTS.get(f"{n}_H", (NOT_MEASURED,))[0] == NOT_MEASURED:
            continue
        time = FACTS.get(f"lat{p[-1]}_total", (NOT_MEASURED,))[0] if basis == "X" else NOT_MEASURED
        put(f"the predecoder: relay-BP with the head's priors ({basis} memory only)", "`predecoder/`", "secondary" if not mark else "extension", p,
            f"{basis}: {FACTS[n + '_H'][0]} shot{'' if FACTS[n + '_H'][0] == '1' else 's'}" + (f" · {time} ms per X shot (validation sample)" if time != NOT_MEASURED else ""))
    if FACTS.get("pair_H", (NOT_MEASURED,))[0] != NOT_MEASURED:
        put("the predecoder in both bases, the kit's pair score", "`predecoder/`", "extension", "0.005", f"{FACTS['pair_H'][0]} pairs")
    for p, basis, _ in PREDECODER_FALLBACK:
        n = f"fb{p[-1]}{basis}"
        if FACTS.get(f"{n}_rule", (NOT_MEASURED,))[0] == NOT_MEASURED:
            continue
        label = "after-design" if (p, basis) in (("0.003", "X"), ("0.005", "Z")) else "post hoc"
        put(f"the predecoder with the fallback to TCBP-JOBS ({basis} memory only)", "`predecoder/fallback.py`", label, p,
            f"{basis}: {FACTS[n + '_rule'][0]} shot{'' if FACTS[n + '_rule'][0] == '1' else 's'}")
    if FACTS.get("pair_rule", (NOT_MEASURED,))[0] != NOT_MEASURED:
        put("the predecoder with the fallback, the kit's pair score", "`predecoder/fallback.py`", "post hoc", "0.005", f"{FACTS['pair_rule'][0]} pairs")
    for p, basis in PTCBP_TEST:
        n = f"ptt{p[-1]}{basis}"
        if FACTS.get(f"{n}_TH", (NOT_MEASURED,))[0] == NOT_MEASURED:
            continue
        control = FACTS[f"{n}_fresh"][0]
        put(f"TCBP-JOBS with the head's priors ({basis} memory only)", "`predecoder/` (formerly branch `predecoder-tcbp`)", "ptcbp", p,
            f"{basis}: {FACTS[n + '_TH'][0]} shots" + (f" (same-run control, the kit's priors: {control})" if control != NOT_MEASURED else ""))
    if FACTS.get("ptpair_TH", (NOT_MEASURED,))[0] != NOT_MEASURED:
        put("TCBP-JOBS with the head's priors, the kit's pair score", "`predecoder/` (formerly branch `predecoder-tcbp`)", "ptcbp", "0.005",
            f"{FACTS['ptpair_TH'][0]} pairs (TCBP-JOBS's stored file: {FACTS['ptpair_T'][0]})")
    selector = DATA / "background" / "selector_README.md"
    found = re.search(r"It failed (\d+) of 10,000 pairs, against TCBP-JOBS's (\d+): paired,\s+(\d+) against (\d+), p = (\d+(?:\.\d+)?)",
                      selector.read_text(encoding="utf-8")) if selector.is_file() else None
    if found:
        mine, theirs, only_mine, only_theirs, p_value = found.groups()
        tcbp = measured["0.005"].get("tcbp")
        if tcbp is not None and int(theirs) != tcbp["failures_pair"]:
            raise Mismatch("background/selector_README.md: TCBP-JOBS's failed pairs differ from its row")
        put("a learned selector among the ensemble's members", "`selector/`", "selector", "0.005",
            f"{int(mine):,} pairs (against TCBP-JOBS's {int(theirs):,}: only the selector {only_mine}, only TCBP-JOBS {only_theirs}, exact p = {p_value})")
        fact("sel_pairs", mine, "data/background/selector_README.md")
        fact("sel_p", p_value, "data/background/selector_README.md")
    else:
        fact("sel_pairs", None, "data/background/selector_README.md")
        fact("sel_p", None, "data/background/selector_README.md")
    lines = [[method, folder, label] + [cells.get(p, "") for p in P_VALUES] for (method, folder, label), cells in groups.items()]
    return write("master_comparison",
                 "**Every method we ran on the test shots, in one table.** Each cell is that method's own measured row: failed pairs of 10,000 (the kit's "
                 "score, with its X failures beside it) and the mean time per pair, or, for a method run on one basis only, its failures of 10,000 shots of "
                 "that basis. The **Label** says what kind of number it is; rows of different labels are not equals, so this table picks no best method. "
                 "Times: one process on the team's server unless the cell says otherwise; TCBP-JOBS's is a GPU decode call; the predecoder's is a clean "
                 "one-thread sample on 1,000 validation shots (exploration shots). The paired tests are in the tables each row comes from.\n\n"
                 + markdown(["Method", "Folder", "Label"] + [f"p = {p}" for p in P_VALUES], lines)
                 + "\n\nNot in this table, because it decodes its own simulated shots, not the saved test files: `neutral_atom_noise/` (exploration; "
                   "a noise model built from Infleqtion's published figures, not Infleqtion's noise model, decoded with the kit's relay-BP and BP-OSD; "
                   "`neutral_atom_noise/RESULTS.md`).")


def predecoder_tcbp_table(measured):
    """TCBP-JOBS given the head's per-shot priors (branch predecoder-tcbp): its validation score, exploration shots, and its test rows
    when their files land. Designed after the test results were seen; every row says so."""
    source = "tables/predecoder_tcbp.md"
    lines, test_lines = [], []
    score = DATA / "exploration" / "predecoder_tcbp" / "tcbp_validation_X_p0.005_score.json"
    order = [("TCBP-JOBS, head priors per shot", "TCBP-JOBS with the head's priors, per shot (TH)", "TH"),
             ("TCBP-JOBS, fresh priors per shot", "TCBP-JOBS with the kit's priors, per shot, the same run (the like-for-like control)", "fresh"),
             ("TCBP-JOBS alone (stored)", "TCBP-JOBS with the kit's priors, the stored file (another fit)", "stored"),
             ("relay-BP, head priors per shot (H)", "relay-BP with the head's priors (H, the predecoder)", "H"),
             ("the frozen fallback rule: H where H converged, else T (stored)", "the frozen fallback rule (post hoc on test X at p = 0.005)", "rule"),
             ("fallback b2: H where H converged, else TCBP-JOBS with head priors", "H where H converged, else TH (designed after the test results)", "b2")]
    if score.is_file():
        record = json.loads(score.read_text(encoding="utf-8"))
        if record["data"]["basis"] != "X" or record["data"]["seed"] != 201 or abs(record["data"]["p"] - 0.005) > 1e-12:
            raise Mismatch("exploration/predecoder_tcbp: not the validation X file at p = 0.005")
        arms, pairs, shots = record["arms"], record["pairs"], record["shots"]
        for key, label, short in order:
            count = arms[key]["failures"]
            low, high = stats.wilson(count, shots)
            pair = pairs.get(f"TCBP-JOBS, head priors per shot | {key}")
            lines.append([label, f"{count} of {shots:,}", f"{count / shots:.2%} ({low:.2%} to {high:.2%})",
                          "" if pair is None else f"{pair['both']} / {pair['only_first']} / {pair['only_second']} (p = {stats.p_text(pair['p'])})"])
            fact(f"ptv_{short}", count, source)
            if pair is not None:
                fact(f"ptv_{short}_only_TH", pair["only_first"], source)
                fact(f"ptv_{short}_only_other", pair["only_second"], source)
                fact(f"ptv_{short}_p", stats.p_text(pair["p"]), source)
    TH, T, FRESH = "TCBP-JOBS, head priors per shot", "TCBP-JOBS alone (stored)", "TCBP-JOBS, fresh priors per shot"
    for p, basis in PTCBP_TEST:
        n = f"ptt{p[-1]}{basis}"
        file = DATA / "test" / "predecoder_tcbp" / f"tcbp_test_{basis}_p{p}_score.json"
        if not file.is_file():
            test_lines.append([p, basis] + ["not measured yet"] * 5)
            for name in ("TH", "T", "fresh", "vs_T_only_TH", "vs_T_only_T", "vs_T_p", "vs_fresh_p"):
                fact(f"{n}_{name}", None, source)
            continue
        record = json.loads(file.read_text(encoding="utf-8"))
        if record["data"]["sha256"] != TEST_SHA256[p][basis] or record["data"]["basis"] != basis:
            raise Mismatch(f"test/predecoder_tcbp/{file.name}: not the team's test file")
        arms, pairs = record["arms"], record["pairs"]
        tcbp = measured[p].get("tcbp")
        if tcbp is not None and arms[T]["failures"] != tcbp[f"failures_{basis}"]:
            raise Mismatch(f"test/predecoder_tcbp/{file.name}: the stored TCBP-JOBS fails {arms[T]['failures']} shots there and {tcbp[f'failures_{basis}']} in its row")
        declared, fresh = pairs[f"{TH} | {T}"], pairs.get(f"{TH} | {FRESH}")
        test_lines.append([p, basis, f"{arms[TH]['failures']} of {record['shots']:,}", f"{arms[T]['failures']}",
                           f"{declared['both']} / {declared['only_first']} / {declared['only_second']} (p = {stats.p_text(declared['p'])})",
                           f"{arms[FRESH]['failures']}" if FRESH in arms else "not run (X at p = 0.005 only)",
                           "" if fresh is None else f"{fresh['both']} / {fresh['only_first']} / {fresh['only_second']} (p = {stats.p_text(fresh['p'])})"])
        for name, value in (("TH", arms[TH]["failures"]), ("T", arms[T]["failures"]), ("fresh", arms[FRESH]["failures"] if FRESH in arms else None),
                            ("vs_T_only_TH", declared["only_first"]), ("vs_T_only_T", declared["only_second"]), ("vs_T_p", stats.p_text(declared["p"])),
                            ("vs_fresh_p", fresh and stats.p_text(fresh["p"]))):
            fact(f"{n}_{name}", value, source)
    time_lines = []
    network_file = DATA / "exploration" / "predecoder_tcbp" / "gpu_network_decode_val1000_fp32_summary.json"
    job5_file = DATA / "test" / "predecoder_tcbp" / "job5_test_X_p0.005_summary.json"
    if network_file.is_file() and job5_file.is_file():
        network = json.loads(network_file.read_text(encoding="utf-8"))
        job5 = json.loads(job5_file.read_text(encoding="utf-8"))
        if job5["data"]["sha256"] != TEST_SHA256["0.005"]["X"] or network["data"]["basis"] != "X" or network["data"]["seed"] != 201:
            raise Mismatch("predecoder_tcbp timing: not the files they should be (test X at p = 0.005; validation X)")
        if job5["arms"]["head"]["failures"] != int(FACTS["ptt5X_TH"][0]):
            raise Mismatch("predecoder_tcbp timing: job 5's head arm fails otherwise than its score file")
        per = network["logits"]["per_shot_seconds"]
        head = job5["arms"]["head"]
        total = per["mean"] * 1e3 + head["build_decode_ms_mean"]
        time_lines = [["the network and the head on the A100, 32-bit floats", ms(per["mean"] * 1e3), ms(per["median"] * 1e3), ms(per["p99"] * 1e3),
                       f"the first {per['shots']:,} validation X shots at p = 0.005 (exploration shots; server job 4)"],
                      ["TCBP-JOBS built with the head's priors and run, per shot", ms(head["build_decode_ms_mean"]), ms(head["build_decode_ms_median"]),
                       ms(head["build_decode_ms_p99"]), f"the 10,000 test X shots at p = 0.005 (server job 5; build {ms(head['build_ms_mean'])}, decode "
                                                        f"{ms(head['decode_ms_mean'])} ms on average)"],
                      ["the two together", ms(total), "", "", ""]]
        for name, value in (("net", ms(per["mean"] * 1e3)), ("bd", ms(head["build_decode_ms_mean"])), ("build", ms(head["build_ms_mean"])),
                            ("decode", ms(head["decode_ms_mean"])), ("total", ms(total))):
            fact(f"pttime_{name}", value, source)
    else:
        for name in ("net", "bd", "build", "decode", "total"):
            fact(f"pttime_{name}", None, source)
    landed = [(p, basis) for p, basis in PTCBP_TEST if FACTS[f"ptt{p[-1]}{basis}_TH"][0] != NOT_MEASURED]
    fact("ptt_statement", landed and ("On the test shots (" + PTCBP_LABEL + "; one basis at a time, accuracy only; its knobs were fitted at p = 0.005 and "
         "applied unchanged at the lower p, with no validation there): " + "; ".join(
             f"{basis} at p = {p}, TH fails {FACTS[f'ptt{p[-1]}{basis}_TH'][0]} against TCBP-JOBS's {FACTS[f'ptt{p[-1]}{basis}_T'][0]} (only TH "
             f"{FACTS[f'ptt{p[-1]}{basis}_vs_T_only_TH'][0]}, only TCBP-JOBS {FACTS[f'ptt{p[-1]}{basis}_vs_T_only_T'][0]}, exact p = "
             f"{FACTS[f'ptt{p[-1]}{basis}_vs_T_p'][0]}" + (": no claim" if float(FACTS[f'ptt{p[-1]}{basis}_vs_T_p'][0]) >= 0.05 else "")
             + (f"; the same run's kit-priors arm fails {FACTS[f'ptt{p[-1]}{basis}_fresh'][0]}" if FACTS[f"ptt{p[-1]}{basis}_fresh"][0] != NOT_MEASURED else "")
             + ")" for p, basis in landed) + ". TCBP-JOBS's stored counts carry its fit's spread (above).") or "Its test rows have not landed yet.", source)
    pair_file = DATA / "test" / "predecoder_tcbp" / "tcbp_pair_test_p0.005.json"
    if pair_file.is_file():
        record = json.loads(pair_file.read_text(encoding="utf-8"))
        tcbp = measured["0.005"].get("tcbp")
        if tcbp is not None and record["T"]["failed_pairs"] != tcbp["failures_pair"]:
            raise Mismatch("test/predecoder_tcbp/tcbp_pair_test_p0.005.json: TCBP-JOBS's failed pairs differ from its row")
        pair = record["pair"]
        test_lines.append(["0.005", "pairs", f"{record['TH']['failed_pairs']} of {record['TH']['pairs']:,} pairs", f"{record['T']['failed_pairs']}",
                           f"{pair['both']} / {pair['only_TH']} / {pair['only_T']} (p = {stats.p_text(pair['p'])})", "", ""])
        fact("ptpair_TH", record["TH"]["failed_pairs"], source)
        fact("ptpair_T", record["T"]["failed_pairs"], source)
        fact("ptpair_p", stats.p_text(pair["p"]), source)
    else:
        for name in ("TH", "T", "p"):
            fact(f"ptpair_{name}", None, source)
    return write("predecoder_tcbp",
                 f"**TCBP-JOBS with the head's priors: {PTCBP_LABEL}.** TCBP-JOBS (NVIDIA's GPU decoder) given, for each shot, the priors our "
                 "network's head computes from that shot's detectors, in place of the kit's fixed priors.\n\n"
                 "**Exploration shots, not test shots:** the 5,000 shots of `data/p0.005/validation_X.npz`. TH against each row: both fail / only TH "
                 "/ only that row (exact p), on the same shots.\n\n"
                 + (markdown(["Decoder", "Failed shots", "Share (95% interval)", "TH against it: both / only TH / only it (exact p)"], lines) if lines
                    else NOT_MEASURED)
                 + f"\n\n**Test shots: {PTCBP_LABEL}.** One basis at a time (X until the Z run lands), accuracy only. Its knobs were fitted at "
                 "p = 0.005 and applied unchanged at p = 0.004 and 0.003, with no validation there. The declared pair is TH against TCBP-JOBS's stored "
                 "test file; at p = 0.005 the same run's kit-priors arm is the like-for-like control.\n\n"
                 + markdown(["p", "Basis", "TH fails", "TCBP-JOBS, its stored test file (its row in this report), fails",
                             "TH against it, the declared pair: both / only TH / only TCBP-JOBS (exact p)",
                             "TCBP-JOBS with the kit's priors in the same run (the like-for-like control) fails", "TH against that control (exact p)"],
                            test_lines)
                 + ("\n\n**Its time per X shot: not a clean timing.** An A100 MIG slice, one process, one shot at a time, 1 CPU thread; server job 5 "
                    "ran beside job 6, another TCBP-JOBS process on the same slice (`runner_gpu.out`, `runner_cpu.out`); the network's time is from "
                    "validation shots. Per X shot, not per pair; ms: mean / median / 99th percentile. For scale: TCBP-JOBS alone takes "
                    f"{FACTS['lat5_tcbp_x'][0]} ms per X shot (a GPU decode call), relay-BP's timed row {FACTS['lat5_relay_x'][0]}, and the relay-BP "
                    f"predecoder {FACTS['lat5_total'][0]} (`predecoder_latency.md`).\n\n"
                    + markdown(["Part", "Mean", "Median", "99th percentile", "Shots"], time_lines) if time_lines else "")
                 + "\n\nSource: `report/data/exploration/predecoder_tcbp/` (branch `predecoder-tcbp` at `dad4fd50`; job 4 from `server-logs`) and "
                   "`report/data/test/predecoder_tcbp/` (job 5 from `server-logs`).")


def ours_at(measured, p):
    """Our timed decoder at p: the one-core row, or the two-core row (checked to give the same answers) if the first has not landed."""
    return measured[p].get("hierarchy") or measured[p].get("hierarchy-parallel")


def slide_facts(measured):
    """The titles and lines of the result slides, written from the data so that they stay true as rows land."""
    against, fewer, level, tcbp_level, tcbp_ahead, tcbp_behind = [], [], [], [], [], []
    for p in P_VALUES:
        ours, relay, tcbp = ours_at(measured, p), measured[p].get(REFERENCE), measured[p].get("tcbp")
        if ours is None or relay is None or ours["failed_pair"] is None or relay["failed_pair"] is None:
            continue
        paired = stats.paired(relay["failed_pair"], ours["failed_pair"])
        against.append(f"p = {p}: {ours['failures_pair']} against {relay['failures_pair']} (exact p = {stats.p_text(paired['p'])})")
        (fewer if ours["failures_pair"] < relay["failures_pair"] and paired["p"] < 0.05 else level).append(p)
        if tcbp is not None and tcbp["failed_pair"] is not None:
            versus = stats.paired(ours["failed_pair"], tcbp["failed_pair"])
            if versus["p"] >= 0.05:
                tcbp_level.append(p)
            else:
                (tcbp_ahead if tcbp["failures_pair"] < ours["failures_pair"] else tcbp_behind).append(p)
    join = lambda levels: ", ".join(levels[:-1]) + (" and " if len(levels) > 1 else "") + levels[-1] if levels else ""
    if against:
        title = f"On the test shots our decoder fails fewer pairs than relay-BP at p = {join(fewer)}" if fewer else "On the test shots our decoder is level with relay-BP"
        if fewer and level:
            title += f", and as many within the error at p = {join(level)}"
        parts = ([f"level with TCBP-JOBS at p = {join(tcbp_level)}"] if tcbp_level else []) + \
                ([f"behind it at p = {join(tcbp_ahead)}"] if tcbp_ahead else []) + ([f"ahead of it at p = {join(tcbp_behind)}"] if tcbp_behind else [])
        if parts:
            title += "; it is " + ", and ".join(parts)
        fact("slide4_title", title, "tables/test_p0.003.md, test_p0.004.md, test_p0.005.md")
        fact("slide4_relay", "Failed pairs of 10,000, ours against relay-BP on the same pairs: " + "; ".join(against) + ".", "tables/test_p<p>.md")
    else:
        fact("slide4_title", "On the test shots: pending the measured rows", "tables/test_p<p>.md")
        fact("slide4_relay", "Pending: our rows on the test shots.", "tables/test_p<p>.md")
    one, two = [], []
    for p in P_VALUES:
        relay = measured[p].get(REFERENCE)
        if relay is None or relay["mean_ms"] is None or not relay.get("time_clean", True):
            continue
        for key, bucket in (("hierarchy", one), ("hierarchy-parallel", two)):
            row = measured[p].get(key)
            if row is not None and row["mean_ms"] is not None and row.get("time_clean", True):
                bucket.append((p, row["mean_ms"] / relay["mean_ms"], row, relay))
    span = lambda bucket: (f"{min(r for _, r, _, _ in bucket):.2f}" if len(bucket) == 1 else
                           f"{min(r for _, r, _, _ in bucket):.2f} to {max(r for _, r, _, _ in bucket):.2f}")
    if one or two:
        title = "Our decoder takes " + " and ".join(part for part in (
            f"{span(one)} of relay-BP's mean time per pair on one core" if one else "",
            f"{span(two)} with its two fast decodes on two cores" if two else "") if part)
        fact("slide5_title", title, "tables/test_time.md")
        lines = [f"p = {p}: ours {ms(row['mean_ms'])} ms mean and {ms(row['p99_ms'])} ms at the 99th percentile, relay-BP {ms(relay['mean_ms'])} and {ms(relay['p99_ms'])}"
                 for p, _, row, relay in (one or two)]
        fact("slide5_lines", "; ".join(lines) + ".", "tables/test_time.md")
    else:
        fact("slide5_title", "Time: pending the measured rows", "tables/test_time.md")
        fact("slide5_lines", "Pending.", "tables/test_time.md")


# ---------- the section for the top-level README ----------

README_HEADING = "## Our part: a hierarchical decoder with a learned tie-breaker, and a study of atom loss"


def refresh_top_readme(section):
    """Once our section has been appended to the top-level README.md, keep it equal to the generated one.

    Only the text from our heading to the end of the file is replaced; every line above it (the teammates') stays as
    it is. Nothing is appended here: the first append is done by hand, once. Not run by the tests (REPORT_OUT).
    """
    if not DOCUMENTS:
        return
    readme = HERE.parent.parent / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.is_file() else ""
    if text.count(README_HEADING) != 1:
        return
    head = text[:text.index(README_HEADING)]
    if not head.endswith("\n\n"):
        raise Mismatch("the top-level README.md: our heading does not follow a blank line")
    section = section.rstrip("\n") + "\n"                 # as in tables/readme_section.md
    if head + section != text:
        readme.write_text(head + section, encoding="utf-8")


def readme_section(measured):
    """The one section appended to the top-level README.md: at most 25 lines, every number from the tables above."""
    value = lambda name: FACTS[name][0]
    tcbp = value("tcbp_statement")
    verdict = (f"on the test shots that network is {value('abl_verdict')} the best hand-written tie-break, so the gain over relay-BP is "
               "PriorTilt's and the hierarchy's" if value("abl_verdict") != "better than" else value("slide3_close")[0].lower() + value("slide3_close")[1:-1])
    result = value("slide4_title") + ("; " + value("ensemble_short") if value("ensemble_short") != NOT_MEASURED else "") + "."
    lines = [README_HEADING, "",
             "Two runs of a fast decoder (IonQ's PriorTilt) settle the shots they agree on; where they disagree, relay-BP joins and a neural",
             f"network picks the most likely answer (`challenge-1-ml-decoder-kit/hierarchy/`, `neural/`), though {verdict}. " + result,
             f"Under atom loss (our own simplified model), relay-BP with the loss flags fails {value('loss_aware_share')} of shots where a loss-blind one",
             f"fails {value('loss_blind_share')}; our loss-time tracker, which estimates from the syndrome when each atom was lost, recovers "
             f"{value('trk_gain')} of what knowing the true loss time would add (`loss/`, `loss_tracker/`).",
             "",
             "Every version we built is in one table, each number with its label ([report/tables/master_comparison.md]"
             "(challenge-1-ml-decoder-kit/report/tables/master_comparison.md)): the frozen hierarchy above (pre-registered); the same hierarchy with "
             "a hand-written tie-break in place of the network (`hierarchy/`; the best hand-written rule pre-registered as timed rows at p = 0.003 "
             "and 0.004 and offline at 0.005, plain lowest weight pre-registered offline at every p; their other timed rows after-test ablations); "
             "TCBP-JOBS in place of relay-BP in it (an after-test ablation); a learned predecoder that gives relay-BP per-shot priors from the network "
             "(`predecoder/`; X memory at p = 0.005 a secondary result, chosen after its own gate failed, and the other rows a registered extension run "
             "after it) and its fallback to TCBP-JOBS where relay-BP does not converge (`predecoder/fallback.py`; designed after the first test result)"
             + ("; and TCBP-JOBS given the same per-shot priors (" + PTCBP_LABEL + ")" if FACTS.get("ptv_TH", (NOT_MEASURED,))[0] != NOT_MEASURED else "")
             + "; a learned selector among the ensemble's members (`selector/`; frozen on validation, its line not met on the test shots); and, on its "
               "own simulated shots, noise shaped like neutral-atom hardware (`neutral_atom_noise/`; exploration, the kit's decoders). None of these "
               "rows replaces the frozen decoder.",
             ""]
    header = ["Decoder"] + [f"p = {p}" for p in P_VALUES]
    table = []
    for key in ("bposd", "relay-bp", "priortilt", "hierarchy", "tcbp"):
        cells = []
        for p in P_VALUES:
            row = ours_at(measured, p) if key == "hierarchy" else measured[p][key]
            cells.append(NOT_MEASURED if row is None else f"{stats.sci(row['rate'])} ± {stats.sci(row['error'], 1)}, "
                         + (f"{ms(row['mean_ms'])} ms" + (" (2 cores)" if row is measured[p].get("hierarchy-parallel") else "")
                            if row["mean_ms"] is not None and not row.get("offline") else "offline, no time"))
        table.append([DECODERS[key].label] + cells)
    lines += markdown(header, table).split("\n")
    lines += ["",
              "Test shots (10,000 pairs per p): logical error rate per cycle ± the kit's error, mean time per pair (CPU rows: the whole decode of "
              "a pair in one process; TCBP-JOBS: a GPU decode call alone, not comparable with the CPU rows). " + tcbp,
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
            pattern = re.compile(rf"(<!-- BEGIN table:{re.escape(name)} -->\n)(?:.*?\n)?(<!-- END table:{re.escape(name)} -->)", re.S)
            text = pattern.sub(lambda found: found.group(1) + table.rstrip("\n") + "\n" + found.group(2), text)
        empty = re.findall(r"<!-- BEGIN table:(\S+) -->\n<!-- END table:", text)
        if empty:
            raise Mismatch(f"{document.name}: tables left empty: {empty}")
        missing = sorted(set(re.findall(r"<!-- BEGIN table:(\S+) -->", text)) - set(tables))
        if missing:
            raise Mismatch(f"{document.name} asks for tables that do not exist: {missing}")
        unknown = sorted(set(re.findall(r"<!--f:(\w+)-->", text)) - set(FACTS))
        if unknown:
            raise Mismatch(f"{document.name} quotes facts that do not exist: {unknown}")
        value = lambda name: FACTS[name][0]
        text = re.sub(r"<!--f:(\w+)-->.*?<!--/f-->", lambda found: f"<!--f:{found.group(1)}-->{value(found.group(1))}<!--/f-->", text, flags=re.S)
        document.write_text(text, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)
    try:
        measured, extra, combined = load_test_rows()
        per_p, (sweep, sweep_slide) = test_tables(measured)
        ladder, ablated = ladder_tables(combined)
        earlier, cross_checks = earlier_table(measured)
        exploration, _ = exploration_table()
        tables = {f"test_p{p}": per_p[p] for p in P_VALUES} | {"test_time": per_p["time"]} | {
            "test_sweep": sweep, "test_sweep_slide": sweep_slide, "earlier_runs": earlier, "tcbp_own_shots": tcbp_own_table(), "exploration_p0.004": exploration,
            "test_comparisons": comparisons_table(measured), "test_ladder": ladder, "test_ablation": ablated,
            "test_ladder_short": (TABLES / "test_ladder_short.md").read_text(encoding="utf-8"),
            "where_failures": failure_split_table(combined, measured), "exploration_p0.005": exploration_p5_table(), "time_split": time_split_table(measured),
            "exploration_ensemble": exploration_ensemble_table(), "ablation_80k": ablation_80k_table(),
            "loss": loss_table(), "networks": networks_table(), "background": background_table(),
            "resources": resources_table(measured), "test_after_swap": swap_table(combined)}
        tables["test_after_swap_short"] = (TABLES / "test_after_swap_short.md").read_text(encoding="utf-8")
        tables["test_predecoder"] = predecoder_table(measured)
        tables["test_network_timed"] = network_timed_table(measured)
        tables["predecoder_latency"] = predecoder_latency_table(measured)
        tables["operating_rule"] = operating_rule_table(measured)
        tables["predecoder_tcbp"] = predecoder_tcbp_table(measured)
        tables["master_comparison"] = master_table(measured, combined)
        test_job_facts()
        tables["checks"] = checks_text(measured, extra, cross_checks, mode_checks(measured, combined))
        tcbp_statement(measured)
        tcbp_fit_caveat()
        ensemble_statement(measured, combined)
        swap_statement()
        timed_ablation(measured)
        findings(measured)
        slide_facts(measured)
        refresh_top_readme(readme_section(measured))
        write("facts", "Single numbers quoted in the prose of `REPORT.md` and `SLIDES.md` (written there as `<!--f:name-->value<!--/f-->`), "
                       "each with the generated table it is taken from.\n\n"
              + markdown(["Name", "Value", "From"], [[f"`{name}`", value, f"`{source}`"] for name, (value, source) in sorted(FACTS.items())]))
        inject(tables)
    except Mismatch as problem:
        print(f"STOPPED: {problem}", file=sys.stderr)
        return 1
    timed = sum(row is not None and not row.get("offline") for p in P_VALUES for row in measured[p].values())
    offline = sum(row is not None and bool(row.get("offline")) for p in P_VALUES for row in measured[p].values())
    print(f"wrote {len(tables) + 1} tables to {TABLES}/; of {len(TEST_ROWS) * len(P_VALUES)} test rows, {timed} timed and {offline} offline")
    for name in extra:
        print(f"  note: report/data/test/{name} is not a registered row")
    return 0


if __name__ == "__main__":
    sys.exit(main())
