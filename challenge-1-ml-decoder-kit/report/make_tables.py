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
from pathlib import Path

import numpy as np

from report import loaders, stats
from report.loaders import DATA, Mismatch
from report.rows import DECODERS, EARLIER_ROWS, LOSS_CONDITIONS, P_VALUES, REFERENCE, TCBP_OWN, TEST_ROWS, TEST_SHA256

HERE = Path(__file__).resolve().parent
# REPORT_OUT sends the tables elsewhere and leaves the documents alone: used by the tests only.
TABLES = Path(os.environ.get("REPORT_OUT") or HERE) / "tables"
DOCUMENTS = [] if os.environ.get("REPORT_OUT") else [HERE / "REPORT.md", HERE / "SLIDES.md"]
NOT_MEASURED = "not measured"


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

def load_test_rows(data=DATA):
    """{p: {decoder key: row or None}} for every registered row, checked against the test files' identity."""
    log = loaders.clean_rows_log(data / "test" / "clean_rows.out")
    measured = {}
    for p in P_VALUES:
        measured[p] = {}
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
            elif spec.kind == "ensemble":
                row = loaders.ensemble(folder, p) if hasattr(loaders, "ensemble") else None
            if row is not None:
                if abs(float(row["p"]) - float(p)) > 1e-12:
                    raise Mismatch(f"{folder}: the file says p = {row['p']}, the folder says {p}")
                if row["test_sha256"] != TEST_SHA256[p]:
                    raise Mismatch(f"{folder}: these are not the team's test shots for p = {p} (sha256 {row['test_sha256']})")
                run = log.get((p, spec.folder.format(p=p).removeprefix(f"eval_p{p}_")))
                row["beside"] = run["beside"] if run else None
                row["run"] = run
            measured[p][spec.decoder] = row
        reference = measured[p].get(REFERENCE)
        for key, row in measured[p].items():
            if row is not None:
                row["paired"] = (stats.paired(reference["failed_pair"], row["failed_pair"])
                                 if key != REFERENCE and reference is not None and reference["failed_pair"] is not None
                                 and row["failed_pair"] is not None else None)
                row["interval"] = stats.per_cycle_interval(row["failures_pair"], row["shots"])
    known = {spec.folder.format(p=p) for spec in TEST_ROWS for p in P_VALUES}
    extra = sorted(path.name for path in (data / "test").glob("*") if path.is_dir() and path.name not in known) if (data / "test").is_dir() else []
    return measured, extra


def where_text(row):
    text = f"{row['device']}, {row['workers']} process, {row['site']}"
    if row.get("beside") is not None:
        text += f"; {len(row['beside'])} other evaluator rows running beside it" if row["beside"] else "; no other evaluator row beside it"
    return text


def paired_text(row, key):
    if key == REFERENCE:
        return "the reference"
    if row.get("paired") is None:
        return "no per-shot file"
    paired = row["paired"]
    return f"{paired['both']} / {paired['row_only']} / {paired['reference_only']} (p = {stats.p_text(paired['p'])})"


def test_tables(measured):
    texts, records = {}, []
    header = ["Decoder", "Built by", "Failures X, Z", "Failed pairs", "Rate per cycle ± error", "95% interval",
              "Time per pair, ms: mean / median / p99", "Against relay-BP, same pairs: both fail / only this row / only relay-BP (exact p)", "Where it ran"]
    for p in P_VALUES:
        lines = []
        for key, row in measured[p].items():
            decoder = DECODERS[key]
            record = {"p": p, "decoder": key, "label": decoder.label, "built_by": decoder.whose, "measured": row is not None}
            if row is None:
                lines.append([decoder.label, decoder.whose] + [NOT_MEASURED] * 7)
            else:
                times = " / ".join(ms(row[name]) for name in ("mean_ms", "median_ms", "p99_ms")) if row["mean_ms"] is not None else NOT_MEASURED
                lines.append([decoder.label, decoder.whose, f"{row['failures_X']}, {row['failures_Z']}",
                              f"{row['failures_pair']} of {row['shots']:,}", rate_text(row["rate"], row["error"]),
                              interval_text(*row["interval"]), times, paired_text(row, key), where_text(row)])
                paired = row.get("paired") or {}
                record |= {"pairs": row["shots"], "failures_X": row["failures_X"], "failures_Z": row["failures_Z"],
                           "failures_pair": row["failures_pair"], "rate_per_cycle": row["rate"], "kit_error": row["error"],
                           "wilson95_low": row["interval"][0], "wilson95_high": row["interval"][1],
                           "mean_ms_per_pair": row["mean_ms"], "median_ms_per_pair": row["median_ms"], "p99_ms_per_pair": row["p99_ms"],
                           "both_fail": paired.get("both"), "only_this_row_fails": paired.get("row_only"),
                           "only_relay_bp_fails": paired.get("reference_only"), "exact_p": paired.get("p"),
                           "device": row["device"], "processes": row["workers"], "machine": row["site"],
                           "rows_beside": "; ".join(row["beside"]) if row.get("beside") else "",
                           "peak_rss_mb_parent": row.get("peak_rss_mb"), "source": row["source"],
                           "test_X_sha256": row["test_sha256"]["X"], "test_Z_sha256": row["test_sha256"]["Z"]}
            records.append(record)
        note = (f"Test shots, p = {p}: `data/p{p}/test_X.npz` and `test_Z.npz` (10,000 shots per basis, Stim seeds 301 and 302; "
                f"sha256 `{TEST_SHA256[p]['X'][:12]}…` and `{TEST_SHA256[p]['Z'][:12]}…`). Every measured row decoded these two files.")
        texts[p] = write(f"test_p{p}", note + "\n\n" + markdown(header, lines), [r for r in records if r["p"] == p])
    write("test_all", "\n\n".join(texts[p] for p in P_VALUES), records)

    # the sweep in one table: a decoder per row, a noise level per column
    lines = []
    for key in DECODERS:
        cells = []
        for p in P_VALUES:
            row = measured[p][key]
            if row is None:
                cells.append(NOT_MEASURED)
            else:
                time = f"{ms(row['mean_ms'])} ms" if row["mean_ms"] is not None else "time " + NOT_MEASURED
                cells.append(f"{rate_text(row['rate'], row['error'])} · {row['failures_pair']} pairs · {time}")
        lines.append([DECODERS[key].label] + cells)
    sweep = write("test_sweep", markdown(["Decoder"] + [f"p = {p}" for p in P_VALUES], lines)
                  + "\n\nEach cell: logical error rate per cycle ± the kit's error · failed pairs of 10,000 · mean time per pair. "
                    "Test shots; `report/tables/test_p<p>.md` has the intervals, the paired counts and where each row ran.")
    return texts, sweep


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
            checks.append(f"p = {p}: relay-BP's timed row has not landed, so it cannot be compared with the teammates' earlier run ({where}).")
        else:
            differ = int((timed["failed_pair"] != earlier["failed_pair"]).sum())
            checks.append(f"p = {p}: relay-BP's timed row and the teammates' earlier run ({where}) give "
                          + ("the same outcome on every one of the 10,000 pairs." if differ == 0 else
                             f"a different outcome on {differ} of the 10,000 pairs ({timed['failures_pair']} against {earlier['failures_pair']} failed pairs)."))
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
    rule = {basis: loaders.weight_rule(shots[basis]) for basis in "XZ"}
    network = {step: loaders.network_rule_table(folder / f"results_network_select_step{step}.txt") for step in (30000, 40000, 50000)}

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
        facts[f"disagree_{basis}"] = int((~agreed).sum())
        if basis == "X":
            for step, table in network.items():
                if table and "b. network breaks ties" in table["rows"]:
                    count, time = table["rows"]["b. network breaks ties"]
                    if table["rows"]["R alone"][0] != facts["R_X"]:
                        raise Mismatch(f"{table['source']}: relay-BP's failures differ from the per-shot files")
                    lines.append([scope, f"our hierarchy, neural tie-break (network at training step {step:,})", "ours",
                                  f"{count} of 25,000", percent(count, 25000, 3), "in the laptop's table: not paired here", ms(time)])
                    records.append({"shots": scope, "row": f"hierarchy, network step {step}", "built_by": "ours", "failures": count, "of": 25000,
                                    "mean_ms_per_shot": time})
                    facts[f"N{step}_X"] = count
    # pairs: shot k of X with shot k of Z, in the same file order (validation, then train)
    pair = {name: shots["X"][f"failed_{name}"].astype(bool) | shots["Z"][f"failed_{name}"].astype(bool) for name in ("R", "P1", "P2")}
    pair["W"] = rule["X"][0] | rule["Z"][0]
    pair_lines = []
    for name, label, whose in (("R", "relay-BP (kit baseline)", "the kit"), ("P1", "PriorTilt, seed A", "IonQ"), ("P2", "PriorTilt, seed B", "IonQ"),
                               ("W", "our hierarchy, lowest-weight tie-break", "ours")):
        count, total = int(pair[name].sum()), len(pair[name])
        rate, error = stats.per_cycle_rate(count, total)
        paired = stats.paired(pair["R"], pair[name]) if name != "R" else None
        seconds = ((shots["X"][f"seconds_{name}"] + shots["Z"][f"seconds_{name}"]) if name != "W" else rule["X"][2] + rule["Z"][2])
        pair_lines.append([label, whose, f"{count} of {total:,}", rate_text(rate, error), interval_text(*stats.per_cycle_interval(count, total)),
                           "the reference" if paired is None else f"{paired['both']} / {paired['row_only']} / {paired['reference_only']} (p = {stats.p_text(paired['p'])})",
                           ms(float(seconds.mean()) * 1e3)])
        facts[f"pairs_{name}"] = count
        facts[f"rate_{name}"] = rate
        records.append({"shots": "25,000 pairs", "row": label, "built_by": whose, "failures": count, "of": total, "rate_per_cycle": rate, "kit_error": error,
                        "both_fail": paired and paired["both"], "only_this_row": paired and paired["row_only"],
                        "only_relay_bp": paired and paired["reference_only"], "exact_p": paired and paired["p"],
                        "mean_ms_per_pair": float(seconds.mean()) * 1e3})
    # where the failures are: the shots on which the two PriorTilt runs agree, and those on which they differ
    agreed = np.concatenate([rule[basis][1] for basis in "XZ"])
    first = np.concatenate([shots[basis]["failed_P1"].astype(bool) for basis in "XZ"])
    left = np.concatenate([rule[basis][0] for basis in "XZ"])
    split_lines = [[label, f"{int(mask.sum()):,}", f"{int((mask & agreed).sum()):,}", f"{int((mask & ~agreed).sum()):,}",
                    f"{(mask & ~agreed).sum() / mask.sum():.1%}"]
                   for label, mask in (("all shots (X and Z memory)", np.ones(len(agreed), bool)), ("shots PriorTilt, seed A, gets wrong", first),
                                       ("shots still wrong after the lowest-weight tie-break", left))]
    facts |= {"shots": len(agreed), "disagree": int((~agreed).sum()), "P1_wrong_disagree": int((first & ~agreed).sum()),
              "left_agree": int((left & agreed).sum())}
    text = ("**Exploration shots, not test shots**: the team's train and validation files at p = 0.004 (`data/p0.004/train_*.npz`, "
            "`validation_*.npz`; 25,000 shots per basis). The rules were chosen on these same shots, which flatters them. "
            "Decoded on a laptop (Ryzen 9 8945HS) beside a GPU training: compare the times with each other only. "
            "In this first table a time is per shot of one basis.\n\n"
            + markdown(["Shots", "Decoder", "Built by", "Failures", "Share of shots (95% interval)",
                        "Against relay-BP, same shots: both fail / only this row / only relay-BP (exact p)", "Mean time per shot, ms"], lines)
            + "\n\nThe same decodes as pairs (shot k of X with shot k of Z), the kit's score; times per pair:\n\n"
            + markdown(["Decoder", "Built by", "Failed pairs", "Rate per cycle ± error", "95% interval",
                        "Against relay-BP: both fail / only this row / only relay-BP (exact p)", "Mean time per pair, ms"], pair_lines)
            + "\n\nWhere the failures are: the shots on which the two PriorTilt runs predict the same 12 bits (\"agree\"), and the others:\n\n"
            + markdown(["Set", "Shots", "The two runs agree", "They disagree", "Share in \"disagree\""], split_lines)
            + "\n\nSources: `report/data/exploration/priortilt/selector_*.npz` (per shot: relay-BP and PriorTilt under two seeds), and for the "
              "rows with the network `results_network_select_step*.txt` (the laptop's own table).")
    return write("exploration_p0.004", text, records), facts


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
    text = ("X memory, p = 0.004, relay-BP at the kit's settings on shots simulated with Clifft under **our own simplified loss model** "
            "(after every CX each of its two atoms is lost with probability s; flags are perfect). It is not Infleqtion's hardware model. "
            "These are exploration numbers on shots made for this study; a shot here is one X-memory shot, not a pair. "
            "Each loss rate and machine has its own dataset, so rows of different datasets are not paired with each other.\n\n"
            + markdown(["s per atom per CX", "Dataset", "Decoder", "Failures", "Share of shots (95% interval)", "Paired on the same shots"], lines)
            + "\n\nAn ORACLE row is given the true loss round on purpose: it is a ceiling for a tracker of the loss time, not a decoder result. "
              "Sources: `report/data/loss/pc/*.json`, `report/data/loss/server/*/tables.md`.")
    return write("loss", text, records)


# ---------- the networks ----------

def networks_table():
    lines, records = [], []
    scores = loaders.scores_md(DATA / "exploration" / "cascade_X_s1" / "SCORES.md")
    for name, where in (("cascade_X_s1", "a laptop GPU (RTX 4060, 8 GB)"), ("cascade_Z_s1", "the ICRN server (A100 slice, 20 GB)")):
        log = loaders.train_log(DATA / "networks" / f"{name}.train.log")
        if log is None or not log["reports"]:
            lines.append([name] + [NOT_MEASURED] * 6)
            continue
        last = log["reports"][-1]
        wrong = " / ".join(f"{last['wrong'][p]:.1%}" for p in P_VALUES)
        speed = sorted(log["shots_per_second"])[len(log["shots_per_second"]) // 2]
        start = "the X network's weights, mirrored" if log["warm_start"] else "random weights"
        lines.append([f"{log['basis']} memory (`{name}`)", where, f"width {log['width']}, depth {log['depth']}, {log['parameters']:,} parameters",
                      start, f"{last['step']:,} of {last['of']:,}", f"{last['step'] * log['batch']:,}", f"{speed:,.0f}", wrong])
        records.append({"network": name, "basis": log["basis"], "machine": where, "parameters": log["parameters"], "last_step_in_log": last["step"],
                        "planned_steps": last["of"], "shots_seen": last["step"] * log["batch"], "median_shots_per_second": speed,
                        **{f"wrong_p{p}": last["wrong"][p] for p in P_VALUES}, "source": log["source"]})
    text = ("The two networks, from their training logs (`report/data/networks/*.train.log`). \"Shots wrong\" is the share of shots with at least "
            "one of the 12 logical bits wrong, on error-model shots drawn for the training report: a training diagnostic, not a result on Stim shots.\n\n"
            + markdown(["Network", "Trained on", "Size", "Started from", "Training step (last in the copied log)", "Shots seen",
                        "Shots per second (median of the log)", "Shots wrong at p = 0.003 / 0.004 / 0.005"], lines))
    if scores:
        steps = sorted({step for step, _ in scores})
        files = [name for name in ("validation_X_p0.003", "train_X_p0.003", "validation_X_p0.004", "train_X_p0.004", "validation_X_p0.005", "train_X_p0.005")
                 if any((step, name) in scores for step in steps)]
        score_lines = [[f"{step:,}"] + [f"{scores[(step, name)][0]:.2%} of {scores[(step, name)][3]:,}" if (step, name) in scores else "" for name in files]
                       for step in steps]
        text += ("\n\nThe X network alone, on the team's Stim shot files (exploration shots; `report/data/exploration/cascade_X_s1/SCORES.md`), "
                 "share of shots wrong:\n\n" + markdown(["Training step"] + [f"`{name}`" for name in files], score_lines))
    return write("networks", text, records)


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
        document.write_text(text, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)
    try:
        measured, extra = load_test_rows()
        per_p, sweep = test_tables(measured)
        earlier, cross_checks = earlier_table(measured)
        exploration, _ = exploration_table()
        tables = {f"test_p{p}": per_p[p] for p in P_VALUES} | {
            "test_sweep": sweep, "earlier_runs": earlier, "tcbp_own_shots": tcbp_own_table(), "exploration_p0.004": exploration,
            "loss": loss_table(), "networks": networks_table()}
        tables["checks"] = checks_text(measured, extra, cross_checks)
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
