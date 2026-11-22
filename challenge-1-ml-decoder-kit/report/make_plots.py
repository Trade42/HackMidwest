"""Make every figure of the report from the files under report/data/.

Run from challenge-1-ml-decoder-kit/:

  python -m report.make_plots

It writes report/figures/<name>.png and <name>.svg, one figure per file:

  ler_vs_p          logical error rate per cycle against p, a line per decoder (the required plot)
  time_vs_p         mean decoding time per pair against p
  ler_vs_time       error rate against time, a point per decoder and p: the trade-off
  loss              share of failed shots against the loss rate: loss-blind, flag-aware, ORACLE, tracker
  learning_curves   the two networks: share of shots wrong against the training step
  where_failures    how often the two fast decodes agree, and where the failures are
  front_ler_vs_p    the README's front page: the error rate against p with the predecoder and the TCBP version
  front_x_vs_time   the README's front page: X failures against time per X shot at p = 0.005, the hardware in the legend

A decoder keeps its colour and marker in every figure (report/rows.py). A row that is not
measured is not drawn, and the figure says so under the plot. Running the script twice gives
the same files.
"""
import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

from report import loaders, stats  # noqa: E402
from report.loaders import DATA, Mismatch  # noqa: E402
from report.make_tables import failure_split, load_test_rows  # noqa: E402
from report.rows import DECODERS, LOSS_CONDITIONS, P_RAMP, P_VALUES, REGISTERED_AFTER_TEST, TCBP_OWN  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGURES = Path(os.environ.get("REPORT_OUT") or HERE) / "figures"   # REPORT_OUT: used by the tests only

INK, SECONDARY, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
NEUTRAL = "#c3c2b7"

plt.rcParams.update({
    "figure.figsize": (11, 6.6), "figure.dpi": 100, "savefig.dpi": 180,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 15, "axes.labelsize": 16, "xtick.labelsize": 15, "ytick.labelsize": 15,
    "legend.fontsize": 13.5, "legend.frameon": False,
    "axes.edgecolor": AXIS, "axes.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "axes.axisbelow": True,
    "text.color": INK, "axes.labelcolor": SECONDARY, "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelcolor": SECONDARY, "ytick.labelcolor": SECONDARY,
    "lines.linewidth": 2.4, "lines.markersize": 11, "lines.solid_capstyle": "round",
    "svg.hashsalt": "report", "svg.fonttype": "path",
})


# ---------- shared pieces ----------

def new_figure(title, subtitle, size=None):
    figure, axes = plt.subplots(figsize=size)
    figure.subplots_adjust(left=0.105, right=0.78, top=0.83, bottom=0.21)
    figure.text(0.02, 0.955, title, fontsize=20, fontweight="bold", va="center")
    figure.text(0.02, 0.895, subtitle, fontsize=14, color=SECONDARY, va="center")
    return figure, axes


def footnote(figure, text, room=True):
    """A note under the plot, in muted ink; the plot's bottom margin grows with the number of lines."""
    lines = textwrap.wrap(text, 120)
    figure.text(0.02, 0.02, "\n".join(lines), fontsize=11.5, color=MUTED, va="bottom", linespacing=1.35)
    if room:
        figure.subplots_adjust(bottom=0.155 + 0.034 * len(lines))


def save(figure, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{name}.png", metadata={"Software": None})
    figure.savefig(FIGURES / f"{name}.svg", metadata={"Date": None, "Creator": None})
    plt.close(figure)
    return name


def marker_style(colour, hollow=False):
    return {"markerfacecolor": SURFACE if hollow else colour, "markeredgecolor": colour if hollow else SURFACE, "markeredgewidth": 2.0}


def end_labels(axes, items, x=1.02):
    """Direct labels right of the plot, one per series, pushed apart where they would collide."""
    if not items:
        return
    low, high = axes.get_ylim()
    place = lambda value: (np.log10(value) - np.log10(low)) / (np.log10(high) - np.log10(low)) if axes.get_yscale() == "log" else (value - low) / (high - low)
    spots = sorted(([place(value), text, colour, marker, hollow] for value, text, colour, marker, hollow in items), key=lambda item: item[0])
    height = [0.03 + 0.05 * (1 + text.count("\n")) for _, text, *_ in spots]     # a label's height, in axes units
    for index in range(1, len(spots)):                      # push up where two labels would touch
        spots[index][0] = max(spots[index][0], spots[index - 1][0] + (height[index] + height[index - 1]) / 2)
    limit = 1.0
    for index in range(len(spots) - 1, -1, -1):             # then pull the stack back inside the frame
        spots[index][0] = min(spots[index][0], limit)
        limit = spots[index][0] - (height[index] + height[index - 1]) / 2 if index else limit
    for position, text, colour, marker, hollow in spots:
        axes.plot([x], [position], marker=marker, color=colour, transform=axes.transAxes, clip_on=False, markersize=9, linestyle="none",
                  **marker_style(colour, hollow))
        axes.text(x + 0.022, position, text, transform=axes.transAxes, va="center", ha="left", fontsize=13.5, color=INK)


def p_axis(axes):
    axes.set_xticks([float(p) for p in P_VALUES], [p for p in P_VALUES])
    axes.set_xlim(0.0027, 0.0053)
    axes.set_xlabel("physical error rate p")
    axes.grid(axis="x", visible=False)


def power_label(value, _position=None):
    exponent = int(np.floor(np.log10(value) + 1e-9))
    mantissa = value / 10 ** exponent
    return f"$10^{{{exponent}}}$" if abs(mantissa - 1) < 1e-6 else ""


def ms_label(value, _position=None):
    if value >= 1000:
        return f"{value / 1000:g} s"
    return f"{value:g} ms"


def tcbp_own_points():
    """TCBP-JOBS on its own sampled shots, as logged: used only while its row on the test shots is missing."""
    points = {}
    for p, file, usable, _why in TCBP_OWN:
        if (DATA / file).is_file():
            row = loaders.tcbp_log(DATA / file)
            points[p] = {"rate": row["rate_printed"], "low": max(row["rate_printed"] - stats.Z95 * row["error_printed"], 1e-9),
                         "high": row["rate_printed"] + stats.Z95 * row["error_printed"], "mean_ms": row["mean_ms"] if usable else None}
    return points


def series(measured):
    """Per decoder, the measured points in order of p: [(p, row)]."""
    return {key: [(p, measured[p][key]) for p in P_VALUES if measured[p][key] is not None] for key in DECODERS}


def missing_note(measured, own):
    absent = [DECODERS[key].label for key in DECODERS if not any(measured[p][key] is not None for p in P_VALUES)
              and not (key == "tcbp" and own) and DECODERS[key].figures == "all"]
    partial = [f"{DECODERS[key].label} at p = {p}" for key in DECODERS for p in P_VALUES
               if measured[p][key] is None and any(measured[q][key] is not None for q in P_VALUES) and DECODERS[key].figures == "all"]
    parts = []
    if absent:
        parts.append("Not measured on the test shots: " + "; ".join(absent) + ".")
    if partial:
        parts.append("Not measured: " + "; ".join(partial) + ".")
    return " ".join(parts)


def after_note(measured, drawn):
    """The points drawn that come from rows run after the test results were seen, at the team's request (ablations, not our decoder)."""
    rows = [f"{DECODERS[key].short} at p = {p}" for p in P_VALUES for key, row in measured[p].items()
            if row and row.get("after_test") and DECODERS[key].figures in drawn]
    return (" Run after the test results were seen, at the team's request (an ablation, not our decoder): " + "; ".join(rows) + ".") if rows else ""


def unclean_note(measured):
    rows = [f"{DECODERS[key].short} at p = {p}" for p in P_VALUES for key, row in measured[p].items() if row and not row.get("time_clean", True)]
    return (" Not drawn, the time not clean (the server throttled the row): " + "; ".join(rows) + ".") if rows else ""


_SPACED = [key for key in DECODERS if key not in REGISTERED_AFTER_TEST]
DODGE = {key: (index - (len(_SPACED) - 1) / 2) * 0.000016 for index, key in enumerate(_SPACED)}


# ---------- 1. the required plot ----------

def ler_vs_p(measured):
    figure, axes = new_figure("Logical error rate per cycle on the test shots",
                              "10,000 pairs of an X and a Z shot per noise level; bars: 95% interval; lower is better")
    lines, own, labels, values, limits = series(measured), {}, [], [], False
    for key, points in lines.items():
        decoder = DECODERS[key]
        if not points:
            continue
        if decoder.figures != "all":
            continue
        hollow = key == "hierarchy-converged"
        xs = [float(p) + DODGE[key] for p, _ in points]
        # a row without a single failed pair has no rate to draw: it is drawn at the upper end of its interval, as a limit
        ys = [row["rate"] if row["failures_pair"] else row["interval"][1] for _, row in points]
        errors = [[max(y - row["interval"][0], 0) if row["failures_pair"] else 0 for y, (_, row) in zip(ys, points)],
                  [max(row["interval"][1] - y, 0) for y, (_, row) in zip(ys, points)]]
        axes.errorbar(xs, ys, yerr=errors, color=decoder.colour, marker=decoder.marker, linestyle="--" if hollow else "-",
                      capsize=4, elinewidth=1.6, zorder=3, **marker_style(decoder.colour, hollow))
        for x, y, (_, row) in zip(xs, ys, points):
            if not row["failures_pair"]:
                limits = True
                axes.plot([x], [y], marker="v", color=decoder.colour, markersize=13, zorder=4, **marker_style(decoder.colour, hollow=True))
        labels.append((ys[-1], decoder.short, decoder.colour, decoder.marker, hollow))
        values += ys + [row["interval"][0] for _, row in points if row["interval"][0] > 0] + [row["interval"][1] for _, row in points]
    if not lines["tcbp"]:
        own = tcbp_own_points()
        if own:
            decoder, ps = DECODERS["tcbp"], sorted(own)
            xs = [float(p) + DODGE["tcbp"] for p in ps]
            ys = [own[p]["rate"] for p in ps]
            axes.errorbar(xs, ys, yerr=[[own[p]["rate"] - own[p]["low"] for p in ps], [own[p]["high"] - own[p]["rate"] for p in ps]],
                          color=decoder.colour, marker=decoder.marker, linestyle=":", capsize=4, elinewidth=1.6, zorder=3,
                          **marker_style(decoder.colour, hollow=True))
            labels.append((ys[-1], "TCBP-JOBS (GPU),\nits own shots", decoder.colour, decoder.marker, True))
            values += ys + [own[p]["low"] for p in ps] + [own[p]["high"] for p in ps]
    axes.set_yscale("log")
    if values:
        axes.set_ylim(10 ** np.floor(np.log10(min(values)) - 0.05), 10 ** np.ceil(np.log10(max(values)) + 0.05))
    else:
        axes.set_ylim(1e-5, 1e-1)
        axes.text(0.5, 0.5, "not measured yet", transform=axes.transAxes, ha="center", va="center", fontsize=18, color=MUTED)
    axes.yaxis.set_major_formatter(FuncFormatter(power_label))
    axes.yaxis.set_minor_formatter(NullFormatter())
    axes.set_ylabel("logical error rate per cycle")
    p_axis(axes)
    end_labels(axes, labels)
    note = missing_note(measured, own)
    offline = {}
    for p in P_VALUES:
        for key, row in measured[p].items():
            if row is not None and row.get("offline") and DECODERS[key].figures == "all":
                offline.setdefault(DECODERS[key].short, []).append(p)
    if offline:
        note += (" Combined offline from saved answers (no time; a fresh relay-BP per 500 shots): "
                 + "; ".join(f"{name} at p = {', '.join(str(p) for p in ps)}" for name, ps in offline.items()) + ".")
    if own:
        note += " TCBP-JOBS is drawn from the teammates' run on its own sampled shots (hollow, dotted; interval from the logged error)."
    if limits:
        note += " A hollow triangle marks a row without a single failed pair: it is the upper end of the 95% interval, not a measured rate."
    note += after_note(measured, ("all",))
    footnote(figure, note.strip())
    return save(figure, "ler_vs_p")


# ---------- 2. time against p ----------

def time_vs_p(measured):
    figure, axes = new_figure("Mean decoding time per pair on the test shots",
                              "one X decode plus one Z decode, every stage counted; one process per row; lower is better")
    labels, values, lines = [], [], series(measured)
    for key, points in lines.items():
        decoder = DECODERS[key]
        points = [(p, row) for p, row in points if row["mean_ms"] is not None and not row.get("offline") and row.get("time_clean", True)]   # timed in one process, not throttled
        if not points or decoder.figures == "none":
            continue
        hollow = key == "hierarchy-converged"
        xs, ys = [float(p) for p, _ in points], [row["mean_ms"] for _, row in points]
        axes.plot(xs, ys, color=decoder.colour, marker=decoder.marker, linestyle="--" if hollow else "-", zorder=3, **marker_style(decoder.colour, hollow))
        device = " (GPU)" if points[0][1]["device"] == "GPU" and "GPU" not in decoder.short else ""
        labels.append((ys[-1], decoder.short + device, decoder.colour, decoder.marker, hollow))
        values += ys
    own = {} if lines["tcbp"] else {p: point for p, point in tcbp_own_points().items() if point["mean_ms"] is not None}
    if own:
        decoder = DECODERS["tcbp"]
        ps = sorted(own)
        axes.plot([float(p) for p in ps], [own[p]["mean_ms"] for p in ps], color=decoder.colour, marker=decoder.marker, linestyle=":",
                  zorder=3, **marker_style(decoder.colour, hollow=True))
        labels.append((own[ps[-1]]["mean_ms"], "TCBP-JOBS (GPU),\nits own shots", decoder.colour, decoder.marker, True))
        values += [own[p]["mean_ms"] for p in ps]
    axes.set_yscale("log")
    if values:
        axes.set_ylim(10 ** np.floor(np.log10(min(values)) - 0.02), 10 ** np.ceil(np.log10(max(values)) + 0.02))
    else:
        axes.set_ylim(1, 1e4)
        axes.text(0.5, 0.5, "not measured yet", transform=axes.transAxes, ha="center", va="center", fontsize=18, color=MUTED)
    axes.yaxis.set_major_formatter(FuncFormatter(ms_label))
    axes.yaxis.set_minor_formatter(NullFormatter())
    axes.set_ylabel("mean time per pair")
    p_axis(axes)
    end_labels(axes, labels)
    note = missing_note(measured, own) + unclean_note(measured) + after_note(measured, ("all", "time"))
    if not lines["tcbp"]:
        note += (" TCBP-JOBS: only its p = 0.003 time on its own shots is usable; at p = 0.004 and 0.005 its GPU was shared with a training of ours."
                 if own else " TCBP-JOBS has no usable time yet.")
    else:
        note += (" TCBP-JOBS runs on a GPU (an A100 slice) and its timer covers the decode call alone, so its line is not comparable with the CPU rows,"
                 " which time the whole decode of a pair in one process.")
    footnote(figure, note.strip())
    return save(figure, "time_vs_p")


# ---------- 3. the trade-off ----------

def ler_vs_time(measured):
    figure, axes = new_figure("Accuracy against time on the test shots",
                              "each line is one decoder at p = 0.003, 0.004 and 0.005; lower and further left is better")
    labels, xs_all, ys_all = [], [], []
    for key, points in series(measured).items():
        decoder = DECODERS[key]
        points = [(p, row) for p, row in points if row["mean_ms"] is not None and row["rate"] > 0 and not row.get("offline") and row.get("time_clean", True)]
        if not points or decoder.figures == "none":
            continue
        hollow = key == "hierarchy-converged"
        xs, ys = [row["mean_ms"] for _, row in points], [row["rate"] for _, row in points]
        axes.plot(xs, ys, color=decoder.colour, marker=decoder.marker, linestyle="--" if hollow else "-", linewidth=1.6, zorder=3,
                  **marker_style(decoder.colour, hollow))
        if key == "relay-bp":
            for (p, _), x, y in zip(points, xs, ys):
                axes.annotate(f"p = {p}", (x, y), textcoords="offset points", xytext=(10, -16), fontsize=12.5, color=SECONDARY)
        labels.append((ys[-1], decoder.short + (" (GPU)" if points[0][1]["device"] == "GPU" and "GPU" not in decoder.short else ""),
                       decoder.colour, decoder.marker, hollow))
        xs_all += xs
        ys_all += ys
    axes.set_xscale("log")
    axes.set_yscale("log")
    if xs_all:
        axes.set_xlim(10 ** np.floor(np.log10(min(xs_all)) - 0.02), 10 ** np.ceil(np.log10(max(xs_all)) + 0.02))
        axes.set_ylim(10 ** np.floor(np.log10(min(ys_all)) - 0.05), 10 ** np.ceil(np.log10(max(ys_all)) + 0.05))
    else:
        axes.set_xlim(1, 1e4)
        axes.set_ylim(1e-5, 1e-1)
        axes.text(0.5, 0.5, "not measured yet", transform=axes.transAxes, ha="center", va="center", fontsize=18, color=MUTED)
    axes.xaxis.set_major_formatter(FuncFormatter(ms_label))
    axes.xaxis.set_minor_formatter(NullFormatter())
    axes.yaxis.set_major_formatter(FuncFormatter(power_label))
    axes.yaxis.set_minor_formatter(NullFormatter())
    axes.set_xlabel("mean time per pair")
    axes.set_ylabel("logical error rate per cycle")
    end_labels(axes, labels)
    footnote(figure, (missing_note(measured, {}) + unclean_note(measured) + after_note(measured, ("all", "time")) + " CPU rows: one process each on the server, timing the whole decode of a pair. TCBP-JOBS (GPU) times its decode call alone: not comparable. "
                      "Rows combined offline from per-shot files have no time measured in one process and are not drawn.").strip())
    return save(figure, "ler_vs_time")


# ---------- 4. loss ----------

def loss():
    pc, server = loaders.loss_pc(DATA / "loss" / "pc"), loaders.loss_server(DATA / "loss" / "server")
    tracked, shares = loaders.loss_tracker(DATA / "loss" / "tracker")
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 6.6), gridspec_kw={"width_ratios": [1.3, 1]})
    figure.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.25, wspace=0.55)
    figure.text(0.02, 0.955, "Atom loss: the flags give most of it back, the tracker more", fontsize=20, fontweight="bold", va="center")
    figure.text(0.02, 0.895, "X memory, p = 0.004, relay-BP; our simplified loss model; bars: 95% interval; lower is better",
                fontsize=14, color=SECONDARY, va="center")

    # left: loss-blind against flag-aware over the loss rate, one dataset per rate (the largest of the loss study)
    rows = [row for row in list(pc.values()) + list(server.values())]
    labels = []
    for condition in [c for c in LOSS_CONDITIONS if c.key in ("blind", "starter")]:
        by_s = {}
        for row in rows:
            if row["condition"] == condition.key:
                by_s.setdefault(row["s"], []).append(row)
        if not by_s:
            continue
        best = {value: max(found, key=lambda row: row["shots"]) for value, found in by_s.items()}
        xs = sorted(best)
        ys = [best[x]["failures"] / best[x]["shots"] for x in xs]
        bounds = [stats.wilson(best[x]["failures"], best[x]["shots"]) for x in xs]
        left.errorbar(xs, ys, yerr=[[y - low for y, (low, _) in zip(ys, bounds)], [high - y for y, (_, high) in zip(ys, bounds)]],
                      color=condition.colour, marker=condition.marker, capsize=4, elinewidth=1.6, zorder=3, **marker_style(condition.colour))
        labels.append((ys[-1], {"blind": "ignores\nthe flags", "starter": "uses the\nflags"}[condition.key], condition.colour, condition.marker, False))
    lossless = DATA / "loss" / "pc" / "L0_X_p0.004.json"
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_ylim(0.004, 1.3)
    left.set_xlim(1.5e-4, 2.6e-3)
    if lossless.is_file():
        base = json.loads(lossless.read_text(encoding="utf-8"))["relay_bp_on_clifft"]
        share = base["failures"] / base["shots"]
        left.axhline(share, color=MUTED, linewidth=1.2, zorder=2)
        left.text(2.5e-3, share * 0.88, f"no loss: {share:.2%}", fontsize=12.5, color=SECONDARY, va="top", ha="right")
    ticks = [2e-4, 5e-4, 1e-3, 2e-3]
    left.set_xticks(ticks, [f"{tick:g}" for tick in ticks])
    left.xaxis.set_minor_formatter(NullFormatter())
    left.set_yticks([0.01, 0.03, 0.1, 0.3, 1.0], ["1%", "3%", "10%", "30%", "100%"])
    left.yaxis.set_minor_formatter(NullFormatter())
    left.set_xlabel("loss probability s, per atom per gate")
    left.set_ylabel("share of shots that fail")
    end_labels(left, labels, x=1.03)
    left.set_title("the flags", fontsize=14, color=SECONDARY, loc="left")

    # right: the tracker's own test shots at s = 0.001, every condition on the same shots
    right.set_title("knowing when, at s = 0.001", fontsize=14, color=SECONDARY, loc="left")
    test = {key[-1]: row for key, row in tracked.items() if key[1] == 7300}
    order = [("starter", "flags,\npublished", "#6250d6"), ("exact", "flags,\nexact\nweights", "#8b7fe0"),
             ("tracker", "ours:\nwith the\ntracker", "#eb6834"), ("oracle-round", "ORACLE\n(a ceiling)", "#52514e")]
    present = [(key, label, colour) for key, label, colour in order if key in test]
    if not present:
        right.axis("off")
        right.text(0.5, 0.5, "the tracker: not measured", ha="center", va="center", fontsize=15, color=MUTED, transform=right.transAxes)
    else:
        for index, (key, label, colour) in enumerate(present):
            row = test[key]
            share = row["failures"] / row["shots"]
            low, high = stats.wilson(row["failures"], row["shots"])
            right.bar(index, share * 100, width=0.62, color=colour, edgecolor="none", zorder=3)
            right.errorbar([index], [share * 100], yerr=[[share * 100 - low * 100], [high * 100 - share * 100]], color=INK, capsize=4, elinewidth=1.4, zorder=4)
            right.text(index, high * 100 + 0.15, f"{row['failures']}", ha="center", va="bottom", fontsize=13, color=INK)
        right.set_xticks(range(len(present)), [label for _, label, _ in present], fontsize=12.5)
        right.set_ylim(0, max(stats.wilson(test[key]["failures"], test[key]["shots"])[1] for key, _, _ in present) * 100 * 1.25)
        right.set_ylabel(f"% of {test[present[0][0]]['shots']:,} shots that fail")
        right.grid(axis="x", visible=False)
    entry = shares.get("test", {}).get(("tracker", "exact"))
    note = "Left: the largest dataset of the loss study at each loss rate. Right: the tracker's own test shots (seed 7300), the tracker frozen beforehand; numbers on the bars are failures."
    if entry:
        note += (f" The tracker recovers {entry['share']:.0%} ({entry['share95'][0]:.0%} to {entry['share95'][1]:.0%}) of the gap between the flag-aware decoder "
                 "with exact weights and the ORACLE.")
    footnote(figure, note, room=False)
    return save(figure, "loss")


# ---------- 5. the networks' learning curves ----------

def learning_curves():
    logs = {"X": loaders.train_log(DATA / "networks" / "cascade_X_s1.train.log"), "Z": loaders.train_log(DATA / "networks" / "cascade_Z_s1.train.log")}
    scores = loaders.scores_md(DATA / "exploration" / "cascade_X_s1" / "SCORES.md")
    figure, panels = plt.subplots(1, 2, figsize=(11, 6.6), sharey=True)
    figure.subplots_adjust(left=0.095, right=0.875, top=0.80, bottom=0.225, wspace=0.42)
    figure.text(0.02, 0.955, "The two networks while they train", fontsize=20, fontweight="bold", va="center")
    figure.text(0.02, 0.895, "share of shots with any logical bit wrong, on fresh error-model shots; lower is better",
                fontsize=14, color=SECONDARY, va="center")
    for axes, basis in zip(panels, "XZ"):
        log = logs[basis]
        axes.set_title(f"{basis} memory" + (" (starts from the X network, mirrored)" if log and log["warm_start"] else ""), fontsize=14, color=SECONDARY, loc="left")
        axes.set_yscale("log")
        axes.set_ylim(0.01, 1.3)
        axes.set_yticks([0.01, 0.03, 0.1, 0.3, 1.0], ["1%", "3%", "10%", "30%", "100%"])
        axes.yaxis.set_minor_formatter(NullFormatter())
        axes.set_xlabel("training step (thousands)")
        if not log or not log["reports"]:
            axes.text(0.5, 0.5, "not measured", transform=axes.transAxes, ha="center", va="center", fontsize=16, color=MUTED)
            continue
        labels = []
        for p in P_VALUES:
            steps = [report["step"] / 1000 for report in log["reports"] if p in report["wrong"] and report["wrong"][p] > 0]
            wrong = [report["wrong"][p] for report in log["reports"] if p in report["wrong"] and report["wrong"][p] > 0]
            axes.plot(steps, wrong, color=P_RAMP[p], marker="o", markersize=5, markeredgewidth=0)
            labels.append((wrong[-1], f"p = {p}", P_RAMP[p], "o", False))
            if basis == "X":
                stim = sorted((step / 1000, value[0]) for (step, name), value in scores.items() if name == f"validation_X_p{p}")
                if stim:
                    axes.plot([x for x, _ in stim], [y for _, y in stim], linestyle="none", marker="D", markersize=9, **marker_style(P_RAMP[p], hollow=True))
        end_labels(axes, labels, x=1.03)
        axes.set_xlim(0, max(report["step"] for report in log["reports"]) / 1000 * 1.02)
    panels[0].set_ylabel("share of shots wrong")
    footnote(figure, "Diamonds: the same X network scored on the team's Stim validation shots (exploration shots). "
                     "The networks alone stay far above relay-BP; they are used only to choose among candidates.", room=False)
    return save(figure, "learning_curves")


# ---------- 6. where the failures are ----------

def where_failures(combined, measured):
    sets = failure_split(combined, measured)
    name = next((name for name in sets if name.startswith("test shots, p = 0.004")), None) or next(iter(sets), None)
    figure, axes = plt.subplots(figsize=(11, 5.6 if not name or len(sets[name]) < 4 else 6.6))
    figure.subplots_adjust(left=0.03, right=0.97, top=0.76, bottom=0.2)
    figure.text(0.02, 0.945, "Most failures sit where two fast decodes disagree", fontsize=20, fontweight="bold", va="center")
    axes.axis("off")
    if name is None:
        axes.text(0.5, 0.5, "not measured", ha="center", va="center", fontsize=18, color=MUTED, transform=axes.transAxes)
        return save(figure, "where_failures")
    bars = sets[name]
    figure.text(0.02, 0.875, f"{name}, X and Z memory; two runs of PriorTilt with different random seeds", fontsize=14, color=SECONDARY, va="center")
    scale = max(bars[1][1] + bars[1][2], 1)
    axes.set_xlim(0, 1)
    axes.set_ylim(-0.6, len(bars) - 0.25)
    orange = DECODERS["hierarchy"].colour
    for index, (what, same, differ) in enumerate(bars):
        y = len(bars) - 1 - index
        total = max(same + differ, 1)
        label = f"All {total:,} shots" if index == 0 else f"The {total:,} {what}"
        width = 1.0 if index == 0 else total / scale     # the failure bars share one scale; the first bar is all shots
        left = same / total * width
        axes.barh(y, max(left - 0.003, 0), height=0.34, color=NEUTRAL, edgecolor="none")
        axes.barh(y, differ / total * width, left=left, height=0.34, color=orange, edgecolor="none")
        axes.text(0, y + 0.3, label, fontsize=15, va="bottom", ha="left", color=INK)
        axes.text(max(left / 2, 0.001), y, f"agree: {same:,}", fontsize=13.5, va="center", ha="center" if left > 0.2 else "left", color=INK)
        text = f"disagree: {differ:,} ({differ / total:.2%})" if differ / total < 0.01 else f"disagree: {differ:,} ({differ / total:.1%})"
        if differ / total * width > 0.25:
            axes.text(left + differ / total * width / 2, y, text, fontsize=13.5, va="center", ha="center", color="#ffffff")
        else:
            axes.text(min(width + 0.012, 0.99), y - 0.33, text, fontsize=13.5, va="center", ha="right", color=INK)
    axes.plot([], [], color=NEUTRAL, linewidth=9, label="the two fast decodes give the same answer: it is final")
    axes.plot([], [], color=orange, linewidth=9, label="they disagree: a third decoder joins and a tie-break chooses")
    axes.legend(loc="upper left", bbox_to_anchor=(0.0, -0.02), ncol=1, fontsize=13.5)
    footnote(figure, "The lower bars share one scale. Shots that are wrong although the two decodes agree cannot be fixed by any tie-break.", room=False)
    return save(figure, "where_failures")


# ---------- 7. the front page's two figures (the top-level README.md) ----------

def generated_facts():
    """The single numbers make_tables.py wrote to tables/facts.md (run make_tables first): name -> value text."""
    file = FIGURES.parent / "tables" / "facts.md"   # REPORT_OUT in the tests, else report/
    if not file.is_file():
        raise Mismatch("tables/facts.md is missing: run python -m report.make_tables first")
    facts = {}
    for line in file.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.split(" | ")]
        if len(cells) >= 3 and cells[0].startswith("| `"):
            facts[cells[0][3:-1]] = cells[1]
    return facts


def number(text):
    try:
        return float(text.replace(",", "").removeprefix("about "))
    except (AttributeError, ValueError):
        return None


TH_COLOUR, PRE_COLOUR = "#b8224f", "#7a5c00"


def front_ler_vs_p(measured):
    figure, axes = new_figure("Logical error rate per cycle on the test shots",
                              "the kit's score: 10,000 pairs of an X and a Z shot per noise level; bars: 95% interval; lower is better", size=(16, 7.6))
    figure.subplots_adjust(left=0.085, right=0.6)
    values = []

    def draw(xs, failures, label, colour, marker, style="-", hollow=False):
        rates = [stats.per_cycle_rate(f, 10000)[0] for f in failures]
        bounds = [stats.per_cycle_interval(f, 10000) for f in failures]
        axes.errorbar(xs, rates, yerr=[[r - b[0] for r, b in zip(rates, bounds)], [b[1] - r for r, b in zip(rates, bounds)]], color=colour,
                      marker=marker, linestyle=style, capsize=4, elinewidth=1.6, zorder=3, label=label, markersize=19 if marker == "*" else 11,
                      **marker_style(colour, hollow))
        values.extend(rates + [b[0] for b in bounds if b[0] > 0] + [b[1] for b in bounds])

    shown = (("bposd", "BP-OSD (the kit's baseline)"), ("relay-bp", "relay-BP (the kit's baseline)"), ("router", "relay-BP with a learned router (teammates)"),
             ("tcbp", "TCBP-JOBS alone (NVIDIA's GPU decoder)"), ("hierarchy", "our first pipeline, frozen before the test run"))
    for index, (key, label) in enumerate(shown):
        points = [(p, measured[p][key]) for p in P_VALUES if measured[p].get(key) is not None and not measured[p][key].get("offline")]
        if points:
            draw([float(p) + (index - 3) * 0.000018 for p, _ in points], [row["failures_pair"] for _, row in points], label,
                 DECODERS[key].colour, DECODERS[key].marker)
    pair = DATA / "test" / "predecoder" / "pair_test_p0.005.json"
    if pair.is_file():
        draw([0.005 + 0.000054], [json.loads(pair.read_text(encoding="utf-8"))["decoders"]["H"]["failed_pairs"]],
             "ours: the relay-BP predecoder (p = 0.005 only; secondary)", PRE_COLOUR, "h")
    ths = []
    for p in P_VALUES:
        file = DATA / "test" / "predecoder_tcbp" / f"tcbp_pair_test_p{p}.json"
        if file.is_file():
            ths.append((float(p) + 0.000072, json.loads(file.read_text(encoding="utf-8"))["TH"]["failed_pairs"]))
    if ths:
        draw([x for x, _ in ths], [f for _, f in ths], "ours: the TCBP version (designed after\nthe test results were seen)", TH_COLOUR, "*")
    axes.set_yscale("log")
    if values:
        axes.set_ylim(10 ** np.floor(np.log10(min(values)) - 0.05), 10 ** np.ceil(np.log10(max(values)) + 0.05))
    axes.yaxis.set_major_formatter(FuncFormatter(power_label))
    axes.yaxis.set_minor_formatter(NullFormatter())
    axes.set_ylabel("logical error rate per cycle")
    p_axis(axes)
    axes.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=12, frameon=False, borderaxespad=0)
    low = DATA / "test" / "predecoder_tcbp" / "tcbp_pair_test_p0.003.json"
    low = json.loads(low.read_text(encoding="utf-8")) if low.is_file() else None
    footnote(figure, "Every point is the same 10,000 test pairs at its p. The TCBP version: its knobs were fitted on X memory at p = 0.005 and applied "
                     "unchanged elsewhere, so there is no validation at p = 0.004 and 0.003"
                     + (f"; at p = 0.003 it fails {low['TH']['failed_pairs']} pair(s) against TCBP-JOBS's {low['T']['failed_pairs']}, too few for a claim"
                        if low else "") + ". The relay-BP predecoder was scored as pairs at p = 0.005 only. TCBP-JOBS's counts carry the spread of its model fit.")
    return save(figure, "front_ler_vs_p")


def front_x_vs_time(measured):
    facts = generated_facts()
    figure, axes = new_figure("Accuracy against time per X shot at p = 0.005",
                              "the 10,000 X-memory test shots; mean time per X shot; lower and further left is better", size=(16, 7.6))
    figure.subplots_adjust(left=0.085, right=0.56)
    points = []
    cpu = "CPU, 1 process (team's server)"
    for key, label in (("bposd", "BP-OSD (kit)"), ("relay-bp", "relay-BP (kit)"), ("router", "router (teammates)"), ("priortilt", "PriorTilt (IonQ)"),
                       ("hierarchy", "our first pipeline"), ("hierarchy-parallel", "our first pipeline, 2 cores")):
        row = measured["0.005"].get(key)
        seconds = None
        if row is not None and row.get("per_shot_file"):
            with np.load(DATA / row["per_shot_file"], allow_pickle=False) as file:
                seconds = float(file["seconds_total_X"].mean()) if "seconds_total_X" in file.files else None
        if seconds is not None:
            where = cpu.replace("1 process", "1 process, 2 cores") if key == "hierarchy-parallel" else cpu
            points.append((seconds * 1e3, row["failures_X"], f"{label}: {where}", DECODERS[key].colour, DECODERS[key].marker, False))
    tcbp = measured["0.005"].get("tcbp")
    if tcbp is not None and number(facts.get("lat_tcbp_x")) is not None:
        points.append((number(facts["lat_tcbp_x"]), tcbp["failures_X"], "TCBP-JOBS alone (NVIDIA): A100 slice, the decode call only",
                       DECODERS["tcbp"].colour, DECODERS["tcbp"].marker, False))
    if number(facts.get("lat_total")) is not None and number(facts.get("pre5X_H")) is not None:
        points.append((number(facts["lat_total"]), number(facts["pre5X_H"]), "ours, relay-BP predecoder: CPU, 1 thread (time on 1,000 validation shots)",
                       PRE_COLOUR, "h", False))
    if number(facts.get("pttime_total")) is not None and number(facts.get("ptt5X_TH")) is not None:
        points.append((number(facts["pttime_total"]), number(facts["ptt5X_TH"]), "ours, the TCBP version: A100 slice, network + build + decode\n"
                       "(as registered; not a clean timing)", TH_COLOUR, "*", False))
        if number(facts.get("ptspeed_new")) is not None:
            points.append((number(facts["ptspeed_new"]), number(facts["ptt5X_TH"]), "the same after the speed work (parts timed alone on\n"
                           "1,000 validation shots; same answers there)", TH_COLOUR, "*", True))
    for x, y, label, colour, marker, hollow in points:
        axes.plot([x], [y], marker=marker, color=colour, linestyle="none", markersize=20 if marker == "*" else 12, label=label, zorder=3,
                  **marker_style(colour, hollow))
    axes.set_xscale("log")
    axes.set_yscale("log")
    if points:
        xs, ys = [x for x, *_ in points], [y for _, y, *_ in points]
        axes.set_xlim(10 ** np.floor(np.log10(min(xs)) - 0.02), 10 ** np.ceil(np.log10(max(xs)) + 0.02))
        axes.set_ylim(10 ** np.floor(np.log10(min(ys)) - 0.05), 10 ** np.ceil(np.log10(max(ys)) + 0.05))
    axes.xaxis.set_major_formatter(FuncFormatter(ms_label))
    axes.xaxis.set_minor_formatter(NullFormatter())
    axes.yaxis.set_major_formatter(FuncFormatter(lambda value, _position=None: f"{value:g}"))
    axes.yaxis.set_minor_formatter(NullFormatter())
    axes.set_xlabel("mean time per X shot (log scale)")
    axes.set_ylabel("X test shots failed, of 10,000")
    axes.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=11, frameon=False, borderaxespad=0, labelspacing=0.9)
    footnote(figure, "CPU and GPU times come from different machines and timers: read them as orders of magnitude, not as equals. The CPU rows time "
                     "the whole decode of each test X shot in one process. The predecoder's and the TCBP version's failures are on the test shots, "
                     "their times on validation shots where marked. The TCBP version was designed after the test results were seen.")
    return save(figure, "front_x_vs_time")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)
    try:
        measured, _, combined = load_test_rows()
        names = [ler_vs_p(measured), time_vs_p(measured), ler_vs_time(measured), loss(), learning_curves(), where_failures(combined, measured),
                 front_ler_vs_p(measured), front_x_vs_time(measured)]
    except Mismatch as problem:
        print(f"STOPPED: {problem}", file=sys.stderr)
        return 1
    print(f"wrote {len(names)} figures (png and svg) to {FIGURES}/: {', '.join(names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
