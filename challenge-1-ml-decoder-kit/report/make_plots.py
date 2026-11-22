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
from report.make_tables import load_test_rows  # noqa: E402
from report.rows import DECODERS, LOSS_CONDITIONS, P_RAMP, P_VALUES, TCBP_OWN  # noqa: E402

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
    lines = textwrap.wrap(text, 132)
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
              and not (key == "tcbp" and own)]
    partial = [f"{DECODERS[key].label} at p = {p}" for key in DECODERS for p in P_VALUES
               if measured[p][key] is None and any(measured[q][key] is not None for q in P_VALUES)]
    parts = []
    if absent:
        parts.append("Not measured on the test shots: " + "; ".join(absent) + ".")
    if partial:
        parts.append("Not measured: " + "; ".join(partial) + ".")
    return " ".join(parts)


DODGE = {key: (index - (len(DECODERS) - 1) / 2) * 0.000016 for index, key in enumerate(DECODERS)}


# ---------- 1. the required plot ----------

def ler_vs_p(measured):
    figure, axes = new_figure("Logical error rate per cycle on the test shots",
                              "10,000 pairs of an X and a Z shot per noise level; bars: 95% interval; lower is better")
    lines, own, labels, values = series(measured), {}, [], []
    for key, points in lines.items():
        decoder = DECODERS[key]
        if not points:
            continue
        hollow = key == "hierarchy-weight"
        xs = [float(p) + DODGE[key] for p, _ in points]
        ys = [max(row["rate"], row["interval"][1] / 30) for _, row in points]
        errors = [[max(y - row["interval"][0], 0) for y, (_, row) in zip(ys, points)], [max(row["interval"][1] - y, 0) for y, (_, row) in zip(ys, points)]]
        axes.errorbar(xs, ys, yerr=errors, color=decoder.colour, marker=decoder.marker, linestyle="--" if hollow else "-",
                      capsize=4, elinewidth=1.6, zorder=3, **marker_style(decoder.colour, hollow))
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
    if own:
        note += " TCBP-JOBS is drawn from the teammates' run on its own sampled shots (hollow, dotted; interval from the logged error)."
    footnote(figure, note.strip())
    return save(figure, "ler_vs_p")


# ---------- 2. time against p ----------

def time_vs_p(measured):
    figure, axes = new_figure("Mean decoding time per pair on the test shots",
                              "one X decode plus one Z decode, every stage counted; one process per row; lower is better")
    labels, values, lines = [], [], series(measured)
    for key, points in lines.items():
        decoder = DECODERS[key]
        points = [(p, row) for p, row in points if row["mean_ms"] is not None]
        if not points:
            continue
        hollow = key == "hierarchy-weight"
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
    note = missing_note(measured, own)
    if not lines["tcbp"]:
        note += (" TCBP-JOBS: only its p = 0.003 time on its own shots is usable; at p = 0.004 and 0.005 its GPU was shared with a training of ours."
                 if own else " TCBP-JOBS has no usable time yet.")
    footnote(figure, note.strip())
    return save(figure, "time_vs_p")


# ---------- 3. the trade-off ----------

def ler_vs_time(measured):
    figure, axes = new_figure("Accuracy against time on the test shots",
                              "each line is one decoder at p = 0.003, 0.004 and 0.005; lower and further left is better")
    labels, xs_all, ys_all = [], [], []
    for key, points in series(measured).items():
        decoder = DECODERS[key]
        points = [(p, row) for p, row in points if row["mean_ms"] is not None and row["rate"] > 0]
        if not points:
            continue
        hollow = key == "hierarchy-weight"
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
    footnote(figure, (missing_note(measured, {}) + " CPU rows: one process each on the server; a GPU row is marked.").strip())
    return save(figure, "ler_vs_time")


# ---------- 4. loss ----------

def loss():
    pc, server = loaders.loss_pc(DATA / "loss" / "pc"), loaders.loss_server(DATA / "loss" / "server")
    points = {}
    for row in list(pc.values()) + list(server.values()):
        points.setdefault(row["condition"], {}).setdefault(row["s"], []).append(row)
    figure, axes = new_figure("Atom loss: what the loss flags give back",
                              "X memory, p = 0.004, relay-BP; our simplified loss model; bars: 95% interval; lower is better")
    labels, drawn = [], []
    for condition in LOSS_CONDITIONS:
        by_s = points.get(condition.key)
        if not by_s:
            continue
        best = {s: max(rows, key=lambda row: row["shots"]) for s, rows in by_s.items()}
        xs = sorted(best)
        ys = [best[s]["failures"] / best[s]["shots"] for s in xs]
        bounds = [stats.wilson(best[s]["failures"], best[s]["shots"]) for s in xs]
        axes.errorbar(xs, ys, yerr=[[y - low for y, (low, _) in zip(ys, bounds)], [high - y for y, (_, high) in zip(ys, bounds)]],
                      color=condition.colour, marker=condition.marker, linestyle=condition.line, capsize=4, elinewidth=1.6, zorder=3,
                      **marker_style(condition.colour))
        for s, rows in by_s.items():                       # a second dataset at the same loss rate: its own hollow point, never pooled
            for row in rows:
                if row is not best[s]:
                    low, high = stats.wilson(row["failures"], row["shots"])
                    share = row["failures"] / row["shots"]
                    axes.errorbar([s * 1.07], [share], yerr=[[share - low], [high - share]], color=condition.colour, marker=condition.marker,
                                  linestyle="none", capsize=3, elinewidth=1.2, zorder=3, **marker_style(condition.colour, hollow=True))
        short = {"blind": "loss-blind", "starter": "flag-aware", "tracker": "ours: with the\nloss-time tracker", "oracle-round": "ORACLE: true loss\nround (a ceiling,\nnot a decoder)"}[condition.key]
        labels.append((ys[-1], short, condition.colour, condition.marker, False))
        drawn.append(condition.key)
    lossless = DATA / "loss" / "pc" / "L0_X_p0.004.json"
    if lossless.is_file():
        base = json.loads(lossless.read_text(encoding="utf-8"))["relay_bp_on_clifft"]
        axes.axhline(base["failures"] / base["shots"], color=MUTED, linewidth=1.2, zorder=2)
        axes.text(1.5e-4, base["failures"] / base["shots"] * 1.08, f"no loss: {base['failures'] / base['shots']:.2%} of shots fail", fontsize=12.5, color=SECONDARY, va="bottom")
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_ylim(0.005, 1.3)
    axes.set_xlim(1.4e-4, 2.8e-3)
    ticks = [2e-4, 5e-4, 1e-3, 2e-3]
    axes.set_xticks(ticks, [f"{tick:g}" for tick in ticks])
    axes.xaxis.set_minor_formatter(NullFormatter())
    axes.set_yticks([0.01, 0.03, 0.1, 0.3, 1.0], ["1%", "3%", "10%", "30%", "100%"])
    axes.yaxis.set_minor_formatter(NullFormatter())
    axes.set_xlabel("loss probability s, per atom per CX gate")
    axes.set_ylabel("share of shots that fail")
    end_labels(axes, labels)
    note = "Hollow points: a second dataset at the same loss rate (another machine and seed), not pooled."
    if "tracker" not in drawn:
        note += " The loss-time tracker is not measured yet."
    footnote(figure, note)
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

def where_failures():
    folder = DATA / "exploration" / "priortilt"
    shots = loaders.selector_files(folder, ["selector_validation_X.npz", "selector_train_X.npz", "selector_validation_Z.npz", "selector_train_Z.npz"])
    figure, axes = plt.subplots(figsize=(11, 5.6))
    figure.subplots_adjust(left=0.03, right=0.97, top=0.74, bottom=0.2)
    figure.text(0.02, 0.945, "Most failures sit where two fast decodes disagree", fontsize=20, fontweight="bold", va="center")
    figure.text(0.02, 0.87, "p = 0.004, 50,000 exploration shots (X and Z memory); two runs of PriorTilt with different random seeds", fontsize=14, color=SECONDARY, va="center")
    axes.axis("off")
    if shots is None:
        axes.text(0.5, 0.5, "not measured", ha="center", va="center", fontsize=18, color=MUTED, transform=axes.transAxes)
        return save(figure, "where_failures")
    failed, agreed, _ = loaders.weight_rule(shots)
    first = shots["failed_P1"].astype(bool)
    bars = [(f"All {len(agreed):,} shots", int(agreed.sum()), int((~agreed).sum())),
            (f"The {int(first.sum())} shots one fast decode gets wrong", int((first & agreed).sum()), int((first & ~agreed).sum())),
            (f"The {int(failed.sum())} shots still wrong after the tie-break (lowest weight)", int((failed & agreed).sum()), int((failed & ~agreed).sum()))]
    scale = max(first.sum(), 1)
    axes.set_xlim(0, 1)
    axes.set_ylim(-0.6, len(bars) - 0.25)
    orange = DECODERS["hierarchy"].colour
    for index, (label, same, differ) in enumerate(bars):
        y = len(bars) - 1 - index
        total = same + differ
        width = 1.0 if index == 0 else total / scale     # the two failure bars share one scale; the first bar is all shots
        left = same / total * width
        axes.barh(y, max(left - 0.003, 0), height=0.34, color=NEUTRAL, edgecolor="none")
        axes.barh(y, differ / total * width, left=left, height=0.34, color=orange, edgecolor="none")
        axes.text(0, y + 0.3, label, fontsize=15, va="bottom", ha="left", color=INK)
        axes.text(max(left / 2, 0.001), y, f"agree: {same:,}", fontsize=13.5, va="center", ha="center" if left > 0.2 else "left", color=INK)
        text = f"disagree: {differ:,} ({differ / total:.1%})"
        if differ / total * width > 0.25:
            axes.text(left + differ / total * width / 2, y, text, fontsize=13.5, va="center", ha="center", color="#ffffff")
        else:
            axes.text(min(width + 0.012, 0.99), y - 0.33, text, fontsize=13.5, va="center", ha="right", color=INK)
    axes.plot([], [], color=NEUTRAL, linewidth=9, label="the two fast decodes give the same answer: it is final")
    axes.plot([], [], color=orange, linewidth=9, label="they disagree: a third decoder joins and a tie-break chooses")
    axes.legend(loc="upper left", bbox_to_anchor=(0.0, -0.02), ncol=1, fontsize=13.5)
    footnote(figure, "The two lower bars share one scale. Shots that are wrong although the two decodes agree cannot be fixed by any tie-break.", room=False)
    return save(figure, "where_failures")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)
    try:
        measured, _ = load_test_rows()
        names = [ler_vs_p(measured), time_vs_p(measured), ler_vs_time(measured), loss(), learning_curves(), where_failures()]
    except Mismatch as problem:
        print(f"STOPPED: {problem}", file=sys.stderr)
        return 1
    print(f"wrote {len(names)} figures (png and svg) to {FIGURES}/: {', '.join(names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
