"""Figures for the report and the slides. Every number drawn comes from a results file.

    fig1_accuracy_by_lane      exact match per lane, single selection vs 3-selection vote
    fig2_noise_to_accuracy     feasibility, approximation ratio, gold recall, exact match per tier
    fig3_redundant_edges       shots kept and approximation ratio against redundant edges
    fig4_two_qubit_gates       CZ gates per tier and connectivity
    fig5_roadmap               qubits, gates and brute-force time for larger problems
    fig6_fault_budget          what the checks catch: every single fault scored
    fig7_noise_sweep           approximation ratio and shots kept against CZ error rate

Style: one colour per tier everywhere (ideal blue, bare orange, error-detected aqua,
the textbook encoding grey as context), thin marks, hairline grid, labels in ink.
The three tier colours pass a colour-vision check as a set; aqua is light on a white
surface, so every series is also labelled directly and the numbers are in the tables.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
BLUE, BLUE_LIGHT = "#2a78d6", "#86b6ef"
ORANGE, AQUA, GREY = "#eb6834", "#1baf7a", "#8a8982"
TIER_COLOR = {"ideal": BLUE, "bare": ORANGE, "bus": AQUA, "direct": GREY}
TIER_NAME = {"ideal": "Ideal (no noise)", "bare": "Bare QAOA", "bus": "Error-detected (bus)",
             "direct": "Textbook Iceberg"}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False,
        "legend.fontsize": 9, "lines.linewidth": 2.0, "lines.markersize": 7,
    })


def _save(fig, path: Path, note: str | None = None) -> Path:
    if note:
        fig.text(0.01, 0.005, note, fontsize=7.5, color=INK2, ha="left", va="bottom", wrap=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _dot(ax, x, y, color, *, lo=None, hi=None, horizontal=False, size=8, zorder=3) -> None:
    """A marker with a surface-coloured ring and an optional interval."""
    if lo is not None and hi is not None:
        if horizontal:
            ax.plot([lo, hi], [y, y], color=color, linewidth=1.6, solid_capstyle="round", zorder=zorder - 1)
        else:
            ax.plot([x, x], [lo, hi], color=color, linewidth=1.6, solid_capstyle="round", zorder=zorder - 1)
    ax.plot([x], [y], marker="o", markersize=size, markerfacecolor=color, markeredgecolor=SURFACE,
            markeredgewidth=1.6, linestyle="none", zorder=zorder)


def _load(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------- fig 1
def fig_accuracy(summary: Mapping[str, Any], out: Path) -> Path | None:
    lanes = [lane for lane in summary["lanes"] if "@" not in lane]
    if not lanes:
        return None
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(lanes) + 1.6))
    ypos = np.arange(len(lanes))[::-1]
    for y, lane in zip(ypos, lanes):
        s = summary["lanes"][lane]
        single, vote = s["em_single"], s["em_vote"]
        ax.plot([single["mean"], vote["mean"]], [y + 0.12, y - 0.12], color=GRID, linewidth=1.2, zorder=1)
        _dot(ax, single["mean"], y + 0.12, BLUE, lo=single["lo"], hi=single["hi"], horizontal=True)
        _dot(ax, vote["mean"], y - 0.12, BLUE_LIGHT, lo=vote["lo"], hi=vote["hi"], horizontal=True)
        ax.text(max(single["hi"], vote["hi"]) + 0.012, y, f"{single['mean']:.2f} / {vote['mean']:.2f}",
                va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{lane}  {summary['lanes'][lane]['label']}" for lane in lanes], color=INK)
    ax.set_xlabel("Exact match (share of questions)")
    ax.set_xlim(0, max(0.8, ax.get_xlim()[1] + 0.06))
    ax.grid(axis="y", visible=False)
    ax.plot([], [], marker="o", color=BLUE, linestyle="none", markeredgecolor=SURFACE, label="Single best selection")
    ax.plot([], [], marker="o", color=BLUE_LIGHT, linestyle="none", markeredgecolor=SURFACE, label="Vote over 3 selections")
    if "oracle" in lanes:            # upper-bound diagnostic, set apart from the selectors
        ax.axhline(ypos[lanes.index("oracle")] + 0.5, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.legend(loc="upper right")
    ax.set_title(f"Answer accuracy by lane ({summary['questions']} questions)")
    return _save(fig, out, "Bars are 95% bootstrap intervals over questions. Numbers at right: single / vote.")


# ------------------------------------------------------------------- fig 2
def fig_chain(summary: Mapping[str, Any], out: Path) -> Path | None:
    tiers = [("ideal", "3a"), ("bare", "3b"), ("bus", "3c")]
    if any(lane not in summary["lanes"] for _, lane in tiers):
        return None
    budgets = sorted({lane.split("@")[1] for lane in summary["lanes"] if lane.startswith("3b@")}, key=int)
    panels = [("feasibility", "Valid samples before repair"), ("approx_ratio", "Approximation ratio"),
              ("gold_recall", "Gold-paragraph recall"), ("em_single", "Exact match")]
    rows = [("", "all shots")] + [(f"@{b}", f"{b} raw shots") for b in budgets]
    fig, axes = plt.subplots(len(rows), 4, figsize=(11.5, 2.5 * len(rows) + 0.6), squeeze=False)
    for ri, (suffix, row_label) in enumerate(rows):
        for ci, (metric, title) in enumerate(panels):
            ax = axes[ri][ci]
            vals = []
            for xi, (tier, lane) in enumerate(tiers):
                cell = summary["lanes"].get(lane + suffix, {}).get(metric)
                if not cell or cell["n"] == 0:
                    continue
                _dot(ax, xi, cell["mean"], TIER_COLOR[tier], lo=cell["lo"], hi=cell["hi"])
                digits = 4 if metric == "approx_ratio" else 3
                ax.text(xi + 0.13, cell["mean"], f"{cell['mean']:.{digits}f}", va="center", fontsize=8, color=INK2)
                vals += [cell["lo"], cell["hi"]]
            if vals:
                pad = max(1e-3, 0.35 * (max(vals) - min(vals)))
                ax.set_ylim(min(vals) - pad, max(vals) + pad)
            ax.set_xlim(-0.5, 2.9)
            ax.set_xticks(range(3))
            ax.set_xticklabels(["Ideal", "Bare", "Error-\ndetected"], fontsize=8.5)
            ax.grid(axis="x", visible=False)
            if ri == 0:
                ax.set_title(title)
            if ci == 0:
                ax.set_ylabel(row_label, color=INK, fontsize=9.5)
    handles = [plt.Line2D([], [], marker="o", color=TIER_COLOR[t], linestyle="none", markeredgecolor=SURFACE,
                          label=TIER_NAME[t]) for t, _ in tiers]
    fig.legend(handles=handles, loc="upper right", ncol=3, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("From circuit noise to the final answer", x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return _save(fig, out, "Each panel has its own scale. Intervals are 95% bootstrap over questions. "
                           "Noise is the local stand-in model, not Sqale.")


# ------------------------------------------------------------------- fig 3
def fig_edges(study: Mapping[str, Any], out: Path) -> Path | None:
    pts = study.get("points") or []
    if not pts:
        return None
    edges = np.array([p["edges"] for p in pts], dtype=float)
    bins = [(0, 6), (6, 12), (12, 18), (18, 24), (24, 30), (30, 46)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))

    def binned(values: np.ndarray):
        xs, ms, los, his = [], [], [], []
        for a, b in bins:
            sel = (edges >= a) & (edges < b) & ~np.isnan(values)
            if sel.sum() < 3:
                continue
            v = values[sel]
            se = v.std(ddof=1) / np.sqrt(len(v))
            xs.append(edges[sel].mean()); ms.append(v.mean()); los.append(v.mean() - 1.96 * se); his.append(v.mean() + 1.96 * se)
        return np.array(xs), np.array(ms), np.array(los), np.array(his)

    ax = axes[0]
    for tier, key in (("bus", "bus_kept"), ("direct", "direct_kept")):
        vals = np.array([p[key] for p in pts], dtype=float)
        ax.plot(edges, vals, "o", markersize=3, color=TIER_COLOR[tier], alpha=0.25, markeredgewidth=0)
        xs, ms, los, his = binned(vals)
        ax.plot(xs, ms, color=TIER_COLOR[tier], label=TIER_NAME[tier])
        for x, m, lo, hi in zip(xs, ms, los, his):
            _dot(ax, x, m, TIER_COLOR[tier], lo=lo, hi=hi)
    ax.set_title("Share of shots kept")
    ax.set_xlabel("Redundant edges in the question's graph")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")

    ax = axes[1]
    for tier, key in (("ideal", "ideal_ar"), ("bare", "bare_ar"), ("bus", "bus_ar"), ("direct", "direct_ar")):
        vals = np.array([p[key] for p in pts], dtype=float)
        xs, ms, los, his = binned(vals)
        ax.plot(xs, ms, color=TIER_COLOR[tier], label=TIER_NAME[tier])
        for x, m, lo, hi in zip(xs, ms, los, his):
            _dot(ax, x, m, TIER_COLOR[tier], lo=lo, hi=hi, size=6)
    ax.set_title("Approximation ratio")
    ax.set_xlabel("Redundant edges in the question's graph")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[2]
    gain = np.array([p["bus_ar"] - p["bare_ar"] for p in pts], dtype=float)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.plot(edges, gain, "o", markersize=3, color=AQUA, alpha=0.25, markeredgewidth=0)
    xs, ms, los, his = binned(gain)
    ax.plot(xs, ms, color=AQUA)
    for x, m, lo, hi in zip(xs, ms, los, his):
        _dot(ax, x, m, AQUA, lo=lo, hi=hi)
    ax.set_title("Gain from error detection")
    ax.set_xlabel("Redundant edges in the question's graph")
    ax.set_ylabel("Approx. ratio: error-detected minus bare")
    noise = study.get("noise", {})
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out, f"Each faint dot is one question at one tau. Lines are binned means with 95% intervals. "
                           f"CZ error {noise.get('p2', '?')}, readout {noise.get('p_meas', '?')} (assumed).")


# ------------------------------------------------------------------- fig 4
def fig_gates(stats: Mapping[str, Any], out: Path) -> Path | None:
    mean = stats.get("mean")
    if not mean:
        return None
    bars = [("Bare\nall-to-all", mean["bare_all_to_all"], ORANGE), ("Bare\n4x6 grid", mean["bare_grid"], ORANGE),
            ("Textbook Iceberg\nall-to-all", mean["direct_all_to_all"], GREY),
            ("Textbook Iceberg\n4x6 grid, routed", mean["direct_grid"], GREY),
            ("Bus Iceberg\n4x6 grid, no SWAPs", mean["bus_grid"], AQUA)]
    sq = stats.get("superstaq")
    if sq:
        def avg(which: str, parity: int) -> float | None:
            vals = [r[which]["two_qubit_gates"] for i, r in enumerate(sq) if i % 2 == parity and which in r and "two_qubit_gates" in r[which]]
            return float(np.mean(vals)) if vals else None
        for label, which, parity, color in (("Bare\nSqale (cq_compile)", "sqale", 0, ORANGE),
                                            ("Bus Iceberg\nSqale (cq_compile)", "sqale", 1, AQUA),
                                            ("Bare\nIBM compile", "ibm", 0, ORANGE), ("Bus Iceberg\nIBM compile", "ibm", 1, AQUA)):
            v = avg(which, parity)
            if v is not None:
                bars.append((label, v, color))
    fig, ax = plt.subplots(figsize=(1.45 * len(bars) + 1.5, 3.9))
    for i, (label, value, color) in enumerate(bars):
        ax.bar(i, value, width=0.42, color=color, zorder=2)
        ax.text(i, value + max(v for _, v, _ in bars) * 0.015, f"{value:.0f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=8.5, color=INK)
    ax.set_ylabel("CZ gates per circuit (mean)")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Two-qubit gates for 10 paragraphs, {mean['edges']:.0f} redundant edges on average")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return _save(fig, out, "Colour marks the tier: orange bare, grey textbook Iceberg, aqua bus Iceberg. "
                           "The textbook mapping needs SWAP chains on a grid; the bus mapping needs none.")


# ------------------------------------------------------------------- fig 5
def fig_roadmap(road: Mapping[str, Any], out: Path) -> Path | None:
    rows = road.get("rows") or []
    if not rows:
        return None
    n = np.array([r["paragraphs"] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

    ax = axes[0]
    timed = [(r["paragraphs"], r["brute_force_seconds"]) for r in rows if r["brute_force_seconds"]]
    if timed:
        ax.plot([t[0] for t in timed], [t[1] for t in timed], color=INK2, marker="o", markeredgecolor=SURFACE, label="Measured")
    fit = road.get("brute_force_fit")
    if fit:
        xs = np.linspace(n.min(), n.max(), 50)
        ax.plot(xs, 2 ** (fit["log2_seconds_per_paragraph"] * xs + fit["intercept"]), color=INK2, linewidth=1.2,
                alpha=0.55, label="Exponential fit")
        ax.axvline(fit["paragraphs_at_1s"], color=GRID, linewidth=1)
        ax.text(fit["paragraphs_at_1s"] + 0.3, ax.get_ylim()[0] * 3 if ax.get_ylim()[0] > 0 else 1e-4,
                f"1 s at about {fit['paragraphs_at_1s']:.0f}", fontsize=8, color=INK2, rotation=90, va="bottom")
    ax.axhline(1.0, color=GRID, linewidth=1)
    ax.set_yscale("log")
    ax.set_title("Brute-force time")
    ax.set_xlabel("Paragraphs")
    ax.set_ylabel("Seconds (exhaustive enumeration)")
    ax.legend(loc="upper left")

    ax = axes[1]
    xs = np.arange(10, 101, 2)
    ax.plot(xs, xs, color=ORANGE, label="Bare")
    ax.plot(xs, xs + (xs % 2) + 4, color=GREY, label="Textbook Iceberg")
    ax.plot(xs, 1.5 * (xs + xs % 2) + 3, color=AQUA, label="Bus Iceberg")
    ax.axhline(24, color=INK2, linewidth=0.8)
    ax.text(100, 26, "Sqale grid today: 24", ha="right", fontsize=8, color=INK2)
    ax.axvline(100, color=INK2, linewidth=0.8)
    ax.text(98.5, 150, "100 logical qubits\n(Infleqtion 2028 target)", ha="right", va="top", fontsize=8, color=INK2)
    ax.set_title("Physical qubits")
    ax.set_xlabel("Paragraphs (logical qubits)")
    ax.legend(loc="upper left")

    ax = axes[2]
    ax.plot(n, [r["bare_cz_grid"] for r in rows], color=ORANGE, marker="o", markeredgecolor=SURFACE, label="Bare, grid-routed")
    ax.plot(n, [r["direct_cz_all_to_all"] for r in rows], color=GREY, marker="o", markeredgecolor=SURFACE, label="Textbook Iceberg, all-to-all")
    ax.plot(n, [r["bus_cz"] for r in rows], color=AQUA, marker="o", markeredgecolor=SURFACE, label="Bus Iceberg, nearest neighbour")
    ax.set_title(f"CZ gates per circuit (p = {road.get('p', 1)})")
    ax.set_xlabel("Paragraphs")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, out, "Synthetic problems with the evaluation set's edge density. Brute force here is plain enumeration, "
                           "not the best classical solver. The bus patch is 3 rows wide and fits Sqale's 6x4 grid up to 10 paragraphs.")


# ------------------------------------------------------------------- fig 6
def fig_fault_budget(budget: Mapping[str, Any], out: Path) -> Path | None:
    tiers = budget.get("tiers")
    if not tiers:
        return None
    order = ["bare", "direct", "bus"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    panels = [("undetected_share", "Two-qubit faults that slip through", "Share of all single CZ faults", 100.0, "{:.1f}%"),
              ("ar_loss_slope", "Approximation ratio lost", "Loss per unit CZ error rate", 1.0, "{:.2f}"),
              ("discard_slope", "Shots discarded", "Discarded share per unit CZ error rate", 1.0, "{:.0f}")]
    for ax, (key, title, ylabel, scale, fmt) in zip(axes, panels):
        vals = [tiers[t][key] * scale for t in order]
        for i, (t, v) in enumerate(zip(order, vals)):
            ax.bar(i, v, width=0.42, color=TIER_COLOR[t], zorder=2)
            ax.text(i, v + max(vals) * 0.02, fmt.format(v), ha="center", va="bottom", fontsize=9.5, color=INK)
        ax.set_xticks(range(3))
        short = {"bare": "Bare QAOA", "direct": "Textbook\nIceberg", "bus": "Error-detected\n(bus)"}
        ax.set_xticklabels([f"{short[t]}\n{tiers[t]['cz']:.0f} CZ" for t in order], fontsize=8.5, color=INK)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(vals) * 1.15 if max(vals) > 0 else 1)
        ax.grid(axis="x", visible=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, out, f"Every single fault injected and scored exactly, CZ level, mean of {budget.get('instances', '?')} "
                           "evaluation questions. Valid in the low-noise limit (at most one fault per shot).")


# ------------------------------------------------------------------- fig 7
def fig_noise_sweep(sweep: Mapping[str, Any], out: Path, kind: str = "depolarizing") -> Path | None:
    rows = [r for r in sweep.get("rows", []) if r["noise"] == kind]
    if not rows:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7))
    for tier in ("bare", "direct", "bus"):
        rs = sorted([r for r in rows if r["tier"] == tier], key=lambda r: r["p2"])
        x = [100 * r["p2"] for r in rs]
        label = f"{TIER_NAME[tier]} ({rs[0]['cz']:.0f} CZ)"
        axes[0].plot(x, [r["ar"] for r in rs], color=TIER_COLOR[tier], marker="o", markeredgecolor=SURFACE, label=label)
        if tier != "bare":
            axes[1].plot(x, [r["kept"] for r in rs], color=TIER_COLOR[tier], marker="o", markeredgecolor=SURFACE, label=TIER_NAME[tier])
        axes[2].plot(x, [r["p_opt_x_kept"] for r in rs], color=TIER_COLOR[tier], marker="o", markeredgecolor=SURFACE, label=TIER_NAME[tier])
    axes[0].axhline(sweep["ideal_ar"], color=BLUE, linewidth=1.2)
    axes[0].text(axes[0].get_xlim()[1], sweep["ideal_ar"], " ideal", color=INK2, fontsize=8, va="bottom", ha="right")
    axes[0].set_title("Approximation ratio of kept shots")
    axes[1].set_title("Share of shots kept")
    axes[1].set_ylim(0, 1)
    axes[2].set_title("Optimum found per raw shot")
    axes[2].set_ylabel("P(optimum) x share kept")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("CZ error rate (%)")
        ax.set_xticks(sorted({100 * r["p2"] for r in rows}))
        ax.set_xticklabels([f"{100 * p:g}" for p in sorted({r["p2"] for r in rows})])
        ax.minorticks_off()
        ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    title = "uniform two-qubit Pauli errors" if kind == "depolarizing" else "phase errors only (IZ, ZI, ZZ)"
    return _save(fig, out, f"{sweep['instances']} evaluation questions, {sweep['shots']} shots each, {title}. "
                           f"{sweep.get('note', '')}")


# -------------------------------------------------------------------- driver
def make_figures(cfg: Mapping[str, Any]) -> list[Path]:
    """Draw every figure whose input file exists; returns the files written."""
    _style()
    res = Path(cfg["output"])
    out = Path(cfg["figures"])
    made: list[Path | None] = []
    summary = _load(res / "summary.json")
    if summary:
        made.append(fig_accuracy(summary, out / "fig1_accuracy_by_lane.png"))
        made.append(fig_chain(summary, out / "fig2_noise_to_accuracy.png"))
    studies = res / "studies"
    for name, fn, target in (("edge_sweep", fig_edges, "fig3_redundant_edges.png"),
                             ("compile_stats", fig_gates, "fig4_two_qubit_gates.png"),
                             ("roadmap", fig_roadmap, "fig5_roadmap.png"),
                             ("fault_budget", fig_fault_budget, "fig6_fault_budget.png")):
        data = _load(studies / f"{name}.json")
        if data:
            made.append(fn(data, out / target))
    sweep = _load(studies / "noise_sweep.json")
    if sweep:
        made.append(fig_noise_sweep(sweep, out / "fig7_noise_sweep.png", "depolarizing"))
        made.append(fig_noise_sweep(sweep, out / "fig7b_noise_sweep_phase_errors.png", "dephasing"))
    return [p for p in made if p is not None]
