"""Tables and the plot from the result files of run.py.

  python -m neutral_atom_noise.summarize --results OUTDIR/results --fit OUTDIR/fit.json --out OUTDIR

It writes OUTDIR/tables.md, summary.json, rates.csv and failure_vs_factor.png.

- A failure fraction comes with its 95% Wilson interval, and as a rate per cycle, 1 - (1 - P)^(1/12).
- Two rows of one point are compared on the shots both decoded: the shots only one of them gets wrong, and the
  exact two-sided sign test on those shots (a binomial test with chance 1/2).
- Iterations are belief-propagation iterations per shot. They do not depend on the machine; milliseconds do.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

CYCLES = 12
DECODER_NAMES = {"relay": "relay-BP", "bposd": "BP-OSD"}


def wilson(failures, shots, z=1.959964):
    """95% Wilson interval of a fraction."""
    if shots == 0:
        return 0.0, 1.0
    p = failures / shots
    centre = (p + z * z / (2 * shots)) / (1 + z * z / shots)
    half = z * np.sqrt(p * (1 - p) / shots + z * z / (4 * shots * shots)) / (1 + z * z / shots)
    return max(0.0, centre - half), min(1.0, centre + half)


def per_cycle(fraction):
    return 1 - (1 - fraction) ** (1 / CYCLES)


def paired(a_failed, b_failed):
    """Shots only a gets wrong, shots only b gets wrong, and the exact two-sided sign test on them."""
    n = min(len(a_failed), len(b_failed))
    a, b = a_failed[:n], b_failed[:n]
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"shots": n, "a_failures": int(a.sum()), "b_failures": int(b.sum()), "only_a": only_a, "only_b": only_b, "p": float(p)}


def load_results(folder):
    """{(shots file, decoder, priors, flags): row}. A row joins the parts of one run, in shot order."""
    parts = {}
    for path in sorted(Path(folder).glob("*.npz")):
        if path.name.endswith(".partial.npz"):
            continue
        with np.load(path) as f:
            run = json.loads(str(f["meta"]))
            arrays = {k: f[k] for k in ("shots", "failed", "converged", "iterations", "ms")}
        parts.setdefault((run["shots_file"], run["decoder"], run["priors"], run["flags"]), []).append((run, arrays))
    rows = {}
    for key, runs in parts.items():
        runs.sort(key=lambda item: item[0]["start"])
        joined = {k: np.concatenate([arrays[k] for _, arrays in runs]) for k in runs[0][1]}
        if not (joined["shots"] == np.arange(len(joined["shots"]))).all():
            raise SystemExit(f"the parts of {key} do not cover shots 0..N-1 exactly once")
        first, meta = runs[0][0], runs[0][0]["shots_meta"]
        rows[key] = {**joined, "shots_file": key[0], "decoder": key[1], "priors": key[2], "flags": key[3],
                     "basis": meta["basis"], "factor": meta["factor"], "uniform_p": meta["uniform_p"], "loss": meta["loss"],
                     "seed": meta["seed"], "shots_machine": meta["machine"], "shots_sha256": first["shots_sha256"],
                     "machines": sorted({run["machine"] for run, _ in runs}), "workers": sorted({run["workers"] for run, _ in runs}),
                     "noise": meta["noise"]}
    return rows


def describe(row, fit):
    """A plain name for the picture of the noise a row decodes with."""
    two_qubit = _two_qubit_error(row["noise"])
    best = fit.get(row["shots_file"], {}).get(row["decoder"] + ("/" + row["flags"] if row["flags"] else ""), {})
    if row["priors"] == "matched":
        name = "matched"
    elif row["priors"] == "generic-2q":
        name = f"generic, same two-qubit error (p = {two_qubit:.4f})"
        if best.get("multiplier") == 1:
            name += ", also the best p on train shots"
    else:
        p = float(row["priors"].removeprefix("uniform:"))
        fitted = best and abs(best["p"] - p) < 1e-12
        name = f"generic, best p on train shots (p = {p:.4f})" if fitted else f"generic, uniform p = {p:.4f}"
    return (row["flags"] + " + " if row["flags"] else "") + name


def _two_qubit_error(noise):
    return 1 - (1 - noise["cx_depolarize2"]) * (1 - noise["cx_phase"]) ** 2 * (1 - noise["cx_scatter"]) ** 2


def rate_record(row, fit):
    n, failures = len(row["failed"]), int(row["failed"].sum())
    low, high = wilson(failures, n)
    return {"basis": row["basis"], "factor": row["factor"], "loss": row["loss"], "decoder": row["decoder"], "priors": row["priors"],
            "flags": row["flags"], "picture": describe(row, fit), "shots": n, "failures": failures,
            "fraction": failures / n, "wilson_low": low, "wilson_high": high, "per_cycle": per_cycle(failures / n),
            "unconverged": int((~row["converged"]).sum()), "mean_iterations": float(row["iterations"].mean()),
            "median_iterations": float(np.median(row["iterations"])), "mean_ms": float(row["ms"].mean()),
            "shots_file": row["shots_file"], "seed": row["seed"], "shots_sha256": row["shots_sha256"],
            "shots_machine": row["shots_machine"], "decode_machines": row["machines"], "workers": row["workers"]}


def _order(row):
    rank = {"matched": 0, "generic-2q": 1}.get(row["priors"], 2)
    return (row["basis"], row["loss"], row["factor"] or 0, row["decoder"] != "relay", row["flags"] or "", rank, row["priors"])


def summarize(rows, fit):
    """(rate records, paired records). Every generic row is paired with the matched row of its point."""
    ordered = sorted(rows.values(), key=_order)
    rates = [rate_record(row, fit) for row in ordered]
    pairs = []
    for row in ordered:
        reference = rows.get((row["shots_file"], row["decoder"], "matched", row["flags"]))
        if row["priors"] == "matched" or reference is None:
            continue
        test = paired(row["failed"], reference["failed"])
        n = test["shots"]
        pairs.append({"basis": row["basis"], "factor": row["factor"], "loss": row["loss"], "decoder": row["decoder"],
                      "flags": row["flags"], "generic": describe(row, fit), "generic_priors": row["priors"], **test,
                      "iterations_generic": float(row["iterations"][:n].mean()),
                      "iterations_matched": float(reference["iterations"][:n].mean())})
    for point in {(p["basis"], p["factor"], p["loss"], p["decoder"], p["flags"]) for p in pairs}:
        group = [p for p in pairs if (p["basis"], p["factor"], p["loss"], p["decoder"], p["flags"]) == point]
        best = min(group, key=lambda p: (p["a_failures"], p["iterations_generic"]))
        for p in group:
            p["better_generic"] = p is best
    return rates, pairs


def _percent(x):
    return f"{100 * x:.2f}%" if x >= 0.001 or x == 0 else f"{100 * x:.3f}%"


def markdown(rates, pairs):
    lines = ["| Basis | Factor | Decoder | Picture of the noise | Failures / shots | Failure fraction (95% Wilson) | Per cycle | "
             "Unconverged | Mean iterations | Mean ms per shot |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rates:
        lines.append(f"| {r['basis']}{' + loss' if r['loss'] else ''} | {r['factor']:g} | {DECODER_NAMES[r['decoder']]} | {r['picture']} | "
                     f"{r['failures']} / {r['shots']} | {_percent(r['fraction'])} ({_percent(r['wilson_low'])}–{_percent(r['wilson_high'])}) | "
                     f"{r['per_cycle']:.2e} | {r['unconverged']} | {r['mean_iterations']:.0f} | {r['mean_ms']:.0f} |")
    lines += ["", "Paired, on the same shots (the generic row against the matched row of its point; **bold** marks the better "
              "generic row of a point):", "",
              "| Basis | Factor | Decoder | Generic picture | Shots | Failures: generic, matched | Only generic wrong | "
              "Only matched wrong | Exact sign test p | Iterations: generic ÷ matched |", "|---|---|---|---|---|---|---|---|---|---|"]
    for p in pairs:
        name = f"**{p['generic']}**" if p["better_generic"] else p["generic"]
        ratio = p["iterations_generic"] / p["iterations_matched"]
        lines.append(f"| {p['basis']}{' + loss' if p['loss'] else ''} | {p['factor']:g} | {DECODER_NAMES[p['decoder']]} | {name} | {p['shots']} | "
                     f"{p['a_failures']}, {p['b_failures']} | {p['only_a']} | {p['only_b']} | {p['p']:.2g} | "
                     f"{p['iterations_generic']:.0f} ÷ {p['iterations_matched']:.0f} = {ratio:.1f} |")
    return "\n".join(lines) + "\n"


# ---------- the plot ----------

SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
SERIES = [("matched", "matched priors", "#2a78d6", "o"),             # colour follows the picture of the noise
          ("generic-2q", "generic priors, same two-qubit error", "#eb6834", "s"),
          ("fit", "generic priors, best p on train shots", "#1baf7a", "D")]


def plot(rates, path, basis="X", loss=False, published_factor=1.0):
    """Failure fraction against the factor: one panel per decoder, one line per picture of the noise."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rates = [r for r in rates if r["basis"] == basis and r["loss"] == loss and not r["flags"]]
    decoders = [d for d in DECODER_NAMES if any(r["decoder"] == d for r in rates)]
    if not decoders:
        return False
    figure, axes = plt.subplots(1, len(decoders), figsize=(5.6 * len(decoders), 4.6), sharey=True, squeeze=False)
    figure.patch.set_facecolor(SURFACE)
    factors = sorted({r["factor"] for r in rates} | {published_factor})
    floor = min([r["wilson_high"] for r in rates] + [1e-3]) / 2
    for axis, decoder in zip(axes[0], decoders, strict=True):
        axis.set_facecolor(SURFACE)
        axis.axhspan(1e-3, 1e-1, color=GRID, alpha=0.45, lw=0)
        axis.axvline(published_factor, color=MUTED, lw=1)
        axis.text(published_factor, 0.985, "published figures ", color=MUTED, fontsize=8, ha="right", va="top",
                  rotation=90, transform=axis.get_xaxis_transform())
        for key, label, colour, marker in SERIES:
            points = [r for r in rates if r["decoder"] == decoder and key in _series(r)]
            points.sort(key=lambda r: r["factor"])
            if not points:
                continue
            seen = [r for r in points if r["failures"] > 0]
            axis.plot([r["factor"] for r in seen], [r["fraction"] for r in seen], color=colour, lw=2, zorder=3)
            axis.errorbar([r["factor"] for r in seen], [r["fraction"] for r in seen],
                          yerr=[[r["fraction"] - r["wilson_low"] for r in seen], [r["wilson_high"] - r["fraction"] for r in seen]],
                          fmt=marker, color=colour, ms=7, mec=SURFACE, mew=1.5, elinewidth=1, capsize=0, zorder=4)
            none = [r for r in points if r["failures"] == 0]     # no failure seen: show the 95% upper limit
            axis.plot([r["factor"] for r in none], [r["wilson_high"] for r in none], ls="none", marker="v", ms=7,
                      mfc=SURFACE, mec=colour, mew=1.5, zorder=4)
        axis.set_yscale("log")
        axis.minorticks_off()
        axis.set_ylim(floor, 1.0)
        axis.set_xticks(factors)
        axis.set_xticklabels([f"{f:g}" for f in factors])
        axis.set_xlabel("noise factor (1 = the published figures)", color=MUTED)
        axis.set_title(DECODER_NAMES[decoder], color=INK, fontsize=11, loc="left")
        axis.grid(True, which="major", color=GRID, lw=1)
        axis.tick_params(colors=MUTED, length=0)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color(AXIS)
        axis.text(0.02, 1e-2, "0.1% to 10%", color=MUTED, fontsize=8, va="center", transform=axis.get_yaxis_transform())
    axes[0][0].set_ylabel("shots with a wrong logical bit", color=MUTED)
    axes[0][0].set_yticks([1e-3, 1e-2, 1e-1, 1.0])
    axes[0][0].set_yticklabels(["0.1%", "1%", "10%", "100%"])
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=colour, lw=2, marker=marker, ms=7, mec=SURFACE, mew=1.5, label=label)
               for key, label, colour, marker in SERIES if any(key in _series(r) for r in rates)]
    figure.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9, labelcolor=INK)
    shots = sorted({r["shots"] for r in rates})
    figure.suptitle(f"{basis} memory{' with atom loss' if loss else ''}, Gross code, 12 cycles: failures against the noise factor\n"
                    f"{' or '.join(str(s) for s in shots)} shots per point, 95% Wilson intervals; "
                    "a hollow triangle is the upper limit where no shot failed", color=INK, fontsize=10, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0.06, 1, 0.9))
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return True


def _series(rate):
    """The lines a row belongs to: matched, generic-2q, the fitted generic p (which can be the generic-2q row)."""
    lines = [rate["priors"]] if rate["priors"] in ("matched", "generic-2q") else []
    return lines + (["fit"] if "best p on train shots" in rate["picture"] else [])


def write_all(results, fit_path, out):
    """Write tables.md, summary.json, rates.csv and the plot into `out`. Returns the markdown."""
    out = Path(out)
    fit = json.loads(Path(fit_path).read_text()) if fit_path and Path(fit_path).exists() else {}
    rates, pairs = summarize(load_results(results), fit)
    text = markdown(rates, pairs)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables.md").write_text(text)
    (out / "summary.json").write_text(json.dumps({"rates": rates, "paired": pairs, "fit": fit}, indent=1))
    with open(out / "rates.csv", "w", newline="") as f:
        columns = ["basis", "factor", "loss", "decoder", "priors", "flags", "picture", "shots", "failures", "fraction",
                   "wilson_low", "wilson_high", "per_cycle", "unconverged", "mean_iterations", "mean_ms", "seed", "shots_sha256"]
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rates)
    try:
        for basis in sorted({r["basis"] for r in rates}):
            plot(rates, out / f"failure_vs_factor_{basis}.png", basis)
    except ImportError:
        print("matplotlib is not installed: no plot written")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, help="folder of run.py result files")
    parser.add_argument("--fit", default=None, help="fit.json written by job.py")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(write_all(args.results, args.fit, args.out))


if __name__ == "__main__":
    main()
