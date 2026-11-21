"""Plot completed baseline or saved-shot evaluation summaries."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load_rows(paths):
    rows = []
    for path in paths:
        with Path(path).open(newline="") as file:
            for row in csv.DictReader(file):
                if row.get("status", "completed") != "completed":
                    continue
                rate = row.get("per_cycle_rate") or row.get("reported_per_cycle_rate")
                mean = row.get("mean_pair_ms") or row.get("reported_mean_ms")
                p99 = row.get("p99_pair_ms") or row.get("reported_p99_ms")
                if not all((row.get("p"), row.get("decoder"), rate, mean, p99)):
                    continue
                validation_only = row.get("validation_only", "").lower() in ("true", "1")
                source = f"{Path(path).parent.name}/{Path(path).stem}"
                if validation_only:
                    source += " validation-only"
                rows.append({"source": source, "validation_only": validation_only,
                             "decoder": row["decoder"], "p": float(row["p"]),
                             "rate": float(rate), "error": float(row.get("standard_error") or row.get("reported_standard_error") or 0),
                             "mean_ms": float(mean), "p99_ms": float(p99), "threshold": row.get("threshold")})
    return rows


def plot(paths, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_rows(paths)
    if not rows:
        raise ValueError("no completed summary rows to plot")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for row in rows:
        label = row["decoder"] + (f" threshold={row['threshold']}" if row["threshold"] else "")
        groups[(row["source"], label)].append(row)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (source, decoder), values in sorted(groups.items()):
        values.sort(key=lambda x: x["p"])
        ax.errorbar([x["p"] for x in values], [x["rate"] for x in values],
                    yerr=[x["error"] for x in values], marker="o", capsize=3, label=f"{decoder} ({source})")
    ax.set(xlabel="Physical noise probability p", ylabel="Logical error rate per cycle", title="Completed summary rows")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(out_dir / "logical_rate.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for (source, decoder), values in sorted(groups.items()):
        values.sort(key=lambda x: x["p"])
        for ax, key, title in zip(axes, ("mean_ms", "p99_ms"), ("Mean", "p99"), strict=True):
            ax.plot([x["p"] for x in values], [x[key] for x in values], marker="o",
                    label=f"{decoder} ({source})")
            ax.set(xlabel="Physical noise probability p", ylabel="X+Z pair decode time (ms)", title=title)
            ax.grid(True, alpha=0.3)
    axes[1].legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(out_dir / "latency.png", dpi=180)
    plt.close(fig)
    thresholds = [row for row in rows if row["threshold"]]
    if thresholds:
        curves = defaultdict(list)
        for row in thresholds:
            curves[(row["decoder"], row["p"], row["validation_only"])].append(row)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for (decoder, p, validation_only), values in sorted(curves.items()):
            values.sort(key=lambda x: float(x["threshold"]))
            ax.plot([x["mean_ms"] for x in values], [x["rate"] for x in values], marker="o",
                    label=f"{decoder} p={p:g}" + (" validation-only" if validation_only else ""))
            for value in values:
                ax.annotate(value["threshold"], (value["mean_ms"], value["rate"]), fontsize=7)
        ax.set(xlabel="Mean X+Z pair decode time (ms)", ylabel="Logical error rate per cycle",
               title="Router threshold sweep")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")
        fig.tight_layout()
        fig.savefig(out_dir / "router_thresholds.png", dpi=180)
        plt.close(fig)
    return len(rows), bool(thresholds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    count, thresholds = plot(args.summaries, args.out)
    print(f"plotted {count} completed rows; router threshold plot: {thresholds}")


if __name__ == "__main__":
    main()
