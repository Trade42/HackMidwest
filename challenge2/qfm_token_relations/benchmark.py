"""Compare the fixed IQP map with classical pronoun-resolution baselines."""

import argparse
import csv
from pathlib import Path
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from feature_map import encode, quantum_features, nonentangling_features, circuit_resources


DATA = Path(__file__).resolve().parent / "data/pronouns.csv"
PLOT = Path(__file__).resolve().parent / "output/benchmark.png"


def load_data(path: Path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len({r["group_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate semantic configurations")
    if any(len(r["sentence"].split()) > 15 for r in rows):
        raise ValueError("Sentence exceeds the 15-word budget")
    world = lambda row: row["group_id"].rsplit("-", 1)[0]
    original_worlds = {world(r) for r in rows if r["split"] != "fresh_test"}
    validation = [r for r in rows if r["split"] == "validation"]
    validation_worlds = {world(r) for r in validation}
    training = [r for r in rows if r["split"] == "train" and world(r) not in validation_worlds]
    test = [r for r in rows if r["split"] == "fresh_test" and world(r) not in original_worlds]
    if not training or not validation or not test:
        raise ValueError("Missing train, validation, or strictly unseen test cases")
    return training, validation, test


def balanced_order(labels: np.ndarray, seed: int) -> np.ndarray:
    """Nested, class-balanced training prefixes for one sampling seed."""
    rng = np.random.default_rng(seed)
    halves = [rng.permutation(np.flatnonzero(labels == label)) for label in (0, 1)]
    count = min(map(len, halves))
    return np.column_stack((halves[0][:count], halves[1][:count])).reshape(-1)


def arrays(rows):
    x = np.array([encode(r["sentence"], r["candidate_0"], r["candidate_1"]) for r in rows])
    y = np.array([int(r["label"]) for r in rows])
    return x, y


def select_classifier(name: str, train_x, train_y, validation_x, validation_y, seed: int):
    if name == "MLP":
        choices = [MLPClassifier(hidden_layer_sizes=width, alpha=alpha, solver="lbfgs",
                                 max_iter=1000, random_state=seed)
                   for width in ((32,), (32, 16)) for alpha in (0.001, 0.01, 0.1)]
    elif name == "RBF SVM":
        choices = [SVC(kernel="rbf", C=c) for c in (0.1, 1, 10, 100)]
    else:
        raise ValueError(f"Unknown classifier: {name}")
    best, best_accuracy = None, -1.0
    for estimator in choices:
        model = make_pipeline(VarianceThreshold(threshold=1e-24), StandardScaler(), estimator)
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(train_x, train_y)
        score = float(np.mean(model.predict(validation_x) == validation_y))
        if score > best_accuracy:
            best, best_accuracy = model, score
    return best, best_accuracy


def plot_results(results, sizes, seeds: int, path: Path, test_size: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"RBF SVM": "#687782", "MLP": "#30343b",
              "non-entangling IQP": "#dc8d36", "entangling IQP": "#287e96"}
    for name, runs in results.items():
        scores = np.asarray(runs)
        mean = scores.mean(axis=0)
        std = scores.std(axis=0, ddof=1) if seeds > 1 else np.zeros(len(sizes))
        ax.plot(sizes, mean, marker="o", linewidth=2, label=name, color=colors[name])
        ax.fill_between(sizes, np.maximum(0, mean - std), np.minimum(1, mean + std),
                        color=colors[name], alpha=0.12)
        print(f"{name:20s} " + "  ".join(
            f"n={size}: {avg:.3f}±{spread:.3f}"
            for size, avg, spread in zip(sizes, mean, std)))
    ax.axhline(0.5, color="#aab0b5", linestyle=":", linewidth=1, label="Chance")
    ax.set(xscale="log", xlabel="Distinct labeled training configurations",
           ylabel="Accuracy on unseen semantic configurations",
           title=f"Mean ± one standard deviation across {seeds} training-subset seeds\n"
                 f"{test_size} fixed unseen-world test cases",
           ylim=(0.4, 1.0))
    ax.set_xticks(sizes, [str(size) for size in sizes])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--sizes", nargs="+", type=int, default=[8, 16, 32, 64, 128, 256])
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--plot", type=Path, default=PLOT)
    args = parser.parse_args()
    if args.seeds < 1 or any(size < 4 or size % 2 for size in args.sizes):
        parser.error("Use at least one seed and even training sizes of at least 4")
    sizes = sorted(set(args.sizes))
    pool, validation, test = load_data(args.data)
    x_train, y_train = arrays(pool)
    if sizes[-1] > 2 * min(np.bincount(y_train)):
        parser.error("Largest training size exceeds the balanced training pool")
    x_validation, y_validation = arrays(validation)
    x_test, y_test = arrays(test)
    print(f"Controlled pronouns: training sizes {sizes}, {args.seeds} seeds, "
          f"{len(validation)} validation, {len(test)} unseen-world test cases", flush=True)

    methods = (
        ("RBF SVM", "RBF SVM", lambda x: x),
        ("MLP", "MLP", lambda x: x),
        ("non-entangling IQP", "RBF SVM", lambda x: np.column_stack((x, nonentangling_features(x)))),
        ("entangling IQP", "RBF SVM", lambda x: np.column_stack((x, quantum_features(x)))),
    )
    results = {}
    for name, classifier, transform in methods:
        started = time.perf_counter()
        train_features, validation_features, test_features = (
            transform(x_train), transform(x_validation), transform(x_test))
        feature_seconds = time.perf_counter() - started
        results[name] = []
        print(f"{name}: features={feature_seconds:.2f}s", flush=True)
        for seed in range(args.seed, args.seed + args.seeds):
            order = balanced_order(y_train, seed)
            scores = []
            for size in sizes:
                model, _ = select_classifier(
                    classifier, train_features[order[:size]], y_train[order[:size]],
                    validation_features, y_validation, seed)
                scores.append(float(np.mean(model.predict(test_features) == y_test)))
            results[name].append(scores)
            print(f"  seed {seed}: " + ", ".join(f"{score:.3f}" for score in scores), flush=True)
    qubits, gates, depth = circuit_resources()
    print(f"Entangling circuit: {qubits} qubits, {gates} logical gates, depth {depth}; exact simulator")
    print("The encoder uses explicit cues in a synthetic grammar. This is not general NLP or quantum advantage.")
    plot_results(results, sizes, args.seeds, args.plot, len(test))
    print(f"Plot: {args.plot}")


if __name__ == "__main__":
    main()
