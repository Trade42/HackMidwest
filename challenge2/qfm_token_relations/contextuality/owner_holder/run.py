"""Owner/holder sentences in a two-question, one-qubit memory game."""

import csv
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pennylane as qml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline


HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset.csv"
PLOT = HERE / "output" / "accuracy.png"
NAMES = ("Mia", "Zoe", "Kira", "Nora", "Bea", "Lana", "Iris", "Eva")
ITEMS = ("book", "coat", "camera", "key", "folder", "tablet", "scarf", "charger")
SIZES = (8, 16, 32, 64, 128, 256)
SEEDS = range(5)


def write_dataset():
    """Split name-item scenarios before making any sentence variants."""
    scenarios = list(product(NAMES, ITEMS))
    np.random.default_rng(42).shuffle(scenarios)
    rows = []
    for index, (other, item) in enumerate(scenarios):
        split = "train" if index < 40 else "validation" if index < 48 else "test"
        for owns, holds in product((0, 1), repeat=2):
            owner = "Ava" if owns else other
            holder = "Ava" if holds else other
            sentences = (
                f"{owner} owns the {item}; {holder} carries it.",
                f"{holder} holds the {item}, which belongs to {owner}.",
            )
            if owner == holder:
                other_person = other if owner == "Ava" else "Ava"
                sentences = (
                    sentences[0] + f" {other_person} neither owns nor carries it.",
                    sentences[1] + f" {other_person} neither owns nor holds it.",
                )
            for sentence in sentences:
                rows.append(dict(scenario=f"{other}/{item}", split=split,
                                 sentence=sentence, owns=owns, holds=holds))
    with DATA.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def load_dataset():
    with DATA.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups = {split: [row for row in rows if row["split"] == split]
              for split in ("train", "validation", "test")}
    train_scenarios, validation_scenarios, test_scenarios = (
        {row["scenario"] for row in group} for group in groups.values())
    if len({row["sentence"] for row in rows}) != len(rows) or (
            train_scenarios & validation_scenarios or
            train_scenarios & test_scenarios or
            validation_scenarios & test_scenarios):
        raise ValueError("Sentences or scenarios overlap across splits")
    return [(np.array([row["sentence"] for row in group]),
             np.array([[int(row["owns"]), int(row["holds"])] for row in group]))
            for group in groups.values()]


def balanced_order(labels, seed):
    rng = np.random.default_rng(seed)
    buckets = [rng.permutation(np.flatnonzero(labels[:, 0] * 2 + labels[:, 1] == code))
               for code in range(4)]
    return np.column_stack(buckets).ravel()


def fit_parser(sentences, labels, validation, answers):
    best_model, best_score = None, -1
    for strength in (0.1, 1.0, 10.0):
        model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2)),
            MultiOutputClassifier(LogisticRegression(C=strength, max_iter=300)),
        )
        model.fit(sentences, labels)
        score = np.mean(model.predict(validation) == answers)
        if score > best_score:
            best_model, best_score = model, score
    return best_model


def qubit_probabilities():
    device = qml.device("default.qubit", wires=1)

    @qml.qnode(device)
    def measure(angle, question):
        qml.RY(angle, wires=0)
        if question == 0:
            qml.Hadamard(wires=0)
        return qml.probs(wires=0)

    table = np.empty((4, 2, 2))
    for first, second in product((0, 1), repeat=2):
        angle = np.arctan2(1 - 2 * first, 1 - 2 * second)
        for question in (0, 1):
            table[first * 2 + second, question] = measure(angle, question)
    return table


def parity_guess(predicted, truth, stored):
    """Best chance of inferring answer parity from either message."""
    parity = truth[:, 0] ^ truth[:, 1]
    classical = [np.mean(predicted[parity == value, stored]) for value in (0, 1)]
    bit_guess = 0.5 + abs(classical[0] - classical[1]) / 2
    bloch = (1 - 2 * predicted) / np.sqrt(2)
    average = [bloch[parity == value].mean(axis=0) for value in (0, 1)]
    qubit_guess = 0.5 + np.linalg.norm(average[0] - average[1]) / 4
    return bit_guess, qubit_guess


def plot_results(results, test_count):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, color in (("Classical bit", "#687782"), ("Qubit", "#287e96")):
        values = np.asarray(results[name])
        mean = values.mean(axis=0)
        spread = values.std(axis=0, ddof=1)
        ax.plot(SIZES, mean, marker="o", linewidth=2, color=color, label=f"{name} (± SD)")
        ax.fill_between(SIZES, mean - spread, mean + spread, color=color, alpha=0.15)
        print(f"{name}: " + "  ".join(f"{size}: {score:.1%} ± {sd:.1%}"
                                      for size, score, sd in zip(SIZES, mean, spread)))
    ax.axhline(0.75, color="black", linestyle=":", linewidth=1, label="Classical limit, parity hidden")
    ax.set(xscale="log", xticks=SIZES, xticklabels=[str(size) for size in SIZES],
           ylim=(0.4, 1.0), xlabel="Training sentences",
           ylabel="Accuracy on a later owner/holder question",
           title=f"Owner and holder: {test_count} held-out sentences")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    PLOT.parent.mkdir(exist_ok=True)
    fig.savefig(PLOT, dpi=160)
    plt.close(fig)


def main():
    write_dataset()
    (train_x, train_y), (validation_x, validation_y), (test_x, test_y) = load_dataset()
    probabilities = qubit_probabilities()
    results = {"Classical bit": [], "Qubit": []}
    parity_at_largest = []
    parser_at_largest = []
    for seed in SEEDS:
        order = balanced_order(train_y, seed)
        run = {name: [] for name in results}
        for size in SIZES:
            model = fit_parser(train_x[order[:size]], train_y[order[:size]],
                               validation_x, validation_y)
            predicted_validation = model.predict(validation_x)
            stored = int(np.argmax(np.mean(predicted_validation == validation_y, axis=0)))
            predicted = model.predict(test_x)
            classical = 0.25 + 0.5 * np.mean(predicted[:, stored] == test_y[:, stored])
            codes = predicted[:, 0] * 2 + predicted[:, 1]
            quantum = np.mean([probabilities[codes, question, test_y[:, question]]
                               for question in (0, 1)])
            run["Classical bit"].append(classical)
            run["Qubit"].append(quantum)
            if size == SIZES[-1]:
                parity_at_largest.append(parity_guess(predicted, test_y, stored))
                parser_at_largest.append(np.mean(predicted == test_y))
        for name in results:
            results[name].append(run[name])
    print(f"Train: {len(train_x)}; validation: {len(validation_x)}; test: {len(test_x)} "
          "sentences (two possible questions each).")
    plot_results(results, len(test_x))
    parity = np.mean(parity_at_largest, axis=0)
    print(f"Held-out fact accuracy at {SIZES[-1]} training sentences: "
          f"{np.mean(parser_at_largest):.1%}.")
    print(f"Parity-guess accuracy at {SIZES[-1]} training sentences: "
          f"classical {parity[0]:.1%}, qubit {parity[1]:.1%} (ideal is 50%).")
    print("The text parser is trained; the one-bit and one-qubit memory protocols are fixed.")
    print(f"Plot: {PLOT}")


if __name__ == "__main__":
    main()
