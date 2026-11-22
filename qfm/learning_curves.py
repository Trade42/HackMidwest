"""Sample-efficiency benchmark on unseen semantic combinations of short sentences.

This uses an explicit parser of a synthetic grammar. It tests feature-map
inductive bias, not general NLP or a computational quantum advantage. Quantum
features are fixed; there are no quantum training epochs.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import platform
import re
import time
import warnings

import numpy as np
import pennylane as qml
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
from threadpoolctl import threadpool_limits

from feature_preprocessing import FEATURE_VARIANCE_FLOOR, make_feature_classifier
from qfp import IQPFeatureMap, classical_interaction_control

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("rbf", "polynomial", "mlp", "classical_interactions", "iqp_plain", "iqp_entangling")
PATTERN = re.compile(
    r"(?P<a>\w+) starts (?P<margin>\d+) points (?P<direction>above|below) (?P<b>\w+); "
    r"(?P<action>gains|loses) (?P<amount>\d+) (?P<condition>if|unless) blocked\. "
    r"(?P<status>Blocked|Unblocked); choose her, (?P<query>higher|lower)\."
)


@dataclass(frozen=True)
class Row:
    group_id: str
    sentence: str
    candidate_0: str
    candidate_1: str
    label: int


def make_dataset(seed, count):
    """One row per semantic group; names/candidate ordering do not define groups."""
    rng = np.random.default_rng(seed)
    rows = []
    names = (("Ava", "Mia"), ("Zoe", "Bea"), ("Kira", "Lana"))
    # Odd initial gaps and even changes prevent ties. Seven interacting inputs
    # span 2,048 distinct configurations, before candidate-order variation.
    for index, case in enumerate(product(range(1, 16, 2), (0, 1), range(2, 17, 2),
                                          (0, 1), (0, 1), (0, 1), (0, 1))):
        margin, above, amount, gains, if_blocked, blocked, query_high = case
        active = bool(blocked) if if_blocked else not bool(blocked)
        difference = (margin if above else -margin) + (amount if gains else -amount) * active
        left_is_answer = (difference > 0) == bool(query_high)
        a, b = names[index % len(names)]
        # Keep orientation across paired higher/lower queries so labels balance.
        if index % 2 == 0:
            reversed_candidates = bool(rng.integers(2))
        candidates = (b, a) if reversed_candidates else (a, b)
        answer = a if left_is_answer else b
        sentence = (f"{a} starts {margin} points {'above' if above else 'below'} {b}; "
                    f"{'gains' if gains else 'loses'} {amount} {'if' if if_blocked else 'unless'} blocked. "
                    f"{'Blocked' if blocked else 'Unblocked'}; choose her, {'higher' if query_high else 'lower'}.")
        rows.append(Row("-".join(map(str, case)), sentence, *candidates, candidates.index(answer)))
    pools = [[row for row in rows if row.label == label] for label in (0, 1)]
    if any(len(pool) < count // 2 for pool in pools):
        raise ValueError("Requested more balanced cases than the generator supports")
    selected = []
    for pool in pools:
        selected.extend(pool[i] for i in rng.permutation(len(pool))[:count // 2])
    return [selected[i] for i in rng.permutation(len(selected))]


def encode(row, layout):
    """Read text and candidates only; never read Row.label or Row.group_id."""
    match = PATTERN.fullmatch(row.sentence)
    if match is None:
        raise ValueError(f"Unsupported controlled sentence: {row.sentence}")
    fields = match.groupdict()
    if {row.candidate_0, row.candidate_1} != {fields['a'], fields['b']}:
        raise ValueError("Candidates do not match the sentence")
    values = (float(fields["margin"]) / 15, float(fields["direction"] == "above"),
              float(fields["amount"]) / 16, float(fields["action"] == "gains"),
              float(fields["condition"] == "if"), float(fields["status"] == "Blocked"),
              float(fields["query"] == "higher"), float(row.candidate_0 != fields["a"]))
    angles = np.array([0.2 + 0.85 * value + 0.01 * i for i, value in enumerate(values)])
    if layout == "meaning":
        return angles
    # One qubit per word position, with a fixed 15-word budget. Numbers count as
    # words; the first name carries the candidate-order marker. Other filler is zero.
    tokens = re.findall(r"\d+|[a-z]+(?:'[a-z]+)?", row.sentence.lower())
    if len(tokens) > 15:
        raise ValueError("Sentence exceeds 15 words")
    result = np.zeros(15)
    positions = (2, 4, 7, 6, 8, 10, 13, 0)
    for position, angle in zip(positions, angles):
        result[position] = angle
    return result


def split_indices(rows, seed, unit='world'):
    """Balanced 75/12.5/12.5 split, keeping both queries about a world together.

    The legacy configuration split is retained only for reproducing old runs.
    The generator has singleton worlds and opposite-label query pairs. Stratify
    these two types to retain exact class balance without splitting a world.
    """
    rng = np.random.default_rng(seed)
    labels = np.array([r.label for r in rows])
    validation, test, train = [], [], []
    per_class_heldout = len(rows) // 16
    if unit == 'configuration':
        for label in (0, 1):
            indices = rng.permutation(np.flatnonzero(labels == label))
            validation.extend(indices[:per_class_heldout])
            test.extend(indices[per_class_heldout:2 * per_class_heldout])
            train.extend(indices[2 * per_class_heldout:])
    elif unit == 'world':
        worlds = {}
        for index, row in enumerate(rows):
            worlds.setdefault(row.group_id.rsplit('-', 1)[0], []).append(index)
        paired, singles = [], [[], []]
        for indices in worlds.values():
            if len(indices) == 2 and set(labels[indices]) == {0, 1}:
                paired.append(indices)
            elif len(indices) == 1:
                singles[labels[indices[0]]].append(indices[0])
            else:
                raise ValueError('Expected singleton worlds or opposite-label query pairs')
        if len(singles[0]) != len(singles[1]):
            raise ValueError('World stratification requires a balanced source corpus')
        paired = [paired[i] for i in rng.permutation(len(paired))]
        pairs_heldout = round(len(paired) / 8)
        singles_heldout = per_class_heldout - pairs_heldout
        for destination, pairs in ((validation, paired[:pairs_heldout]),
                                    (test, paired[pairs_heldout:2*pairs_heldout]),
                                    (train, paired[2*pairs_heldout:])):
            destination.extend(i for pair in pairs for i in pair)
        for singleton_class in singles:
            indices = rng.permutation(singleton_class)
            validation.extend(indices[:singles_heldout])
            test.extend(indices[singles_heldout:2*singles_heldout])
            train.extend(indices[2*singles_heldout:])
    else:
        raise ValueError('Split unit must be world or configuration')
    result = {name: np.array(rng.permutation(indices), dtype=int)
              for name, indices in (("train", train), ("validation", validation), ("test", test))}
    groups = [{rows[i].group_id.rsplit('-', 1)[0] if unit == 'world' else rows[i].group_id
               for i in result[name]} for name in result]
    if any(a & b for a, b in combinations(groups, 2)):
        raise ValueError("Semantic groups overlap across splits")
    return result


def balanced_order(train, y, seed):
    rng = np.random.default_rng(seed)
    zero = rng.permutation(train[y[train] == 0])
    one = rng.permutation(train[y[train] == 1])
    return np.column_stack((zero, one)).reshape(-1)


def select_model(method, x, y, train, validation, seed):
    if method in ("mlp", "mlp_lbfgs"):
        choices = [{"hidden_layer_sizes": width, "alpha": alpha}
                   for width in ((32,), (32, 16)) for alpha in (0.0001, 0.001, 0.01, 0.1)]
    elif method == "polynomial":
        choices = [{"C": c, "degree": degree} for c in (1, 100) for degree in (2, 3, 4, 6)]
    elif method == "linear":
        choices = [{"C": c} for c in (0.0001, 0.001, 0.01, 0.1, 1, 10, 30, 100)]
    else:
        choices = [{"C": c, "gamma": gamma} for c in (0.1, 1, 10, 100) for gamma in ("scale", 0.1)]
    best = None
    best_score = -1.0
    warning_count = 0
    started = time.perf_counter()
    for params in choices:
        estimator = (MLPClassifier(**params, random_state=seed,
                                   max_iter=2000 if method == "mlp_lbfgs" else 500,
                                   solver="lbfgs" if method == "mlp_lbfgs" else "adam",
                                   early_stopping=False)
                     if method in ("mlp", "mlp_lbfgs") else
                     LinearSVC(**params, random_state=seed, max_iter=5000, dual="auto")
                     if method == "linear" else
                     SVC(kernel="poly" if method == "polynomial" else "rbf", coef0=1.0, **params))
        model = make_feature_classifier(estimator)
        # Tiny matrices run much slower when BLAS dispatches many CPU threads
        # for every optimizer step. Bound each fit instead of oversubscribing.
        with threadpool_limits(limits=1):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(x[train], y[train])
            score = float(model.score(x[validation], y[validation]))
        warning_count += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        if score > best_score:
            best, best_score, best_params = model, score, params
    return best, best_score, best_params, time.perf_counter() - started, warning_count


def feature_cache_path(x, layout, method, cache_dir, repeats):
    # Preserve the original key: preprocessing changes do not change raw maps.
    key = hashlib.sha256(x.tobytes() + f"{method}/{repeats}/{qml.__version__}/v1".encode()).hexdigest()
    return cache_dir / f"{layout}_{method}_{key[:16]}.npz"


def mapped_features(x, layout, method, output, repeats, *, cache_dir=None, cached_only=False):
    if method in ("rbf", "polynomial", "mlp"):
        return x, {"feature_seconds": 0.0, "cache_hit": False, "quantum": False}
    cache_dir = cache_dir if cache_dir is not None else output / "features"
    cache = feature_cache_path(x, layout, method, cache_dir, repeats)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            return saved["features"], {**json.loads(str(saved["info"])), "cache_hit": True}
    if cached_only:
        raise FileNotFoundError(f"--cached-only: missing feature cache {cache}")
    started = time.perf_counter()
    info = {"cache_hit": False, "quantum": method.startswith("iqp")}
    if method == "classical_interactions":
        features = classical_interaction_control(x)
    else:
        qfm = IQPFeatureMap(x.shape[1], repeats=repeats, entangling=method == "iqp_entangling",
                            batch_size=16, entangling_pairs=list(combinations(range(x.shape[1]), 2)))
        active_count = int(np.any(x != 0.0, axis=0).sum())
        print(f"  simulating {active_count}/{x.shape[1]} qubits; zero-angle wires factored out", flush=True)

        def progress(done, total):
            if done == min(16, total) or done == total or done % 128 == 0:
                print(f"  circuits {done}/{total}; elapsed {time.perf_counter() - started:.1f}s", flush=True)

        features = qfm.transform(x, progress=progress)
        resources = qml.specs(qfm._circuit, level="device")(x[0])["resources"]
        info.update(qubits=x.shape[1], gates=resources.num_gates, depth=resources.depth,
                    shots=None, unique_circuits=qfm.last_unique_circuits_,
                    simulated_qubits=qfm.last_simulated_qubits_, qnode_calls=qfm.last_qnode_calls_)
    info["feature_seconds"] = time.perf_counter() - started
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, features=features, info=json.dumps(info))
    return features, info


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(records, output):
    summary = []
    keys = sorted({(r["layout"], r["method"], r["n_train"]) for r in records})
    for layout, method, n in keys:
        selected = [r for r in records if (r["layout"], r["method"], r["n_train"]) == (layout, method, n)]
        scores = np.array([r["test_accuracy"] for r in selected])
        summary.append({"layout": layout, "method": method, "n_train": n,
                        "seeds": len(scores), "test_mean": float(scores.mean()),
                        "test_std": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
                        "fit_seconds_mean": float(np.mean([r["fit_and_tune_seconds"] for r in selected]))})
    save_csv(output / "summary.csv", summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    layouts = sorted({r["layout"] for r in summary})
    fig, axes = plt.subplots(1, len(layouts), figsize=(7 * len(layouts), 5), squeeze=False)
    for ax, layout in zip(axes[0], layouts):
        for method in METHODS:
            points = [r for r in summary if r["layout"] == layout and r["method"] == method]
            if not points:
                continue
            x = np.array([r["n_train"] for r in points])
            mean = np.array([r["test_mean"] for r in points])
            std = np.array([r["test_std"] for r in points])
            ax.plot(x, mean, marker="o", label=method)
            ax.fill_between(x, np.maximum(0, mean - std), np.minimum(1, mean + std), alpha=0.1)
        ax.axhline(0.5, color="gray", linestyle=":", label="Chance (50%)")
        ax.set(xscale="log", ylim=(0, 1.02), xlabel="Distinct labeled training configurations",
               ylabel="Accuracy on unseen semantic configurations", title=layout)
        sizes = sorted({r["n_train"] for r in summary})
        ax.set_xticks(sizes, [str(n) for n in sizes])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("Mean ± one standard deviation across training-subset seeds")
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts", nargs="+", choices=("meaning", "words"), default=["meaning", "words"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS),
                        help="Run only these models; default: all six")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--sizes", nargs="+", type=int, default=[8, 16, 32, 64, 128, 256])
    parser.add_argument("--cases", type=int, default=1024)
    parser.add_argument("--dataset-seed", type=int, default=241122)
    parser.add_argument("--split-unit", choices=("world", "configuration"), default="world",
                        help="Keep both queries about each world together; configuration reproduces legacy splits")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "qfm" / "results" / "learning_curves_worlds")
    parser.add_argument("--cache-dir", type=Path,
                        help="Reuse raw features from this directory; default: OUTPUT/features")
    parser.add_argument("--cached-only", action="store_true",
                        help="Fail before fitting if a requested feature cache is missing; never simulate")
    args = parser.parse_args()
    if args.cases < 64 or args.cases % 16 or args.cases > 2048:
        parser.error("cases must be a multiple of 16 between 64 and 2048")
    if args.seeds < 1 or args.repeats < 1 or any(n < 4 or n % 2 or n > args.cases * 3 // 4 for n in args.sizes):
        parser.error("Use positive seeds/repeats and even training sizes from 4 to 75% of cases")
    args.sizes = sorted(set(args.sizes))
    args.layouts = list(dict.fromkeys(args.layouts))
    args.methods = list(dict.fromkeys(args.methods))
    cache_dir = args.cache_dir if args.cache_dir is not None else args.output / "features"
    rows = make_dataset(args.dataset_seed, args.cases)
    y = np.array([r.label for r in rows])
    split = split_indices(rows, args.dataset_seed + 1, args.split_unit)
    inputs = {layout: np.array([encode(row, layout) for row in rows]) for layout in args.layouts}
    if args.cached_only:
        missing = [feature_cache_path(inputs[layout], layout, method, cache_dir, args.repeats)
                   for layout in args.layouts for method in args.methods
                   if method not in ("rbf", "polynomial", "mlp")
                   and not feature_cache_path(inputs[layout], layout, method, cache_dir, args.repeats).exists()]
        if missing:
            parser.error("--cached-only: missing feature caches:\n" + "\n".join(map(str, missing)))
    args.output.mkdir(parents=True, exist_ok=True)
    count = len(args.layouts) * len(args.methods) * args.seeds * len(args.sizes)
    print(f"{count} training configurations; 8 validation fits each. "
          f"Cached features only: {args.cached_only}", flush=True)
    assignments = {int(i): name for name, indices in split.items() for i in indices}
    save_csv(args.output / "dataset.csv", [{**asdict(row), "split": assignments[i]} for i, row in enumerate(rows)])
    protocol = {"dataset_seed": args.dataset_seed, "cases": args.cases, "sizes": args.sizes,
                "seeds": args.seeds, "layouts": args.layouts, "methods": args.methods,
                "repeats": args.repeats, "cache_dir": str(cache_dir), "cached_only": args.cached_only,
                "split_unit": args.split_unit,
                "split_sizes": {name: len(indices) for name, indices in split.items()},
                "versions": {"python": platform.python_version(), "pennylane": qml.__version__,
                             "numpy": np.__version__, "sklearn": sklearn.__version__},
                "selection": "Validation only; all methods get identical nested training subsets",
                "tuning_budget": "Eight hyperparameter candidates per method and training subset",
                "feature_cache": "Fixed unsupervised maps computed once; scaler fit only on each training subset",
                "preprocessing": {"variance_floor": FEATURE_VARIANCE_FLOOR,
                                  "fit_on": "Each training subset only; remove near-constant features before StandardScaler"},
                "uncertainty": "Standard deviation over training-subset seeds; fixed corpus and test split",
                "scope": "Hand-written grammar parser; fixed quantum features; no quantum training epochs",
                "timing": "Report feature computation separately from classifier fitting/tuning; caching is not a speedup claim",
                "feature_maps": {}}
    records = []
    for layout in args.layouts:
        x = inputs[layout]
        encoded_sets = [{tuple(x[i]) for i in split[name]} for name in split]
        if any(a & b for a, b in combinations(encoded_sets, 2)):
            raise ValueError("Encoded inputs overlap across semantic splits")
        if len(np.unique(x, axis=0)) != len(rows):
            raise ValueError("Encoding collapsed distinct semantic cases")
        print(f"\n{layout}: {x.shape[1]} qubits; {len(rows)} distinct inputs; no split overlap", flush=True)
        for method in args.methods:
            print(f"Preparing {layout}/{method} features...", flush=True)
            features, info = mapped_features(x, layout, method, args.output, args.repeats,
                                            cache_dir=cache_dir, cached_only=args.cached_only)
            if info["cache_hit"]:
                print(f"  loaded {len(features)} rows from cache; no feature generation", flush=True)
            protocol["feature_maps"][f"{layout}/{method}"] = info
            (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
            for seed in range(args.seeds):
                order = balanced_order(split["train"], y, seed)
                for n in args.sizes:
                    train = order[:n]
                    model, validation_accuracy, params, elapsed, warnings_count = select_model(
                        method, features, y, train, split["validation"], seed)
                    started = time.perf_counter()
                    predicted = model.predict(features[split["test"]])
                    predict_seconds = time.perf_counter() - started
                    record = {"layout": layout, "method": method, "seed": seed, "n_train": n,
                              "train_accuracy": float(model.score(features[train], y[train])),
                              "validation_accuracy": validation_accuracy,
                              "test_accuracy": float(np.mean(predicted == y[split["test"]])),
                              "fit_and_tune_seconds": elapsed, "test_head_seconds": predict_seconds,
                              "parameters": json.dumps(params), "convergence_warnings": warnings_count}
                    record["features_total"] = int(features.shape[1])
                    record["features_kept"] = int(model.named_steps["variancethreshold"].get_support().sum())
                    records.append(record)
                    save_csv(args.output / "runs.csv", records)
                print(f"  {method}: seed {seed + 1}/{args.seeds} complete", flush=True)
    summarize(records, args.output)
    print(f"\nSaved runs.csv, summary.csv, protocol.json, dataset.csv and learning_curves.png to {args.output}")


if __name__ == "__main__":
    main()
