"""Compare word-position and semantic-cue layouts on the <=15-word corpus.

Inputs come from an explicit parser of the synthetic grammar. No labels or
symbolic metadata enter either encoder. All methods use the same clues; this
tests circuit input layouts, not general NLP ability or quantum advantage.
"""

import argparse
import csv
import hashlib
import json
import platform
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pennylane as qml
import sklearn
from sklearn.svm import SVC

from corpus_experiment import load_split, validate_splits
from controlled_features import MeaningCueEncoder, WordPositionEncoder, tokenize
from feature_preprocessing import FEATURE_VARIANCE_FLOOR, make_feature_classifier
from qfp import IQPFeatureMap, classical_interaction_control

ROOT = Path(__file__).resolve().parents[1]
C_GRID = (0.1, 1.0, 10.0, 100.0)


def accuracy(labels, predictions, rows):
    result = {"accuracy": float(np.mean(labels == predictions)), "n": len(labels)}
    for tier in ("1", "2", "3"):
        mask = np.array([r["tier"] == tier for r in rows])
        result[f"tier_{tier}"] = float(np.mean(labels[mask] == predictions[mask]))
    return result


def choose_classifier(train_x, labels, val_x, val_labels, kernel):
    # Same validation search and training-only standardization for all methods.
    best, best_accuracy, best_c = None, -1.0, None
    for c in C_GRID:
        classifier = make_feature_classifier(
            SVC(C=c, kernel=kernel, degree=3, coef0=1.0)
        ).fit(train_x, labels)
        score = float(np.mean(classifier.predict(val_x) == val_labels))
        if score > best_accuracy:
            best, best_accuracy, best_c = classifier, score, c
    return best, best_c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "quantum_token_corpus" / "short")
    parser.add_argument("--full", action="store_true", help="Use 1080 training rows instead of 144")
    parser.add_argument("--output", type=Path, help="Directory for results.json and summary.csv")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16, help="Quantum circuit batch size")
    parser.add_argument("--topology", choices=("complete", "ring"), default="complete",
                        help="Complete connects distant word clues; ring preserves original topology")
    args = parser.parse_args()
    if args.repeats < 1 or args.batch_size < 1:
        parser.error("repeats and batch-size must be positive")
    output = args.output or ROOT / "qfm" / "results" / ("short_full" if args.full else "short_small")
    names = ("train", "validation", "test", "test_ood")
    data = {split: load_split(args.corpus, split, small=not args.full and split != "test_ood")
            for split in names}
    validate_splits(data)
    labels = {split: np.array([e.label for e in data[split][0]]) for split in names}
    lengths = [len(tokenize(e.sentence)) for split in names for e in data[split][0]]
    if max(lengths) > 15:
        parser.error("Use the short corpus: sentences must contain at most 15 words")
    files = {split: args.corpus / f"{split}_mixed{'_small' if not args.full and split != 'test_ood' else ''}.csv"
             for split in names}
    report = {
        "corpus": str(args.corpus), "full": args.full,
        "word_range": [min(lengths), max(lengths)],
        "split_sizes": {split: len(data[split][0]) for split in names},
        "sha256": {split: hashlib.sha256(path.read_bytes()).hexdigest() for split, path in files.items()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pennylane": qml.__version__, "scikit_learn": sklearn.__version__},
        "protocol": {"C_grid": C_GRID, "selection": "validation accuracy; smallest C breaks ties",
                     "classifier_scaling": "Remove near-constant features then StandardScaler; fit on training only",
                     "feature_variance_floor": FEATURE_VARIANCE_FLOOR,
                     "shots": None, "device": "default.qubit", "repeats": args.repeats,
                     "batch_size": args.batch_size,
                     "entangling_topology": args.topology,
                     "encoding": "Explicit lexical cues; filler zero; no SVD/PCA or angle rescaling",
                     "limitation": "Controlled grammar; candidate_0 precedes candidate_1; not general NLP"},
        "layouts": {}, "models": [],
    }
    print(f"Short corpus: {report['split_sizes']}; {min(lengths)}–{max(lengths)} words", flush=True)
    summary = []
    for layout, encoder in (("word_positions", WordPositionEncoder(15)),
                            ("meaning_cues", MeaningCueEncoder())):
        encode_start = time.perf_counter()
        encoded = {split: encoder.transform(data[split][0]) for split in names}
        training_inputs = {tuple(x) for x in encoded["train"]}
        report["layouts"][layout] = {
            "qubits": encoder.n_qubits, "unique_training_inputs": len(training_inputs),
            "encode_seconds_all_splits": time.perf_counter() - encode_start,
            "seen_training_input_fraction": {
                split: float(np.mean([tuple(x) in training_inputs for x in encoded[split]]))
                for split in names[1:]},
        }
        print(f"\n{layout}: {encoder.n_qubits} qubits, {len(training_inputs)} distinct training inputs", flush=True)
        for method in ("rbf", "poly3", "classical_interactions", "iqp_no_entanglement", "iqp_entangling"):
            resources = None
            if method.startswith("iqp"):
                mapping = IQPFeatureMap(encoder.n_qubits, repeats=args.repeats,
                                        entangling=method == "iqp_entangling", batch_size=args.batch_size,
                                        entangling_pairs=list(combinations(range(encoder.n_qubits), 2))
                                        if args.topology == "complete" else None)
                transform = mapping.transform
                specs = qml.specs(mapping._circuit, level="device")(np.zeros(encoder.n_qubits))["resources"]
                resources = {"qubits": encoder.n_qubits, "gates": specs.num_gates, "depth": specs.depth,
                             "readout_features": len(mapping.feature_names), "shots": None}
            elif method == "classical_interactions":
                transform = classical_interaction_control
            else:
                transform = lambda x: x
            features, feature_seconds, executions = {}, {}, {}
            # Select the classifier using only train/validation before evaluating test data.
            for split in ("train", "validation"):
                start = time.perf_counter()
                features[split] = transform(encoded[split])
                feature_seconds[split] = time.perf_counter() - start
                if resources:
                    executions[split] = mapping.last_unique_circuits_
            start = time.perf_counter()
            classifier, c = choose_classifier(features["train"], labels["train"], features["validation"],
                                               labels["validation"], "poly" if method == "poly3" else "rbf")
            fit_seconds = time.perf_counter() - start
            model_result = {"layout": layout, "method": method, "selected_C": c,
                            "resources": resources, "fit_and_selection_seconds": fit_seconds,
                            "feature_seconds": feature_seconds, "unique_circuits": executions, "scores": {}}
            for split in names:
                if split not in features:
                    start = time.perf_counter()
                    features[split] = transform(encoded[split])
                    feature_seconds[split] = time.perf_counter() - start
                    if resources:
                        executions[split] = mapping.last_unique_circuits_
                start = time.perf_counter()
                prediction = classifier.predict(features[split])
                score = accuracy(labels[split], prediction, data[split][1])
                score["svm_prediction_seconds"] = time.perf_counter() - start
                model_result["scores"][split] = score
                summary.append({"layout": layout, "method": method, "split": split,
                                "C": c, **{k: score[k] for k in ("accuracy", "tier_1", "tier_2", "tier_3")}})
            report["models"].append(model_result)
            scores = model_result["scores"]
            print(f"  {method:24s} train={scores['train']['accuracy']:.3f} "
                  f"val={scores['validation']['accuracy']:.3f} test={scores['test']['accuracy']:.3f} "
                  f"ood={scores['test_ood']['accuracy']:.3f} C={c:g} "
                  f"feature_time={sum(feature_seconds.values()):.2f}s", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nResults: {output}", flush=True)
    print("Both encoders use an explicit grammar parser. See seen_training_input_fraction in results.json.")


if __name__ == "__main__":
    main()
