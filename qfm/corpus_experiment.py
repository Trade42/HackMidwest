"""Run the existing IQP feature extractor on the teammate's fixed corpus splits.

This is a prototype using the shared TF-IDF/SVD encoder, not a frozen language
model. Replace only the encoder to test the intended semantic-embedding claim.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pennylane as qml
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from qfp import (
    Example,
    QuantumPronounClassifier,
    RobustAngleScaler,
    ToyTextEncoder,
    classical_interaction_control,
)

DEFAULT_CORPUS = Path(__file__).resolve().parents[1] / "quantum_token_corpus"


def load_split(corpus: Path, split: str, small: bool) -> tuple[list[Example], list[dict[str, str]]]:
    suffix = "_small" if small else ""
    path = corpus / f"{split}_mixed{suffix}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty split: {path}")
    required = {"id", "tier", "sentence", "candidate_0", "candidate_1", "label", "scenario_id"}
    if not required.issubset(rows[0]):
        raise ValueError(f"Missing required columns in {path}: {required - set(rows[0])}")
    examples = [
        Example(r["sentence"], r["candidate_0"], r["candidate_1"], int(r["label"]), r["scenario_id"])
        for r in rows
    ]
    return examples, rows


def validate_splits(splits: dict[str, tuple[list[Example], list[dict[str, str]]]]) -> None:
    ids: set[str] = set()
    scenarios: set[str] = set()
    for split, (_, rows) in splits.items():
        split_ids = {r["id"] for r in rows}
        split_scenarios = {r["scenario_id"] for r in rows}
        if len(split_ids) != len(rows) or ids & split_ids:
            raise ValueError(f"Duplicate or overlapping IDs in {split}")
        if scenarios & split_scenarios:
            raise ValueError(f"Scenario leakage into {split}")
        ids.update(split_ids)
        scenarios.update(split_scenarios)


def load_embeddings(path: Path, train_rows: list[dict[str, str]],
                    eval_rows: list[dict[str, str]], eval_split: str,
                    n_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    """Load frozen embeddings in CSV row order, checking IDs before fitting PCA."""
    with np.load(path, allow_pickle=False) as data:
        arrays = {}
        for split, rows in (("train", train_rows), (eval_split, eval_rows)):
            x_key, id_key = f"{split}_X", f"{split}_ids"
            if x_key not in data or id_key not in data:
                raise ValueError(f"{path} needs {x_key} and {id_key}")
            x = np.asarray(data[x_key], dtype=float)
            ids = [str(value) for value in data[id_key]]
            if x.ndim != 2 or x.shape[0] != len(rows) or ids != [r["id"] for r in rows]:
                raise ValueError(f"{split} embeddings or IDs do not match corpus CSV order")
            if not np.isfinite(x).all():
                raise ValueError(f"{split} embeddings contain non-finite values")
            arrays[split] = x
    train_x, eval_x = arrays["train"], arrays[eval_split]
    if train_x.shape[1] != eval_x.shape[1]:
        raise ValueError("Train and evaluation embedding widths differ")
    if train_x.shape[1] < n_qubits:
        raise ValueError("Embedding width is smaller than qubit count")
    if train_x.shape[1] > n_qubits:
        reducer = PCA(n_components=n_qubits, random_state=42).fit(train_x)
        train_x, eval_x = reducer.transform(train_x), reducer.transform(eval_x)
    return train_x, eval_x


def report(name: str, labels: np.ndarray, prediction: np.ndarray, rows: list[dict[str, str]],
           fit_seconds: float, predict_seconds: float) -> None:
    print(f"{name}: overall={accuracy_score(labels, prediction):.3f} "
          f"fit={fit_seconds:.2f}s predict={predict_seconds:.2f}s")
    for tier in ("1", "2", "3"):
        mask = np.array([r["tier"] == tier for r in rows])
        print(f"  tier {tier}: {accuracy_score(labels[mask], prediction[mask]):.3f} ({mask.sum()} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--full", action="store_true", help="Use full corpus instead of starter subset")
    parser.add_argument("--eval", choices=("validation", "test", "test_ood"), default="validation")
    parser.add_argument("--qubits", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--embeddings", type=Path,
                        help="NPZ with train_X/train_ids and <eval>_X/<eval>_ids")
    parser.add_argument("--predictions", type=Path, help="Write id,prediction CSV for the selected split")
    args = parser.parse_args()
    if not args.full and args.eval == "test_ood":
        parser.error("The corpus has no small test_ood split; use --full")

    splits = {
        split: load_split(args.corpus, split, small=not args.full)
        for split in ("train", args.eval)
    }
    validate_splits(splits)
    train, train_rows = splits["train"]
    evaluation, eval_rows = splits[args.eval]
    y_train = np.array([e.label for e in train])
    y_eval = np.array([e.label for e in evaluation])

    if args.embeddings:
        X_train, X_eval = load_embeddings(
            args.embeddings, train_rows, eval_rows, args.eval, args.qubits
        )
        encoder_name = f"frozen embeddings from {args.embeddings}"
    else:
        encoder = ToyTextEncoder(n_features=args.qubits).fit(train)
        X_train, X_eval = encoder.transform(train), encoder.transform(evaluation)
        encoder_name = "shared TF-IDF/SVD placeholder"
    scaler = RobustAngleScaler().fit(X_train)
    train_angles, eval_angles = scaler.transform(X_train), scaler.transform(X_eval)
    print(f"Corpus: {len(train)} train, {len(evaluation)} {args.eval}; "
          f"{encoder_name}; {args.qubits} features")

    for name, kernel in (("Raw angles + RBF SVM", "rbf"),
                         ("Raw angles + degree-3 polynomial SVM", "poly")):
        start = time.perf_counter()
        classifier = SVC(kernel=kernel, C=1.0, degree=3).fit(train_angles, y_train)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        prediction = classifier.predict(eval_angles)
        report(name, y_eval, prediction, eval_rows, fit_seconds,
               time.perf_counter() - start)

    # Classical interaction control has the same input angles and output width.
    start = time.perf_counter()
    classical = SVC(kernel="rbf", C=1.0).fit(
        classical_interaction_control(train_angles), y_train
    )
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    prediction = classical.predict(classical_interaction_control(eval_angles))
    report("Classical interaction + RBF SVM", y_eval, prediction, eval_rows,
           fit_seconds, time.perf_counter() - start)

    for entangling in (False, True):
        name = "Entangling IQP" if entangling else "Non-entangling IQP"
        model = QuantumPronounClassifier(
            n_qubits=args.qubits, repeats=args.repeats, entangling=entangling
        ).fit_encoded(X_train, y_train)
        start = time.perf_counter()
        prediction = model.predict_encoded(X_eval)
        report(name, y_eval, prediction, eval_rows, model.fit_seconds,
               time.perf_counter() - start)
        resources = qml.specs(model.qfm._circuit, level="device")(
            np.zeros(args.qubits)
        )["resources"]
        print(f"  quantum resources: {args.qubits} qubits, {args.repeats} IQP repeats, "
              f"{resources.num_gates} gates, depth {resources.depth}, "
              f"{len(model.qfm.feature_names)} readout features, analytic simulation")
        if entangling and args.predictions:
            args.predictions.parent.mkdir(parents=True, exist_ok=True)
            with args.predictions.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(("id", "prediction"))
                writer.writerows((r["id"], int(p)) for r, p in zip(eval_rows, prediction))
            print(f"  predictions: {args.predictions}")

    if not args.embeddings:
        print("Prototype only: TF-IDF/SVD inputs are not frozen language-model embeddings.")


if __name__ == "__main__":
    main()
