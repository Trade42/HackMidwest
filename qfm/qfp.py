"""Quantum feature map for two-candidate pronoun resolution.

Adapted from CedarvilleJimmySRNL-final/srnl/quantum_feature_map.py:
    raw features -> RobustAngleScaler -> IQPEmbedding -> Z/ZZ readouts
                 -> classical SVM -> candidate index (0 or 1)

The quantum map is a *feature extractor*, not a quantum kernel. Ring-connected
IQP embeddings and Z/ZZ readouts reproduce the structure of the Quantathon map.

Install:
    pip install 'pennylane==0.45.*' numpy scikit-learn

Run:
    python quantum_pronoun_qfm.py
    python quantum_pronoun_qfm.py --ablation --draw
    python quantum_pronoun_qfm.py --sentence "The trophy doesn't fit in the suitcase because it is too large." --candidate-0 trophy --candidate-1 suitcase

CSV input columns:
    sentence,candidate_0,candidate_1,label,group_id

This prototype includes a TF-IDF + SVD encoder solely to make the file runnable.
Replace it with Developer 1's *frozen* language-model embedding/feature encoder.
The sample data are too small for scientific claims about language understanding,
quantum advantage, or quantum contextuality.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import SVC

SEED = 42
N_QUBITS = 6


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("Expected a 2D feature matrix of shape (examples, features)")
    return x


@dataclass(frozen=True)
class Example:
    sentence: str
    candidate_0: str
    candidate_1: str
    label: int  # 0 = candidate_0; 1 = candidate_1
    group_id: str = ""

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError("label must be 0 or 1")


class ToyTextEncoder:
    """Demo-only shared input encoder; fit exclusively on training examples."""

    def __init__(self, n_features: int = N_QUBITS):
        self.n_features = n_features
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.svd: TruncatedSVD | None = None

    @staticmethod
    def _text(example: Example) -> str:
        return (
            f"{example.sentence} "
            f"candidate_zero {example.candidate_0} "
            f"candidate_one {example.candidate_1}"
        )

    def fit(self, examples: Sequence[Example]) -> "ToyTextEncoder":
        tfidf = self.vectorizer.fit_transform(self._text(e) for e in examples)
        n_components = min(self.n_features, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
        if n_components < 1:
            raise ValueError("Need at least two training rows and two vocabulary terms")
        self.svd = TruncatedSVD(n_components=n_components, random_state=SEED)
        self.svd.fit(tfidf)
        return self

    def transform(self, examples: Sequence[Example]) -> np.ndarray:
        if self.svd is None:
            raise RuntimeError("Call encoder.fit(train) before transform")
        tfidf = self.vectorizer.transform(self._text(e) for e in examples)
        raw = self.svd.transform(tfidf)
        if raw.shape[1] < self.n_features:
            raw = np.pad(raw, ((0, 0), (0, self.n_features - raw.shape[1])))
        return np.asarray(raw, dtype=float)


@dataclass
class RobustAngleScaler:
    """Train-only robust quantile scaling, adapted from the Quantathon map.

    Transforms each raw feature into [-max_angle, +max_angle].
    """

    quantile_range: tuple[float, float] = (0.05, 0.95)
    max_angle: float = 1.0
    preserve_sparse: bool = True

    def fit(self, x: np.ndarray) -> "RobustAngleScaler":
        x = _as_2d(x)
        lo, hi = self.quantile_range
        if not 0 <= lo < hi <= 1:
            raise ValueError("quantile_range must satisfy 0 <= low < high <= 1")
        self.fill_ = np.nanmedian(x, axis=0)
        if np.any(~np.isfinite(self.fill_)):
            raise ValueError("Each column must contain at least one finite value")
        filled = np.where(np.isfinite(x), x, self.fill_)
        self.lower_ = np.quantile(filled, lo, axis=0)
        self.upper_ = np.quantile(filled, hi, axis=0)
        if self.preserve_sparse:
            collapsed = np.isclose(self.lower_, self.upper_)
            fallback = collapsed & ~np.isclose(np.min(filled, axis=0), np.max(filled, axis=0))
            self.lower_[fallback] = np.min(filled, axis=0)[fallback]
            self.upper_[fallback] = np.max(filled, axis=0)[fallback]
        self.constant_ = np.isclose(self.lower_, self.upper_)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "fill_"):
            raise RuntimeError("Fit RobustAngleScaler on training features first")
        x = _as_2d(x)
        if x.shape[1] != len(self.fill_):
            raise ValueError("Input width differs from scaler's training data")
        filled = np.where(np.isfinite(x), x, self.fill_)
        span = np.where(self.constant_, 1.0, self.upper_ - self.lower_)
        unit = np.clip((filled - self.lower_) / span, 0.0, 1.0)
        angles = self.max_angle * (2.0 * unit - 1.0)
        angles[:, self.constant_] = 0.0
        return angles

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def ring_pairs(n_features: int) -> list[tuple[int, int]]:
    """Identical ring topology to the Quantathon feature map."""
    if n_features < 2:
        raise ValueError("Need at least two qubits/features")
    if n_features == 2:
        return [(0, 1)]
    return [(i, (i + 1) % n_features) for i in range(n_features)]


class IQPFeatureMap:
    """PennyLane IQP embedding plus commuting Z / ZZ expectation readouts.

    transform: (N, n_qubits) -> (N, n_qubits + len(ring_pairs))
    These measurements are real-valued quantum-generated *features*.
    """

    def __init__(
        self,
        n_features: int = N_QUBITS,
        *,
        repeats: int = 2,
        entangling: bool = True,
        shots: int | None = None,
        device=None,
        batch_size: int = 1024,
        entangling_pairs: Sequence[tuple[int, int]] | None = None,
        readout_pairs: Sequence[tuple[int, int]] | None = None,
        optimize_inactive: bool = True,
    ):
        if repeats < 1 or batch_size < 1 or n_features < 1:
            raise ValueError("repeats, batch_size and n_features must be positive")
        self.n_features = n_features
        self.repeats = repeats
        self.entangling = entangling
        self.shots = shots
        self.device = device
        self.batch_size = batch_size
        default_pairs = ring_pairs(n_features) if n_features > 1 else []
        self.pairs = default_pairs if readout_pairs is None else list(readout_pairs)
        self.entangling_pairs = default_pairs if entangling_pairs is None else list(entangling_pairs)
        for pairs in (self.pairs, self.entangling_pairs):
            if any(i == j or not (0 <= i < n_features and 0 <= j < n_features) for i, j in pairs):
                raise ValueError("Interaction and readout pairs must connect distinct, valid qubits")
        self.optimize_inactive = optimize_inactive
        self._circuit = self._make_circuit()
        self.last_unique_circuits_ = 0
        self.last_qnode_calls_ = 0
        self.last_simulated_qubits_ = n_features

    def _make_circuit(self):
        try:
            import pennylane as qml
        except ImportError as exc:
            raise ImportError("Install PennyLane: pip install 'pennylane==0.45.*'") from exc

        wires = list(range(self.n_features))
        device = self.device or qml.device(
            "default.qubit", wires=self.n_features, shots=self.shots
        )

        @qml.qnode(device)
        def circuit(angles):
            if self.entangling:
                qml.IQPEmbedding(
                    features=angles,
                    wires=wires,
                    n_repeats=self.repeats,
                    pattern=self.entangling_pairs,
                )
            else:
                # Ablation: same single-qubit IQP-like layers, no ZZ gates.
                for _ in range(self.repeats):
                    for wire in wires:
                        qml.Hadamard(wires=wire)
                    for wire in wires:
                        qml.RZ(angles[..., wire], wires=wire)

            # Same shared measurement setting as original Quantathon code.
            for wire in wires:
                qml.Hadamard(wires=wire)
            observables = [qml.Z(wire) for wire in wires]
            observables.extend(qml.Z(i) @ qml.Z(j) for i, j in self.pairs)
            return [qml.expval(obs) for obs in observables]

        return circuit

    @property
    def feature_names(self) -> list[str]:
        names = [f"q_z_{i}" for i in range(self.n_features)]
        names.extend(f"q_zz_{i}_{j}" for i, j in self.pairs)
        return names

    def transform(
        self, angles: np.ndarray, *, progress: Callable[[int, int], None] | None = None
    ) -> np.ndarray:
        angles = _as_2d(angles)
        if angles.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} angle features")
        if not np.isfinite(angles).all():
            raise ValueError("Quantum inputs must be finite")
        self.last_unique_circuits_ = 0
        self.last_qnode_calls_ = 0
        self.last_simulated_qubits_ = self.n_features
        if len(angles) == 0:
            return np.empty((0, len(self.feature_names)))

        # For exact expectations on our default simulator, a wire whose input
        # is identically zero is disconnected: every incident angle product is
        # zero. Its remaining r+1 Hadamards give <Z> = 1 for odd r, 0 for even r.
        # Factor it out, then reconstruct the ORIGINAL Z and ring-ZZ readouts.
        # Finite shots and externally supplied devices keep the original path.
        active = np.flatnonzero(np.any(angles != 0.0, axis=0))
        if (self.optimize_inactive and self.shots is None and self.device is None
                and len(active) < self.n_features):
            return self._transform_active(angles, active, progress)

        unique, inverse = np.unique(angles, axis=0, return_inverse=True)
        chunks = []
        for start in range(0, len(unique), self.batch_size):
            batch = unique[start : start + self.batch_size]
            # One (batch,) return array per observable, as in the original.
            chunks.append(np.asarray(self._circuit(batch), dtype=float).T)
            if progress is not None:
                progress(min(start + self.batch_size, len(unique)), len(unique))
        mapped = np.vstack(chunks)
        self.last_unique_circuits_ = len(unique)
        self.last_qnode_calls_ = len(chunks)
        return mapped[inverse]

    def _transform_active(self, angles, active, progress):
        isolated_z = float(self.repeats % 2)
        features = np.full((len(angles), len(self.feature_names)), isolated_z)
        self.last_simulated_qubits_ = len(active)
        if len(active) == 0:
            self.last_unique_circuits_ = 1
            if progress is not None:
                progress(1, 1)
            return features  # isolated_z**2 == isolated_z for every pair

        indices = {int(wire): i for i, wire in enumerate(active)}
        interactions = [(indices[i], indices[j]) for i, j in self.entangling_pairs
                        if i in indices and j in indices]
        measured_pairs = [(indices[i], indices[j]) for i, j in self.pairs
                          if i in indices and j in indices]
        reduced_map = IQPFeatureMap(
            len(active), repeats=self.repeats, entangling=self.entangling,
            batch_size=self.batch_size, entangling_pairs=interactions,
            readout_pairs=measured_pairs, optimize_inactive=False,
        )
        reduced = reduced_map.transform(angles[:, active], progress=progress)
        features[:, active] = reduced[:, :len(active)]
        pair_column = len(active)
        for offset, (i, j) in enumerate(self.pairs):
            if i in indices and j in indices:
                features[:, self.n_features + offset] = reduced[:, pair_column]
                pair_column += 1
            else:
                features[:, self.n_features + offset] = features[:, i] * features[:, j]
        self.last_unique_circuits_ = reduced_map.last_unique_circuits_
        self.last_qnode_calls_ = reduced_map.last_qnode_calls_
        return features

    def draw(self, angles: np.ndarray | None = None) -> str:
        import pennylane as qml

        if angles is None:
            angles = np.full(self.n_features, 0.3)
        angles = np.asarray(angles, dtype=float).reshape(-1)
        if len(angles) != self.n_features:
            raise ValueError("Wrong angle width")
        return str(qml.draw(self._circuit)(angles))


def classical_interaction_control(angles: np.ndarray) -> np.ndarray:
    """Same-width nonlinear control from the Quantathon implementation."""
    angles = _as_2d(angles)
    pairs = ring_pairs(angles.shape[1])
    singles = np.cos(angles)
    interaction_terms = np.column_stack(
        [np.cos(angles[:, i] * angles[:, j]) for i, j in pairs]
    )
    return np.column_stack((singles, interaction_terms))


class QuantumPronounClassifier:
    """Quantum feature extractor + classical SVM for candidate index 0/1."""

    def __init__(
        self,
        n_qubits: int = N_QUBITS,
        repeats: int = 2,
        entangling: bool = True,
        C: float = 1.0,
        shots: int | None = None,
    ):
        self.scaler = RobustAngleScaler()
        self.qfm = IQPFeatureMap(
            n_qubits, repeats=repeats, entangling=entangling, shots=shots
        )
        self.classifier = SVC(kernel="rbf", C=C)
        self.fit_seconds: float | None = None

    def fit_encoded(
        self, X_train: np.ndarray, labels: Sequence[int]
    ) -> "QuantumPronounClassifier":
        X_train = _as_2d(X_train)
        labels = np.asarray(labels, dtype=int)
        if len(X_train) != len(labels) or set(labels.tolist()) != {0, 1}:
            raise ValueError("Training examples must match labels and include 0/1")
        start = time.perf_counter()
        angles = self.scaler.fit_transform(X_train)
        quantum_features = self.qfm.transform(angles)
        self.classifier.fit(quantum_features, labels)
        self.fit_seconds = time.perf_counter() - start
        return self

    def predict_encoded(self, X: np.ndarray) -> np.ndarray:
        features = self.qfm.transform(self.scaler.transform(X))
        return self.classifier.predict(features).astype(int)

    def decision_encoded(self, X: np.ndarray) -> np.ndarray:
        features = self.qfm.transform(self.scaler.transform(X))
        return np.asarray(self.classifier.decision_function(features))


def read_examples_csv(path: str | Path) -> list[Example]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("No labeled examples found in CSV")
    return [
        Example(
            sentence=row["sentence"],
            candidate_0=row["candidate_0"],
            candidate_1=row["candidate_1"],
            label=int(row["label"]),
            group_id=row.get("group_id") or f"row-{i}",
        )
        for i, row in enumerate(rows)
    ]


def toy_examples() -> list[Example]:
    """Small contrastive example set; not a validated semantic benchmark."""
    scenarios = [
        ("trophy", "suitcase", "doesn't fit in", "large", "small"),
        ("package", "mailbox", "doesn't fit in", "large", "small"),
        ("painting", "frame", "doesn't fit in", "wide", "narrow"),
        ("sofa", "doorway", "doesn't fit through", "wide", "narrow"),
        ("truck", "tunnel", "cannot pass through", "tall", "low"),
        ("book", "bag", "doesn't fit in", "large", "small"),
        ("piano", "elevator", "doesn't fit in", "large", "small"),
        ("boat", "canal", "cannot pass through", "wide", "narrow"),
        ("cabinet", "hallway", "cannot pass through", "wide", "narrow"),
        ("luggage", "compartment", "doesn't fit in", "large", "small"),
        ("table", "doorway", "cannot pass through", "wide", "narrow"),
        ("vase", "box", "doesn't fit in", "tall", "short"),
    ]
    data: list[Example] = []
    for i, (item, container, relation, larger, smaller) in enumerate(scenarios):
        for adjective, referent in ((larger, item), (smaller, container)):
            sentence = (
                f"The {item} {relation} the {container} "
                f"because it is too {adjective}."
            )
            for candidate_0, candidate_1 in ((item, container), (container, item)):
                label = 0 if candidate_0 == referent else 1
                data.append(
                    Example(sentence, candidate_0, candidate_1, label, f"scenario-{i}")
                )
    return data


def split_examples(
    examples: Sequence[Example], seed: int = SEED
) -> tuple[list[Example], list[Example]]:
    groups = [ex.group_id or f"row-{i}" for i, ex in enumerate(examples)]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros(len(examples)), groups=groups))
    return [examples[i] for i in train_idx], [examples[i] for i in test_idx]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="Labeled CSV from Developer 1")
    parser.add_argument("--qubits", type=int, default=N_QUBITS)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--draw", action="store_true")
    parser.add_argument("--sentence")
    parser.add_argument("--candidate-0")
    parser.add_argument("--candidate-1")
    args = parser.parse_args()

    data = read_examples_csv(args.csv) if args.csv else toy_examples()
    train, test = split_examples(data)
    encoder = ToyTextEncoder(n_features=args.qubits).fit(train)
    X_train, X_test = encoder.transform(train), encoder.transform(test)
    y_train = np.array([ex.label for ex in train])
    y_test = np.array([ex.label for ex in test])

    print(f"Training examples: {len(train)}; test examples: {len(test)}")
    print(f"Qubits: {args.qubits}; IQP repeats: {args.repeats}")

    for entangling in ([True, False] if args.ablation else [True]):
        name = "Entangling IQP" if entangling else "Non-entangling ablation"
        print(f"\n--- {name} ---")
        model = QuantumPronounClassifier(
            n_qubits=args.qubits,
            repeats=args.repeats,
            entangling=entangling,
        ).fit_encoded(X_train, y_train)
        start = time.perf_counter()
        y_pred = model.predict_encoded(X_test)
        elapsed = time.perf_counter() - start
        print(f"Held-out accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print(f"Training runtime: {model.fit_seconds:.3f}s")
        print(f"Test runtime: {elapsed:.3f}s")
        print(f"Quantum readout features: {len(model.qfm.feature_names)}")
        print(f"Unique evaluation circuits: {model.qfm.last_unique_circuits_}")

        for example, prediction in list(zip(test, y_pred))[:3]:
            referent = (example.candidate_0, example.candidate_1)[int(prediction)]
            print(f"  {example.sentence} => {referent}")

        if args.draw:
            angles = model.scaler.transform(X_train[:1])[0]
            print("\nCircuit:\n" + model.qfm.draw(angles))

        if args.sentence:
            if not args.candidate_0 or not args.candidate_1:
                parser.error("--sentence requires --candidate-0 and --candidate-1")
            sample = Example(args.sentence, args.candidate_0, args.candidate_1, 0)
            raw = encoder.transform([sample])
            answer_index = int(model.predict_encoded(raw)[0])
            answer = (sample.candidate_0, sample.candidate_1)[answer_index]
            print(f"\nInput: {sample.sentence}")
            print(f"Predicted referent: {answer} (candidate_index={answer_index})")

    print()
    print("NOTE: Toy TF-IDF features are not a semantic language model.")
    print("Replace them with frozen LM embeddings before evaluating QFM claims.")
    print("This does not constitute a test of quantum contextuality.")


if __name__ == "__main__":
    main()
