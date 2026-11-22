"""IQP feature map for a controlled pronoun-resolution task."""

from itertools import combinations
import re

import numpy as np
import pennylane as qml


N_QUBITS = 8
PAIRS = list(combinations(range(N_QUBITS), 2))
SENTENCE = re.compile(
    r"(?P<a>\w+) starts (?P<margin>\d+) points (?P<direction>above|below) (?P<b>\w+); "
    r"(?P<action>gains|loses) (?P<amount>\d+) (?P<condition>if|unless) blocked\. "
    r"(?P<status>Blocked|Unblocked); choose her, (?P<query>higher|lower)\."
)
_OBJECT = r"(?P<object>book|coat|charger|camera|scarf|tablet|umbrella|folder)"
_ENDING = (r"; she (?P<action>returned it|kept it|reclaimed it|asked for it back) "
           r"(?P<time>after class|before lunch|that evening|the next day)\.")
NUANCED_SENTENCES = (
    (re.compile(r"(?P<a>[A-Z][a-z]+) lent (?P<b>[A-Z][a-z]+) the " + _OBJECT + _ENDING), 0, 0),
    (re.compile(r"(?P<a>[A-Z][a-z]+) borrowed (?P<b>[A-Z][a-z]+)'s " + _OBJECT + _ENDING), 1, 0),
    (re.compile(r"(?P<b>[A-Z][a-z]+) was lent the " + _OBJECT + r" by (?P<a>[A-Z][a-z]+)" + _ENDING), 0, 1),
    (re.compile(r"(?P<b>[A-Z][a-z]+)'s " + _OBJECT + r" was borrowed by (?P<a>[A-Z][a-z]+)" + _ENDING), 1, 1),
)
NAMES = (("Ava", "Mia"), ("Zoe", "Bea"), ("Kira", "Lana"), ("Nora", "Iris"))
OBJECTS = ("book", "coat", "charger", "camera", "scarf", "tablet", "umbrella", "folder")
TIMES = ("after class", "before lunch", "that evening", "the next day")
ACTIONS = (("returned it", "kept it"), ("reclaimed it", "asked for it back"))


def encode(sentence: str, candidate_0: str, candidate_1: str) -> np.ndarray:
    """Parse only the sentence and candidate order; labels and IDs never enter."""
    match = SENTENCE.fullmatch(sentence)
    if match is not None:
        fields = match.groupdict()
        b = fields["b"]
        values = (
            int(fields["margin"]) / 15,
            float(fields["direction"] == "above"),
            int(fields["amount"]) / 16,
            float(fields["action"] == "gains"),
            float(fields["condition"] == "if"),
            float(fields["status"] == "Blocked"),
            float(fields["query"] == "higher"),
            float(candidate_0 != fields["a"]),
        )
    else:
        parsed = next(((match, transfer, voice) for pattern, transfer, voice in NUANCED_SENTENCES
                       if (match := pattern.fullmatch(sentence)) is not None), None)
        if parsed is None:
            raise ValueError(f"Unsupported sentence: {sentence}")
        match, transfer, voice = parsed
        fields = match.groupdict()
        b = fields["b"]
        if (fields["a"], b) not in NAMES:
            raise ValueError("Unsupported name pair")
        role, synonym = next((role, synonym) for role in range(2) for synonym in range(2)
                             if ACTIONS[role][synonym] == fields["action"])
        values = (
            float(transfer),
            float(role),
            float(synonym),
            float(voice),
            OBJECTS.index(fields["object"]) / (len(OBJECTS) - 1),
            TIMES.index(fields["time"]) / (len(TIMES) - 1),
            NAMES.index((fields["a"], b)) / (len(NAMES) - 1),
            float(candidate_0 != (fields["a"] if voice == 0 else b)),
        )
    if {candidate_0, candidate_1} != {fields["a"], b}:
        raise ValueError("Candidates do not match the sentence")
    return np.array([0.2 + 0.85 * value + 0.01 * i for i, value in enumerate(values)])


def _circuit():
    device = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(device)
    def circuit(angles):
        qml.IQPEmbedding(angles, wires=range(N_QUBITS), n_repeats=2, pattern=PAIRS)
        for wire in range(N_QUBITS):
            qml.Hadamard(wires=wire)
        return tuple([qml.expval(qml.Z(i)) for i in range(N_QUBITS)] +
                     [qml.expval(qml.Z(i) @ qml.Z(j)) for i, j in PAIRS])

    return circuit


def quantum_features(inputs: np.ndarray, batch_size: int = 16) -> np.ndarray:
    """Return 8 Z and 28 ZZ expectations from the entangling circuit."""
    inputs = np.asarray(inputs, dtype=float)
    if inputs.ndim != 2 or inputs.shape[1] != N_QUBITS or not np.isfinite(inputs).all():
        raise ValueError("Expected a finite (examples, 8) feature matrix")
    circuit = _circuit()
    unique, inverse = np.unique(inputs, axis=0, return_inverse=True)
    chunks = [np.asarray(circuit(batch * 0.55), dtype=float).T
              for batch in np.array_split(unique, range(batch_size, len(unique), batch_size))]
    return np.vstack(chunks)[inverse]


def nonentangling_features(inputs: np.ndarray) -> np.ndarray:
    """Exact analytic readouts of the matching two-repeat circuit without ZZ gates."""
    singles = np.sin(np.asarray(inputs) * 0.55) ** 2
    pairs = np.column_stack([singles[:, i] * singles[:, j] for i, j in PAIRS])
    return np.column_stack((singles, pairs))


def circuit_resources() -> tuple[int, int, int]:
    resources = qml.specs(_circuit(), level="device")(np.zeros(N_QUBITS))["resources"]
    return N_QUBITS, resources.num_gates, resources.depth
