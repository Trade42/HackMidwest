"""Explicit encodings for the short synthetic grammar; no fitted text compression.

Both layouts receive the same lexical clues. The positional layout places each
clue at its word position; the semantic layout places it in a named feature slot.
Filler words get zero angles. This is a controlled representation experiment,
not a pretrained word embedding or a general language-understanding system.
"""

from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np

from qfp import Example

SLOTS = ("cover_vs_fit", "large_vs_small", "initial_high", "swapped", "query_high")
HIGH = {"brighter", "warmer", "faster", "louder", "higher"}
LOW = {"dimmer", "cooler", "slower", "quieter", "lower"}


def tokenize(sentence: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", sentence.lower())


@dataclass(frozen=True)
class Cue:
    slot: int
    position: int
    value: int

    @property
    def angle(self) -> float:
        # Two distinct, positive angles per named feature. Avoid +/- aliases
        # in the original even-valued non-entangling IQP readouts.
        return 0.25 + 0.02 * self.slot + 0.75 * self.value


def extract_cues(example: Example) -> tuple[list[str], list[Cue]]:
    tokens = tokenize(example.sentence)
    positions = []
    for candidate in (example.candidate_0, example.candidate_1):
        candidate_tokens = tokenize(candidate)
        matches = [i for i in range(len(tokens) - len(candidate_tokens) + 1)
                   if tokens[i:i + len(candidate_tokens)] == candidate_tokens]
        if not candidate_tokens or not matches:
            raise ValueError(f"Candidate not found in sentence: {candidate}")
        positions.append(matches[0])
    if positions[0] >= positions[1]:
        raise ValueError("This controlled grammar requires candidate_0 before candidate_1")

    if "too" in tokens:
        size_pos = tokens.index("too") + 1
        if size_pos >= len(tokens) or tokens[size_pos] not in {"large", "small"}:
            raise ValueError("Expected 'too large' or 'too small'")
        relation_pos = [i for i in range(positions[0] + len(tokenize(example.candidate_0)), positions[1])
                        if tokens[i] in {"fit", "enter", "go", "cover"}]
        if len(relation_pos) != 1:
            raise ValueError("Expected one fit/cover relation")
        pos = relation_pos[0]
        cues = [Cue(0, pos, int(tokens[pos] == "cover")),
                Cue(1, size_pos, int(tokens[size_pos] == "large"))]
    else:
        comparisons = [i for i, word in enumerate(tokens) if word in HIGH | LOW]
        changes = [i for i, word in enumerate(tokens)
                   if word in {"swapped", "exchanged", "unchanged"}]
        if len(comparisons) != 2 or len(changes) != 1 or "than" not in tokens or "now" not in tokens:
            raise ValueError("Expected an initial comparison, state update and final comparison")
        first, last = comparisons
        changed = changes[0]
        if not first < changed < last:
            raise ValueError("Comparison and update order is invalid")
        cues = [Cue(2, first, int(tokens[first] in HIGH)),
                Cue(3, changed, int(tokens[changed] != "unchanged")),
                Cue(4, last, int(tokens[last] in HIGH))]
    return tokens, cues


class MeaningCueEncoder:
    n_qubits = len(SLOTS)

    def transform(self, examples: Sequence[Example]) -> np.ndarray:
        result = np.zeros((len(examples), self.n_qubits))
        for i, example in enumerate(examples):
            _, cues = extract_cues(example)
            for cue in cues:
                result[i, cue.slot] = cue.angle
        return result


class WordPositionEncoder:
    def __init__(self, max_words: int = 15):
        self.n_qubits = max_words

    def transform(self, examples: Sequence[Example]) -> np.ndarray:
        result = np.zeros((len(examples), self.n_qubits))
        for i, example in enumerate(examples):
            tokens, cues = extract_cues(example)
            if len(tokens) > self.n_qubits:
                raise ValueError(f"{len(tokens)} words exceed the {self.n_qubits}-qubit budget")
            for cue in cues:
                result[i, cue.position] = cue.angle
        return result
