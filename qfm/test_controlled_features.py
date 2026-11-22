"""Check the short corpus, clue preservation and the configurable circuit topology."""

import csv
from dataclasses import replace
from itertools import combinations
import json
from pathlib import Path
import unittest

import numpy as np

from corpus_experiment import load_split, validate_splits
from controlled_features import MeaningCueEncoder, WordPositionEncoder, extract_cues, tokenize
from qfp import IQPFeatureMap

CORPUS = Path(__file__).resolve().parents[1] / "quantum_token_corpus"


class ControlledFeaturesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = {s: load_split(CORPUS / "short", s, False)
                      for s in ("train", "validation", "test", "test_ood")}
        cls.examples = [e for examples, _ in cls.splits.values() for e in examples]

    def test_length_and_original_labels_and_splits_are_preserved(self):
        validate_splits(self.splits)
        sentences = set()
        for split, (examples, rows) in self.splits.items():
            _, originals = load_split(CORPUS, split, False)
            for e, row, original in zip(examples, rows, originals):
                self.assertLessEqual(len(tokenize(e.sentence)), 15)
                self.assertNotIn(e.sentence, sentences)
                sentences.add(e.sentence)
                self.assertEqual({k: v for k, v in row.items() if k != "sentence"},
                                 {k: v for k, v in original.items() if k != "sentence"})

    def test_parser_recovers_every_factor_from_text(self):
        with (CORPUS / "metadata_all.csv").open() as handle:
            metadata = {r["id"]: json.loads(r["factors"]) for r in csv.DictReader(handle)}
        for examples, rows in self.splits.values():
            for example, row in zip(examples, rows):
                f = metadata[row["id"]]
                if row["tier"] in ("1", "2"):
                    expected = {0: int(f.get("relation") == "cover"), 1: int(f["size_cue"] == "large")}
                else:
                    expected = {2: f["initial_left_high"], 3: f["swapped"], 4: f["query_high"]}
                _, cues = extract_cues(example)
                self.assertEqual({c.slot: c.value for c in cues}, expected)

    def test_both_layouts_preserve_target_distinctions_without_reading_labels(self):
        for encoder in (MeaningCueEncoder(), WordPositionEncoder()):
            encoded = encoder.transform(self.examples)
            seen = {}
            for row, example in zip(encoded, self.examples):
                key = tuple(row)
                self.assertEqual(seen.setdefault(key, example.label), example.label)
            changed_labels = [replace(e, label=1 - e.label) for e in self.examples[:12]]
            np.testing.assert_array_equal(encoder.transform(changed_labels), encoded[:12])
        with self.assertRaisesRegex(ValueError, "exceed"):
            WordPositionEncoder(3).transform(self.examples[:1])

    def test_positive_angle_encoding_survives_quantum_readout(self):
        x = MeaningCueEncoder().transform(self.examples)
        unique, indices = np.unique(x, axis=0, return_index=True)
        labels = [self.examples[i].label for i in indices]
        for entangling in (False, True):
            mapper = IQPFeatureMap(5, entangling=entangling,
                                   entangling_pairs=list(combinations(range(5), 2)))
            features = mapper.transform(unique)
            seen = {}
            for row, label in zip(features, labels):
                key = tuple(np.round(row, 9))
                self.assertEqual(seen.setdefault(key, label), label)

    def test_complete_topology_connects_nonadjacent_clues(self):
        angles = np.array([[0.25, 0.0, 1.04, 0.0, 0.0]])
        plain = IQPFeatureMap(5, entangling=False).transform(angles)
        ring = IQPFeatureMap(5).transform(angles)
        complete = IQPFeatureMap(5, entangling_pairs=list(combinations(range(5), 2))).transform(angles)
        np.testing.assert_allclose(plain, ring, atol=1e-12)
        self.assertFalse(np.allclose(plain, complete))


if __name__ == "__main__":
    unittest.main()
