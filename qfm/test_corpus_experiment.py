"""Checks the fixed corpus interface and both quantum feature maps."""

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np

from corpus_experiment import DEFAULT_CORPUS, load_embeddings, load_split, validate_splits
from qfp import IQPFeatureMap


class CorpusIntegrationTests(unittest.TestCase):
    def test_starter_splits_are_disjoint_and_balanced(self):
        splits = {
            name: load_split(DEFAULT_CORPUS, name, small=True)
            for name in ("train", "validation", "test")
        }
        validate_splits(splits)
        self.assertEqual([len(splits[n][0]) for n in splits], [144, 72, 72])
        for examples, _ in splits.values():
            self.assertEqual(sum(e.label for e in examples), len(examples) // 2)

    def test_quantum_maps_have_expected_shape_and_different_outputs(self):
        angles = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]] * 2)
        plain = IQPFeatureMap(entangling=False)
        entangled = IQPFeatureMap(entangling=True)
        plain_features = plain.transform(angles)
        entangled_features = entangled.transform(angles)
        self.assertEqual(plain_features.shape, (2, 12))
        self.assertEqual(entangled_features.shape, (2, 12))
        self.assertEqual(entangled.last_unique_circuits_, 1)
        self.assertTrue(np.isfinite(entangled_features).all())
        self.assertFalse(np.allclose(plain_features, entangled_features))

    def test_external_embeddings_require_matching_ids(self):
        _, train_rows = load_split(DEFAULT_CORPUS, "train", small=True)
        _, eval_rows = load_split(DEFAULT_CORPUS, "validation", small=True)
        train_ids = np.array([r["id"] for r in train_rows])
        eval_ids = np.array([r["id"] for r in eval_rows])
        rng = np.random.default_rng(42)
        train_embeddings = rng.normal(size=(144, 8))
        eval_embeddings = rng.normal(size=(72, 8))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.npz"
            np.savez(path, train_X=train_embeddings, train_ids=train_ids,
                     validation_X=eval_embeddings, validation_ids=eval_ids)
            train_x, eval_x = load_embeddings(path, train_rows, eval_rows, "validation", 6)
            self.assertEqual(train_x.shape, (144, 6))
            self.assertEqual(eval_x.shape, (72, 6))
            np.savez(path, train_X=train_embeddings, train_ids=train_ids[::-1],
                     validation_X=eval_embeddings, validation_ids=eval_ids)
            with self.assertRaisesRegex(ValueError, "do not match"):
                load_embeddings(path, train_rows, eval_rows, "validation", 6)


if __name__ == "__main__":
    unittest.main()
