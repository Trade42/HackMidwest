"""Regressions for roundoff amplification and bounded cached reruns."""

import json
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.svm import SVC

from feature_preprocessing import make_feature_classifier
from learning_curves import feature_cache_path, mapped_features
from qfp import IQPFeatureMap


class FeaturePreprocessingTests(unittest.TestCase):
    def test_roundoff_padding_does_not_change_predictions(self):
        rng = np.random.default_rng(17)
        signal = rng.uniform(-1, 1, (80, 2))
        labels = (signal[:, 0] + signal[:, 1] > 0).astype(int)
        noisy = np.column_stack((signal, rng.normal(0, 1e-17, (80, 20))))
        clean_model = make_feature_classifier(SVC(C=10)).fit(signal[:40], labels[:40])
        noisy_model = make_feature_classifier(SVC(C=10)).fit(noisy[:40], labels[:40])
        self.assertEqual(noisy_model.named_steps['variancethreshold'].get_support().sum(), 2)
        np.testing.assert_array_equal(clean_model.predict(signal[40:]), noisy_model.predict(noisy[40:]))
        np.testing.assert_allclose(clean_model.decision_function(signal[40:]),
                                   noisy_model.decision_function(noisy[40:]), atol=1e-12)

    def test_selection_and_scaling_use_training_rows_only(self):
        train = np.array([[1., 1e-17, 0.], [2., -1e-17, 0.],
                          [3., 2e-17, 0.], [4., -2e-17, 0.]])
        model = make_feature_classifier(SVC()).fit(train, [0, 0, 1, 1])
        heldout = np.array([[10., 100., 100.], [20., -100., -100.]])
        transformed = model[:-1].transform(heldout)
        self.assertEqual(transformed.shape, (2, 1))
        np.testing.assert_allclose(transformed[:, 0], (heldout[:, 0] - 2.5) / np.std(train[:, 0]))
        np.testing.assert_array_equal(model.named_steps['variancethreshold'].get_support(),
                                      [True, False, False])

    def test_small_real_signal_is_retained(self):
        train = np.array([[0., 1e-17], [1e-7, -1e-17], [2e-7, 2e-17], [3e-7, -2e-17]])
        model = make_feature_classifier(SVC()).fit(train, [0, 0, 1, 1])
        np.testing.assert_array_equal(model.named_steps['variancethreshold'].get_support(), [True, False])

    def test_cached_only_missing_file_never_starts_simulation(self):
        with TemporaryDirectory() as directory, patch('learning_curves.IQPFeatureMap') as mapper:
            with self.assertRaisesRegex(FileNotFoundError, 'cached-only'):
                mapped_features(np.zeros((2, 3)), 'words', 'iqp_entangling', Path(directory),
                                2, cached_only=True)
            mapper.assert_not_called()

    def test_raw_cache_reused_with_separate_output_directory(self):
        with TemporaryDirectory() as directory, patch('learning_curves.IQPFeatureMap') as mapper:
            cache_dir = Path(directory) / 'old_features'
            cache_dir.mkdir()
            x = np.array([[0.2, 0.4], [0.6, 0.8]])
            expected = np.array([[0.1, 0.2], [0.3, 0.4]])
            cache = feature_cache_path(x, 'meaning', 'iqp_plain', cache_dir, 2)
            np.savez(cache, features=expected, info=json.dumps({'cache_hit': False, 'quantum': True}))
            result, info = mapped_features(x, 'meaning', 'iqp_plain', Path(directory) / 'new_results',
                                           2, cache_dir=cache_dir, cached_only=True)
            np.testing.assert_array_equal(result, expected)
            self.assertTrue(info['cache_hit'])
            mapper.assert_not_called()

    def test_progress_counts_unique_inputs(self):
        x = np.array([[0.2, 0.4], [0.4, 0.6], [0.2, 0.4], [0.6, 0.8]])
        mapper = IQPFeatureMap(2, entangling=False, batch_size=2)
        progress = []
        result = mapper.transform(x, progress=lambda done, total: progress.append((done, total)))
        self.assertEqual(progress, [(2, 3), (3, 3)])
        self.assertEqual(result.shape, (4, 3))
        np.testing.assert_array_equal(result[0], result[2])

    def test_zero_wire_factorization_preserves_readouts(self):
        # Include both parities, a wire that is zero in only one row, no active
        # wires, and a single active wire. These are tiny circuits, not a sweep.
        inputs = (
            np.array([[0.2, 0., 0.4, 0.], [0.6, 0., 0., 0.]]),
            np.array([[0., 0., 0., 0.]]),
            np.array([[0., 0.3, 0., 0.]]),
        )
        for x in inputs:
            for repeats in (1, 2):
                for entangling in (False, True):
                    with self.subTest(active=int(np.any(x != 0., axis=0).sum()),
                                      repeats=repeats, entangling=entangling):
                        kwargs = dict(repeats=repeats, entangling=entangling,
                                      entangling_pairs=list(combinations(range(4), 2)))
                        reference = IQPFeatureMap(4, optimize_inactive=False, **kwargs)
                        optimized = IQPFeatureMap(4, **kwargs)
                        np.testing.assert_allclose(optimized.transform(x), reference.transform(x),
                                                   rtol=0, atol=1e-12)
                        self.assertEqual(optimized.last_simulated_qubits_, np.any(x != 0., axis=0).sum())

    def test_custom_device_keeps_original_simulation(self):
        import pennylane as qml
        mapper = IQPFeatureMap(2, device=qml.device('default.qubit', wires=2))
        mapper.transform(np.array([[0.2, 0.]]))
        self.assertEqual(mapper.last_simulated_qubits_, 2)


if __name__ == '__main__':
    unittest.main()
