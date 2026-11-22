"""Leakage, representation and finite-shot checks; no model download required."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import torch

from distilbert_experiment import cohort_indices, fit, load_frozen, save_csv, write_json
from frozen_encoder import FrozenDistilBertEncoder, TrainingOnlyReducer, input_fingerprint, masked_mean, text_pair
from nlp_features import (FourierSurrogate, exact_features, iqp_probabilities, make_iqp,
                          readout_eigenvalues, resource_report, sample_readouts)


class DistilBertPipelineTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('QFM_TEST_PRETRAINED') == '1', 'Opt in to loading the pretrained checkpoint')
    def test_real_encoder_is_frozen_and_padding_does_not_change_the_embedding(self):
        encoder = FrozenDistilBertEncoder()
        short = dict(sentence='Ava is taller than Mia.', candidate_0='Ava', candidate_1='Mia')
        long = dict(short, sentence='Ava starts three points below Mia and gains four unless the move is blocked.')
        single, _ = encoder.encode([short])
        padded, _ = encoder.encode([short, long])
        self.assertEqual(single.shape, (1, 768))
        self.assertFalse(encoder.model.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in encoder.model.parameters()))
        np.testing.assert_allclose(single[0], padded[0], atol=3e-6, rtol=0)

    def test_encoder_input_excludes_labels_and_includes_ordered_candidates(self):
        row = dict(sentence='Ava is taller than Mia.', candidate_0='Ava', candidate_1='Mia', label=0, id='1')
        changed = dict(row, label=1, id='test-secret', world_id='different')
        self.assertEqual(text_pair(row), text_pair(changed))
        self.assertEqual(input_fingerprint([row], {}), input_fingerprint([changed], {}))
        self.assertNotEqual(input_fingerprint([row], {}), input_fingerprint([dict(row, candidate_0='Mia')], {}))

    def test_masked_pooling_ignores_padding_and_special_tokens(self):
        hidden = torch.tensor([[[999., 999.], [1., 2.], [3., 6.], [-99., -99.], [999., 999.]]])
        actual = masked_mean(hidden, torch.tensor([[1, 1, 1, 1, 0]]), torch.tensor([[1, 0, 0, 1, 1]]))
        np.testing.assert_allclose(actual.numpy(), [[2., 4.]])

    def test_pca_and_scaling_are_unchanged_by_heldout_data(self):
        rng = np.random.default_rng(17)
        train = rng.normal(size=(32, 16))
        reducer = TrainingOnlyReducer().fit(train)
        state = [reducer.pca.mean_.copy(), reducer.pca.components_.copy(), reducer.scaler.lower_.copy()]
        reducer.transform(rng.normal(1000, 10, (15, 16)))
        np.testing.assert_allclose(reducer.pca.mean_, train.mean(axis=0))
        for before, after in zip(state, [reducer.pca.mean_, reducer.pca.components_, reducer.scaler.lower_]):
            np.testing.assert_array_equal(before, after)
        self.assertEqual(reducer.n_fit_rows, 32)

    def test_cohorts_preserve_assignments_and_exclude_shared_worlds(self):
        rows = [dict(split=s, world_id=w) for s, w in [('train','a'), ('validation','a'),
                 ('validation','b'), ('test','a'), ('test','b'), ('test','c')]]
        train, validation, cohorts = cohort_indices(rows)
        np.testing.assert_array_equal(train, [0])
        np.testing.assert_array_equal(validation, [2])
        np.testing.assert_array_equal(cohorts['test'], [3, 4, 5])
        np.testing.assert_array_equal(cohorts['test_strict'], [5])
        self.assertEqual(rows[1]['split'], 'validation')

    def test_joint_shots_preserve_correlations(self):
        probabilities = np.zeros((1, 256))
        probabilities[0, [0, 255]] = .5
        measured = sample_readouts(probabilities, 1000, np.random.default_rng(7))
        np.testing.assert_array_equal(measured[0, :8], np.repeat(measured[0, 0], 8))
        np.testing.assert_array_equal(measured[0, 8:], np.ones(28))

    def test_probabilities_reproduce_existing_iqp_readouts(self):
        angles = np.random.default_rng(15).uniform(-.55, .55, (3, 8))
        for entangling in (True, False):
            exact = make_iqp(entangling).transform(angles)
            via_probabilities = iqp_probabilities(angles, entangling) @ readout_eigenvalues()
            np.testing.assert_allclose(exact, via_probabilities, atol=1e-12)
        singles = np.sin(angles)**2
        np.testing.assert_allclose(make_iqp(False).transform(angles)[:, :8], singles, atol=1e-12)

    def test_compiled_resource_counts(self):
        quantum = resource_report(True)
        self.assertEqual(quantum['compiled_two_qubit_gates'], 112)
        self.assertEqual(quantum['compiled_gates'], 208)
        self.assertEqual(resource_report(False)['compiled_two_qubit_gates'], 0)

    def test_trained_surrogate_inference_requires_no_quantum_calls(self):
        angles = np.random.default_rng(18).uniform(-.55, .55, (20, 8))
        surrogate = FourierSurrogate().fit(angles[:12], make_iqp().transform(angles[:12]))
        with patch('nlp_features.make_iqp', side_effect=AssertionError('Unexpected quantum inference')):
            features, _, _ = exact_features(angles[12:], ['surrogate'], surrogate=surrogate)
        self.assertEqual(features['surrogate'].shape, (8, 44))

    def test_changed_test_embeddings_and_labels_cannot_change_fitted_models(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(22, 16))
        rows = [dict(id=str(i), sentence=f'Sentence {i}', candidate_0='A', candidate_1='B',
                     label=i % 2, world_id=str(i), original_split=s, split=s)
                for i, s in enumerate(['train']*12 + ['validation']*4 + ['test']*6)]
        states = []
        with TemporaryDirectory() as directory:
            for changed in (False, True):
                output = Path(directory) / str(changed)
                output.mkdir()
                data = x.copy()
                source = [dict(r) for r in rows]
                if changed:
                    data[16:] += 1000
                    for row in source[16:]:
                        row['label'] = 1-row['label']
                save_csv(output / 'dataset.csv', source)
                write_json(output / 'protocol.json', dict(training_sizes=[12], subset_seeds=[301],
                                                         methods=['rbf', 'surrogate'], encoder={}))
                np.savez(output / 'embeddings.npz', all_X=data, all_ids=np.array([r['id'] for r in rows]))
                fit(output)
                frozen = load_frozen(output)
                bundle = joblib.load(output / frozen['bundles'][0]['path'])
                states.append(bundle)
                self.assertFalse((output / 'test_results.csv').exists())
                # Tampering after freeze must be caught before any evaluation.
                (output / 'protocol.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'changed after model freeze'):
                    load_frozen(output)
        np.testing.assert_array_equal(states[0]['reducer'].pca.components_, states[1]['reducer'].pca.components_)
        for method in ('rbf', 'surrogate'):
            np.testing.assert_array_equal(states[0]['heads'][method][-1].dual_coef_,
                                          states[1]['heads'][method][-1].dual_coef_)
        np.testing.assert_array_equal(states[0]['surrogate'].regressor[-1].coef_,
                                      states[1]['surrogate'].regressor[-1].coef_)


if __name__ == '__main__':
    unittest.main()
