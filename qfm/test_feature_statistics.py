"""Small mathematical checks for the exploratory readouts and uncertainty audit."""

import unittest

import numpy as np

from confirm_feature_gain import paired_interval
from feature_search import pauli_features_from_states
from learning_curves import make_dataset, split_indices


class FeatureStatisticsTests(unittest.TestCase):
    def test_world_split_keeps_related_questions_together_and_balanced(self):
        for count in (64, 1024, 2048):
            rows = make_dataset(241122, count)
            split = split_indices(rows, 241123)
            world_sets = []
            for name, indices in split.items():
                expected_count = count * 3 // 4 if name == 'train' else count // 8
                self.assertEqual(len(indices), expected_count)
                self.assertEqual(sum(rows[i].label for i in indices), expected_count // 2)
                world_sets.append({rows[i].group_id.rsplit('-', 1)[0] for i in indices})
            self.assertEqual(sum(map(len, world_sets)), len(set.union(*world_sets)))
            self.assertEqual(len(set(np.concatenate(list(split.values())))), count)

    def test_duplicate_queries_within_world_do_not_add_independent_evidence(self):
        differences = np.array([[1., 0., -1.], [1., 1., -1.], [0., 1., 0.]])
        original = paired_interval(differences, comparisons=1, resamples=100,
                                   cluster_ids=['a', 'b', 'c'])
        duplicate = paired_interval(np.repeat(differences, 2, axis=1), comparisons=1, resamples=100,
                                    cluster_ids=['a', 'a', 'b', 'b', 'c', 'c'])
        np.testing.assert_allclose(original, duplicate, atol=1e-15)

    def test_identical_models_have_zero_gain_interval(self):
        self.assertEqual(paired_interval(np.zeros((4, 6)), comparisons=6, resamples=20,
                                         cluster_ids=['a', 'a', 'b', 'b', 'c', 'c']), (0., 0.))

    def test_pauli_correlations_of_bell_and_complex_product_states(self):
        bell = np.array([1., 0., 0., 1.], dtype=complex) / np.sqrt(2)
        product = np.kron(np.array([1., 1j]) / np.sqrt(2), np.array([1., 0.]))
        features = pauli_features_from_states(np.array([bell, product]), 2)
        expected = np.zeros((2, 15))
        expected[0, [6, 10, 14]] = [1., -1., 1.]  # XX, YY, ZZ
        expected[1, [1, 5, 11]] = 1.  # YI, IZ, YZ
        np.testing.assert_allclose(features, expected, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
