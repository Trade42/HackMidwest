"""Checks for the operational constraints that make the contextuality claim valid."""

import unittest

import numpy as np

from contextual_memory import (
    BITS, circuit_crosscheck, classical_noncontextual_optimum, expected_success,
    parity_guess_probability, parse_sentence, prepare, probability_table, sentence_for,
)


class ContextualMemoryTests(unittest.TestCase):
    def test_sentence_encoding_preserves_both_independent_cues(self):
        for bits in BITS:
            sentence = sentence_for(bits)
            self.assertLessEqual(len(sentence.split()), 15)
            np.testing.assert_array_equal(parse_sentence(sentence), bits)

    def test_states_are_physical_and_every_parity_mixture_is_identical(self):
        for visibility in (0., .4, 1.):
            states = np.array([prepare(bits, visibility) for bits in BITS])
            for state in states:
                np.testing.assert_allclose(state, state.conj().T)
                self.assertAlmostEqual(np.trace(state).real, 1.)
                self.assertGreaterEqual(np.linalg.eigvalsh(state).min(), -1e-12)
            self.assertAlmostEqual(parity_guess_probability(states), .5)

    def test_exact_advantage_and_noise_threshold(self):
        for visibility in (0., 1 / np.sqrt(2), 1.):
            table = probability_table(np.array([prepare(bits, visibility) for bits in BITS]))
            expected = .5 + visibility / (2 * np.sqrt(2))
            self.assertAlmostEqual(expected_success(table), expected)
            for i, bits in enumerate(BITS):
                for query in range(2):
                    self.assertAlmostEqual(table[i, query, bits[query]], expected)

    def test_classical_bound_optimizes_all_response_assignments(self):
        optimum, channel = classical_noncontextual_optimum()
        self.assertAlmostEqual(optimum, .75)
        np.testing.assert_allclose(channel.sum(axis=1), 1.)
        np.testing.assert_allclose(channel[[0, 3]].mean(axis=0), channel[[1, 2]].mean(axis=0))
        self.assertGreaterEqual(channel.min(), -1e-12)

    def test_dephasing_destroys_the_violation(self):
        states = np.array([prepare(bits, dephase=True) for bits in BITS])
        for a in states:
            for b in states:
                np.testing.assert_allclose(a @ b - b @ a, 0.)
        self.assertLessEqual(expected_success(probability_table(states)), .75)
        self.assertAlmostEqual(parity_guess_probability(states), .5)

    def test_independent_circuit_implementation_matches_bloch_calculation(self):
        for visibility in (0., .4, 1.):
            self.assertLess(circuit_crosscheck(visibility), 1e-10)


if __name__ == '__main__':
    unittest.main()
