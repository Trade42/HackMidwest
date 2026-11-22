"""Majority vote over the answers from the best selections."""

from __future__ import annotations

import pytest

from qctx.vote import majority_vote


def test_majority_wins_even_when_the_best_selection_disagrees():
    answer, info = majority_vote(["Cruz Azul", "Liga MX", "liga mx."], [-3.0, -2.0, -1.0])
    assert answer == "Liga MX"                 # raw answer of the lowest-energy member of the winning group
    assert info["winner_index"] == 1
    assert info["votes"] == {"cruz azul": 1, "liga mx": 2}
    assert info["tie"] is False and info["winner_votes"] == 2 and info["n"] == 3


def test_normalisation_groups_equivalent_answers():
    answer, info = majority_vote(["The Beatles", "beatles!", "Rolling Stones"], [0.5, -1.0, -2.0])
    assert answer == "beatles!"                # lower energy than "The Beatles"
    assert info["winner_index"] == 1


def test_three_way_tie_goes_to_lowest_energy():
    answer, info = majority_vote(["A1", "B2", "C3"], [-1.0, -5.0, -2.0])
    assert answer == "B2" and info["tie"] is True and info["winner_index"] == 1


def test_two_group_tie_goes_to_group_with_lowest_energy_selection():
    answer, info = majority_vote(["x", "y", "x", "y"], [-1.0, -4.0, -2.0, 0.0])
    assert answer == "y" and info["tie"] is True and info["winner_index"] == 1


def test_equal_energies_fall_back_to_position():
    answer, info = majority_vote(["first", "second"], [1.0, 1.0])
    assert answer == "first" and info["winner_index"] == 0


def test_empty_answers_are_a_normal_group():
    answer, info = majority_vote(["", "", "Paris"], [-1.0, -2.0, -9.0])
    assert answer == "" and info["votes"] == {"": 2, "paris": 1} and info["winner_index"] == 1


def test_single_answer_and_no_answers():
    assert majority_vote(["only"], [0.0])[0] == "only"
    answer, info = majority_vote([], [])
    assert answer == "" and info["winner_index"] == -1 and info["n"] == 0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        majority_vote(["a", "b"], [0.0])
