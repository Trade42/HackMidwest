"""Majority vote over the answers from the best distinct selections.

The model answers once per selection. Answers that normalise to the same string
form one group, the largest group wins, and ties go to the group that contains
the lowest-energy selection.
"""

from __future__ import annotations

from typing import Sequence

from .metrics import normalize_answer


def majority_vote(answers: Sequence[str], energies: Sequence[float]) -> tuple[str, dict]:
    """Combine answers by majority.

    Args:
        answers: one answer per selection, in any order.
        energies: QUBO energy of the selection behind each answer (lower is better).

    Returns:
        ``(answer, info)``. ``answer`` is the raw answer of the lowest-energy member
        of the winning group. ``info`` holds ``votes`` (normalised answer -> count),
        ``winner_index`` (position of the returned answer in ``answers``), ``tie``
        (the top vote count was shared), ``winner_votes`` and ``n``.
    """
    answers = [str(a) for a in answers]
    energies = [float(e) for e in energies]
    if len(answers) != len(energies):
        raise ValueError("answers and energies must have the same length")
    if not answers:
        return "", {"votes": {}, "winner_index": -1, "tie": False, "winner_votes": 0, "n": 0}

    groups: dict[str, list[int]] = {}
    for i, a in enumerate(answers):
        groups.setdefault(normalize_answer(a), []).append(i)

    def best_member(members: list[int]) -> int:
        """Lowest energy first, earliest position second."""
        return min(members, key=lambda i: (energies[i], i))

    top = max(len(m) for m in groups.values())
    leaders = [key for key, members in groups.items() if len(members) == top]
    winner_key = min(leaders, key=lambda key: (energies[best_member(groups[key])], best_member(groups[key])))
    winner_index = best_member(groups[winner_key])
    info = {
        "votes": {key: len(members) for key, members in groups.items()},
        "winner_index": winner_index,
        "tie": len(leaders) > 1,
        "winner_votes": top,
        "n": len(answers),
    }
    return answers[winner_index], info
