"""One-hot translation selection as an upper-triangular QUBO."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .translator import Candidate


@dataclass
class Qubo:
    sizes: list[int]
    coefficients: dict[tuple[int, int], float]
    offset: float = 0.0

    @property
    def n_bits(self) -> int:
        return sum(self.sizes)

    def energy(self, bits) -> float:
        return self.offset + sum(value * bits[i] * bits[j] for (i, j), value in self.coefficients.items())

    def selection(self, bits) -> list[int] | None:
        result, start = [], 0
        for size in self.sizes:
            chosen = [j for j, bit in enumerate(bits[start:start + size]) if bit]
            if len(chosen) != 1:
                return None
            result.append(chosen[0])
            start += size
        return result

    def bits_for(self, selection: list[int]) -> tuple[int, ...]:
        if len(selection) != len(self.sizes) or any(not 0 <= choice < size for choice, size in zip(selection, self.sizes)):
            raise ValueError("invalid candidate selection")
        return tuple(int(j == choice) for size, choice in zip(self.sizes, selection) for j in range(size))

    def hamiltonian(self):
        import pennylane as qml

        constant = self.offset
        linear = [0.0] * self.n_bits
        quadratic: dict[tuple[int, int], float] = {}
        for (i, j), value in self.coefficients.items():
            if i == j:
                constant += value / 2
                linear[i] -= value / 2
            else:
                constant += value / 4
                linear[i] -= value / 4
                linear[j] -= value / 4
                quadratic[i, j] = quadratic.get((i, j), 0.0) + value / 4
        coeffs, operators = [constant], [qml.Identity(0)]
        for i, value in enumerate(linear):
            if abs(value) > 1e-12:
                coeffs.append(value)
                operators.append(qml.Z(i))
        for (i, j), value in quadratic.items():
            if abs(value) > 1e-12:
                coeffs.append(value)
                operators.append(qml.Z(i) @ qml.Z(j))
        return qml.Hamiltonian(coeffs, operators)


def build_qubo(candidates: list[list[Candidate]], consistency_weight: float = 1.0) -> Qubo:
    if not candidates or any(not row for row in candidates):
        raise ValueError("each sentence needs at least one candidate")
    if consistency_weight < 0:
        raise ValueError("consistency_weight must be nonnegative")
    sizes = [len(row) for row in candidates]
    starts = [sum(sizes[:i]) for i in range(len(sizes))]
    quality_costs = [[max(c.mean_log_prob for c in row) - c.mean_log_prob for c in row] for row in candidates]
    pair_costs: dict[tuple[int, int], float] = {}
    for sentence, row in enumerate(candidates):
        for later in range(sentence + 1, len(candidates)):
            for j, first in enumerate(row):
                for k, second in enumerate(candidates[later]):
                    common = first.terms.keys() & second.terms.keys()
                    mismatches = sum(first.terms[t] != second.terms[t] for t in common)
                    if mismatches:
                        pair_costs[starts[sentence] + j, starts[later] + k] = consistency_weight * mismatches
    # Every valid assignment costs less than this bound; any one-hot violation costs at least penalty.
    penalty = 1.0 + sum(max(row) for row in quality_costs) + sum(pair_costs.values())
    coeffs: dict[tuple[int, int], float] = {}
    for i, row in enumerate(quality_costs):
        indices = list(range(starts[i], starts[i] + sizes[i]))
        for j, cost in zip(indices, row):
            coeffs[j, j] = cost - penalty
        for a, j in enumerate(indices):
            for k in indices[a + 1:]:
                coeffs[j, k] = 2 * penalty
    for pair, cost in pair_costs.items():
        coeffs[pair] = coeffs.get(pair, 0.0) + cost
    return Qubo(sizes, coeffs, penalty * len(sizes))


def exact_solution(qubo: Qubo) -> tuple[list[int], float]:
    best = min(product(*(range(size) for size in qubo.sizes)), key=lambda picks: qubo.energy(qubo.bits_for(list(picks))))
    return list(best), qubo.energy(qubo.bits_for(list(best)))


def local_solution(qubo: Qubo, starts: int = 8, seed: int = 0) -> tuple[list[int], float]:
    import random

    rng = random.Random(seed)
    best_selection, best_cost = None, float("inf")
    for _ in range(starts):
        current = [rng.randrange(size) for size in qubo.sizes]
        changed = True
        while changed:
            changed = False
            for i, size in enumerate(qubo.sizes):
                choice = min(range(size), key=lambda j: qubo.energy(qubo.bits_for(current[:i] + [j] + current[i + 1:])))
                if choice != current[i]:
                    current[i], changed = choice, True
        cost = qubo.energy(qubo.bits_for(current))
        if cost < best_cost:
            best_selection, best_cost = current.copy(), cost
    return best_selection, best_cost
