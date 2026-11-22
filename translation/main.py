"""Run the document translation candidate-selection demo."""

from __future__ import annotations

import argparse
import time

from .objective import build_qubo, exact_solution, local_solution
from .optimizer import qaoa_solution
from .translator import MarianTranslator, split_sentences


DEFAULT_TEXT = "The technician inspected the seal on the pump. The seal was damaged. She replaced it."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=DEFAULT_TEXT, help="English paragraph to translate")
    parser.add_argument("--candidates", type=int, default=2, help="Marian candidates per sentence (2–4)")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sentences = split_sentences(args.text)
    if not 2 <= len(sentences) <= 3:
        parser.error("start with 2–3 sentences")
    started = time.perf_counter()
    candidates = MarianTranslator().generate(sentences, args.candidates)
    generation_time = time.perf_counter() - started
    qubo = build_qubo(candidates, args.consistency_weight)
    baseline = [max(range(len(row)), key=lambda j: row[j].mean_log_prob) for row in candidates]
    baseline_objective = qubo.energy(qubo.bits_for(baseline))

    started = time.perf_counter()
    exact, exact_objective = exact_solution(qubo)
    exact_time = time.perf_counter() - started
    started = time.perf_counter()
    heuristic, heuristic_objective = local_solution(qubo, seed=args.seed)
    heuristic_time = time.perf_counter() - started
    started = time.perf_counter()
    quantum = qaoa_solution(qubo, args.layers, args.steps, args.shots, args.seed)
    quantum_time = time.perf_counter() - started

    def report(label: str, picks: list[int], objective: float) -> None:
        chosen = [candidates[i][j] for i, j in enumerate(picks)]
        mismatches = sum(
            first.terms[term] != second.terms[term]
            for i, first in enumerate(chosen)
            for second in chosen[i + 1:]
            for term in first.terms.keys() & second.terms.keys()
        )
        quality_loss = sum(max(c.mean_log_prob for c in candidates[i]) - item.mean_log_prob for i, item in enumerate(chosen))
        print(f"\n{label}: objective={objective:.4f}, quality loss={quality_loss:.4f}, term mismatches={mismatches}")
        print(" ".join(candidates[i][j].text for i, j in enumerate(picks)))
        for i, j in enumerate(picks):
            item = candidates[i][j]
            print(f"  sentence {i + 1}: candidate {j + 1}, mean log P/token={item.mean_log_prob:.3f}, terms={item.terms}")

    print(f"Input: {args.text}")
    for i, row in enumerate(candidates):
        print(f"Sentence {i + 1}: {sentences[i]}")
        for j, item in enumerate(row):
            print(f"  [{j + 1}] {item.text} | mean log P/token={item.mean_log_prob:.3f} | beam score={item.beam_score:.3f} | terms={item.terms}")
        if len(row) < args.candidates:
            print("  Candidate limitation: MarianMT returned fewer distinct translations than requested.")
    has_conflict = any(
        first.terms[term] != second.terms[term]
        for i, row in enumerate(candidates)
        for later in candidates[i + 1:]
        for first in row
        for second in later
        for term in first.terms.keys() & second.terms.keys()
    )
    if not has_conflict:
        print("Terminology limitation: candidate translations offer no cross-sentence term conflict for the optimizer to resolve.")
    report("Independent Marian choice", baseline, baseline_objective)
    report("Exact QUBO optimum", exact, exact_objective)
    report("Classical local search", heuristic, heuristic_objective)
    report("QAOA best valid measured bitstring", quantum.selection, quantum.objective)
    print(f"\nTiming: Marian={generation_time:.2f}s, exact={exact_time:.3f}s, local={heuristic_time:.3f}s, QAOA simulator={quantum_time:.2f}s")
    print(f"Quantum resources: {qubo.n_bits} qubits, {quantum.depth} circuit depth, {quantum.two_qubit_gates} two-qubit gates, {quantum.valid_shots}/{quantum.shots} valid shots")
    print("Objective improvements measure this scoring rule, not verified translation accuracy or hardware speedup.")


if __name__ == "__main__":
    main()
