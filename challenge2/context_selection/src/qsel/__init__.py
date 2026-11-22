"""Quantum context selection: QUBO, classical solvers, QAOA."""

from .angles import TunedAngles, tune_angles
from .classical import brute_force, greedy, greedy_candidates, optimum, repair, simulated_annealing, top_k
from .qaoa import Angles, QaoaSimulator, qaoa_circuit
from .qubo import Qubo, build_qubo, random_qubo

__all__ = [
    "Angles", "QaoaSimulator", "Qubo", "TunedAngles", "brute_force", "build_qubo", "greedy",
    "greedy_candidates", "optimum", "qaoa_circuit", "random_qubo", "repair",
    "simulated_annealing", "top_k", "tune_angles",
]

# ---- sampling, backends, select, router (appended)
from .backends import (Backend, IdealBackend, LocalNoisyBackend, circuit_fingerprint, circuit_stats,
                       make_backend)
from .compile_stats import compile_table, roadmap_point
from .router import RouteDecision, estimate_two_qubit_gates, predicted_kept_share, route
from .sampling import (RankedSelection, SampleStats, expected_hit_curve, rank_selections,
                       sample_stats, subsample_counts)
from .select import (CLASSICAL_SOLVERS, QUANTUM_TIERS, SelectionResult, finish, prepare_circuit,
                     select, select_qubo)

__all__ += [
    "Backend", "CLASSICAL_SOLVERS", "IdealBackend", "LocalNoisyBackend", "QUANTUM_TIERS",
    "RankedSelection", "RouteDecision", "SampleStats", "SelectionResult", "circuit_fingerprint",
    "circuit_stats", "compile_table", "estimate_two_qubit_gates", "expected_hit_curve", "finish",
    "make_backend", "predicted_kept_share", "prepare_circuit", "rank_selections", "roadmap_point",
    "route", "sample_stats", "select", "select_qubo", "subsample_counts",
]
