"""Compile statistics: what each tier costs on each target.

``compile_table`` feeds ``circuits/stats.json`` and the gate-count figure (bare vs
error-detected, per backend). ``roadmap_point`` is one point of the compile-only
projection: how the same problem grows from 10 to 30 paragraphs.
"""

from __future__ import annotations

import math
import time
from typing import Any, Mapping

from qiskit import QuantumCircuit

from .backends import Backend, LocalNoisyBackend, circuit_stats
from .qaoa import Angles, qaoa_circuit
from .qubo import Qubo, random_qubo


def _encode(bare: QuantumCircuit, encoder_opts: Mapping[str, Any] | None) -> QuantumCircuit:
    import iceberg

    return iceberg.encode(bare, **dict(encoder_opts or {}))


def compile_table(
    qubo: Qubo,
    angles: Angles,
    backends: Mapping[str, Backend],
    *,
    encoder_opts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resources of the bare and error-detected circuits, before and after compilation.

    Returns ``{"n", "edges", "p", "bare": {...}, "iceberg": {...}}`` where each tier
    maps ``"logical"`` (the circuit as built) and every backend name to
    :func:`qsel.backends.circuit_stats`. A backend whose compile fails contributes
    ``{"error": message}`` instead of aborting the table.
    """
    bare = qaoa_circuit(qubo, angles)
    circuits = {"bare": bare, "iceberg": _encode(bare, encoder_opts)}
    table: dict[str, Any] = {"n": qubo.n, "edges": qubo.num_edges, "p": angles.p}
    for tier, circuit in circuits.items():
        row: dict[str, Any] = {"logical": circuit_stats(circuit)}
        for name, backend in backends.items():
            try:
                row[name] = circuit_stats(backend.compile(circuit))
            except Exception as exc:
                row[name] = {"error": f"{type(exc).__name__}: {exc}"}
        table[tier] = row
    return table


def near_square_grid(qubits: int) -> tuple[int, int]:
    """Smallest near-square ``(rows, cols)`` grid with at least ``qubits`` sites."""
    rows = max(1, int(math.floor(math.sqrt(qubits))))
    cols = int(math.ceil(qubits / rows))
    return rows, cols


def roadmap_point(
    n: int,
    *,
    edge_fraction: float = 0.2,
    p: int = 1,
    seed: int = 0,
    grid: bool = True,
    layout_trials: int = 4,
    max_brute_force_n: int = 22,
    encoder_opts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One point of the compile-only roadmap projection for ``n`` paragraphs.

    A synthetic problem with the same structure as the real ones is built, then:

    * brute force is timed (exhaustive enumeration) up to ``max_brute_force_n``
      paragraphs; above that ``brute_force_seconds`` is ``None`` and the caller
      extrapolates (the cost doubles per paragraph);
    * the bare and error-detected circuits are compiled to CZ gates with all-to-all
      connectivity and, when ``grid`` is set, on a nearest-neighbour grid: the patch
      the bus-style encoding occupies (3 rows, ``K/2 + 1`` columns), or the smallest
      near-square grid that fits for the direct style. The bare circuit is routed on
      the same grid.

    Nothing is simulated, so this works well past the sizes we can run.
    """
    qubo = random_qubo(n, edge_fraction=edge_fraction, seed=seed)
    brute: float | None = None
    if n <= max_brute_force_n:
        t0 = time.perf_counter()
        energies = qubo.energies_all()
        int(energies.argmin())
        brute = time.perf_counter() - t0
        del energies

    angles = Angles((0.4,) * p, (0.3,) * p)     # gate counts do not depend on the angles
    bare = qaoa_circuit(qubo, angles)
    encoded = _encode(bare, encoder_opts)
    layout = (encoded.metadata or {}).get("iceberg", {})
    if layout.get("style") == "bus" and layout.get("positions"):
        rows = 1 + max(r for r, _ in layout["positions"]) - min(r for r, _ in layout["positions"])
        cols = 1 + max(c for _, c in layout["positions"]) - min(c for _, c in layout["positions"])
        shape = (rows, cols)
    else:
        shape = near_square_grid(encoded.num_qubits)
    full = LocalNoisyBackend(coupling="full", seed=seed)
    on_grid = LocalNoisyBackend(coupling="grid", grid_shape=shape, layout_trials=layout_trials,
                                seed=seed) if grid else None

    out: dict[str, Any] = {
        "n": n,
        "edges": qubo.num_edges,
        "p": p,
        "seed": seed,
        "edge_fraction": edge_fraction,
        "grid_shape": list(shape) if grid else None,
        "style": layout.get("style", "direct"),
        "brute_force_seconds": brute,
    }
    for tier, circuit in (("bare", bare), ("iceberg", encoded)):
        row = {
            "logical_qubits": n,
            "physical_qubits": circuit.num_qubits,
            "all_to_all": circuit_stats(full.compile(circuit)),
        }
        if on_grid is not None:
            row["grid"] = circuit_stats(on_grid.compile(circuit))
        row["two_qubit_gates"] = row["grid" if on_grid is not None else "all_to_all"]["two_qubit_gates"]
        out[tier] = row
    return out
