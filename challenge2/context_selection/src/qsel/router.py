"""Route a problem to a classical solver, bare QAOA or error-detected QAOA.

Rule, in order:

1. if the bare circuit needs more two-qubit gates than the budget, solve classically
   (exact enumeration when the problem is small, greedy otherwise);
2. otherwise use the error-detected tier when the predicted share of shots kept stays
   above a floor;
3. otherwise run bare QAOA.

The kept-share prediction is a deliberately simple model (independent faults, a fixed
fraction of them detected). It is good enough to decide a route; measured kept shares
are what the evaluation reports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from qiskit import QuantumCircuit

from .qaoa import Angles, qaoa_circuit
from .qubo import Qubo


@dataclass(frozen=True)
class RouteDecision:
    tier: str        #: "brute_force", "greedy", "physical" or "iceberg"
    reason: str
    estimates: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_two_qubit_gates(qubo: Qubo, p: int, *, encoded: bool, style: str = "bus") -> int:
    """Two-qubit (CZ) gate count, analytic.

    Bare: ``2 * edges * p``, before routing.

    Iceberg, ``style="direct"`` (textbook mapping, before routing): every edge, field
    and mixer term is a two-qubit rotation (two CZ each), plus the chain that prepares
    the start state and its flag check, plus the ``S_X`` measurement and its flag.

    Iceberg, ``style="bus"`` (grid-native, no routing needed): with ``K`` code-logical
    qubits and ``C = K/2 + 1`` bus qubits, each layer costs about ``4`` CZ per edge (a
    conservative figure; neighbouring pairs need fewer), ``2C + 2K`` for the fields and
    ``2C + 2K`` for the mixer; the start state costs ``2C + 3`` and the ``S_X`` check
    ``4C - 1``.
    """
    edges = qubo.num_edges
    if not encoded:
        return 2 * edges * p
    n_code = qubo.n + (qubo.n % 2)
    if style == "direct":
        return 2 * (edges + 2 * n_code) * p + (n_code + 1) + 2 + (n_code + 2) + 2
    if style == "bus":
        bus = n_code // 2 + 1
        per_layer = 4 * edges + 2 * (2 * bus + 2 * n_code)
        return per_layer * p + (2 * bus + 3) + (4 * bus - 1)
    raise ValueError("style must be 'bus' or 'direct'")


def measured_qubits(n: int, style: str = "bus") -> int:
    """Qubits read out by the error-detected circuit (every one is a chance to discard)."""
    n_code = n + (n % 2)
    return n_code + 2 + (n_code // 2 + 1 if style == "bus" else 2)


def predicted_kept_share(
    two_qubit_gates: int,
    p2: float,
    *,
    detect_fraction: float = 0.8,
    measured_qubits: int = 14,
    p_meas: float = 0.0,
) -> float:
    """Expected share of shots that pass the checks.

    ``(1 - detect_fraction * p2) ** gates * (1 - p_meas) ** measured_qubits``: each
    two-qubit gate fails with probability ``p2`` and a fixed fraction of those faults
    is detected; any readout flip is detected. One-qubit errors and idling are ignored.
    """
    gate_term = (1.0 - detect_fraction * p2) ** int(two_qubit_gates)
    return float(gate_term * (1.0 - p_meas) ** int(measured_qubits))


def route(
    qubo: Qubo,
    *,
    p: int = 1,
    p2: float = 0.005,
    p_meas: float = 0.005,
    max_two_qubit_gates: int = 120,
    keep_floor: float = 0.05,
    max_brute_force_n: int = 20,
    gate_counter: Callable[[QuantumCircuit], int] | None = None,
    encoder_opts: dict[str, Any] | None = None,
    style: str = "bus",
) -> RouteDecision:
    """Pick a solver tier for ``qubo``.

    Args:
        p: QAOA depth the quantum tiers would use.
        p2, p_meas: assumed two-qubit and readout error rates of the target.
        max_two_qubit_gates: budget for the bare circuit; above it the problem is
            solved classically.
        keep_floor: minimum predicted share of shots kept for the error-detected tier.
        max_brute_force_n: largest problem solved by exact enumeration.
        gate_counter: optional ``circuit -> two-qubit gate count`` using a real compiler
            (local transpile or ``cq_compile``). Without it the analytic pre-routing
            estimate is used.
        encoder_opts: keyword arguments for ``iceberg.encode`` when counting real gates.
        style: Iceberg encoding style, ``"bus"`` (default) or ``"direct"``. An explicit
            ``style`` in ``encoder_opts`` wins.
    """
    opts = dict(encoder_opts or {})
    style = opts.setdefault("style", style)
    source = "analytic"
    bare_gates = estimate_two_qubit_gates(qubo, p, encoded=False)
    ice_gates = estimate_two_qubit_gates(qubo, p, encoded=True, style=style)
    measured = measured_qubits(qubo.n, style)
    if gate_counter is not None:
        import iceberg

        probe = Angles((0.4,) * p, (0.3,) * p)        # gate counts do not depend on the angles
        bare = qaoa_circuit(qubo, probe)
        encoded = iceberg.encode(bare, **opts)
        bare_gates = int(gate_counter(bare))
        ice_gates = int(gate_counter(encoded))
        measured = encoded.num_clbits
        source = "compiled"
    kept = predicted_kept_share(ice_gates, p2, measured_qubits=measured, p_meas=p_meas)
    estimates = {
        "n": qubo.n,
        "edges": qubo.num_edges,
        "p": p,
        "style": style,
        "bare_two_qubit_gates": bare_gates,
        "iceberg_two_qubit_gates": ice_gates,
        "predicted_kept_share": kept,
        "gate_count_source": source,
        "max_two_qubit_gates": max_two_qubit_gates,
        "keep_floor": keep_floor,
    }
    if bare_gates > max_two_qubit_gates:
        tier = "brute_force" if qubo.n <= max_brute_force_n else "greedy"
        reason = (f"bare circuit needs {bare_gates} two-qubit gates, over the budget of "
                  f"{max_two_qubit_gates}: solve classically")
    elif kept >= keep_floor:
        tier = "iceberg"
        reason = (f"{bare_gates} two-qubit gates is within budget and the error-detected circuit "
                  f"({ice_gates} gates) is predicted to keep {kept:.0%} of shots, above the "
                  f"{keep_floor:.0%} floor")
    else:
        tier = "physical"
        reason = (f"{bare_gates} two-qubit gates is within budget but the error-detected circuit "
                  f"({ice_gates} gates) is predicted to keep only {kept:.1%} of shots, below the "
                  f"{keep_floor:.0%} floor: run bare QAOA")
    return RouteDecision(tier, reason, estimates)
