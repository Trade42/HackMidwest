"""Local noisy backend built on the exact Pauli-frame simulator.

This is the offline stand-in for the Sqale noisy simulator. It applies circuit-level
Pauli noise at the CZ level (every CNOT counted as ``H CZ H``) and returns physical
bitstrings, so encoded circuits go through ``iceberg.postselect`` exactly as hardware
counts would. The noise parameters are assumptions, not Sqale calibration data.

Connectivity model:

* bus-encoded circuits are already nearest-neighbour and run as built;
* bare circuits are routed onto the grid with SWAPs (best of several SABRE seeds);
* direct-encoded circuits are run with free all-to-all connectivity, which flatters
  them; they are a reference point, not a hardware proposal.
"""

from __future__ import annotations

from typing import Any, Sequence

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

from iceberg.frames import FrameSimulator, PauliNoise

_ROUTE_BASIS = ["h", "rz", "rx", "rzz", "swap"]


def route_on_grid(circuit: QuantumCircuit, grid_shape: tuple[int, int] = (4, 6), *,
                  trials: int = 16, seed: int = 0) -> QuantumCircuit:
    """Place and route a bare QAOA circuit on a grid without touching its rotations.

    Only layout and SWAP insertion run (no resynthesis), so every ``rz``/``rzz``/``rx``
    keeps its angle and the frame simulator can still attribute it to a logical term.
    """
    cmap = CouplingMap.from_grid(*grid_shape)
    best, best_swaps = None, None
    for s in range(trials):
        out = transpile(circuit, coupling_map=cmap, basis_gates=_ROUTE_BASIS, optimization_level=0,
                        layout_method="sabre", routing_method="sabre", seed_transpiler=seed + s)
        swaps = out.count_ops().get("swap", 0)
        if best_swaps is None or swaps < best_swaps:
            best, best_swaps = out, swaps
            if swaps == 0:
                break
    best.metadata = {**(circuit.metadata or {}), "routed_grid": list(grid_shape), "swaps": int(best_swaps)}
    return best


class PauliFrameBackend:
    """Sampling backend with circuit-level Pauli noise (see module docstring)."""

    def __init__(self, noise: PauliNoise | dict | None = None, *, level: str = "cz",
                 route_bare: bool = True, grid_shape: tuple[int, int] = (4, 6),
                 route_trials: int = 16, seed: int = 0, name: str | None = None):
        if isinstance(noise, dict):
            noise = PauliNoise(**noise)
        self.noise = noise or PauliNoise()
        self.level = level
        self.route_bare = route_bare
        self.grid_shape = tuple(grid_shape)
        self.route_trials = route_trials
        self.seed = seed
        self.name = name or f"local-pauli(p2={self.noise.p2:g})"
        self._runs = 0
        self.last_stats: list[dict[str, Any]] = []

    def compile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        meta = circuit.metadata or {}
        if "iceberg" in meta or not self.route_bare:
            return circuit
        return route_on_grid(circuit, self.grid_shape, trials=self.route_trials, seed=self.seed)

    def stats(self, circuit: QuantumCircuit) -> dict[str, Any]:
        """Qubit and gate counts of what would execute."""
        compiled = self.compile(circuit)
        fs = FrameSimulator(compiled, level=self.level)
        active = {compiled.find_bit(q).index for inst in compiled.data for q in inst.qubits
                  if inst.operation.name != "barrier"}
        meta = compiled.metadata or {}
        style = (meta.get("iceberg") or {}).get("style", "bare")
        return {
            "qubits": len(active),
            "two_qubit_gates": fs.two_qubit_gates,
            "one_qubit_gates": fs.one_qubit_gates,
            "swaps": int(meta.get("swaps", 0)),
            "style": style,
            "connectivity": "all-to-all (optimistic)" if style == "direct" else f"grid {self.grid_shape}",
        }

    def run(self, circuits: Sequence[QuantumCircuit] | QuantumCircuit, shots: int) -> list[dict[str, int]]:
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        out = []
        self.last_stats = []
        for circuit in circuits:
            compiled = self.compile(circuit)
            fs = FrameSimulator(compiled, level=self.level)
            out.append(fs.sample(shots, self.noise, seed=self.seed * 1_000_003 + self._runs))
            self.last_stats.append({"two_qubit_gates": fs.two_qubit_gates,
                                    "swaps": int((compiled.metadata or {}).get("swaps", 0))})
            self._runs += 1
        return out
