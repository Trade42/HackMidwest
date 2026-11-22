"""Execution backends: where a sampling circuit actually runs.

* :class:`IdealBackend`: exact statevector, the ideal tier.
* ``PauliFrameBackend`` (in :mod:`qsel.pauli_backend`): the default local noisy
  backend, an exact and fast Pauli-noise sampler.
* :class:`LocalNoisyBackend`: Qiskit transpiler plus Aer with the same kind of noise.
  Slower and written independently, it is the cross-check for the one above.
* ``SuperstaqBackend`` (in :mod:`qsel.superstaq_backend`): Sqale through Superstaq.

The two local backends are offline stand-ins for the Sqale noisy simulator. Neither
is a model of Sqale.

All of them take circuits with terminal measurements and return Qiskit-style counts
(bit 0 rightmost), one dictionary per circuit.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
from qiskit import QuantumCircuit

_DIRECTIVES = ("barrier", "delay")


# ------------------------------------------------------------------ utilities
def circuit_fingerprint(circuit: QuantumCircuit) -> str:
    """Stable hash of a circuit's structure (gates, qubits, classical bits, parameters).

    It does not depend on the Qiskit version or on metadata, so it can key caches that
    are committed to the repository.
    """
    h = hashlib.sha256()
    h.update(f"{circuit.num_qubits}|{circuit.num_clbits}".encode())
    for inst in circuit.data:
        op = inst.operation
        if op.name == "barrier":
            continue
        qs = ",".join(str(circuit.find_bit(q).index) for q in inst.qubits)
        cs = ",".join(str(circuit.find_bit(c).index) for c in inst.clbits)
        ps = ",".join(f"{float(p):.12g}" for p in op.params)
        h.update(f";{op.name}:{qs}:{cs}:{ps}".encode())
    return h.hexdigest()


def circuit_stats(circuit: QuantumCircuit) -> dict[str, Any]:
    """Resource summary of a circuit: qubits in use, depth and two-qubit gates.

    Barriers are ignored and idle qubits are not counted. Any two-qubit operation counts
    as a two-qubit gate, whatever its name.
    """
    active: set[int] = set()
    two = 0
    for inst in circuit.data:
        if inst.operation.name in _DIRECTIVES:
            continue
        active.update(circuit.find_bit(q).index for q in inst.qubits)
        if len(inst.qubits) == 2:
            two += 1
    ops = {str(k): int(v) for k, v in circuit.count_ops().items() if k not in _DIRECTIVES}
    return {
        "qubits": len(active),
        "depth": int(circuit.depth(lambda i: i.operation.name not in _DIRECTIVES)),
        "two_qubit_gates": int(two),
        "two_qubit_depth": int(circuit.depth(
            lambda i: len(i.qubits) == 2 and i.operation.name not in _DIRECTIVES)),
        "count_ops": ops,
    }


def compact_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    """Copy of ``circuit`` without idle qubits (classical bits are kept as they are)."""
    active = sorted({circuit.find_bit(q).index for inst in circuit.data for q in inst.qubits
                     if inst.operation.name not in _DIRECTIVES})
    remap = {old: new for new, old in enumerate(active)}
    out = QuantumCircuit(len(active), circuit.num_clbits, name=circuit.name)
    for inst in circuit.data:
        if inst.operation.name in _DIRECTIVES:
            continue
        qs = [out.qubits[remap[circuit.find_bit(q).index]] for q in inst.qubits]
        cs = [out.clbits[circuit.find_bit(c).index] for c in inst.clbits]
        out.append(inst.operation, qs, cs)
    return out


def _clean_counts(counts: Mapping[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, value in counts.items():
        k = str(key).replace(" ", "")
        out[k] = out.get(k, 0) + int(round(float(value)))
    return out


def _child_seed(seed: int, circuit: QuantumCircuit) -> int:
    """Per-circuit seed that does not depend on the circuit's position in a batch."""
    digest = hashlib.sha256(f"{seed}|{circuit_fingerprint(circuit)}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


@runtime_checkable
class Backend(Protocol):
    """Anything that can run sampling circuits."""

    name: str

    def run(self, circuits: Sequence[QuantumCircuit], shots: int) -> list[dict[str, int]]:
        ...

    def compile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        ...


# ---------------------------------------------------------------------- ideal
class IdealBackend:
    """Exact statevector simulation followed by multinomial sampling (the ideal tier)."""

    def __init__(self, seed: int = 0):
        self.name = "ideal"
        self.seed = int(seed)

    def compile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        return circuit

    def probabilities(self, circuit: QuantumCircuit) -> np.ndarray:
        """Exact distribution over the circuit's classical bits (index = integer value)."""
        from iceberg.sim import outcome_probabilities, run, to_program

        program = to_program(circuit)
        if not program.measures:
            raise ValueError("circuit has no measurements to sample")
        probs = outcome_probabilities(program, run(program))
        return probs / probs.sum()

    def run(self, circuits: Sequence[QuantumCircuit], shots: int) -> list[dict[str, int]]:
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        out = []
        for circuit in circuits:
            probs = self.probabilities(circuit)
            rng = np.random.default_rng(_child_seed(self.seed, circuit))
            draws = rng.multinomial(int(shots), probs)
            width = circuit.num_clbits
            out.append({format(int(k), f"0{width}b"): int(draws[k]) for k in np.flatnonzero(draws)})
        return out


# ---------------------------------------------------------------- local noisy
class LocalNoisyBackend:
    """Qiskit transpiler + Aer noise simulation: the independent cross-check backend.

    The noise parameters are assumptions chosen for testing, not Sqale calibration
    data. Results from this backend must be labelled as a local stand-in.

    Compilation: ``qiskit.transpile`` to ``basis`` on a ``grid_shape`` nearest-neighbour
    grid (``coupling="grid"``) or with all-to-all connectivity (``coupling="full"``).
    ``layout_trials`` transpiler seeds are tried and the circuit with the fewest
    two-qubit gates is kept. Bus-style Iceberg circuits are already nearest-neighbour:
    they are placed on their own grid positions, so no SWAP is added, and they are
    compiled at ``bus_optimization_level`` so that the check gadgets keep their shape.

    Noise: after every two-qubit basis gate, one of the 15 two-qubit Paulis with total
    probability ``p2``; after every one-qubit basis gate except ``rz``, one of the three
    Paulis with total probability ``p1``; each measured bit flips with probability
    ``p_meas``. ``rz`` is treated as a noiseless frame change and idle qubits do not
    decohere.

    Simulation (``method``):

    * ``"automatic"`` or ``"aer"`` (default): Aer density matrix (exact) when the
      compiled circuit uses at most 12 qubits, statevector trajectories otherwise.
      Trajectories cost about a minute per 4,000 shots at 14 qubits and grow 4x per
      two extra qubits, so keep Aer runs to small problems.
    * any Aer method name (``"statevector"``, ``"density_matrix"``, ...): that method.
    * ``"frame"``: the generic exact sampler in :mod:`qsel.frame_sim`. Same
      distribution as Aer (tested against its density matrix), 10 to 30 times faster
      at 14 qubits. It extends this cross-check to sizes Aer cannot reach.
    """

    reports_compiled = True

    def __init__(
        self,
        p1: float = 0.0005,
        p2: float = 0.005,
        p_meas: float = 0.005,
        coupling: str = "grid",
        grid_shape: tuple[int, int] = (4, 6),
        basis: Sequence[str] = ("cz", "rz", "sx", "x"),
        optimization_level: int = 3,
        layout_trials: int = 8,
        seed: int = 0,
        method: str = "automatic",
        bus_optimization_level: int = 1,
    ):
        if coupling not in ("grid", "full"):
            raise ValueError("coupling must be 'grid' or 'full'")
        self.p1, self.p2, self.p_meas = float(p1), float(p2), float(p_meas)
        self.coupling = coupling
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self.basis = tuple(basis)
        self.optimization_level = int(optimization_level)
        self.layout_trials = max(1, int(layout_trials))
        self.seed = int(seed)
        self.method = method
        self.bus_optimization_level = int(bus_optimization_level)
        self.name = f"local-aer[{coupling},p2={self.p2:g}]"
        self._compiled: dict[str, QuantumCircuit] = {}
        self._frames: dict[str, Any] = {}
        self._noise_model = None
        #: simulation method used for each circuit of the most recent run
        self.last_methods: list[str] = []
        #: stats of the circuits compiled by the most recent :meth:`run` call
        self.last_compiled: list[dict[str, Any]] = []

    # ------------------------------------------------------------ compilation
    def _coupling_map(self):
        if self.coupling == "full":
            return None
        from qiskit.transpiler import CouplingMap

        return CouplingMap.from_grid(*self.grid_shape)

    def _bus_positions(self, circuit: QuantumCircuit) -> list[tuple[int, int]] | None:
        """Grid positions of a bus-style Iceberg circuit, or ``None`` for anything else."""
        meta = (circuit.metadata or {}).get("iceberg")
        if not isinstance(meta, dict) or meta.get("style") != "bus" or not meta.get("positions"):
            return None
        return [tuple(int(v) for v in p) for p in meta["positions"]]

    def _compile_bus(self, circuit: QuantumCircuit, positions: list[tuple[int, int]]) -> QuantumCircuit:
        """Compile a nearest-neighbour bus circuit in place: its own layout, no routing."""
        from qiskit import transpile
        from qiskit.transpiler import CouplingMap

        rows = 1 + max(r for r, _ in positions) - min(r for r, _ in positions)
        cols = 1 + max(c for _, c in positions) - min(c for _, c in positions)
        if self.coupling == "grid":
            fits = (rows <= self.grid_shape[0] and cols <= self.grid_shape[1]) or \
                   (rows <= self.grid_shape[1] and cols <= self.grid_shape[0])
            if not fits:
                raise ValueError(f"bus layout needs a {rows}x{cols} patch, grid is {self.grid_shape}")
        pairs = [(a, b) for a in range(len(positions)) for b in range(len(positions)) if a != b and
                 abs(positions[a][0] - positions[b][0]) + abs(positions[a][1] - positions[b][1]) == 1]
        return transpile(
            circuit,
            basis_gates=list(self.basis),
            coupling_map=CouplingMap(pairs),
            initial_layout=list(range(circuit.num_qubits)),
            routing_method="none",
            optimization_level=min(self.optimization_level, self.bus_optimization_level),
            seed_transpiler=self.seed,
        )

    def compile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Device-level circuit: basis gates, routed onto the grid. Cached per circuit."""
        key = circuit_fingerprint(circuit)
        if key in self._compiled:
            return self._compiled[key]
        from qiskit import transpile

        positions = self._bus_positions(circuit)
        if positions is not None:
            self._compiled[key] = self._compile_bus(circuit, positions)
            return self._compiled[key]

        cmap = self._coupling_map()
        if cmap is not None and circuit.num_qubits > cmap.size():
            raise ValueError(
                f"circuit needs {circuit.num_qubits} qubits, grid {self.grid_shape} has {cmap.size()}"
            )
        trials = self.layout_trials if cmap is not None else 1
        best, best_key = None, None
        for trial in range(trials):
            cand = transpile(
                circuit,
                basis_gates=list(self.basis),
                coupling_map=cmap,
                optimization_level=self.optimization_level,
                seed_transpiler=self.seed + trial,
            )
            stats = circuit_stats(cand)
            rank = (stats["two_qubit_gates"], stats["depth"], trial)
            if best_key is None or rank < best_key:
                best, best_key = cand, rank
        self._compiled[key] = best
        return best

    def compiled_stats(self, circuit: QuantumCircuit) -> dict[str, Any]:
        return circuit_stats(self.compile(circuit))

    # ------------------------------------------------------------------ noise
    def noise_model(self):
        """The Aer noise model described in the class docstring (``None`` if noiseless)."""
        if self._noise_model is not None:
            return self._noise_model
        if self.p1 <= 0 and self.p2 <= 0 and self.p_meas <= 0:
            return None
        from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

        model = NoiseModel(basis_gates=list(self.basis))
        one_qubit = [g for g in self.basis if g not in ("rz", "cz", "cx", "ecr", "rzz", "rxx", "swap")]
        two_qubit = [g for g in self.basis if g in ("cz", "cx", "ecr", "rzz", "rxx", "swap")]
        # Aer's depolarizing parameter includes the identity; rescale so that p1 and p2 are
        # the probabilities of a non-identity Pauli.
        if self.p1 > 0 and one_qubit:
            model.add_all_qubit_quantum_error(depolarizing_error(self.p1 * 4.0 / 3.0, 1), one_qubit)
        if self.p2 > 0 and two_qubit:
            model.add_all_qubit_quantum_error(depolarizing_error(self.p2 * 16.0 / 15.0, 2), two_qubit)
        if self.p_meas > 0:
            pm = self.p_meas
            model.add_all_qubit_readout_error(ReadoutError([[1 - pm, pm], [pm, 1 - pm]]))
        self._noise_model = model
        return model

    # -------------------------------------------------------------------- run
    def _frame_simulator(self, key: str, small: QuantumCircuit):
        """Generic frame sampler for a compiled circuit (raises if the circuit is unsupported)."""
        if key not in self._frames:
            from .frame_sim import GenericFrameSampler, UnsupportedCircuit

            try:
                self._frames[key] = GenericFrameSampler(small, noiseless=("rz", "p", "id"))
            except UnsupportedCircuit:
                self._frames[key] = None
                raise
        if self._frames[key] is None:
            from .frame_sim import UnsupportedCircuit

            raise UnsupportedCircuit("circuit is outside the Clifford + Pauli-rotation set")
        return self._frames[key]

    def _run_aer(self, small: QuantumCircuit, shots: int, seed: int) -> dict[str, int]:
        from qiskit_aer import AerSimulator

        noise = self.noise_model()
        if self.method in ("automatic", "aer"):
            method = "density_matrix" if (noise is not None and small.num_qubits <= 12) else "statevector"
        else:
            method = self.method
        sim = AerSimulator(method=method, noise_model=noise)
        result = sim.run(small, shots=int(shots), seed_simulator=seed).result()
        return _clean_counts(result.get_counts(0))

    def run(self, circuits: Sequence[QuantumCircuit], shots: int) -> list[dict[str, int]]:
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        out: list[dict[str, int]] = []
        self.last_compiled = []
        self.last_methods = []
        for circuit in circuits:
            compiled = self.compile(circuit)
            self.last_compiled.append(circuit_stats(compiled))
            small = compact_circuit(compiled)
            seed = _child_seed(self.seed, circuit)
            if self.method == "frame":
                frame = self._frame_simulator(circuit_fingerprint(circuit), small)
                self.last_methods.append("frame")
                out.append(frame.sample(int(shots), p1=self.p1, p2=self.p2, p_meas=self.p_meas, seed=seed))
            else:
                self.last_methods.append("aer")
                out.append(self._run_aer(small, shots, seed))
        return out


# -------------------------------------------------------------------- factory
def make_backend(cfg: Mapping[str, Any]) -> Backend:
    """Build a backend from a config mapping with a ``kind`` key.

    * ``"ideal"``: :class:`IdealBackend`;
    * ``"local-pauli"``: ``qsel.pauli_backend.PauliFrameBackend``, the default local
      noisy backend (``noise`` holds the Pauli-noise parameters);
    * ``"local-aer"`` (alias ``"local-noisy"``): :class:`LocalNoisyBackend`, the
      independent Aer cross-check;
    * ``"superstaq"``: ``qsel.superstaq_backend.SuperstaqBackend``.

    The remaining keys are passed to the backend's constructor.
    """
    params = dict(cfg)
    kind = str(params.pop("kind", "ideal")).replace("_", "-")
    if kind == "ideal":
        return IdealBackend(**params)
    if kind == "local-pauli":
        from .pauli_backend import PauliFrameBackend

        noise = params.pop("noise", None)
        if params.get("grid_shape") is not None:
            params["grid_shape"] = tuple(params["grid_shape"])
        return PauliFrameBackend(noise=noise, **params)
    if kind in ("local-aer", "local-noisy"):
        if "grid_shape" in params:
            params["grid_shape"] = tuple(params["grid_shape"])
        if "basis" in params:
            params["basis"] = tuple(params["basis"])
        return LocalNoisyBackend(**params)
    if kind == "superstaq":
        from .superstaq_backend import SuperstaqBackend

        if params.get("grid_shape") is not None:
            params["grid_shape"] = tuple(params["grid_shape"])
        return SuperstaqBackend(**params)
    raise ValueError(f"unknown backend kind {kind!r}")
