"""PennyLane QAOA over the same QUBO used by the classical solvers."""

from __future__ import annotations

from dataclasses import dataclass

from .objective import Qubo


@dataclass
class QaoaResult:
    selection: list[int]
    objective: float
    valid_shots: int
    shots: int
    depth: int
    two_qubit_gates: int


def qaoa_solution(qubo: Qubo, layers: int = 2, steps: int = 40, shots: int = 1000, seed: int = 0) -> QaoaResult:
    import pennylane as qml
    from pennylane import numpy as np

    n = qubo.n_bits
    if n > 12:
        raise ValueError("QAOA simulation is limited to 12 candidate variables; reduce sentences/candidates")
    if min(layers, steps, shots) < 1:
        raise ValueError("layers, steps, and shots must be positive")
    cost = qubo.hamiltonian()
    mixer = qml.Hamiltonian([1.0] * n, [qml.X(i) for i in range(n)])

    def circuit(parameters):
        for wire in range(n):
            qml.Hadamard(wires=wire)
        for gamma, beta in parameters:
            qml.qaoa.cost_layer(gamma, cost)
            qml.qaoa.mixer_layer(beta, mixer)

    analytic = qml.device("default.qubit", wires=n)

    @qml.qnode(analytic, interface="autograd")
    def expectation(parameters):
        circuit(parameters)
        return qml.expval(cost)

    rng = np.random.default_rng(seed)
    parameters = np.array(rng.uniform(0, 0.5, (layers, 2)), requires_grad=True)
    optimizer = qml.AdamOptimizer(stepsize=0.12)
    for _ in range(steps):
        parameters = optimizer.step(expectation, parameters)

    sampled = qml.device("default.qubit", wires=n, seed=seed)

    @qml.set_shots(shots=shots)
    @qml.qnode(sampled)
    def sample(parameters):
        circuit(parameters)
        return qml.sample(wires=range(n))

    bitstrings = sample(parameters)
    valid = [(qubo.energy(bits), qubo.selection(bits)) for bits in bitstrings if qubo.selection(bits) is not None]
    if not valid:
        raise RuntimeError("QAOA produced no valid one-hot bitstrings; increase shots or layers")
    objective, selection = min(valid, key=lambda item: item[0])
    decomposed = qml.transforms.decompose(sample, gate_set={"Hadamard", "RZ", "RX", "CNOT", "GlobalPhase"})
    resources = qml.specs(decomposed)(parameters)["resources"]
    two_qubit = resources.gate_types.get("CNOT", 0)
    return QaoaResult(selection, objective, len(valid), shots, resources.depth, two_qubit)
