"""Fixed IQP maps, classical Fourier controls, and shared measurement resources."""

from itertools import combinations
import time

import numpy as np
import pennylane as qml
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from qfp import IQPFeatureMap


def make_iqp(entangling=True, dimensions=8):
    pairs = list(combinations(range(dimensions), 2))
    return IQPFeatureMap(dimensions, repeats=2, entangling=entangling,
                         entangling_pairs=pairs, readout_pairs=pairs, batch_size=32)


def fourier_features(angles):
    """Fixed low harmonics of the single-qubit and ZZ interaction phases.

    This is a truncated classical basis, not an exact dense IQP expansion.
    It has no label dependence, random frequencies or hyperparameter search.
    """
    pairs = list(combinations(range(angles.shape[1]), 2))
    interactions = np.column_stack([angles[:, i]*angles[:, j] for i, j in pairs])
    phases = np.column_stack([angles, interactions])
    chunks = [angles, np.sin(phases), np.cos(phases), np.sin(2*phases), np.cos(2*phases)]
    for left, right in ((np.sin(angles), np.sin(angles)), (np.sin(angles), np.cos(angles)),
                        (np.cos(angles), np.sin(angles)), (np.cos(angles), np.cos(angles))):
        chunks.append(np.column_stack([left[:, i]*right[:, j] for i, j in pairs]))
    return np.column_stack(chunks)


class FourierSurrogate:
    """Approximate IQP readouts using training-input teacher features only.

    Inference is classical. Training still incurs the teacher feature cost.
    A separate Fourier-only classifier tests features requiring no teacher.
    """
    def fit(self, angles, quantum_readouts):
        self.regressor = make_pipeline(StandardScaler(), Ridge(alpha=.001))
        with threadpool_limits(limits=1):
            self.regressor.fit(fourier_features(angles), quantum_readouts)
        return self

    def readouts(self, angles):
        with threadpool_limits(limits=1):
            return self.regressor.predict(fourier_features(angles))

    def transform(self, angles):
        return np.column_stack([angles, self.readouts(angles)])


def exact_features(angles, methods, surrogate=None, train_indices=None):
    """All heads share the same eight angles; quantum heads retain these too."""
    features, timings = {}, {}
    for name in ('iqp_entangling', 'iqp_plain'):
        if name not in methods and not (name == 'iqp_entangling' and 'surrogate' in methods and surrogate is None):
            continue
        started = time.perf_counter()
        features[name] = np.column_stack([angles, make_iqp(name == 'iqp_entangling').transform(angles)])
        timings[name] = time.perf_counter()-started
    if 'fourier' in methods:
        started = time.perf_counter()
        features['fourier'] = fourier_features(angles)
        timings['fourier'] = time.perf_counter()-started
    if 'surrogate' in methods:
        started = time.perf_counter()
        if surrogate is None:
            if train_indices is None:
                raise ValueError('Surrogate fitting needs explicit training indices')
            surrogate = FourierSurrogate().fit(angles[train_indices],
                                               features['iqp_entangling'][train_indices, angles.shape[1]:])
        features['surrogate'] = surrogate.transform(angles)
        timings['surrogate'] = time.perf_counter()-started
    for name in ('rbf', 'polynomial', 'mlp'):
        if name in methods:
            features[name], timings[name] = angles, 0.
    return features, timings, surrogate


def resource_report(entangling=True):
    mapper = make_iqp(entangling)
    tape = qml.workflow.construct_tape(mapper._circuit)(np.full(8, .2))
    scripts, _ = qml.transforms.decompose(tape, gate_set={'Hadamard', 'RZ', 'CNOT'}, max_expansion=5)
    native = scripts[0]
    resources = native.specs['resources']
    return dict(qubits=8, repeats=2, logical_zz_gates=56 if entangling else 0,
                compiled_two_qubit_gates=sum(len(op.wires) == 2 for op in native.operations),
                compiled_gates=resources.num_gates, compiled_depth=resources.depth,
                readout_features=36, measurement_settings=1,
                basis='Hadamard/RZ/CNOT', connectivity='all-to-all; no routing',
                execution='classical exact statevector simulator', hardware_runtime=None)


@qml.transform
def _probability_measurement(tape):
    return [qml.tape.QuantumScript(tape.operations, [qml.probs(wires=range(8))])], lambda results: results[0]


def iqp_probabilities(angles, entangling=True):
    circuit = _probability_measurement(make_iqp(entangling)._circuit)
    return np.vstack([np.asarray(circuit(angles[i:i+32])) for i in range(0, len(angles), 32)])


def readout_eigenvalues(dimensions=8):
    bits = (np.arange(2**dimensions)[:, None] >> np.arange(dimensions-1, -1, -1)) & 1
    z = 1 - 2*bits
    return np.column_stack([z, *[z[:, i]*z[:, j] for i, j in combinations(range(dimensions), 2)]])


def sample_readouts(probabilities, shots, rng):
    """One shared Z-basis histogram per input gives all correlated Z/ZZ estimates.

    Multinomial sampling is exactly the distribution of `shots` independent
    circuit measurements. This optimizes a classical simulator, not hardware.
    """
    if shots < 1:
        raise ValueError('shots must be positive')
    probabilities = np.clip(probabilities, 0, 1)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    counts = np.array([rng.multinomial(shots, p) for p in probabilities])
    return counts @ readout_eigenvalues() / shots
