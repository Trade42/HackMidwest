"""Small validation-only search for useful quantum feature augmentation.

Screen eight-qubit maps while retaining the original inputs. The previous test
split is excluded. Quantum and classical maps use the same subsets and eight
classifier hyperparameter trials; map selection uses validation scores only.
"""

import argparse
import csv
from dataclasses import dataclass, asdict, replace
import hashlib
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from learning_curves import ROOT, Row, balanced_order, encode, save_csv, select_model
from qfp import IQPFeatureMap


@dataclass(frozen=True)
class Configuration:
    name: str
    family: str
    scale: float = 1.0
    repeats: int = 2
    classifier: str = 'rbf'
    centered: bool = False


CONFIGURATIONS = (
    Configuration('raw_rbf', 'raw'),
    Configuration('raw_polynomial', 'raw', classifier='polynomial'),
    Configuration('raw_mlp', 'raw', classifier='mlp'),
    Configuration('classical_cubic', 'cubic'),
    *(Configuration(f'classical_trig_s{s:g}', 'trig', scale=s) for s in (0.7, 1.0, 1.4)),
    *(Configuration(f'classical_product_s{s:g}', 'product', scale=s) for s in (0.7, 1.0, 1.4)),
    *(Configuration(f'quantum_r{r}_s{s:g}', 'quantum', scale=s, repeats=r)
      for r, s in ((1, 0.7), (1, 1.4), (2, 0.7), (2, 1.0), (2, 1.4), (3, 1.0))),
    Configuration('classical_cubic_centered', 'cubic', centered=True, scale=0.7),
    Configuration('classical_trig_centered', 'trig', centered=True, scale=0.7),
    Configuration('classical_product_centered', 'product', centered=True, scale=0.7),
    *(Configuration(f'quantum_centered_r{r}_s{s:g}', 'quantum', scale=s, repeats=r, centered=True)
      for r, s in ((1, 0.7), (2, 0.4), (2, 0.7), (3, 0.7))),
    *(Configuration(f'classical_probs_s{s:g}', 'product_probs', scale=s) for s in (0.7, 1.4)),
    *(Configuration(f'quantum_probs_r{r}_s{s:g}', 'quantum_probs', scale=s, repeats=r)
      for r, s in ((1, 0.7), (2, 0.4), (2, 0.7))),
    Configuration('classical_blend_trig', 'blend_trig', scale=1.4),
    Configuration('classical_blend_cubic', 'blend_cubic'),
    *(Configuration(f'quantum_blend_r2_s{s:g}', 'blend_quantum', scale=s, repeats=2)
      for s in (0.4, 0.7)),
    Configuration('raw_mlp_lbfgs', 'raw', classifier='mlp_lbfgs'),
    Configuration('classical_pauli_s0.7', 'product_pauli', scale=0.7),
    Configuration('quantum_pauli_r2_s0.7', 'quantum_pauli', scale=0.7),
    *(Configuration(f'quantum_r2_s{s:g}', 'quantum', scale=s) for s in (0.4, 0.55, 0.85)),
    *(Configuration(f'classical_product_s{s:g}', 'product', scale=s) for s in (0.4, 0.55, 0.85)),
)


def pauli_features_from_states(states, n_qubits):
    """All one- and two-qubit Pauli expectations from exact state vectors."""
    paulis = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    pairs = list(combinations(range(n_qubits), 2))
    tensors = states.reshape((len(states),) + (2,) * n_qubits)
    chunks = []
    pair_observables = np.array([np.kron(a, b) for a in paulis for b in paulis])
    for subset in [(i,) for i in range(n_qubits)] + pairs:
        order = (0,) + tuple(i + 1 for i in subset) + tuple(i + 1 for i in range(n_qubits) if i not in subset)
        amplitudes = tensors.transpose(order).reshape(len(states), 2 ** len(subset), -1)
        density = amplitudes @ amplitudes.conj().swapaxes(1, 2)
        operators = paulis if len(subset) == 1 else pair_observables
        chunks.append(np.einsum('bij,kji->bk', density, operators).real)
    return np.column_stack(chunks)


def features_for(x, config, cache_dir):
    """Preserve raw cues alongside each nonlinear map; cache unsupervised maps."""
    spec = asdict(config)
    if spec['classifier'] in ('linear', 'mlp', 'mlp_lbfgs') and config.family != 'raw':
        spec['classifier'] = 'rbf'  # Readout choice does not change the raw map.
    if not spec['centered']:
        del spec['centered']  # Preserve cache keys from the first screen.
    fingerprint = hashlib.sha256(x.tobytes() + json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    path = cache_dir / f'{config.name}_{fingerprint}.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            return saved['features'], {**json.loads(str(saved['info'])), 'cache_hit': True}
    start = time.perf_counter()
    # Invert the documented affine cue encoding, then center each cue around
    # zero. This reads input values only; no target or final comparison enters.
    values = (x - 0.2 - 0.01 * np.arange(x.shape[1])) / 0.85
    angles = (2 * values - 1) * config.scale if config.centered else x * config.scale
    pairs = list(combinations(range(x.shape[1]), 2))
    if config.family == 'raw':
        features = x
    elif config.family == 'cubic':
        features = PolynomialFeatures(degree=3, include_bias=False).fit_transform(angles)
    elif config.family == 'trig':
        nonlinear = np.column_stack([np.cos(angles[:, i] * angles[:, j]) for i, j in pairs])
        features = np.column_stack((x, np.cos(angles), nonlinear))
    elif config.family == 'product':
        # Exact classical evaluation of the two-repeat NON-entangling circuit.
        # Include this control so its easily computed features cannot be
        # mistaken for an advantage requiring entanglement.
        singles = np.sin(angles) ** 2
        nonlinear = np.column_stack([singles[:, i] * singles[:, j] for i, j in pairs])
        features = np.column_stack((x, singles, nonlinear))
    elif config.family == 'quantum':
        mapper = IQPFeatureMap(x.shape[1], repeats=config.repeats, entangling=True,
                              batch_size=16, entangling_pairs=pairs, readout_pairs=pairs)
        features = np.column_stack((x, mapper.transform(angles)))
    elif config.family == 'product_probs':
        singles = np.sin(angles) ** 2
        probabilities = np.ones((len(x), 1))
        for wire in range(x.shape[1]):
            p = np.column_stack(((1 + singles[:, wire]) / 2, (1 - singles[:, wire]) / 2))
            probabilities = (probabilities[:, :, None] * p[:, None, :]).reshape(len(x), -1)
        features = np.column_stack((x, probabilities))
    elif config.family == 'product_pauli':
        # Bloch vectors after H-RZ-H-RZ-H (the two-repeat plain map).
        bloch = np.stack((np.cos(angles), np.sin(angles) * np.cos(angles), np.sin(angles) ** 2), axis=-1)
        products = [np.einsum('bi,bj->bij', bloch[:, i], bloch[:, j]).reshape(len(x), 9) for i, j in pairs]
        features = np.column_stack((x, bloch.reshape(len(x), -1), *products))
    elif config.family in ('quantum_probs', 'quantum_pauli'):
        import pennylane as qml
        wires = list(range(x.shape[1]))
        device = qml.device('default.qubit', wires=wires, shots=None)

        @qml.qnode(device)
        def circuit(batch):
            qml.IQPEmbedding(batch, wires=wires, n_repeats=config.repeats, pattern=pairs)
            for wire in wires:
                qml.Hadamard(wires=wire)
            return qml.state() if config.family == 'quantum_pauli' else qml.probs(wires=wires)

        measured = np.vstack([np.asarray(circuit(angles[i:i + 16])) for i in range(0, len(x), 16)])
        nonlinear = pauli_features_from_states(measured, x.shape[1]) if config.family == 'quantum_pauli' else measured
        features = np.column_stack((x, nonlinear))
    elif config.family.startswith('blend_'):
        base, _ = features_for(x, Configuration('classical_probs_s0.7', 'product_probs', scale=0.7), cache_dir)
        if config.family == 'blend_trig':
            extra = Configuration('classical_trig_s1.4', 'trig', scale=1.4)
        elif config.family == 'blend_cubic':
            extra = Configuration('classical_cubic', 'cubic')
        else:
            extra = Configuration(f'quantum_r{config.repeats}_s{config.scale:g}', 'quantum',
                                  scale=config.scale, repeats=config.repeats)
        extra_features, _ = features_for(x, extra, cache_dir)
        # The existing base already contains the original input columns.
        features = np.column_stack((base, extra_features[:, x.shape[1]:]))
    else:
        raise ValueError(f'Unknown family: {config.family}')
    info = {'feature_seconds': time.perf_counter() - start, 'cache_hit': False,
            'n_features': features.shape[1], 'n_rows': len(features)}
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=features, info=json.dumps(info))
    return features, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'qfm/results/learning_curves/dataset.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'qfm/results/feature_search')
    parser.add_argument('--cache-dir', type=Path, help='Reuse maps from another validation screen')
    parser.add_argument('--seeds', type=int, nargs='+', default=[101, 102, 103])
    parser.add_argument('--n-train', type=int, default=128)
    parser.add_argument('--only', nargs='+', choices=[c.name for c in CONFIGURATIONS])
    parser.add_argument('--classifier', choices=['linear', 'rbf', 'mlp', 'mlp_lbfgs'],
                        help='Override the classifier after feature extraction')
    args = parser.parse_args()
    with args.source.open() as handle:
        source = [r for r in csv.DictReader(handle) if r['split'] in ('train', 'validation')]
    rows = [Row(r['group_id'], r['sentence'], r['candidate_0'], r['candidate_1'], int(r['label'])) for r in source]
    x = np.array([encode(r, 'meaning') for r in rows])
    y = np.array([r.label for r in rows])
    train_pool = np.array([i for i, r in enumerate(source) if r['split'] == 'train'])
    validation = np.array([i for i, r in enumerate(source) if r['split'] == 'validation'])
    if args.n_train < 4 or args.n_train % 2 or args.n_train > len(train_pool):
        parser.error('Use an even training size between 4 and the training pool size')
    configs = list(CONFIGURATIONS[:16]) if args.only is None else [c for c in CONFIGURATIONS if c.name in args.only]
    if args.classifier is not None:
        configs = [replace(c, classifier=args.classifier) for c in configs]
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {'source': str(args.source), 'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
                'n_train': args.n_train, 'seeds': args.seeds, 'validation_size': len(validation),
                'configurations': [asdict(c) for c in configs],
                'selection': 'Mean validation accuracy across paired training subsets; test rows excluded',
                'tuning': 'Eight classifier choices per map and training subset',
                'scope': 'Explicit grammar parser; eight meaning cues; fixed quantum maps plus original inputs'}
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    records, summary = [], []
    print(f'Validation screen: {len(configs)} maps x {len(args.seeds)} subsets, n={args.n_train}; no test evaluation', flush=True)
    for config in configs:
        features, info = features_for(x, config, args.cache_dir or args.output / 'features')
        scores = []
        for seed in args.seeds:
            train = balanced_order(train_pool, y, seed)[:args.n_train]
            model, score, params, elapsed, warnings_count = select_model(
                config.classifier, features, y, train, validation, seed)
            scores.append(score)
            records.append({'name': config.name, 'family': config.family, 'seed': seed,
                            'classifier': config.classifier,
                            'n_train': args.n_train, 'validation_accuracy': score,
                            'train_accuracy': float(model.score(features[train], y[train])),
                            'fit_and_tune_seconds': elapsed, 'parameters': json.dumps(params),
                            'convergence_warnings': warnings_count,
                            'features_kept': int(model.named_steps['variancethreshold'].get_support().sum())})
        summary.append({'name': config.name, 'family': config.family,
                        'classifier': config.classifier,
                        'validation_mean': float(np.mean(scores)),
                        'validation_std': float(np.std(scores, ddof=1)) if len(scores) > 1 else 0., **info})
        save_csv(args.output / 'runs.csv', records)
        save_csv(args.output / 'summary.csv', sorted(summary, key=lambda r: r['validation_mean'], reverse=True))
        print(f"  {config.name:28s} validation={np.mean(scores):.3f} map={info['feature_seconds']:.2f}s cached={info['cache_hit']}", flush=True)
    print(f'Saved validation-only results to {args.output}', flush=True)


if __name__ == '__main__':
    main()
