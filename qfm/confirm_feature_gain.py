"""Frozen comparison on semantic configurations absent from the original run.

Choose models using validation screens BEFORE running this script. It evaluates
all chosen models on the same fresh cases and paired training-subset seeds,
then reports simultaneous paired bootstrap intervals for the quantum gains.
"""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from feature_search import CONFIGURATIONS, features_for
from learning_curves import ROOT, Row, balanced_order, encode, make_dataset, save_csv, select_model


def paired_interval(differences, comparisons, resamples=10000, cluster_ids=None):
    """Resample paired subset seeds and paired semantic-world clusters.

    Bonferroni-adjust the percentile intervals for every declared comparison.
    This quantifies variation within this synthetic population and train pool;
    it is not evidence about unrestricted natural-language tasks.
    """
    rng = np.random.default_rng(582001)
    n_seeds, n_cases = differences.shape
    if cluster_ids is None:
        cluster_ids = np.arange(n_cases)
    if len(cluster_ids) != n_cases:
        raise ValueError('One cluster ID is required per test case')
    _, inverse = np.unique(cluster_ids, return_inverse=True)
    sizes = np.bincount(inverse)
    sums = np.array([np.bincount(inverse, weights=row) for row in differences])
    n_clusters = len(sizes)
    estimates = np.empty(resamples)
    for i in range(resamples):
        seeds = rng.integers(n_seeds, size=n_seeds)
        clusters = rng.integers(n_clusters, size=n_clusters)
        estimates[i] = sums[np.ix_(seeds, clusters)].sum() / (n_seeds * sizes[clusters].sum())
    tail = 0.05 / (2 * comparisons)
    low, high = np.quantile(estimates, [tail, 1 - tail])
    return float(low), float(high)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'qfm/results/learning_curves')
    parser.add_argument('--output', type=Path, default=ROOT / 'qfm/results/feature_confirmation')
    parser.add_argument('--quantum', default='quantum_r2_s0.55')
    parser.add_argument('--classical', nargs='+',
                        default=['classical_probs_s0.7', 'classical_product_s0.55', 'raw_mlp_lbfgs'])
    parser.add_argument('--sizes', type=int, nargs='+', default=[64, 128])
    parser.add_argument('--seeds', type=int, default=16)
    args = parser.parse_args()
    available = {c.name: c for c in CONFIGURATIONS}
    if args.quantum not in available or 'quantum' not in available[args.quantum].family:
        parser.error('Choose a quantum map from feature_search.py')
    if not args.classical or any(name not in available or 'quantum' in available[name].family for name in args.classical):
        parser.error('Choose classical control maps from feature_search.py')
    if args.seeds < 2:
        parser.error('Use at least two paired subset seeds')
    args.classical = list(dict.fromkeys(args.classical))
    configs = [available[name] for name in [args.quantum, *args.classical]]
    args.sizes = sorted(set(args.sizes))
    with (args.source / 'dataset.csv').open() as handle:
        original = list(csv.DictReader(handle))
    source_protocol = json.loads((args.source / 'protocol.json').read_text())
    development = [r for r in original if r['split'] in ('train', 'validation')]
    rows = [Row(r['group_id'], r['sentence'], r['candidate_0'], r['candidate_1'], int(r['label']))
            for r in development]
    original_groups = {r['group_id'] for r in original}
    universe = make_dataset(source_protocol['dataset_seed'], 2048)
    fresh = [r for r in universe if r.group_id not in original_groups]
    if not fresh or len({r.group_id for r in fresh}) != len(fresh):
        parser.error('No distinct unused semantic configurations remain')
    train_pool = np.array([i for i, r in enumerate(development) if r['split'] == 'train'])
    validation = np.array([i for i, r in enumerate(development) if r['split'] == 'validation'])
    if any(n < 4 or n % 2 or n > len(train_pool) for n in args.sizes):
        parser.error('Use even training sizes from 4 to the training pool size')
    test = np.arange(len(rows), len(rows) + len(fresh))
    rows += fresh
    x = np.array([encode(r, 'meaning') for r in rows])
    y = np.array([r.label for r in rows])
    if {tuple(a) for a in x[:test[0]]} & {tuple(a) for a in x[test]}:
        raise ValueError('Fresh inputs overlap with development inputs')
    seeds = list(range(201, 201 + args.seeds))
    comparisons = len(args.classical) * len(args.sizes)
    protocol = {
        'source_dataset_sha256': hashlib.sha256((args.source / 'dataset.csv').read_bytes()).hexdigest(),
        'dataset_seed': source_protocol['dataset_seed'], 'n_fresh_cases': len(fresh),
        'fresh_group_ids': [r.group_id for r in fresh],
        'training_sizes': args.sizes, 'subset_seeds': seeds,
        'models_frozen_before_test_evaluation': [asdict(c) for c in configs],
        'selection': 'Validation only; original test cases excluded; all maps retain original input cues',
        'criterion': 'Quantum gain interval strictly above zero against every declared classical control at a training size',
        'uncertainty': 'Paired crossed bootstrap over subset seeds and semantic-world clusters, 10000 draws; Bonferroni-adjusted approximate 95% family coverage',
        'declared_comparisons': comparisons,
        'scope': 'Controlled synthetic grammar and explicit parser; fixed train pool; exact simulated quantum features',
    }
    args.output.mkdir(parents=True, exist_ok=True)
    # Persist the protocol before computing features, fitting or scoring models.
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    assignments = [r['split'] for r in development] + ['fresh_test'] * len(fresh)
    save_csv(args.output / 'dataset.csv', [{**asdict(r), 'split': split} for r, split in zip(rows, assignments)])
    print(f'Frozen confirmation: {len(configs)} models x {len(seeds)} subsets x {len(args.sizes)} sizes; '
          f'{len(fresh)} unused cases', flush=True)
    records, correct, saved_predictions, map_info = [], {}, {}, {}
    for config in configs:
        features, map_info[config.name] = features_for(x, config, args.output / 'features')
        print(f'{config.name}: features ready', flush=True)
        for n in args.sizes:
            predictions = []
            for position, seed in enumerate(seeds):
                train = balanced_order(train_pool, y, seed)[:n]
                model, score, params, elapsed, warning_count = select_model(
                    config.classifier, features, y, train, validation, seed)
                prediction = model.predict(features[test])
                predictions.append(prediction)
                records.append({'model': config.name, 'classifier': config.classifier,
                                'n_train': n, 'seed': seed, 'validation_accuracy': score,
                                'test_accuracy': float(np.mean(prediction == y[test])),
                                'fit_and_tune_seconds': elapsed, 'convergence_warnings': warning_count,
                                'parameters': json.dumps(params)})
                save_csv(args.output / 'runs.csv', records)
                if (position + 1) % 4 == 0:
                    print(f'  n={n}: {position + 1}/{len(seeds)} subsets complete', flush=True)
            predictions = np.array(predictions)
            saved_predictions[f'{config.name}__n{n}'] = predictions
            correct[config.name, n] = (predictions == y[test]).astype(float)
            print(f'  n={n}: mean test accuracy {correct[config.name, n].mean():.4f}', flush=True)
    np.savez_compressed(args.output / 'predictions.npz', labels=y[test],
                        group_ids=np.array([r.group_id for r in fresh]), **saved_predictions)
    summary = []
    for config in configs:
        for n in args.sizes:
            scores = correct[config.name, n].mean(axis=1)
            summary.append({'model': config.name, 'n_train': n, 'test_mean': float(scores.mean()),
                            'test_std': float(scores.std(ddof=1)), 'seeds': len(seeds), 'test_cases': len(fresh)})
    save_csv(args.output / 'summary.csv', summary)
    gains = []
    for n in args.sizes:
        for baseline in args.classical:
            difference = correct[args.quantum, n] - correct[baseline, n]
            low, high = paired_interval(difference, comparisons,
                                        cluster_ids=[r.group_id.rsplit('-', 1)[0] for r in fresh])
            gains.append({'n_train': n, 'quantum': args.quantum, 'baseline': baseline,
                          'gain': float(difference.mean()), 'interval_low': low, 'interval_high': high,
                          'above_noise': low > 0})
    save_csv(args.output / 'paired_gains.csv', gains)
    report = {'feature_maps': map_info, 'comparisons': gains,
              'passes_all_controls_by_size': {str(n): all(r['above_noise'] for r in gains if r['n_train'] == n)
                                             for n in args.sizes}}
    (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig, ax = plt.subplots(figsize=(10, max(4, len(gains) * .65)))
    for i, row in enumerate(gains):
        color = '#13865c' if row['above_noise'] else '#666a77'
        ax.errorbar(row['gain'], i, xerr=[[max(0, row['gain'] - row['interval_low'])],
                                        [max(0, row['interval_high'] - row['gain'])]],
                    fmt='o', color=color, capsize=4)
    ax.axvline(0, color='#333333', linestyle=':')
    ax.set_yticks(range(len(gains)), [f"n={r['n_train']} vs {r['baseline']}" for r in gains])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set(xlabel='Quantum accuracy minus classical accuracy (percentage points)',
           title='Fresh-case comparison: paired, multiplicity-adjusted intervals')
    ax.grid(axis='x', alpha=.2)
    fig.tight_layout()
    fig.savefig(args.output / 'paired_gains.png', dpi=170)
    plt.close(fig)
    print(json.dumps(report['passes_all_controls_by_size']), flush=True)
    print(f'Saved confirmation results to {args.output}', flush=True)


if __name__ == '__main__':
    main()
