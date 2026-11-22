"""Reanalyze saved predictions with world clusters; never fit or tune a model.

Higher/lower queries share an underlying world. The original group identifier
included that query, allowing paired worlds across development/test splits.
Report both all unused configurations and only worlds absent from the entire
original corpus, adjusting intervals for all comparisons in both cohorts.
This is a documented statistical correction after the initial confirmation.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from confirm_feature_gain import paired_interval
from learning_curves import ROOT, save_csv


def world_id(group_id):
    return str(group_id).rsplit('-', 1)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, default=ROOT / 'qfm/results/feature_confirmation')
    parser.add_argument('--original', type=Path, default=ROOT / 'qfm/results/learning_curves/dataset.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'qfm/results/feature_confirmation_clustered')
    args = parser.parse_args()
    protocol = json.loads((args.predictions / 'protocol.json').read_text())
    configs = protocol['models_frozen_before_test_evaluation']
    quantum, *baselines = [config['name'] for config in configs]
    sizes = protocol['training_sizes']
    with args.original.open() as handle:
        original_worlds = {world_id(row['group_id']) for row in csv.DictReader(handle)}
    with np.load(args.predictions / 'predictions.npz', allow_pickle=False) as saved:
        labels = saved['labels']
        groups = saved['group_ids']
        predictions = {key: saved[key] for key in saved.files if '__n' in key}
    worlds = np.array([world_id(group) for group in groups])
    cohorts = {'unused_configurations': np.ones(len(groups), dtype=bool),
               'unseen_worlds': ~np.isin(worlds, list(original_worlds))}
    if not cohorts['unseen_worlds'].any():
        raise ValueError('No entirely unseen worlds remain')
    comparisons = len(baselines) * len(sizes) * len(cohorts)
    amendment = {
        'source_predictions': str(args.predictions),
        'timing': 'Statistical correction after viewing original confirmation; models unchanged',
        'reason': 'Higher/lower questions share a world; original IDs included the question',
        'clusters': 'All factors except the final query bit',
        'strict_cohort': 'Worlds absent from every split of the original 1024-row corpus',
        'method': 'Paired crossed bootstrap of training-subset seeds and world clusters, 10000 draws',
        'coverage': 'Bonferroni-adjusted approximate 95% family coverage',
        'declared_comparisons': comparisons,
        'limitations': 'Fixed synthetic grammar, train pool and validation set; training subsets overlap',
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'amendment.json').write_text(json.dumps(amendment, indent=2) + '\n')
    summary, gains = [], []
    for cohort, mask in cohorts.items():
        for size in sizes:
            correctness = {}
            for model in [quantum, *baselines]:
                correctness[model] = (predictions[f'{model}__n{size}'][:, mask] == labels[mask]).astype(float)
                scores = correctness[model].mean(axis=1)
                summary.append(dict(cohort=cohort, model=model, n_train=size,
                                    accuracy=float(scores.mean()), seed_std=float(scores.std(ddof=1)),
                                    test_cases=int(mask.sum()), worlds=len(np.unique(worlds[mask]))))
            for baseline in baselines:
                difference = correctness[quantum] - correctness[baseline]
                low, high = paired_interval(difference, comparisons, cluster_ids=worlds[mask])
                gains.append(dict(cohort=cohort, n_train=size, quantum=quantum, baseline=baseline,
                                  gain=float(difference.mean()), interval_low=low, interval_high=high,
                                  above_noise=low > 0))
        print(f'{cohort}: {mask.sum()} cases / {len(np.unique(worlds[mask]))} worlds', flush=True)
    save_csv(args.output / 'summary.csv', summary)
    save_csv(args.output / 'paired_gains.csv', gains)
    (args.output / 'results.json').write_text(json.dumps({'summary': summary, 'gains': gains}, indent=2) + '\n')
    plot(gains, args.output)
    for row in gains:
        print(f"{row['cohort']} n={row['n_train']} vs {row['baseline']}: "
              f"{100*row['gain']:+.2f} pp [{100*row['interval_low']:+.2f}, {100*row['interval_high']:+.2f}]")


def plot(gains, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, cohort, title in zip(axes, ['unused_configurations', 'unseen_worlds'],
                                 ['All unused configurations', 'Entirely unseen worlds']):
        rows = [r for r in gains if r['cohort'] == cohort]
        for i, row in enumerate(rows):
            ax.errorbar(row['gain'], i,
                        xerr=[[max(0, row['gain']-row['interval_low'])],
                              [max(0, row['interval_high']-row['gain'])]],
                        fmt='o', capsize=4, color='#13865c' if row['above_noise'] else '#666a77')
        ax.axvline(0, color='#333333', ls=':')
        ax.set_yticks(range(len(rows)), [f"n={r['n_train']} vs {r['baseline']}" for r in rows], fontsize=8)
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set(title=title, xlabel='Quantum accuracy gain (percentage points)')
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Saved predictions: world-cluster bootstrap, adjusted across 12 comparisons', fontsize=11)
    fig.tight_layout()
    fig.savefig(output / 'paired_gains.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
