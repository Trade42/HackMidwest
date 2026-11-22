"""Frozen DistilBERT -> training-only PCA -> fixed quantum/classical features.

Stages make model selection auditable: prepare frozen embeddings, fit every
model using training/validation only, freeze artifacts, then evaluate tests.
Existing row assignments are preserved; strict cohorts exclude shared worlds.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from corpus_experiment import load_split, validate_splits
from frozen_encoder import FrozenDistilBertEncoder, TrainingOnlyReducer, text_pair
from learning_curves import ROOT, balanced_order, save_csv, select_model
from nlp_features import exact_features, resource_report

METHODS = ('iqp_entangling', 'iqp_plain', 'rbf', 'polynomial', 'mlp', 'full_rbf', 'fourier', 'surrogate')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def record_environment(output):
    packages = ['torch', 'transformers', 'huggingface-hub', 'tokenizers', 'safetensors',
                'pennylane', 'scikit-learn', 'numpy', 'scipy', 'joblib']
    write_json(output / 'environment.json', {p: importlib.metadata.version(p) for p in packages})


def load_source(source, extra_test=None):
    """Normalize either existing numeric CSV or the existing pronoun corpus."""
    source = Path(source)
    provenance = {}
    rows = []
    if source.is_dir():
        splits = {name: load_split(source, name, small=False)
                  for name in ('train', 'validation', 'test', 'test_ood')}
        validate_splits(splits)
        for split, (_, data) in splits.items():
            provenance[str(source / f'{split}_mixed.csv')] = digest(source / f'{split}_mixed.csv')
            for r in data:
                rows.append({k: r[k] for k in ('id', 'sentence', 'candidate_0', 'candidate_1', 'label')} |
                            dict(split=split, original_split=split, world_id=r['scenario_id']))
    else:
        provenance[str(source)] = digest(source)
        for r in read_rows(source):
            rows.append({k: r[k] for k in ('sentence', 'candidate_0', 'candidate_1', 'label')} |
                        dict(id=r.get('id', r['group_id']), split=r['split'], original_split=r['split'],
                             world_id=r['group_id'].rsplit('-', 1)[0]))
    if extra_test is not None:
        original_worlds = {r['world_id'] for r in rows}
        provenance[str(extra_test)] = digest(extra_test)
        for r in read_rows(extra_test):
            world = r['group_id'].rsplit('-', 1)[0]
            if r['split'] != 'fresh_test' or world in original_worlds:
                continue
            rows.append({k: r[k] for k in ('sentence', 'candidate_0', 'candidate_1', 'label')} |
                        dict(id=r['group_id'], split='test_unseen_worlds', original_split=r['split'],
                             world_id=world))
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Source IDs overlap')
    text_keys = [tuple(' '.join(t.lower().split()) for t in text_pair(r)) for r in rows]
    if len(set(text_keys)) != len(rows):
        raise ValueError('Repeated sentence/candidate inputs; review corpus before running')
    return rows, provenance


def cohort_indices(rows):
    train = np.array([i for i, r in enumerate(rows) if r['split'] == 'train'])
    val = np.array([i for i, r in enumerate(rows) if r['split'] == 'validation'])
    train_worlds = {rows[i]['world_id'] for i in train}
    val_worlds = {rows[i]['world_id'] for i in val}
    selection = np.array([i for i in val if rows[i]['world_id'] not in train_worlds], dtype=int)
    cohorts = {}
    for split in sorted({r['split'] for r in rows} - {'train', 'validation'}):
        full = np.array([i for i, r in enumerate(rows) if r['split'] == split])
        strict = np.array([i for i in full if rows[i]['world_id'] not in train_worlds | val_worlds], dtype=int)
        cohorts[split] = full
        if len(strict) != len(full):
            cohorts[f'{split}_strict'] = strict
    if not len(selection) or any(not len(indices) for indices in cohorts.values()):
        raise ValueError('Empty strict validation or evaluation cohort')
    return train, selection, cohorts


def reject_repeated_encodings(values, ids, description):
    keys = [tuple(row) for row in np.round(values, decimals=8)]
    seen = {}
    for key, identifier in zip(keys, ids):
        if key in seen:
            raise ValueError(f'Repeated {description}: {seen[key]} and {identifier}')
        seen[key] = identifier


def prepare(args):
    if (args.output / 'protocol.json').exists():
        raise FileExistsError('Run directory already prepared; use fit/evaluate or choose a new --output')
    rows, provenance = load_source(args.source, args.extra_test)
    train, validation, cohorts = cohort_indices(rows)
    if any(n <= 8 or n % 2 or n > len(train) for n in args.sizes):
        raise ValueError('Training sizes must be even, greater than 8 and no larger than the training pool')
    args.output.mkdir(parents=True, exist_ok=True)
    record_environment(args.output)
    save_csv(args.output / 'dataset.csv', rows)
    primary = ('test_unseen_worlds' if 'test_unseen_worlds' in cohorts else
               'test_ood' if 'test_ood' in cohorts else 'test_strict' if 'test_strict' in cohorts else 'test')
    encoder = FrozenDistilBertEncoder(batch_size=args.batch_size, threads=args.threads)
    protocol = dict(source_hashes=provenance, training_sizes=sorted(set(args.sizes)),
                    subset_seeds=list(range(301, 301+args.seeds)), methods=list(args.methods),
                    primary_evaluation=primary, encoder=encoder.specification,
                    pca_components=8, pca_fit='Current training subset only; refit per size and seed',
                    angle_scaling='Training quantiles 5/95%, clipped to [-0.55, 0.55]',
                    feature_design='Quantum heads concatenate the shared eight angles with 36 Z/ZZ readouts',
                    circuit='Fixed two repeats, all-pairs ZZ interactions/readouts; no circuit search',
                    split_counts=dict(Counter(r['split'] for r in rows)),
                    selection_validation_ids=[rows[i]['id'] for i in validation],
                    excluded_validation_ids=[r['id'] for i, r in enumerate(rows)
                                             if r['split'] == 'validation' and i not in set(validation)],
                    evaluation_ids={k: [rows[i]['id'] for i in ix] for k, ix in cohorts.items()},
                    split_policy='Original assignments retained. Strict cohorts exclude shared worlds.',
                    prior_exposure='These synthetic corpora were evaluated in earlier handcrafted-feature experiments.',
                    selection='Eight classifier candidates per model; no test scores used in this experiment',
                    surrogate='Fixed Fourier basis, ridge alpha=0.001 fitted to training IQP outputs only',
                    scope='Frozen pretrained NLP encoder, exact simulated circuits; no contextuality claim',
                    runtime='CPU simulation; no hardware timings', python=platform.python_version())
    write_json(args.output / 'protocol.json', protocol)
    print(f'Encoding {len(rows)} sentences; {len(validation)} world-disjoint validation rows', flush=True)
    start = time.perf_counter()

    def progress(done, total):
        if done == total or done % 256 == 0:
            print(f'  DistilBERT {done}/{total}, {time.perf_counter()-start:.1f}s', flush=True)

    x, info = encoder.encode(rows, cache_dir=args.cache_dir, progress=progress)
    reject_repeated_encodings(x, [r['id'] for r in rows], 'DistilBERT embeddings')
    payload = dict(all_X=x, all_ids=np.array([r['id'] for r in rows]), metadata=json.dumps(info))
    # Retain compatibility with corpus_experiment.py --embeddings and its ID checks.
    for split in sorted({r['split'] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r['split'] == split]
        payload[f'{split}_X'] = x[indices]
        payload[f'{split}_ids'] = np.array([rows[i]['id'] for i in indices])
    np.savez_compressed(args.output / 'embeddings.npz', **payload)
    write_json(args.output / 'encoder.json', info)
    print(f'Embeddings ready: {x.shape}; cache={info["cache_hit"]}', flush=True)


def load_run(output):
    rows = read_rows(output / 'dataset.csv')
    protocol = json.loads((output / 'protocol.json').read_text())
    with np.load(output / 'embeddings.npz', allow_pickle=False) as data:
        if list(data['all_ids']) != [r['id'] for r in rows]:
            raise ValueError('Embedding IDs do not match the saved dataset')
        x = data['all_X']
    return rows, protocol, x


def fit(output):
    if (output / 'frozen.json').exists():
        raise FileExistsError('Models already frozen; use evaluate or a new run directory')
    rows, protocol, x = load_run(output)
    y = np.array([int(r['label']) for r in rows])
    train_pool, validation, _ = cohort_indices(rows)
    records, bundles = [], []
    methods = protocol['methods']
    (output / 'models').mkdir(exist_ok=True)
    for size in protocol['training_sizes']:
        for seed in protocol['subset_seeds']:
            train = balanced_order(train_pool, y, seed)[:size]
            development = np.r_[train, validation]
            local_train, local_val = np.arange(size), np.arange(size, len(development))
            start = time.perf_counter()
            reducer = TrainingOnlyReducer().fit(x[train])
            angles = reducer.transform(x[development])
            pca_seconds = time.perf_counter()-start
            reject_repeated_encodings(angles, [rows[i]['id'] for i in development], 'PCA/angle inputs')
            features, times, surrogate = exact_features(angles, methods, train_indices=local_train)
            if 'full_rbf' in methods:
                features['full_rbf'], times['full_rbf'] = x[development], 0.
            bundle = dict(reducer=reducer, surrogate=surrogate, heads={}, train_indices=train,
                          n_train=size, seed=seed, encoder=protocol['encoder'], methods=methods)
            for method in methods:
                classifier = 'mlp_lbfgs' if method == 'mlp' else 'polynomial' if method == 'polynomial' else 'rbf'
                head, score, params, elapsed, warnings = select_model(
                    classifier, features[method], y[development], local_train, local_val, seed)
                bundle['heads'][method] = head
                records.append(dict(n_train=size, seed=seed, model=method, validation_accuracy=score,
                                    training_accuracy=float(head.score(features[method][local_train], y[train])),
                                    parameters=json.dumps(params), convergence_warnings=warnings,
                                    pca_seconds=pca_seconds, pca_variance=float(reducer.pca.explained_variance_ratio_.sum()),
                                    feature_seconds=times[method], fit_and_tune_seconds=elapsed,
                                    feature_width=features[method].shape[1],
                                    surrogate_teacher_seconds=times.get('iqp_entangling', 0.) if method == 'surrogate' else 0.))
                save_csv(output / 'validation_results.csv', records)
            path = output / 'models' / f'n{size}_seed{seed}.joblib'
            joblib.dump(bundle, path, compress=3)
            bundles.append(dict(path=str(path.relative_to(output)), sha256=digest(path), n_train=size, seed=seed))
            print(f'Fitted n={size}, seed={seed}: {len(methods)} models, PCA retains '
                  f'{100*reducer.pca.explained_variance_ratio_.sum():.1f}% variance', flush=True)
    frozen = dict(protocol_sha256=digest(output / 'protocol.json'), dataset_sha256=digest(output / 'dataset.csv'),
                  embeddings_sha256=digest(output / 'embeddings.npz'), bundles=bundles,
                  selected_application_bundle=bundles[-len(protocol['subset_seeds'])]['path'],
                  application_selection='Largest declared training size, first declared seed; no test selection',
                  test_evaluated=False)
    write_json(output / 'frozen.json', frozen)
    write_json(output / 'resources.json', {name: resource_report(name == 'iqp_entangling')
                                           for name in ('iqp_entangling', 'iqp_plain')})
    print('All models frozen before test evaluation.', flush=True)


def load_frozen(output):
    frozen = json.loads((output / 'frozen.json').read_text())
    for name in ('protocol', 'dataset', 'embeddings'):
        suffix = '.json' if name == 'protocol' else '.csv' if name == 'dataset' else '.npz'
        if digest(output / f'{name}{suffix}') != frozen[f'{name}_sha256']:
            raise ValueError(f'{name} changed after model freeze')
    for bundle in frozen['bundles']:
        if digest(output / bundle['path']) != bundle['sha256']:
            raise ValueError('Model artifact changed after freeze')
    return frozen


def evaluate(output):
    frozen = load_frozen(output)
    rows, protocol, x = load_run(output)
    y = np.array([int(r['label']) for r in rows])
    _, _, cohorts = cohort_indices(rows)
    indices = np.unique(np.concatenate(list(cohorts.values())))
    positions = {int(index): j for j, index in enumerate(indices)}
    records, predictions, approximations = [], {}, []
    for artifact in frozen['bundles']:
        bundle = joblib.load(output / artifact['path'])
        all_angles = bundle['reducer'].transform(x)
        reject_repeated_encodings(all_angles, [r['id'] for r in rows], 'PCA inputs across all splits')
        angles = all_angles[indices]
        features, times, _ = exact_features(angles, bundle['methods'], surrogate=bundle['surrogate'])
        if 'full_rbf' in bundle['methods']:
            features['full_rbf'], times['full_rbf'] = x[indices], 0.
        for method, head in bundle['heads'].items():
            start = time.perf_counter()
            with threadpool_limits(limits=1):
                prediction = head.predict(features[method])
            head_seconds = time.perf_counter()-start
            predictions[f'{method}__n{bundle["n_train"]}__seed{bundle["seed"]}'] = prediction
            for split, original_indices in cohorts.items():
                local = np.array([positions[int(i)] for i in original_indices])
                records.append(dict(model=method, n_train=bundle['n_train'], seed=bundle['seed'], split=split,
                                    test_cases=len(local), accuracy=float(np.mean(prediction[local] == y[original_indices])),
                                    feature_seconds_all_eval=times[method], head_seconds_all_eval=head_seconds,
                                    eval_rows=len(indices)))
        if 'surrogate' in features and 'iqp_entangling' in features:
            error = features['surrogate'][:, 8:] - features['iqp_entangling'][:, 8:]
            approximations.append(dict(n_train=bundle['n_train'], seed=bundle['seed'],
                                       readout_rmse=float(np.sqrt(np.mean(error**2))),
                                       readout_max_error=float(np.abs(error).max())))
        save_csv(output / 'test_results.csv', records)
        print(f'Evaluated frozen n={bundle["n_train"]}, seed={bundle["seed"]}', flush=True)
    np.savez_compressed(output / 'predictions.npz', indices=indices, labels=y[indices],
                        ids=np.array([rows[i]['id'] for i in indices]), **predictions)
    if approximations:
        save_csv(output / 'surrogate_approximation.csv', approximations)
    summary = []
    for split in cohorts:
        for size in protocol['training_sizes']:
            for method in protocol['methods']:
                selected = [r for r in records if r['split'] == split and r['n_train'] == size and r['model'] == method]
                scores = [r['accuracy'] for r in selected]
                summary.append(dict(split=split, n_train=size, model=method, accuracy_mean=float(np.mean(scores)),
                                    accuracy_std=float(np.std(scores, ddof=1)) if len(scores)>1 else 0.,
                                    subset_seeds=len(scores), test_cases=selected[0]['test_cases']))
    save_csv(output / 'summary.csv', summary)
    paired_report(output, rows, protocol, cohorts, indices, predictions, y)
    plot_results(output, summary, protocol['primary_evaluation'])
    for row in summary:
        if row['split'] == protocol['primary_evaluation']:
            print(f'{row["split"]} n={row["n_train"]} {row["model"]}: {row["accuracy_mean"]:.3f}', flush=True)
    write_json(output / 'evaluation.json', dict(frozen_manifest_sha256=digest(output / 'frozen.json'),
                                               primary_evaluation=protocol['primary_evaluation'],
                                               test_selection=False, repeated_evaluation=(output / 'evaluation.json').exists()))


def paired_report(output, rows, protocol, cohorts, indices, predictions, y):
    if 'iqp_entangling' not in protocol['methods']:
        return
    from confirm_feature_gain import paired_interval
    selected = cohorts[protocol['primary_evaluation']]
    local = np.array([int(np.flatnonzero(indices == i)[0]) for i in selected])
    worlds = [rows[i]['world_id'] for i in selected]
    comparisons = (len(protocol['methods'])-1)*len(protocol['training_sizes'])
    gains = []
    for size in protocol['training_sizes']:
        correct = {name: np.array([predictions[f'{name}__n{size}__seed{seed}'][local] == y[selected]
                                   for seed in protocol['subset_seeds']], dtype=float)
                   for name in protocol['methods']}
        for baseline in protocol['methods']:
            if baseline == 'iqp_entangling':
                continue
            differences = correct['iqp_entangling'] - correct[baseline]
            lo, hi = paired_interval(differences, comparisons, resamples=5000, cluster_ids=worlds)
            gains.append(dict(n_train=size, baseline=baseline, gain=float(differences.mean()),
                              interval_low=lo, interval_high=hi, subset_seeds=len(protocol['subset_seeds']),
                              interpretation='Exploratory paired world/seed bootstrap; adjusted approximate intervals'))
    if gains:
        save_csv(output / 'paired_gains.csv', gains)


def plot_results(output, summary, split):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig, ax = plt.subplots(figsize=(10, 5.5))
    methods = list(dict.fromkeys(r['model'] for r in summary))
    for method in methods:
        rows = sorted([r for r in summary if r['split'] == split and r['model'] == method], key=lambda r:r['n_train'])
        ax.errorbar([r['n_train'] for r in rows], [r['accuracy_mean'] for r in rows],
                    yerr=[r['accuracy_std'] for r in rows], marker='o', capsize=3, label=method)
    ax.axhline(.5, color='#999999', ls=':')
    ax.set(xlabel='Labeled training examples; PCA refit for each subset', ylabel='Accuracy',
           title=f'Frozen DistilBERT, {split}\nError bars: SD across training subsets')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / 'learning_curves.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'fit', 'evaluate', 'all'), default='all')
    parser.add_argument('--source', type=Path, default=ROOT / 'qfm/results/learning_curves/dataset.csv')
    parser.add_argument('--extra-test', type=Path, help='Optional existing confirmation CSV; retain only entirely unseen worlds')
    parser.add_argument('--output', type=Path, default=ROOT / 'qfm/results/distilbert_numeric')
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'qfm/cache/distilbert')
    parser.add_argument('--sizes', nargs='+', type=int, default=[128, 256])
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if min(args.seeds, args.batch_size, args.threads) < 1:
        parser.error('seeds, batch-size and threads must be positive')
    args.methods = list(dict.fromkeys(args.methods))
    if args.stage in ('prepare', 'all'):
        prepare(args)
    if args.stage in ('fit', 'all'):
        fit(args.output)
    if args.stage in ('evaluate', 'all'):
        evaluate(args.output)


if __name__ == '__main__':
    main()
