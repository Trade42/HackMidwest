"""Measure finite-shot inference sensitivity of frozen, exactly trained heads.

One computational-basis histogram supplies all commuting Z/ZZ readouts.
Multinomial draws simulate independent measurement shots efficiently. No gate
noise, hardware runtime, shot-based training, or classifier retuning is implied.
"""

import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from distilbert_experiment import ROOT, cohort_indices, load_frozen, load_run, save_csv, write_json
from nlp_features import iqp_probabilities, readout_eigenvalues, resource_report, sample_readouts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'qfm/results/distilbert_numeric')
    parser.add_argument('--shots', nargs='+', type=int, default=[100, 1000, 10000])
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.seeds < 1 or any(s < 1 for s in args.shots):
        parser.error('Positive shot counts and seeds required')
    frozen = load_frozen(args.run)
    rows, protocol, embeddings = load_run(args.run)
    bundle = joblib.load(args.run / frozen['selected_application_bundle'])
    _, _, cohorts = cohort_indices(rows)
    indices = cohorts[protocol['primary_evaluation']]
    y = np.array([int(rows[i]['label']) for i in indices])
    started = time.perf_counter()
    angles = bundle['reducer'].transform(embeddings[indices])
    pca_seconds = time.perf_counter()-started
    output = args.output or args.run / 'shots'
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'protocol.json', dict(
        run=str(args.run), model_artifact=frozen['selected_application_bundle'],
        selection='Largest declared training size, first seed; fixed before test evaluation',
        classifier_training='Exact features; same frozen head at every shot count',
        measurements='One shared Z-basis sample histogram per input, all Z/ZZ derived jointly',
        shot_simulation='Exact multinomial sampling from the simulated Born distribution',
        shot_counts=sorted(set(args.shots)), seeds=list(range(701, 701+args.seeds)),
        primary_evaluation=protocol['primary_evaluation'], examples=len(indices),
        embedding_cost='Frozen embeddings cached; excluded from circuit latency; CLI reports encoder separately',
        execution='CPU statevector simulator, no hardware or gate noise', hardware_runtime=None))
    records, predictions = [], {}
    for method in ('iqp_entangling', 'iqp_plain'):
        if method not in bundle['heads']:
            continue
        head = bundle['heads'][method]
        exact_prediction = None
        exact_readouts = None
        for shots in [0, *sorted(set(args.shots))]:
            started = time.perf_counter()
            probabilities = iqp_probabilities(angles, method == 'iqp_entangling')
            state_seconds = time.perf_counter()-started
            for seed in range(701, 702 if shots == 0 else 701+args.seeds):
                started = time.perf_counter()
                with threadpool_limits(limits=1):
                    readouts = (probabilities @ readout_eigenvalues() if shots == 0 else
                                sample_readouts(probabilities, shots, np.random.default_rng(seed)))
                sample_seconds = time.perf_counter()-started
                started = time.perf_counter()
                with threadpool_limits(limits=1):
                    predicted = head.predict(np.column_stack([angles, readouts]))
                head_seconds = time.perf_counter()-started
                if shots == 0:
                    exact_prediction, exact_readouts = predicted, readouts
                # State simulation is repeated for each budget, reused only across
                # measurement seeds. Report complete timing on the first seed.
                total = pca_seconds+state_seconds+sample_seconds+head_seconds if seed == 701 else None
                records.append(dict(model=method, shots=shots, seed=seed, accuracy=float(np.mean(predicted == y)),
                                    prediction_agreement_with_exact=float(np.mean(predicted == exact_prediction)),
                                    readout_rmse_to_exact=float(np.sqrt(np.mean((readouts-exact_readouts)**2))),
                                    examples=len(y), pca_seconds=pca_seconds, statevector_seconds=state_seconds,
                                    sampling_readout_seconds=sample_seconds, classifier_seconds=head_seconds,
                                    state_cached_across_measurement_seeds=seed != 701,
                                    complete_simulation_seconds=total,
                                    circuit_latency_ms_per_input=(1000*total/len(y)) if total is not None else None,
                                    physical_shot_budget=len(y)*shots if shots else None))
                predictions[f'{method}_shots{shots}_seed{seed}'] = predicted
            save_csv(output / 'results.csv', records)
            selected = [r['accuracy'] for r in records if r['model'] == method and r['shots'] == shots]
            print(f'{method}, {shots or "exact"} shots: {np.mean(selected):.3f}, '
                  f'circuit simulation {state_seconds:.2f}s', flush=True)
    np.savez_compressed(output / 'predictions.npz', labels=y, indices=indices, **predictions)
    write_json(output / 'resources.json', {m: resource_report(m == 'iqp_entangling')
                                           for m in ('iqp_entangling', 'iqp_plain')})
    plot(records, output)


def plot(records, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method in dict.fromkeys(r['model'] for r in records):
        budgets = sorted({r['shots'] for r in records if r['model'] == method and r['shots'] > 0})
        means, stds = [], []
        for budget in budgets:
            values = [r['accuracy'] for r in records if r['model'] == method and r['shots'] == budget]
            means.append(np.mean(values)); stds.append(np.std(values, ddof=1) if len(values)>1 else 0.)
        line = ax.errorbar(budgets, means, yerr=stds, marker='o', capsize=3, label=method)
        exact = next(r['accuracy'] for r in records if r['model'] == method and r['shots'] == 0)
        ax.axhline(exact, color=line[0].get_color(), ls='--', label=f'{method}: exact')
    ax.set(xscale='log', xlabel='Measurement shots per input', ylabel='Accuracy',
           title='Frozen exact-trained heads under simulated measurement noise')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / 'shot_sensitivity.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
