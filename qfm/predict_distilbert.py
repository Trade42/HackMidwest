"""End-to-end sentence/candidate inference with frozen quantum and classical heads."""

import argparse
import json
from pathlib import Path
import time

import joblib
from threadpoolctl import threadpool_limits

from distilbert_experiment import ROOT, load_frozen
from frozen_encoder import FrozenDistilBertEncoder
from nlp_features import exact_features


def predict(run, sentence, candidate_0, candidate_1):
    frozen = load_frozen(run)
    bundle = joblib.load(run / frozen['selected_application_bundle'])
    spec = bundle['encoder']
    encoder = FrozenDistilBertEncoder(model_id=spec['model_id'], revision=spec['revision'],
                                     max_length=spec['max_length'])
    started = time.perf_counter()
    encoder.load()
    load_seconds = time.perf_counter()-started
    row = dict(sentence=sentence, candidate_0=candidate_0, candidate_1=candidate_1)
    embeddings, info = encoder.encode([row])
    started = time.perf_counter()
    angles = bundle['reducer'].transform(embeddings)
    pca_seconds = time.perf_counter()-started
    features, times, _ = exact_features(angles, bundle['methods'], surrogate=bundle['surrogate'])
    if 'full_rbf' in bundle['methods']:
        features['full_rbf'], times['full_rbf'] = embeddings, 0.
    predictions = []
    for method, head in bundle['heads'].items():
        started = time.perf_counter()
        with threadpool_limits(limits=1):
            candidate = int(head.predict(features[method])[0])
        predictions.append(dict(model=method, candidate_index=candidate,
                                candidate=(candidate_0, candidate_1)[candidate],
                                feature_seconds=times[method], head_seconds=time.perf_counter()-started))
    return dict(input=row, predictions=predictions, checkpoint_load_seconds=load_seconds,
                encoder_seconds=info['current_seconds'], pca_seconds=pca_seconds,
                model_artifact=frozen['selected_application_bundle'],
                quantum_execution='Exact CPU simulator; no hardware timing',
                preprocessing='Frozen DistilBERT and saved training-only PCA/scaler')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'qfm/results/distilbert_numeric')
    parser.add_argument('--sentence', required=True)
    parser.add_argument('--candidate-0', required=True)
    parser.add_argument('--candidate-1', required=True)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    report = predict(args.run, args.sentence, args.candidate_0, args.candidate_1)
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(f'{"Model":20s} {"Prediction":20s} {"Feature + head (ms)":>20s}')
    for row in report['predictions']:
        elapsed = 1000*(row['feature_seconds']+row['head_seconds'])
        print(f'{row["model"]:20s} {row["candidate"]:20s} {elapsed:20.2f}')
    print(f'Encoder: {1000*report["encoder_seconds"]:.1f} ms; '
          f'PCA: {1000*report["pca_seconds"]:.1f} ms; '
          f'checkpoint load: {report["checkpoint_load_seconds"]:.2f} s')
    print(report['quantum_execution'])


if __name__ == '__main__':
    main()
