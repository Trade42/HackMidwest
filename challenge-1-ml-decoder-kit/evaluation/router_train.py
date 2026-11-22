"""Fit a logistic hard-shot router; select and freeze its risk threshold on validation only."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from evaluation.router import FEATURE_NAMES, fit_logistic, risk, select_threshold, threshold_report


def load_features(path, split):
    with np.load(path, allow_pickle=False) as file:
        metadata = json.loads(str(file["metadata"].item()))
        features = file["features"]
        wrong = file["wrong"]
        easy = file["first_leg_converged"]
    if metadata["split"] != split or metadata["feature_names"] != list(FEATURE_NAMES):
        raise ValueError(f"{path}: feature split or schema mismatch")
    if features.shape != (len(wrong), len(FEATURE_NAMES)) or easy.shape != wrong.shape:
        raise ValueError(f"{path}: feature arrays have inconsistent shapes")
    return metadata, features, wrong.astype(bool), easy.astype(bool)


def train(train_path, validation_path, max_missed_wrong, out):
    train_meta, train_features, train_wrong, train_easy = load_features(train_path, "train")
    val_meta, val_features, val_wrong, val_easy = load_features(validation_path, "validation")
    for field in ("basis", "p", "matrix_sha256", "matrix_semantic_sha256", "commit_manifest_sha256"):
        if train_meta.get(field) != val_meta.get(field):
            raise ValueError(f"train/validation {field} mismatch")
    if (train_meta["dataset"]["sha256"] == val_meta["dataset"]["sha256"] or
            train_meta["dataset"]["seed"] == val_meta["dataset"]["seed"]):
        raise ValueError("train and validation must use distinct datasets and seeds")
    train_hard, val_hard = ~train_easy, ~val_easy
    start = perf_counter()
    model = fit_logistic(train_features[train_hard], train_wrong[train_hard])
    training_seconds = perf_counter() - start
    scores = risk(model, val_features[val_hard])
    threshold = select_threshold(scores, val_wrong[val_hard], max_missed_wrong)
    validation = threshold_report(scores, val_wrong[val_hard], threshold)
    model.update({"basis": train_meta["basis"], "p": train_meta["p"],
                  "matrix_sha256": train_meta["matrix_sha256"],
                  "matrix_semantic_sha256": train_meta.get("matrix_semantic_sha256"),
                  "threshold": threshold,
                  "commit_manifest_sha256": train_meta.get("commit_manifest_sha256"),
                  "train_dataset": train_meta["dataset"], "validation_dataset": val_meta["dataset"],
                  "training_seconds": training_seconds,
                  "train_hard_shots": int(train_hard.sum()), "train_hard_wrong": int(train_wrong[train_hard].sum()),
                  "validation_hard_shots": int(val_hard.sum()), "validation": validation,
                  "selection_rule": "largest validation risk threshold with at most max_missed_wrong hard candidate errors accepted",
                  "max_missed_wrong": max_missed_wrong})
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out}: frozen router already exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model, indent=2))
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", required=True)
    parser.add_argument("--validation-features", required=True)
    parser.add_argument("--max-missed-wrong", type=int, required=True,
                        help="validation budget for hard wrong candidates accepted without fallback")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    model = train(args.train_features, args.validation_features, args.max_missed_wrong, args.out)
    print(json.dumps({"output": args.out, "validation": model["validation"]}))


if __name__ == "__main__":
    main()
