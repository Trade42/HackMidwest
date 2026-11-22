# QFM Token-Relationship Corpus (v1)

For the **9–15-word edition**, use [`short/`](short/README.md). It preserves
these IDs, labels and splits. Run `qfm/controlled_experiment.py --full` from
the repository root to compare word-position and semantic-cue quantum inputs.

A reproducible, **synthetic** benchmark for investigating whether a quantum feature map is useful for learning **context-dependent coreference** as the number of jointly relevant semantic factors increases.

This is **not** a naturalistic pronoun-resolution corpus or evidence that quantum models have an NLP advantage. The data are intentionally template-controlled and should be complemented with human-authored examples before broader claims are made.

## Quick start

```python
import csv

with open('train_mixed_small.csv', newline='', encoding='utf-8') as f:
    train = list(csv.DictReader(f))

x = [ (r['sentence'], r['candidate_0'], r['candidate_1']) for r in train ]
y = [ int(r['label']) for r in train ]
```

`label=0` means `candidate_0`; `label=1` means `candidate_1`. A model should return a **binary classification**, not arbitrary generated text. The `referent` column is present for convenience but **must not be supplied as an input**. Use the *same* tokenizer, sentence embeddings or semantic inputs across quantum and classical experiments. In a feature-map setup, a classical encoder converts the sentence/candidates to a shared, fixed-length numeric vector, which the quantum backend maps to a state or measurement features; a classical prediction head produces the 0/1 label.

## Split sizes

| Split | Tier 1 | Tier 2 | Tier 3 | Total |
|---|---:|---:|---:|---:|
| `train` | 360 | 360 | 360 | **1,080** |
| `validation` | 72 | 72 | 72 | **216** |
| `test` | 72 | 72 | 72 | **216** |
| `test_ood` | 48 | 48 | 48 | **144** |

**Total: 1,656 examples.** Each tier/split has exactly 50% label 0 and 50% label 1. Scenarios (noun pair + color pairing) are never shared between splits; the underlying noun vocabulary is seen in training. The `test_ood` split additionally uses paraphrases **not present in training**. These are held-out scenario and template tests, not a full lexical out-of-vocabulary test.

### What the tiers measure

**Tier 1 — 1 variable (size cue)**: The item fails to fit inside the container; if it was *too large* the pronoun refers to the item, if *too small* to the container.

> The green trophy did not fit inside the yellow suitcase because it was too large. → `green trophy`.

**Tier 2 — 2 interacting variables (relation × size)**: Fit-on versus cover, crossed with large versus small. The answer follows an XOR-like truth table, so neither relation nor adjective alone predicts the label.

| Relation | Adjective | Correct referent |
|---|---|---|
| fit on | large | surface-covering item (candidate 0) |
| fit on | small | surface (candidate 1) |
| cover | large | surface (candidate 1) |
| cover | small | surface-covering item (candidate 0) |

> The red blanket could not cover the blue bed because it was too large. → `blue bed`.

**Tier 3 — 3 interacting variables (initial comparison × swap × query)**: Two items start with a relative property (brighter/dimmer, warmer/cooler, etc.); their settings either swap or remain the same; a subsequent clause picks one of them by its **final** property. Every one of the three bits is necessary to determine the label; no single bit or pair of bits suffices in the balanced truth table.

> Initially, the green lamp was dimmer than the yellow lamp. During calibration, their brightness settings were exchanged. Afterward, the researcher inspected it because it was the dimmer one. → `yellow lamp`.

**Note:** "1/2/3 variables" refers to the number of **binary experimental factors** determining the outcome, not the number of tokens in the sentence or a proven minimal neural-network interaction order. In particular, Tier 3 intentionally includes symbolic state updates. Success may be attributed to learning simple logical functions rather than to broader semantic understanding.

## Files

- `tier_{1,2,3}_{train,validation,test,test_ood}.csv`: each tier separately; 12 files.
- `{train,validation,test,test_ood}_mixed.csv`: combined tiers; 4 files.
- `{train,validation,test,test_ood}_mixed.jsonl`: same combined examples, newline-delimited JSON; 4 files.
- `{train,validation,test}_mixed_small.csv`: balanced **small** starter subsets; 48 train / 24 validation / 24 test per tier (144 / 72 / 72 total).
- `{test,test_ood}_mixed_unlabeled.csv`: unlabeled inference sets; keep labeled test files for evaluation.
- `metadata_all.csv`: scenario IDs, template IDs, and symbolic factors for audit only. **Do not feed this file or its columns to training.**
- `dataset_manifest.json`: counts and field definitions.
- `generate_corpus.py`: reproduces the full dataset, including deterministic randomization, from seed `241121`.
- `evaluate_predictions.py`: reports accuracy by tier from an `id,prediction` CSV.

All CSVs are UTF-8, with standard headers and integer 0/1 labels. `scenario_id` is metadata to verify split separation; do not use it as a feature. `tier` is for stratification/evaluation, not an input feature.

## Recommended experimental protocol

1. Start with `train_mixed_small.csv` to keep quantum circuit-simulation costs manageable.
2. Fit **all preprocessing and classical/quantum classifiers on training only**. Tune on validation, lock hyperparameters, and evaluate once on test and `test_ood`.
3. Compare (a) classical nonlinear kernel or MLP, (b) non-entangling quantum feature map, (c) entangling quantum feature map under **identical inputs**, with multiple random seeds.
4. Report accuracy by tier, accuracy by training-set size, and time/resources (qubits, shots, depth, simulated/hardware runtime). Evaluate accuracy on `test_ood` separately.
5. Use strong classical baselines that can express pairwise and triple interactions (polynomial-kernel SVMs, an MLP, and ideally a lightweight transformer). A high-dimensional quantum space is not automatically more useful than a classical nonlinear feature map.

**Scientific caveats:** Labels are rule-derived, not human-annotated. Examples are deliberately repetitive and tightly controlled. The corpus does **not** measure quantum contextuality; to claim contextuality, run a separate operational contextuality-witness experiment. Also, entangling gates alone do not establish contextuality or quantum computational advantage. Explicitly encoded higher-order features should be made available to an appropriate classical control to test whether any benefit is genuinely quantum.

## Examples of expected model output

For a row with `sentence = 'The green trophy ... because it was too large.'`, `candidate_0 = 'green trophy'`, `candidate_1 = 'yellow suitcase'`, output `prediction = 0`.

For an OOD test row, inference code reads the same input columns, but its output should contain only `id,prediction` (0 or 1). For example, `evaluate_predictions.py your_predictions.csv --gold test_mixed.csv` evaluates accuracy by tier.
