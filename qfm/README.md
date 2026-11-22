# Quantum feature map corpus experiments

## Frozen DistilBERT benchmark

The current NLP pipeline uses the frozen
[DistilBERT base uncased checkpoint](https://huggingface.co/distilbert/distilbert-base-uncased),
revision `12040accade4e8a0f71eabdb258fecc2e7e948be`. It mean-pools content-token
representations from the sentence plus ordered candidate names, excluding
padding and inserted special tokens. No labels, scenario IDs or handcrafted
meaning cues enter the encoder. Inputs exceeding 128 tokens fail explicitly;
they are never silently truncated.

For each labeled training subset, PCA is fitted to its 768-dimensional
embeddings, retaining eight components. The existing robust angle scaler is
also fitted on that subset, with angles bounded by ±0.55. All reduced-input
models receive these same eight values. Quantum heads retain the angles and
append 36 Z/ZZ readouts. Circuit depth, connectivity and scale are fixed;
there is no circuit search. The full-dimensional RBF baseline bypasses PCA.

### Recorded results

Each task compares eight models at 128 and 256 labeled examples with three
paired training-subset seeds and eight classifier candidates per model. Models
are selected on validation only and frozen before test scoring. The table gives
mean accuracy at **256 training examples**:

| Model | Numeric comparisons: 506 questions, unseen worlds | Pronouns: 144 held-out paraphrases |
| --- | ---: | ---: |
| Entangling QFM + RBF | 50.72% | 50.93% |
| Non-entangling QFM + RBF | 51.05% | 51.62% |
| Eight-dimensional RBF | 50.13% | 50.00% |
| Eight-dimensional polynomial SVM | 50.66% | 49.77% |
| Eight-dimensional MLP | 51.12% | 50.46% |
| Full 768-dimensional RBF | 52.64% | 63.66% |
| Classical Fourier features + RBF | 50.40% | 49.77% |
| Classical surrogate of QFM + RBF | 49.80% | 49.54% |

**The earlier handcrafted-cue improvement did not survive this encoder
replacement. No quantum accuracy or sample-efficiency advantage is demonstrated.**
The full-embedding pronoun baseline has a higher observed mean; the adjusted
world-cluster intervals remain wide, so this small run does not establish a
precise population-level gap. The three-seed results are exploratory.

PCA retains about 95.4% of embedding variance for numeric comparisons and 64.0%
for pronouns at n=256. Retained variance is not a measure of retained label
information. Both the reduced classical and quantum models perform near chance.
Some MLP hyperparameter candidates report convergence warnings; these are
recorded, not discarded. Full results, standard deviations and adjusted paired
bootstrap intervals are in each run directory.

Original split assignments and CSVs are preserved. For the numeric task, 40 of
128 validation rows share a world with training and are excluded from model
selection, leaving 88. Both the original 128-row test and its 73-row strictly
disjoint subset are reported. The primary stronger test uses the existing 506
questions about 253 worlds absent from the original corpus. These corpora were
already inspected in earlier handcrafted-feature experiments; they are not
newly collected untouched data. No test score selects a model in this run.
The second task reuses the teammate's short pronoun corpus and its fixed
train/validation/test/test_ood assignments. Its OOD split holds out paraphrase
templates, not all underlying semantic rules.

### Run and reproduce

Install the optional NLP dependencies into the existing environment (CPU wheels;
the checkpoint download and installation happen once):

```bash
uv pip install --python .qfm-venv/bin/python -r qfm/requirements-nlp.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

Reproduce the bounded comparison in a **new output directory**:

```bash
.qfm-venv/bin/python qfm/distilbert_experiment.py \
  --extra-test qfm/results/feature_confirmation/dataset.csv \
  --output qfm/results/distilbert_numeric_reproduction

.qfm-venv/bin/python qfm/distilbert_experiment.py \
  --source quantum_token_corpus/short \
  --output qfm/results/distilbert_pronouns_reproduction
```

Each command fits 48 final heads (384 classifier candidates), with a fixed
circuit, two training sizes and three seeds. Frozen embeddings are cached;
initial encoding took about 35 seconds for 1,530 numeric rows and 44 seconds for
1,656 pronoun rows on this CPU. Classifier work and circuit simulation add time.
Use `--sizes 128 --seeds 1` for a smaller integration run, not a significance test.
`--stage prepare`, `--stage fit`, and `--stage evaluate` can be run separately.
After preparation, fitting/evaluation read the saved protocol rather than new
CLI size/method values. A frozen run refuses refitting in the same directory.

Recorded runs are `results/distilbert_numeric/` and
`results/distilbert_pronouns/`. Each contains the original assignments, embedding
arrays, encoder revision, protocol, environment versions, training-subset IDs,
validation scores, hashed frozen model bundles, predictions, test results,
summary, paired intervals, surrogate errors, and `learning_curves.png`.
`encoding_audit.json` verifies no repeated full embeddings or PCA inputs across
any split for all six fitted reducers. Numeric feature equality is checked to
eight decimal places.

The exported `embeddings.npz` includes the existing
`train_X/train_ids`, `validation_X/validation_ids`, etc. interface. It remains
compatible with `corpus_experiment.py --embeddings ... --qubits 8` and the
existing ID/order checks. The original experiments remain available below.

### Measurement shots and resource accounting

```bash
.qfm-venv/bin/python qfm/distilbert_shots.py \
  --run qfm/results/distilbert_numeric \
  --output qfm/results/distilbert_numeric/shots_diagnostics_reproduction
```

This compares exact expectations with 100, 1,000 and 10,000 shots, using three
measurement seeds and the frozen n=256/seed=301 model. The head was trained on
exact features and is not retuned for shot noise. All Z/ZZ features come from
one shared computational-basis histogram per input; they do not receive
independent, artificial feature noise.

The simulator samples multinomial Born counts, mathematically equivalent to
independent circuit shots. Its sampling cost need not scale linearly with the
physical shot budget. `results.csv` separates statevector, sampling, PCA and
classifier times. Complete timing is measured on the first seed per budget;
the other seeds reuse probabilities. Latencies are batched CPU wall times,
exclude the cached language encoder, and are **not hardware runtime**.

The entangling circuit has **8 qubits, 56 logical ZZ gates, 112 CNOT gates,
208 total gates and depth 68** after H/RZ/CNOT decomposition, assuming all-to-all
connectivity without routing. The non-entangling control has 40 gates, depth 5,
and zero two-qubit gates. Both use 36 readouts from one measurement setting.

For numeric comparisons, agreement with exact predictions rises from **63.1%**
at 100 shots to **93.2%** at 1,000 and **97.6%** at 10,000. For pronouns the
corresponding values are 78.0%, 93.8%, and 98.6%. Near-chance task accuracy alone
would conceal these substantial changes. The exact heads themselves remain
near chance. These runs include measurement noise only, not device gate noise.
Recorded diagnostics are in `distilbert_numeric/shots_diagnostics/` and
`distilbert_pronouns/shots/`, including resource JSON and a sensitivity plot.

### Classical Fourier control and QFM surrogate

`nlp_features.py` implements fixed low harmonics of the single-qubit and ZZ
gate phases plus pairwise sine/cosine products. The Fourier-only classifier
needs no quantum teacher. The separate ridge surrogate approximates the 36 IQP
readouts from these features, using **only training-input quantum outputs**.
Its inference is fully classical; its training still incurs the teacher feature
cost, which is recorded separately.

At n=256, mean held-out readout RMSE is approximately **0.0213** for numeric
comparisons and **0.0200** for pronouns. Surrogate classification accuracy is
also near chance. These results demonstrate a small-domain approximation of the
simulated features, not a general efficient simulation theorem or a useful
quantum-specific prediction benefit. Timings and both training sizes are saved
alongside the quantum and classical baselines.

### End-to-end interface

```bash
.qfm-venv/bin/python qfm/predict_distilbert.py \
  --run qfm/results/distilbert_pronouns \
  --sentence 'The trophy cannot fit inside the suitcase; it is too large.' \
  --candidate-0 trophy --candidate-1 suitcase
```

This prints quantum and classical predictions side by side. `--json` provides
the same interface for application integration, including separate encoder,
PCA, feature-map and classifier timings. It loads the largest declared training
size and first seed, selected before test evaluation. The frozen model, PCA and
angle scaler are reused; prediction performs no fitting. A recorded invocation
is in `results/distilbert_pronouns/demo_prediction.json`.

Tests cover label exclusion, candidate ordering, padding masks, the real frozen
checkpoint, training-only PCA/scaling/surrogate fitting, unchanged models after
test-data perturbation, artifact tampering, shared shot correlations, circuit
equivalence, gate counts and classical-only surrogate inference:

```bash
QFM_TEST_PRETRAINED=1 .qfm-venv/bin/python -m unittest discover \
  -s qfm -p 'test_distilbert_pipeline.py' -v
.qfm-venv/bin/python -m unittest discover -s qfm -p 'test_corpus_experiment.py' -v
```

The contextuality memory demonstration is independent of this NLP benchmark.

## Earlier findings with handcrafted cues

The numerical failure in the word-position model is fixed (details below).
Small validation-only searches then compared circuit scales, depths, full
probabilities, X/Y/Z readouts, classical feature maps, and converged neural
readouts. The selected quantum map retains the eight input cues and adds eight
Z and 28 ZZ readouts, with two IQP repetitions at angle scale 0.55.

At **128 training examples**, averaged over 16 paired subset seeds, the stricter
evaluation uses **506 questions about 253 worlds absent from the original
corpus**:

| Model | Accuracy | Quantum gain, adjusted interval |
| --- | ---: | ---: |
| Selected entangling map + RBF SVM | 80.50% | — |
| Matched non-entangling map, evaluated analytically | 78.67% | +1.83 points [+0.11, +3.52] |
| Best validation-selected classical feature map | 79.27% | +1.22 points [−1.21, +3.71] |
| Converged classical MLP (L-BFGS) | 80.36% | +0.14 points [−3.92, +4.36] |

This supports a benefit from entangling gates against the matched ablation
within this experiment. It does **not establish an advantage over the stronger
classical controls**, or fewer training examples/epochs to reach equal accuracy.
At 64 examples, every gain interval includes zero. These fixed feature maps
have no trained quantum parameters; counting their quantum training epochs as
zero would give a misleading comparison.

Models were chosen on validation data, then frozen for the confirmation run.
The first confirmation used 1,024 unused configurations, including questions
whose opposite query appeared in the original corpus. The subsequent audit
keeps related questions together in the bootstrap and also reports only
entirely unseen worlds. It reuses saved predictions without fitting models.
Intervals use 10,000 paired bootstrap draws over training seeds and world
clusters, adjusted across all 12 reported comparisons. Coverage is approximate
and conditional on this synthetic population, training pool and validation set.
The statistical correction was made after viewing the first confirmation.

Files:

- `results/feature_confirmation/`: frozen protocol, per-seed scores and predictions.
- `results/feature_confirmation_clustered/`: corrected analysis, stricter cohort,
  amendment record and `paired_gains.png`.
- `results/feature_search*/`: validation screens, including unsuccessful variants.

Recompute the corrected statistics from saved predictions:

```bash
.qfm-venv/bin/python qfm/audit_feature_gain.py
```

A bounded validation screen (two cached maps, three subsets, one training size):

```bash
.qfm-venv/bin/python qfm/feature_search.py \
  --only quantum_r2_s0.55 classical_product_s0.55 \
  --cache-dir qfm/results/feature_search/features \
  --output qfm/results/feature_search_recheck
```

`confirm_feature_gain.py` reproduces the frozen four-model comparison at two
training sizes over 16 subsets. It runs 1,024 small classifier fits; this is
larger than the statistics-only command above. Existing confirmation cases have
already been inspected and should not be described as a new untouched test set.
Small optimizer fits use one BLAS thread to avoid thread-dispatch overhead.

## Contextuality: a separate delayed-question memory experiment

`contextual_memory.py` reproduces the two-bit parity-oblivious multiplexing
protocol from [Spekkens et al.](https://arxiv.org/abs/0805.1463). This changes
the resource constraint and task: it is an illustrative memory experiment,
not a replacement score for the original classifier benchmark.

1. Read a nine-word sentence: **Ava is taller than Mia but younger than Mia.**
2. Extract two independent cues and prepare one qubit before seeing the query.
3. Later receive either **Choose her: taller.** or **Choose her: older.**
4. Measure once in the requested X or Z basis and return Ava or Mia.

The four sentences and two questions are uniform. The message must conceal
the XOR of the two cues: whether the answers to the two questions differ.
Even- and odd-parity mixtures have exactly the same density matrix, so **no
measurement** can recover that parity better than chance. Under this
constraint, the optimal classical accuracy is 75%; one qubit achieves
`(1 + 1/sqrt(2))/2 = 85.3553%`. This is preparation contextuality; entanglement
is not required. The classical bound is checked by a linear program over all
four possible deterministic response assignments and stochastic encoders.

The default simulation gives **85.38% across 4,096 fresh single-shot trials**,
with a 95% exact binomial interval of **84.26–86.44%**. These error bars describe
simulated measurement randomness, not uncertainty about NLP generalization.
An independent PennyLane circuit agrees with the analytic probabilities to
`2.1e-14`. Removing coherence eliminates the violation; an optimally re-encoded
commuting control reaches 75%. An unrestricted classical control retains both
cues and answers **100%** correctly. The parity restriction is essential.

```bash
.qfm-venv/bin/python qfm/contextual_memory.py
```

The run uses eight one-qubit circuits and vectorized sampling; it takes seconds,
including imports and plotting. Outputs are in `results/contextual_memory/`:
`protocol.json`, `results.json`, `sentences.csv`, `simulation.npz`, and
`contextual_memory.png`/`.svg`. Optional `--visibility 0.9` mixes the state with
10% maximally mixed noise. The predicted advantage disappears at a 29.29%
mixture weight for this specified noise model.

The simulator has access to density matrices and probability tables for
verification. The operational decoder receives one qubit, one query and one
measurement outcome. Giving it both expectation values, allowing it to inspect
amplitudes, or sending repeated copies would change the available resources.
Taking the argmax of exact probabilities would likewise change the protocol.
This experiment has no training and demonstrates no hardware speedup.

### What contextuality suggests for the original model

The existing IQP Z/ZZ readouts commute. Adding X/Y/Z readouts alone supplies
more classical features; it does not certify contextuality. That feature
expansion and a neural readout were both tried on validation data without an
improvement over the selected model.

[Bowles et al., Contextuality and inductive bias in quantum machine learning](https://arxiv.org/abs/2302.01365)
connect operational equivalences to constrained multitask probability models,
including a conservation law in the outputs. Applying that idea would require
a task with those constraints and matched classical controls. Our unrestricted,
deterministic pronoun labels do not impose the parity-hiding condition above.
In particular, suppressing joint parity information can destroy information
needed for a question that combines both cues. Semantic context and quantum
contextuality are distinct concepts.

Focused checks for these mechanisms:

```bash
.qfm-venv/bin/python -m unittest discover -s qfm -p 'test_contextual_memory.py' -v
.qfm-venv/bin/python -m unittest discover -s qfm -p 'test_feature_statistics.py' -v
```

## Learning curves on unseen meaning combinations

`learning_curves.py` generates 1,024 distinct controlled configurations and
measures sample efficiency over 8, 16, 32, 64, 128 and 256 labeled training
examples, using five paired training-subset seeds. All sentences have 14 words,
including numbers. Example:

> Ava starts 3 points above Mia; loses 4 unless blocked. Blocked; choose her, higher.

The label depends on the initial numerical gap, direction, change amount,
gain/loss, if/unless condition, blocked/unblocked state, higher/lower query,
and candidate ordering. Odd initial gaps and even changes prevent ties. The
generator has 2,048 distinct semantic configurations, with balanced targets.
The default run samples 1,024 and fixes 768 train-pool, 128 validation, and 128
test cases. It then draws nested balanced training subsets from the train pool.
New runs keep both higher/lower questions about each underlying world in the
same split, while preserving exact class balance and split sizes. Worlds and
encoded vectors must be disjoint across the three splits; the runner checks
these invariants before fitting classifiers. `--split-unit configuration`
reproduces the older split, whose identifiers included the question itself.

```bash
uv pip install --python .qfm-venv/bin/python -r qfm/requirements.txt
.qfm-venv/bin/python qfm/learning_curves.py --layouts meaning words --seeds 5
```

Both layouts retain the same eight input clues: eight meaning qubits, or 15
word-position qubits. An explicit parser extracts them from text and candidates;
it does not read the labels or calculate the final comparison. This remains a
controlled feature-map experiment rather than a learned NLP encoder.

The six comparisons are RBF SVM, polynomial SVM (degrees up to six), MLP,
classical interaction features, non-entangling IQP, and entangling IQP. Every
method gets the same labeled subsets, eight validation hyperparameter trials,
and training-only feature selection and standardization. Features with training
variance at or below `1e-24` (standard deviation `1e-12`) are removed before
scaling. This fixed numerical tolerance prevents simulator roundoff in zero
readouts from becoming unit-variance noise. Test scores do not select parameters.
Quantum maps use exact simulator expectations and all-pairs interactions.

New outputs go to `qfm/results/learning_curves_worlds/`: `dataset.csv`, `runs.csv`,
`summary.csv`, `protocol.json`, and `learning_curves.png`. Shading is one
standard deviation across training-subset seeds; the held-out test split is
fixed. The CSV logs MLP convergence warnings rather than silently treating an
unfinished optimization as a strong baseline. Fixed quantum features are cached
across seeds and training sizes, and their generation cost is reported separately
from classifier fitting/tuning. Cached fitting time is not end-to-end quantum
runtime. Quantum circuits have no trained parameters, so this run does not claim
an epoch-count comparison.

For a smaller first run, use `--layouts meaning --seeds 2 --sizes 8 32 128`.
Use a different `--dataset-seed` to resample the finite corpus and split;
use `--output` to keep those runs in separate directories. This changes the
sample, not the underlying finite set of possible worlds, so changing the seed
does not create a new untouched benchmark after tuning against that population.

### Quick checks without more circuit simulations

Choose `--methods` and reuse existing raw features in a separate output folder.
This command reproduces the historical corrected check on its original splits:

```bash
.qfm-venv/bin/python qfm/learning_curves.py \
  --layouts meaning words --methods iqp_plain iqp_entangling \
  --seeds 5 --sizes 256 --split-unit configuration \
  --cache-dir qfm/results/learning_curves/features --cached-only \
  --output qfm/results/learning_curves_fixed_check
```

This runs only 20 classifier configurations (eight validation fits each).
Omit `--split-unit configuration` and choose a new output folder for the
stronger world-disjoint split; raw feature caches remain valid for either split.
`--cached-only` fails before fitting if any requested cache is missing; it cannot
start a new simulation. Raw-feature cache keys are unchanged by the preprocessing
fix. `runs.csv` records the number of features retained in each training subset.
When simulations are requested, the runner prints circuit counts and elapsed
time during generation. Recorded cached generation time is the original cost,
not time spent in the current run.

On the default exact simulator, wires whose input angles are exactly zero for
every row are now factored out. Their Z and ZZ outputs are reconstructed
analytically, including the repeat-count parity. For this word-position dataset,
the same 15 logical qubits and 30 output features require only an eight-qubit
simulation. Finite-shot runs and custom devices retain the original execution
path. New cache metadata records `simulated_qubits` separately from the logical
`qubits`; old caches retain the original generation timings.

The original word-position quantum results used 20 essentially zero readout
columns whose floating-point errors were amplified by standardization. Treat
those original scores as affected by this preprocessing bug. Corrected outputs
are kept separately; the original CSVs and feature caches are preserved.

The focused corrected check (256 training examples, five paired seeds) gives
word-position test accuracy of 79.7% without entangling gates and 71.9% with
them, versus the original 51.4% and 49.5%. Meaning-layout results are unchanged
at 88.6% and 78.9%. This fixes the numerical failure, without establishing a
quantum advantage. The word layout has eight variable single-qubit readouts and
only two variable ring-pair readouts; the meaning layout has eight of each.

Focused regression checks:

```bash
.qfm-venv/bin/python -m unittest discover -s qfm -p 'test_feature_preprocessing.py' -v
```

## Short sentences: word positions versus meaning cues

The current controlled experiment uses `quantum_token_corpus/short`: 1,656
examples with **9–15 words** each. It compares two input layouts:

- **Word positions:** 15 qubits, one slot per word position. Relevant clues get
  explicit angles; other words and padding get zero. Word identities beyond
  these clues are not embedded.
- **Meaning cues:** five qubits for fit/cover, large/small, initial comparison,
  swap/no swap, and final comparison. Absent clues get zero.

Both layouts use the same explicit lexical parser. It extracts clues without
computing the answer and without reading labels or metadata. These are controlled
grammar encodings, not pretrained language-model embeddings. Tests verify that
all rule-relevant factors are recoverable from every shortened example, and
that neither input encoding merges examples with different labels.

Set up and run from the repository root:

```bash
uv venv .qfm-venv --python 3.12
uv pip install --python .qfm-venv/bin/python -r qfm/requirements.txt

# 144 training rows, 72 validation, 72 test, 144 paraphrase test
.qfm-venv/bin/python qfm/controlled_experiment.py

# 1,080 training rows, 216 validation, 216 test, 144 paraphrase test
.qfm-venv/bin/python qfm/controlled_experiment.py --full

.qfm-venv/bin/python -m unittest discover -s qfm -p 'test_*.py' -v
```

Results are written to `qfm/results/short_small/` or `qfm/results/short_full/` as
`results.json` and `summary.csv`. The JSON records per-tier accuracy, training
accuracy, validation choices, feature generation time, qubits, gates, circuit
depth, software versions, data hashes, and how often evaluation inputs exactly
match an encoded training input.

Every layout is evaluated with an RBF SVM, degree-3 polynomial SVM, classical
interaction features, non-entangling IQP features, and entangling IQP features.
All use the same splits and the same validation search over C = 0.1, 1, 10, 100.
Near-constant feature removal and standardization are fitted on training only.
No SVD/PCA is
used; fixed positive angles avoid the sign alias in the original non-entangling
readout. The simulator uses exact expectations (`shots=None`).

The controlled experiment defaults to an interaction between every pair of
qubits so separated word clues can interact. Z and ring-ZZ readouts are preserved.
Use `--topology ring` to run the original ring topology. With zero-valued filler
words, a ring can leave the actual clues disconnected. `qfp.py` retains its
original ring default for other callers.

Recorded full-corpus results (before numerical-feature filtering was added):

| Layout | Qubits | Model | Test | Paraphrase test |
|---|---:|---|---:|---:|
| Word positions | 15 | RBF SVM | 100% | 84.7% |
| Word positions | 15 | Polynomial SVM | 100% | 86.1% |
| Word positions | 15 | Non-entangling IQP + SVM | 100% | 84.7% |
| Word positions | 15 | Entangling IQP + SVM | 100% | 86.1% |
| Meaning cues | 5 | All five models | 100% | 100% |

The regular test inputs all match encoded training inputs after nouns and
filler wording are discarded. The meaning layout has only **12 distinct inputs**,
all seen in training; the positional layout has 32 training inputs, and 69.4% of
its paraphrase test rows match a training input. These results establish that
the controlled pipeline can retain and learn the clues. They do not show unseen
semantic-combination generalization or a quantum advantage over the classical
controls. Paraphrase scores are conditional on the hand-written parser knowing
the supported words and grammar.

Entangling feature generation over all four full splits took about 26.7 seconds
for 15 qubits and 0.033 seconds for five qubits on this machine. These are simulator
wall times with duplicate inputs evaluated once per split, not hardware latency
or measured quantum speedups. JSON separates encoder, feature, and SVM timing.

## Original TF-IDF or external-embedding experiment

`qfp.py` is the existing IQP feature extractor: a train-fitted angle scaler,
ring-connected IQP circuit, 12 Z/ZZ readouts at six qubits, then an RBF SVM.
`corpus_experiment.py` uses the teammate's fixed corpus splits and compares that
extractor with a non-entangling circuit and classical controls.

Run the original corpus experiment after the setup above:

```bash
.qfm-venv/bin/python qfm/corpus_experiment.py
.qfm-venv/bin/python qfm/corpus_experiment.py --eval test --predictions /tmp/qfm-test.csv
.qfm-venv/bin/python quantum_token_corpus/evaluate_predictions.py /tmp/qfm-test.csv --gold quantum_token_corpus/test_mixed_small.csv
```

The default uses the balanced starter subset: 144 training examples and 72
validation examples. Use `--full --eval test_ood` for the full corpus and
paraphrase test. Fit preprocessing on training rows only; use validation for
model choices and reserve test and test_ood for final reporting.

The runner uses the existing **TF-IDF/SVD placeholder** as the shared input
encoder for every model unless `--embeddings` is supplied. This placeholder
does not test the team's frozen language-model embeddings or support claims
about semantic understanding or quantum advantage. On the starter validation
split, it gives chance-level accuracy even on training rows; use it only to
check the pipeline.

For external frozen embeddings, write an `.npz` file with `train_X`,
`train_ids`, and the selected evaluation split's arrays, for example
`validation_X` and `validation_ids`. The ID arrays must match the corresponding
CSV row order exactly. Use `--embeddings path/to/embeddings.npz`; a PCA reducer
is fit on training embeddings only if their width exceeds the qubit count.
The same reduced input goes to every quantum and classical model.

The quantum interface also accepts external numerical embeddings through
`QuantumPronounClassifier.fit_encoded(X_train, y_train)` and
`predict_encoded(X_eval)`, where each row has `n_qubits` numerical features.

The corpus contains synthetic, rule-derived sentences. Its held-out scenarios
reuse the training noun vocabulary; `test_ood` additionally holds out
paraphrase templates. Neither split is a natural-language benchmark or a full
unseen-vocabulary test. IDs, tiers, scenario IDs, referents, template IDs, and
the factor metadata are excluded from model inputs.
