# Gross code saved-shot evaluation

Run commands from `challenge-1-ml-decoder-kit` with its Python 3.12 environment. The
original `baseline.py`, `decoders.py`, circuits, environment, and `baselines.log` are
unchanged by these tools.

## Parse the active baseline log (safe during the run)

```sh
.venv312/bin/python -m evaluation.parse_baselines baselines.log --out-prefix results/baselines_parsed
.venv312/bin/python -m evaluation.plot_results results/baselines_parsed.csv --out results/plots
```

The parser writes `results/baselines_parsed.json` and `.csv`. Each run has a completion
status. Missing decoder entries are marked unfinished with empty measurement fields;
a partial final line is kept only as `partial_tail` in JSON. The log contains printed,
rounded rate, standard error, mean, and p99 only. It has no shot predictions, paired
failure count, median, or individual timings.

## Saved-shot evaluation (after baselines finish)

Generate independent train, validation, and test NPZ files for X and Z. This example
uses the kit's standard circuits and p=0.004:

```sh
mkdir -p data/p0.004
.venv312/bin/python make_dataset.py --basis X --p 0.004 --shots 20000 --seed 101 --out data/p0.004/train_X.npz
.venv312/bin/python make_dataset.py --basis Z --p 0.004 --shots 20000 --seed 102 --out data/p0.004/train_Z.npz
.venv312/bin/python make_dataset.py --basis X --p 0.004 --shots 5000 --seed 201 --out data/p0.004/validation_X.npz
.venv312/bin/python make_dataset.py --basis Z --p 0.004 --shots 5000 --seed 202 --out data/p0.004/validation_Z.npz
.venv312/bin/python make_dataset.py --basis X --p 0.004 --shots 10000 --seed 301 --out data/p0.004/test_X.npz
.venv312/bin/python make_dataset.py --basis Z --p 0.004 --shots 10000 --seed 302 --out data/p0.004/test_Z.npz
.venv312/bin/python -m evaluation.evaluate --train-x data/p0.004/train_X.npz --train-z data/p0.004/train_Z.npz --validation-x data/p0.004/validation_X.npz --validation-z data/p0.004/validation_Z.npz --test-x data/p0.004/test_X.npz --test-z data/p0.004/test_Z.npz --decoder bposd --decoder relay-bp --decoder relay-first-solution --decoder relay-first-leg-or-full --out results/eval_p0.004
.venv312/bin/python -m evaluation.plot_results results/eval_p0.004/summary.csv --out results/plots
```

`reproduce_eval.sh` runs these as separate stages for p=0.003, 0.004, or 0.005:

```sh
./reproduce_eval.sh dataset 0.004
./reproduce_eval.sh evaluate 0.004
./reproduce_eval.sh plot 0.004
./reproduce_eval.sh parse-log
```

The script does not automatically advance to another stage. Dataset and evaluation
stages can take a long time; do not run them during the active baselines.

The saved-shot runner is single-process and sequential. The kit adapters time
decoder API calls before observable projection, matching the kit's timing boundary.
The five-solution Relay-BP baseline uses the kit's plain `decode()` call, so its
internal convergence and iteration counts are unavailable in that row.
For plugins the default timer wraps `decode_shot`, so router inference and a fallback
rerun are included. Optional
`before_shot(k)` time, decoder construction, and complete evaluation wall time are
separate. Batch throughput includes setup and Python loop overhead. Parent-process
peak RSS is labeled as such; it is not an aggregate worker-memory measurement.
Unconverged results are still scored. A decoding exception fails the run.

## Decoder plugin interface

Pass `--decoder my_module:make_decoder`. The module is imported only when selected.
`make_decoder(problem, options)` returns a session with `decode_shot(full, basis)`.
`full` has 1728 detector bits, `basis` has the kit's 936 basis bits. The `problem`
contains `basis`, `H`, `L`, `priors`, `p`, and `circuit`. Truth is never provided to
the decoder. `decode_shot` returns `evaluation.adapters.ShotResult(predictions, metrics)`
with 12 binary observable predictions. The session may implement `before_shot(k)`;
this executes outside the decode timer. It may expose `setup_seconds` and `settings`.
`--plugin-options` accepts a JSON object file passed to the factory.

The learned logistic router is implemented in `evaluation.router`. Plugins must return
one result for every input shot in order and count repeated decoding inside
`decode_shot`. Useful `metrics` keys are `converged`, `iterations`,
`first_leg_converged`, and `fallback_used`.

Each decoder writes `*_shots.npz` (predictions, failure masks, actual timings) and
`*_metrics.json` (per-shot diagnostics). `summary.csv` and `summary.json` hold rates,
latency summaries, identities, configuration, package versions, and machine details.
For threshold curves, put `"threshold": 0.5` (for example) in a plugin-options
JSON file, evaluate the same held-out X/Z NPZ files at each threshold into separate
output directories, and pass all of their `summary.csv` files to
`evaluation.plot_results`. A threshold curve is drawn only for rows with an actual
threshold. The parser and plotter never fill missing runs with estimates.

## Revised comparison and router v1

`python -m evaluation.comparison_rows --out results/comparison_rows.json` writes the
eleven planned rows with separate `preprocessor_is_ml` and `router_is_ml` flags.
Rows 6–8 remain unavailable until the loss and tracker implementations arrive;
missing rows receive no scores. Row 9 has a non-ML BP preprocessor and an ML router.

The installed relay-bp 0.2.2 detailed API supplies cumulative iterations,
`success`, `posterior_ratios`, and mechanism corrections. Tiny installed-API checks
showed iterations accumulate across legs; positive posterior ratios favor an absent
fault and negative ratios favor a present fault. Router features are first-solution
iterations, convergence, mean absolute posterior ratio (clipped at 100), correction
weight, and 13 per-cycle syndrome weights. A hard shot is one whose first solution
was not found in the first 80-iteration leg. On hard shots, the logistic router
estimates `P(first candidate wrong)` and reruns the kit's five-solution Relay-BP if
that risk is greater than the frozen threshold. Easy shots accept the first result.
The full rerun is counted in total latency. No logical-flip confidence formula is
applied to failure-risk probabilities.
If a router is later trained on predecoder residuals, `router_features` accepts
`--commit-manifest` and labels its candidate after restoring `Lc`. The resulting
router model records that manifest hash; composition rejects a mismatch. Raw-shot
router training remains the row-5 default.

After the baselines finish, generate independent circuit samples and features for
all three p values and both bases. Each `router-features` stage uses the 20,000-shot
train file and the separate 5,000-shot validation file:

```sh
for p in 0.003 0.004 0.005; do ./reproduce_eval.sh dataset "$p"; done
for p in 0.003 0.004 0.005; do ./reproduce_eval.sh router-features "$p"; done
MAX_MISSED_WRONG=0 ./reproduce_eval.sh router-train 0.004
```

`MAX_MISSED_WRONG` is an explicit validation error budget. The trainer uses only
hard shots, reports detection of incorrect candidates, and freezes the largest
validation threshold satisfying that budget. It fails if the validation hard set
contains no incorrect candidate. Repeat `router-train` for the other p values after
checking their validation reports. The model JSON records training/validation
dataset hashes, matrix identity, feature schema, and threshold. It cannot load for
another basis or p.

A measured validation-only risk/latency curve can be produced later with, for
example, `./reproduce_eval.sh router-sweep 0.004 0.1 0.2 0.3` followed by
`.venv312/bin/python -m evaluation.plot_results results/router_p0.004/validation_sweep/summary.csv --out results/router_p0.004/plots`.
Each threshold executes full per-shot decoding and actual fallback reruns; the
curve is labeled validation-only and is not a held-out result.

For paired X/Z evaluation of row 5, create a JSON options file with paths to the
frozen models, for example:

```json
{"router_models":{"X":"results/router_p0.004/model_X.json","Z":"results/router_p0.004/model_Z.json"}}
```

Then run:

```sh
.venv312/bin/python -m evaluation.evaluate --test-x data/p0.004/test_X.npz --test-z data/p0.004/test_Z.npz --decoder evaluation.router:make_tiered_decoder --plugin-options results/router_p0.004/eval_options.json --out results/router_eval_p0.004
```

## Tier 0 layout, data, and frozen commits

The layout check can run with bounded CPU during the baseline run:

```sh
.venv312/bin/python -m evaluation.predecoder_layout --basis X --p 0.004
.venv312/bin/python -m evaluation.predecoder_layout --basis Z --p 0.004
```

`GrossLayout` maps the 936 detector bits to `(13,12,6)` and each of the 8,784
error-mechanism columns to a `(122,12,6)` fault grid by translating its H-column
support. It verifies uniqueness, round trips, all 72 torus translations, and prior
invariance for the actual matrices. `circular_pad_spatial` wraps only the final two
spatial axes; syndrome time is never wrapped. The model outputs probabilities or
logits for **8,784 error mechanisms**, not 144 data qubits.

An optional synthetic training/diagnostic shard is generated separately from the
kit's Stim samples:

```sh
.venv312/bin/python -m evaluation.make_dem_dataset --basis X --p 0.004 --shots 1000 --seed 10001 --shard 0 --out data/dem/p0.004/X_000.npz
.venv312/bin/python -m evaluation.oracle_diagnostic --synthetic-shard data/dem/p0.004/X_000.npz
```

The DEM shard contains sampled mechanism truth and a matrix hash. This truth is
restricted to training and the explicitly labeled oracle algebra diagnostic. It is
never passed to deployed decoder plugins. The final sweep and go/no-go screen use
separate kit circuit-sampled NPZ files.

The CNN owner supplies a lazy `module:function` model factory:

```python
model = factory(problem, model_path, model_config)
values = model.predict_faults(syndrome_grid, prior_grid, context)
```

`values` must be a finite host NumPy array of shape `(122,12,6)` or `(8784,)`.
The frozen manifest declares `output_kind` as `probability` or `logit`; a probability
threshold commits only high fault-presence probabilities. A GPU implementation
must synchronize and transfer its result before returning so batch-size-one
latency includes the device work. Expose `model.device` as `"cpu"` or `"gpu"`
so reports can distinguish the runs. `context` contains `basis`, full detectors, and
optional eight-iteration memory-BP posterior ratios. The CNN owner can train X
first and provide a separate Z model later. No CNN training code or weights are
included here.

`evaluation.predecoder:make_composed_decoder` implements rows 9–11. It uses a
validation-frozen commit manifest per basis. For each committed vector `c`, it
decodes `s XOR Hc` and returns `Lc XOR` the downstream logical prediction. The
downstream choice is `tiered` (rows 9/10) or `bposd` (row 11). All 12 observable
bits are scored, including when the first stage or fallback does not converge.
BP-confidence uses `MinSumBPDecoderF64` with `max_iter=8` and memory strength
0.15 and commits only when the posterior log-odds of *absence* is sufficiently
negative. Positive odds of absence never become a committed fault.
Here `max_iter=8` is the installed API's upper bound: the decoder stops early on
convergence, and it exposes no fixed-eight-iteration switch. Actual iterations
are recorded. A strict eight-update belief feature would need a separately
approved implementation choice.

## Validation and held-out go/no-go

`predecoder_eval.py select` opens **validation only** and freezes a threshold.
It compares candidate thresholds against the no-predecoder tiered control on the
same validation shots, keeps candidates within the supplied added-failure budget,
and selects the one with the lowest measured mean total latency. Example BP control:

```sh
.venv312/bin/python -m evaluation.predecoder_eval select --mode bp --basis X --p 0.004 --validation data/p0.004/validation_X.npz --router-model results/router_p0.004/model_X.json --thresholds 2 4 6 --max-added-failures 0 --out results/commits_p0.004/bp_X.json
```

For CNN commits, use `--mode cnn` with candidate probability or logit thresholds,
`--model-factory team_cnn:load_model`, `--model-path` to the trained X artifact,
`--model-training-manifest` recording the model hash and distinct training dataset
hashes/seeds, and `--output-kind probability` or `logit`. The training manifest
format is `{"basis":"X","p":0.004,"model_sha256":"...","training_datasets":[{"sha256":"...","seed":101}]}`.
Optional `--model-config` supplies a JSON object to the model factory.
The frozen manifest records both the model-file hash and factory-source hash.
The model path and code must exist
before that command runs. The selected manifest records the model hash and
validation data identity. Run the X p=0.004 go/no-go screen only after the BP and
CNN manifests and router model are frozen:

```sh
.venv312/bin/python -m evaluation.predecoder_eval go --test data/p0.004/go_X.npz --router-model results/router_p0.004/model_X.json --bp-manifest results/commits_p0.004/bp_X.json --cnn-manifest results/commits_p0.004/cnn_X.json --out results/go_p0.004
```

The `dataset 0.004` stage creates `go_X.npz` with 5,000 circuit shots and seed 401,
separate from train, validation, and the 10,000-shot final sweep. The go command
constructs and hashes the frozen configurations before opening this file and
rejects overlapping dataset hashes or seeds. It reports hard-shot fractions,
observed logical failures, iteration counts, actual total and kit-compatible
latencies, paired discordant outcomes, uncertainty, and parent-process peak RSS.
The promotion screen requires at least one-third fewer hard shots, no more
observed failures than row 5, fewer hard shots without more failures than the BP
control, and lower measured total latency than row 5. Passing this 5,000-shot
screen does not prove equal underlying error rates.

Total per-shot latency includes memory-BP context, CNN inference and device
transfer/synchronization, thresholding, residual transformation, router work,
downstream decoding, and reruns. Decoder construction and model training are
separate. Results from different devices must be recorded in separate output
directories. If tier 0 passes, evaluate p=0.003 and 0.005 with frozen weights and
a separate Z model. Row 5 remains a standalone comparison regardless of the
tier-0 result. A probabilities-as-priors fallback would be a single, separately
labeled experiment if tier 0 fails; its precise prior-combination rule needs an
explicit decision before implementation.

The loss owner can supply loss simulation, heralds, per-shot priors, and X/Z
support through a decoder plugin and a loss dataset extension. The current
loader covers the kit's circuit NPZ format; a herald schema is still needed.
