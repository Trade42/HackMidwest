# ML-first decoder

A fast ML layer in front, a confidence check, and the real decoder as the fallback. Same shape as the DuQuantum
"ML telescope", moved to the Gross code.

1. A CNN reads the 936-bit basis syndrome (13 rounds on the 12 x 6 torus) and predicts the 12 logical bits.
2. Confidence is the smallest |logit| of the 12. At or above a frozen threshold, the CNN's answer is final.
3. Every other shot goes to relay-BP. By default that is the kit's relay-BP (`decoders.RELAY_SETTINGS`: a first leg
   of 80 iterations, legs of 60, stop at 5 solutions) with seed 1. A faster schedule is an option (`--relay` in
   `ml_first.py`, `"relay"` in the plugin's options). Choose one only on validation shots, per p and per basis, with
   paired counts against the kit's schedule. The one faster schedule measured so far, 40 / 30 / stop at 1, costs
   accuracy (see "Measured so far").
4. Planned, not built: shots unsolved at a leg cap go to TCBP-JOBS on the GPU.

If the CNN is weak, a high threshold sends every shot to relay-BP. That costs one CNN inference per shot, and the
failures are those of the kit's relay-BP at seed 1 on the same shots. The evaluator's `relay-bp` row uses relay-bp's
default seed, 0, and relay-BP's results depend on its seed and on the order of the shots, so the two are close, not
equal.

## Files

| File | What it does |
| --- | --- |
| `train_logical_cnn.py` | Trains the CNN. Samples unlimited data from the kit's error model (DEM-sampled shots) |
| `ml_first.py` | CNN + threshold + relay-BP on real kit shots. Prints the threshold sweep |
| `plugin.py` | The same decoder as a plugin for `evaluation.evaluate` |
| `relay_knobs.py` | Sweeps relay-BP's schedule on kit shots |
| `relay_schedule_paired.py` | The kit's relay-BP schedule against 40 / 30 / stop at 1 on the same shots, paired |
| `stim_seed_check.py` | Whether a Stim seed gives the same shots (see the rules at the end) |
| `test_ml_decoder.py` | Tests of the checkpoint, the plugin's fallback settings and the trainer's seed |
| `requirements.txt` | The packages these scripts need on top of the kit's, and which script needs which |
| `ml_options_p0.0001.json` | The plugin options of the evaluator run below |
| `logs/` | The output behind every number in this README, one file per command |
| `diagnostics/` | The predecoder studies that led here. Oracle and DEM-sampled, never decoder results |

Row 12 (NVIDIA's TCBP-JOBS) is `tcbp_jobs_baseline.py` in the kit folder, on `main` since `59a33aa`.

Run everything from `challenge-1-ml-decoder-kit/`. The scripts with options (`train_logical_cnn.py`, `ml_first.py`,
`relay_schedule_paired.py`) find the kit one folder up from themselves. `relay_knobs.py` and the scripts in
`diagnostics/` take the kit folder as their first argument: pass `.`.

## Install

```sh
pip install -r requirements.txt -r ml_decoder/requirements.txt
```

`ml_decoder/requirements.txt` pins torch 2.15.0. Without a GPU, take the CPU build:
`pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu`.

## Commands

```sh
# training on the GPU, limited by time: prints a report every 10 minutes and saves model_X.pt
python ml_decoder/train_logical_cnn.py --basis X --hours 8 --seed 1
python ml_decoder/train_logical_cnn.py --basis Z --hours 8 --seed 1
# a repeatable CPU check, limited by shots (see "What is tested" below)
python ml_decoder/train_logical_cnn.py --p-train 0.0001 --p-report 0.0001 --width 32 --depth 2 --batch 256 \
    --train-shots 500000 --report-shots 100000 --seed 1 --threads 4 --basis X --out model_X_p0.0001.pt

# threshold sweep on kit shots. Pick the threshold on one seed, report on another
python ml_decoder/ml_first.py --basis X --model model_X.pt --p 0.004 --shots 5000 --seed 1

# paired evaluation with the repository's evaluator, on test shots that tuned nothing.
# data/p0.004/test_{X,Z}.npz are the team's committed test files: use them as they are. Never write
# make_dataset.py output over them: a Stim seed gives other shots on another machine, and the
# paired comparisons need the committed files. To sample your own shots, write them elsewhere:
#   python make_dataset.py --basis X --p 0.004 --shots 10000 --seed 301 --out runs/my_shots/test_X.npz
#   python make_dataset.py --basis Z --p 0.004 --shots 10000 --seed 302 --out runs/my_shots/test_Z.npz
python -m evaluation.evaluate --test-x data/p0.004/test_X.npz --test-z data/p0.004/test_Z.npz \
    --decoder ml_decoder.plugin:make_decoder --plugin-options ml_options.json --out results/ml_first_p0.004
# ml_options.json: {"models": {"X": "model_X.pt", "Z": "model_Z.pt"}, "threshold": 6.0}
# optional keys: "device" (default cuda when torch sees a GPU, else cpu), "threads" (torch CPU threads) and
# "relay" (relay-BP settings to replace; without it, the kit's settings with seed 1)

# row 12, in the kit folder: needs Linux, an NVIDIA GPU and cudaq-qec 0.9.0 or later
python tcbp_jobs_baseline.py --p 0.004 --shots 10000

# relay-BP schedule sweep: X memory only, Stim seed 1. At 10,000 shots these are baseline.py's X shots, so confirm a
# schedule picked here on a validation seed, X and Z, with relay_schedule_paired.py (--basis, --seed)
python ml_decoder/relay_knobs.py . 0.004 10000
```

A run limited by `--hours` or `--minutes` is seeded but not repeatable: the number of steps, the learning-rate schedule
and the curriculum follow the clock. A run limited by `--train-shots` repeats on the same CPU with the same seed and
`--threads`. Checkpoints save the basis, the training grid and the seed. `ml_first.py` and the plugin refuse a
checkpoint of the other basis, and load an older checkpoint without these fields with a warning.

## What is tested and what is not

Machine for everything below: luminaty, an AMD Ryzen 7 7730U, CPU only, Python 3.12.15 and the versions in
`ml_decoder/requirements.txt`, under `nice -n 15` in lane A (CPUs 4 to 7, which are 2 physical cores). Every log in
`logs/` starts with its command. Times (ms per shot, elapsed seconds) depend on the machine and on what else ran: other
jobs shared these cores during most runs, so no time here is a latency result. Everything else in a log comes back
when its command is run again on this machine.

### The trainer learns at very low noise, and a seeded run repeats

```sh
python ml_decoder/train_logical_cnn.py --p-train 0.0001 --p-report 0.0001 --width 32 --depth 2 --batch 256 \
    --train-shots 500000 --report-shots 100000 --seed 1 --threads 4 --basis X --out model_X_p0.0001.pt
```

Log `logs/train_X_p0.0001_seed1.log`. A 0.7 M-parameter model, trained and scored on DEM-sampled shots at
p = 0.0001. Every report scores it on the same 20,224 held-out DEM shots, on which the always-0 guess gets 33.22% of
shots wrong:

| Training shots | Shots wrong | Bits right (always-0: 92.84%) |
| --- | --- | --- |
| 100,096 | 33.22% | 92.84% |
| 200,192 | 33.25% | 92.84% |
| 300,032 | 22.42% | 96.01% |
| 400,128 | 10.56% | 98.28% |
| 500,224 (done) | 8.85% | 98.58% |

The model first learns to say always-0, then learns real corrections, which show from the 300,032-shot report on.

Repeatability, on this CPU:

- The same command with `--train-shots 100000 --report-shots 50000`, run twice, gives the same lines and checkpoints
  with identical bytes (`logs/train_X_p0.0001_seed1_100k.a.log`, `.b.log` and `.repeat-check.log`, which holds the
  `diff` and the `cmp`).
- A second run of the full command above gave the first run's lines through its third report, 300,032 shots, and was
  then stopped to free the shared lane (`logs/train_X_p0.0001_seed1.repeat-stopped.log`).
- Floating-point sums can come out differently with another thread count, so pin `--threads`.

### The threshold sweep runs with that model

```sh
python ml_decoder/ml_first.py --basis X --model model_X_p0.0001.pt --p 0.0001 --shots 2000 --seed 1 --threads 2
```

Log `logs/ml_first_X_p0.0001_2000_seed1.log`. 2,000 Stim shots of the kit's circuit, `sample(circuit, 2000, 1)`. The
fallback is the kit's relay-BP at seed 1, decoding every shot in order, and the log's second line prints its settings.
A shot fails when any of its 12 bits is wrong; for relay-BP, also when it did not converge.

| CNN keeps | Threshold | CNN failures among kept | Total failures of 2,000 |
| --- | --- | --- | --- |
| 0% (relay-BP alone) | | | 0 [0.0e+00, 1.9e-03] |
| 30.4% (the 10% and 25% rows tie) | 15.71 | 0 of 609 [0.0e+00, 6.3e-03] | 0 [0.0e+00, 1.9e-03] |
| 50.0% | 6.92 | 0 of 1,000 [0.0e+00, 3.8e-03] | 0 [0.0e+00, 1.9e-03] |
| 75.0% | 2.62 | 2 of 1,500 [3.7e-04, 4.8e-03] | 2 [2.7e-04, 3.6e-03] |
| 90.0% | 0.62 | 52 of 1,800 [2.2e-02, 3.8e-02] | 52 [2.0e-02, 3.4e-02] |
| 100% (CNN alone) | | 188 of 2,000 [8.2e-02, 1.1e-01] | 188 [8.2e-02, 1.1e-01] |

Brackets are 95% Wilson intervals. The 10% and 25% rows tie: 609 shots share the highest confidence, 15.71. On its
most confident half the CNN makes no mistake on these shots, like relay-BP; kept at 100% it is far worse. These are
the sweep's own shots: they choose a threshold and are no held-out result. This seed served as validation for the
evaluator run below, which froze the threshold at the 50% row, 6.92.

### A Z model, and the plugin in the evaluator on test shots

```sh
python ml_decoder/train_logical_cnn.py --p-train 0.0001 --p-report 0.0001 --width 32 --depth 2 --batch 256 \
    --seed 1 --threads 4 --train-shots 400000 --report-shots 100000 --basis Z --out model_Z_p0.0001.pt
python ml_decoder/ml_first.py --basis Z --model model_Z_p0.0001.pt --p 0.0001 --shots 2000 --seed 2 --threads 2
```

Logs `logs/train_Z_p0.0001_seed1_400k.log` and `logs/ml_first_Z_p0.0001_2000_seed2.log`. The Z model trained on
400,000 shots, not 500,000, to keep one hold of the shared lane under 20 minutes. On its 20,224 held-out DEM shots it
gets 13.29% of shots wrong at the end, against 41.84% for the always-0 guess. On 2,000 Stim shots (seed 2) it fails 0
of the 1,000 shots it is most confident on [0.0e+00, 3.8e-03] and 264 of 2,000 kept at 100%; relay-BP alone fails 0.

Then the plugin, through the repository's evaluator, on test shots that chose nothing. The threshold, 6.92, is the
X sweep's 50% row above (seed 1); the Z sweep's 50% row is at 5.27, so 6.92 keeps fewer Z shots.

```sh
python make_dataset.py --basis X --p 0.0001 --shots 2000 --seed 301 --out data/p0.0001/test_X.npz
python make_dataset.py --basis Z --p 0.0001 --shots 2000 --seed 302 --out data/p0.0001/test_Z.npz
python -m evaluation.evaluate --test-x data/p0.0001/test_X.npz --test-z data/p0.0001/test_Z.npz \
    --decoder ml_decoder.plugin:make_decoder --plugin-options ml_decoder/ml_options_p0.0001.json \
    --out results/ml_first_p0.0001
```

`ml_decoder/ml_options_p0.0001.json` is committed. Log `logs/plugin_eval_p0.0001_merge-repo.log` (this branch merged
with `main` at `c54ce0d`; `main`'s later commits up to `13f7378` change no Python file) and
`logs/plugin_eval_p0.0001_head.log` (this branch alone); both print the same numbers.
Failures [0 0] of 2,000 pairs, 95% Wilson [0.00e+00, 1.92e-03] of pairs, so at most 1.60e-04 per cycle. A share of
0.51725 of the shots went to relay-BP (`fallback_fraction`). The evaluator built both sessions with `create_session`
and checked every shot with `validate_shot_result`. The log also prints the plugin's recorded settings for X and for
Z: the kit's relay-BP schedule, seed 1, and the checkpoint's basis, grid, seed and shots. This is a plumbing check at
p = 0.0001, far below the challenge's 0.003 to 0.005, and no comparison: relay-BP alone also fails 0 of 2,000 X shots
and 0 of 2,000 Z shots in the sweeps above.

### Tests

```sh
python -m pytest -p no:cacheprovider test_kit.py evaluation ml_decoder
```

52 passed at this branch's head (`logs/pytest_head.log`): the kit's and the evaluator's 46, and the 6 in
`test_ml_decoder.py`. 57 passed with `main` (`13f7378`) merged in (`logs/pytest_merge.log`). The 6 check that a
checkpoint keeps its fields and loads with `weights_only=True`, that `load_checkpoint` and the plugin refuse the other
basis, that an older checkpoint loads with a warning, that the plugin's fallback is the kit's relay-BP settings with
seed 1 unless `"relay"` changes them, and that a seeded shot-limited run repeats exactly.

Not tested:

- Whether the CNN learns at p = 0.003 to 0.005. Every shot there has 35 to 60 faults. This is the open question.
- Anything on a GPU. Three code paths are new there and have not run: the report's own `torch.Generator` on cuda,
  `shot_logits`' synchronise, and seeding under mixed precision. A one-minute check on the laptop, from the kit folder:

  ```sh
  python ml_decoder/train_logical_cnn.py --basis X --p-train 0.0001 --p-report 0.0001 --width 32 --depth 2 --batch 256 \
      --train-shots 25600 --report-shots 12800 --shots 2048 --seed 1 --out model_gpu_smoke.pt
  python ml_decoder/ml_first.py --basis X --model model_gpu_smoke.pt --p 0.0001 --shots 200 --seed 1
  ```

  It must print `X memory on cuda, seed 1, ...` first, one report at 12,800 shots, `done: 25600 shots`, then
  `CNN on cuda` and the sweep table, with no error. On a GPU the same seed need not give the same digits twice.
- The NVIDIA calls of row 12 (`tcbp_jobs_baseline.py` in the kit folder).

## Measured so far

### Relay-BP schedules on 400 kit shots

`python ml_decoder/relay_knobs.py . 0.004 400`, log `logs/relay_knobs_X_p0.004_400.log`. X memory, p = 0.004, the 400
Stim shots `sample(circuit, 400, 1)`. Each row decodes them in order on its own fresh decoder with seed 1. Scoring: a
shot fails when a predicted logical bit is wrong or relay-BP did not converge. The kit's `decoders.score` does not
count non-convergence; the paired comparison below gives both rules.

| First leg / later legs / stop rule | Mean iterations | Median | p99 | Failures of 400 |
| --- | --- | --- | --- | --- |
| 80 / 60 / 5 solutions (kit baseline) | 580 | 136 | 10,743 | 4 |
| 80 / 60 / 1 solution | 210 | 21 | 3,435 | 3 |
| 80 / 60 / 2 solutions | 314 | 47 | 5,499 | 4 |
| 40 / 60 / 1 solution | 202 | 21 | 3,218 | 2 |
| 20 / 60 / 1 solution | 227 | 33 | 2,508 | 4 |
| 40 / 30 / 1 solution | 150 | 21 | 2,566 | 3 |
| 20 / 30 / 1 solution | 222 | 32 | 9,020 | 7 |
| 20 / 30 / 2 solutions | 343 | 62 | 9,020 | 6 |

Iterations are the schedule's cost and do not depend on the machine. 400 shots cannot tell these schedules apart in
accuracy: 4 against 3 failures is no evidence either way. The next table shows what it takes.

### The kit's schedule against 40 / 30 / 1 on 5,000 paired shots

`python ml_decoder/relay_schedule_paired.py --p 0.004 --shots 5000 --seed 1`, log
`logs/relay_schedule_paired_X_p0.004_5000_seed1.log`. X memory, p = 0.004, `ml_first.py`'s 5,000 Stim shots
(`sample(circuit, 5000, 1)`). Each schedule decodes all of them in order on one fresh decoder with seed 1.

| | Kit 80 / 60 / 5 | 40 / 30 / 1 |
| --- | --- | --- |
| Failures, kit scoring (`decoders.score`: a wrong logical bit) | 20 of 5,000 [2.59e-03, 6.17e-03] | 49 of 5,000 [7.42e-03, 1.29e-02] |
| Failures, an unconverged shot also failed | 20 of 5,000 [2.59e-03, 6.17e-03] | 51 of 5,000 [7.77e-03, 1.34e-02] |
| Unconverged | 7 | 40 |
| Iterations: mean, median, p99 | 505, 134, 11,306 | 179, 20, 4,895 |

Brackets are 95% Wilson intervals. Paired on the same shots, kit scoring: both fail on 18, only the kit's schedule on
2, only 40 / 30 / 1 on 31; exact McNemar p = 1.3e-07. With unconverged shots failed: 18, 2 and 33, p = 3.7e-08.
40 / 30 / 1 takes far fewer iterations (mean 179 against 505) and fails more than twice as often (49 against 20).
Most of its extra failures are unconverged shots: its budget, 40 + 300 x 30 iterations, is half the kit's
80 + 300 x 60. That is why the kit's schedule is the fallback's default. A faster schedule has to be chosen on
validation shots, per p and per basis, with these paired counts.

### Predecoder diagnostics (DEM-sampled, X memory, p = 0.004, 300 shots)

These shots are drawn from the kit's error model, so the true faults are known. The oracle rows use them by design.
They are diagnostics, not decoder results. Every script decodes each shot's variants one after another on shared
relay-BP decoders (seed 1), so rows of one table are not independent: see the last bullet.

- Headroom, `python ml_decoder/diagnostics/predecoder_oracle.py . 0.004 300 20`
  (`logs/predecoder_oracle_p0.004_300_20.log`): an oracle that removes a random half of all true faults takes the
  shots whose first 80-iteration leg fails from 16.3% to 0.3%, and the failures from 9 to 0 of 300.
- Which faults matter, `python ml_decoder/diagnostics/predecoder_which.py . 0.004 300 20`
  (`logs/predecoder_which_p0.004_300_20.log`). Removing the true faults BP is already sure about (LLR 12 or more after
  8 iterations, 44.9% of the faults) helps little: first leg fails 16.3% to 14.7%, failures 8 to 7 of 300. Removing
  the ones BP is unsure about helps a lot: those that share a detector with another fault (28.8% of the faults) take
  it to 0.0%, the isolated ones (26.3%) to 6.0%.
- A real predecoder, `python ml_decoder/diagnostics/predecoder_real2.py . 0.004 300 300 300 20`
  (`logs/predecoder_real2_p0.004_300_300_300_20.log`; 300 shots each to train, validate and test, relay-BP capped at
  20 legs):
  - BP's own confident picks (LLR 12 or more) are 94.0% precise on the test shots, 21.9 commits per shot, and give no
    gain: hard shots 13.3% to 12.3%, failures 7 and 7 of 300.
  - A gradient-boosted model on BP's 8-step trajectory is calibrated on validation (true-fault share 0.959 in its
    0.9 to 0.99 band) but scored no column at 0.99 or above, in training or validation. At the 99% precision cut it
    committed 4.8 faults per shot on the test shots (98.8% precise) and found 0.1% of the faults BP misses. Hard shots
    went from 13.3% to 14.0%.
  - Differences of a few failures or a few dozen mean iterations between rows are within relay-BP's order effect.
    The "BP-missed only" rows commit almost nothing (0.0 per shot) and still show 93 to 105 mean iterations against
    79, and 5 or 7 failures against 7. Likewise the "none" rows of `predecoder_oracle.py` and `predecoder_which.py`
    decode the same 300 shots and fail 9 and 8 times.

That is why this decoder predicts logical bits and does not commit individual faults.

## Rules that keep the numbers honest

- Training data is DEM-sampled. Final numbers come from the kit's Stim shots.
- Freeze the threshold on validation shots, then evaluate once on the test shots.
- Count CNN inference in the latency. The plugin does, because the evaluator times the whole `decode_shot`.
  `ml_first.py` times the same path, one shot at a time.
- Report paired outcomes against relay-BP on the same shots.
- Relay-BP carries state from one decode to the next, so its failures and iterations depend on the order of the shots
  and on its seed. Every relay-BP number in this README states both. `ml_first.py` decodes every shot with relay-BP,
  in order; the plugin decodes only the shots the CNN does not keep.
- A Stim seed does not give the same shots on every machine. Checked on 2026-11-22: `make_dataset.py --basis X --p 0.004
  --shots 10000 --seed 301` on luminaty differs from the committed `data/p0.004/test_X.npz` in 23.2% of the detector
  bits, as much as two independent samples would. Paired comparisons across machines must use saved shot files, not
  seeds. Also, a 400-shot sample of a seed is not the first 400 shots of its 2,000-, 5,000- or 10,000-shot sample.
  (`python ml_decoder/stim_seed_check.py data/p0.004/test_X.npz`, `logs/stim_seed_check.log`. The file is on `main`,
  not on this branch: run the check once the branch is merged.)
- TCBP-JOBS and CNN training need the same GPU. Run row 12 first.

## Next steps

1. Run row 12 with the kit folder's `tcbp_jobs_baseline.py`, then start X training.
2. While it trains: run `relay_knobs.py` at all three p (X memory, Stim seed 1). Keep a faster schedule only where
   `relay_schedule_paired.py` on a validation seed, X and Z, shows no loss against the kit's schedule.
3. When `model_X.pt` exists: `ml_first.py` on a validation seed, freeze the threshold, then the plugin on test shots.
4. Train the Z model. Add the TCBP-JOBS tail handoff.

Stopping rule for training: if "bits right" at p = 0.003 is still at 50% after 2 hours, raise `--width` to 256 or stop.
