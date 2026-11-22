# ML-first decoder

A fast ML layer in front, a confidence check, and the real decoder as the fallback. Same shape as the DuQuantum
"ML telescope", moved to the Gross code.

1. A CNN reads the 936-bit basis syndrome (13 rounds on the 12 x 6 torus) and predicts the 12 logical bits.
2. Confidence is the smallest |logit| of the 12. At or above a frozen threshold, the CNN's answer is final.
3. Every other shot goes to relay-BP at the tuned schedule: first leg 40, later legs 30, stop at the first solution.
4. Planned, not built: shots unsolved at a leg cap go to TCBP-JOBS on the GPU.

If the CNN is weak, the threshold sends every shot to relay-BP. That costs one CNN inference per shot and no accuracy.

## Files

| File | What it does |
| --- | --- |
| `train_logical_cnn.py` | Trains the CNN. Samples unlimited data on the GPU from the kit's error model |
| `ml_first.py` | CNN + threshold + relay-BP on real kit shots. Prints the threshold sweep |
| `plugin.py` | The same decoder as a plugin for `evaluation.evaluate` |
| `tcbp_jobs_baseline.py` | NVIDIA's TCBP-JOBS on the same shots as `baseline.py` (row 12) |
| `relay_knobs.py` | Sweeps relay-BP's schedule on kit shots |
| `diagnostics/` | The predecoder studies that led here. Oracle and DEM-sampled, never decoder results |

All scripts find the kit one folder up. Run them from `challenge-1-ml-decoder-kit/`.

## Commands

```sh
pip install torch
# 3-minute check, then the real run. Prints a report every 10 minutes and saves model_X.pt
python ml_decoder/train_logical_cnn.py --basis X --minutes 3 --width 32 --depth 2 --batch 256
python ml_decoder/train_logical_cnn.py --basis X --hours 8
python ml_decoder/train_logical_cnn.py --basis Z --hours 8

# threshold sweep on kit shots. Pick the threshold on one seed, report on another
python ml_decoder/ml_first.py --basis X --model model_X.pt --p 0.004 --shots 5000 --seed 1

# paired evaluation with the repository's evaluator
python -m evaluation.evaluate ... --decoder ml_decoder.plugin:make_decoder --plugin-options ml_options.json
# ml_options.json: {"models": {"X": "model_X.pt", "Z": "model_Z.pt"}, "threshold": 6.0, "device": "cuda"}

# row 12, needs Linux, an NVIDIA GPU and cudaq-qec 0.9.0 or later
pip install cudaq-qec
python ml_decoder/tcbp_jobs_baseline.py --p 0.004 --shots 10000

# relay-BP schedule sweep
python ml_decoder/relay_knobs.py . 0.004 10000
```

## What is tested and what is not

Tested on CPU, in a sandbox without a GPU:

- `train_logical_cnn.py` runs end to end and learns at very low noise. At p = 0.0001 a 0.7 M-parameter model went
  from 34.6% of shots wrong (the always-0 guess) to 4.5% in 0.5 M shots.
- `ml_first.py` and `plugin.py` run with that model. On 2,000 Stim shots at p = 0.0001 the CNN kept 50% of shots
  with 0 failures in 1,000 and 75% with 1 failure in 1,500. Kept at 100% it failed 102 of 2,000.
- `plugin.py` passes `evaluation.adapters.create_session` and `validate_shot_result`.
- `tcbp_jobs_baseline.py` with `--decoder relay`, which exercises everything except the NVIDIA calls.

Not tested:

- Whether the CNN learns at p = 0.003 to 0.005. Every shot there has 35 to 60 faults. This is the open question.
- The three NVIDIA calls in `tcbp_jobs_baseline.py`. The option and preset names were read from the 0.9.0 wheel.
- Anything on a GPU, and the Z basis model.

Stopping rule for training: if "bits right" at p = 0.003 is still at 50% after 2 hours, raise `--width` to 256 or stop.

## Measured so far

Relay-BP schedule, 400 kit shots, X memory, p = 0.004 (`relay_knobs.py`):

| First leg / later legs / stop rule | Mean iterations | Median | p99 | Failures of 400 |
| --- | --- | --- | --- | --- |
| 80 / 60 / 5 solutions (kit baseline) | 580 | 136 | 10,743 | 4 |
| 80 / 60 / 1 solution | 210 | 21 | 3,435 | 3 |
| 40 / 60 / 1 solution | 202 | 21 | 3,218 | 2 |
| 40 / 30 / 1 solution | 150 | 21 | 2,566 | 3 |
| 20 / 30 / 1 solution | 222 | 32 | 9,020 | 7 |

Small sample. Rerun on 10,000 shots, X and Z, at each p before quoting it.

Hard-commit predecoders, 300 DEM-sampled shots at p = 0.004 (`diagnostics/predecoder_real2.py`):

- An oracle that removes half of the faults BP misses takes hard shots from 12.7% to 0.3%. That is the headroom.
- BP's own confident picks are 94% precise and give no gain.
- A calibrated model on BP's 8-step trajectory never got above 98% sure of any fault. At 99% precision it committed
  1.7 faults per shot and none of the ones BP misses. Mean iterations went from 89 to 104.
- BP's probabilities fed back in as priors doubled the hard shots, 13% to 26%.

That is why this decoder predicts logical bits and does not commit individual faults.

## Rules that keep the numbers honest

- Training data is DEM-sampled. Final numbers come from the kit's Stim shots.
- Freeze the threshold on validation shots, then evaluate once on the test shots.
- Count CNN inference in the latency. The plugin does, because the evaluator times the whole `decode_shot`.
- Report paired outcomes against relay-BP on the same shots.
- TCBP-JOBS and CNN training need the same GPU. Run row 12 first.

## Next steps

1. Run row 12 with `tcbp_jobs_baseline.py`, then start X training.
2. While it trains: rerun `relay_knobs.py` on 10,000 shots at all three p and keep the best schedule per p.
3. When `model_X.pt` exists: `ml_first.py` on a validation seed, freeze the threshold, then the plugin on test shots.
4. Train the Z model. Add the TCBP-JOBS tail handoff.
