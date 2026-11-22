# predecoder: the network tells BP, shot by shot, which faults to suspect

**The idea in one paragraph.** Relay-BP, the kit's strongest decoder, starts every shot with
the same fixed guess of how likely each possible fault is (the kit's "priors"). Our trained
neural network has already read the whole shot. This folder gives the network a second output
(a "head") that says, for each of the 8,784 possible faults BP considers, how likely it is that
this fault happened in this shot. Those numbers replace the fixed priors for that one shot,
and relay-BP then decodes as usual. If the network is right about where the faults are, BP
should find the answer sooner and fail less.

**The result, in short** (X memory, p = 0.004, the 5,000 shots of `data/p0.004/validation_X.npz`):

- The head learned what it was meant to: on held-out shots its probabilities are calibrated,
  87% better in log-loss than the kit's fixed priors, and better than BP's own beliefs.
- With those priors, relay-BP failed 14 shots where the same relay-BP with the kit's priors
  failed 20. That is **not a significant gain** (6 against 12 shots that only one of them got
  wrong, p = 0.24): **the kill line we fixed in advance is not passed**, so this is a measured
  negative result, and the held-out test shots were not used.
- What it does do, clearly: relay-BP needs **2.9 times fewer iterations** (189 against 554 on
  average; fewer on 4,758 of the 5,000 shots), the shots whose first leg fails fall from 691 to
  163, and the slowest 1% of shots need 2,012 iterations or more instead of 13,940.

- **At p = 0.005 the same head and knobs give a large gain** (a secondary result: the knobs
  were fitted at p = 0.004 and not changed). With the network run on a CPU in 32-bit floats,
  the reproducible setting, relay-BP with the head's priors fails **79** of 5,000 validation
  shots against **206** with the kit's priors, 2.6 times fewer (only the head's 17 against only
  the kit's priors' 144, p = 3e-26), with 3.2 times fewer iterations. The laptop's run, with
  the network on its GPU in 16-bit floats, gave 70 (10 against 146); its logits were not kept,
  so that 70 cannot be checked, and the gap shows how much a count moves with the last digits
  of the priors. At p = 0.003 neither fails much: 2 against 0 (p = 0.5).

- **On the held-out test shots, one run at p = 0.005** (Sal's yes at 13:34 UTC; a secondary
  result, chosen after the p = 0.004 gate failed; X memory only): relay-BP with the head's
  priors fails **151** of 10,000 shots against **433** with the kit's priors, 2.9 times fewer
  (only the head's 26 against only the kit's priors' 308, p = 2e-62), with 3.4 times fewer
  iterations. On the same shots TCBP-JOBS fails 186 (61 against 96, p = 0.0065; chosen after
  the p = 0.004 gate failed, X only, accuracy only, not a pair score).

- **The kit's pair score at p = 0.005, on the held-out test shots** (the registered extension
  that followed, with a Z head trained the same way): relay-BP with the network's per-shot priors
  fails **281** of 10,000 pairs, against **360** for TCBP-JOBS (only ours 114, only TCBP-JOBS 193,
  p = 8e-6) and 845 for relay-BP. Handing the shots where our relay-BP does not converge to
  TCBP-JOBS (a rule designed after the first test result) gives **206**.

Details, commands and logs for every number are under "Numbers".

| file | what it does |
|------|--------------|
| `mapping.py` | links the network's fault list (67,824 mechanisms of the full error model) to the decoder's 8,784 columns, and checks the link |
| `head.py` | the head: the network's internal vectors in, one probability per column out |
| `train.py` | trains the head on shots drawn from the error model, with the truly fired faults as labels |
| `evaluate.py` | runs the network and the head on a shot file, saves every shot's logits, times one shot |
| `priors.py` | turns the head's logits into one shot's priors (two knobs), and fits the knobs on train shots |
| `subset.py` | the two train files of the fit: the shots relay-BP fails, and a sample of the rest |
| `calibration.py` | held-out check of the head against the kit's priors and BP's own beliefs |
| `decode.py` | relay-BP built with one shot's priors; the paired test and the Wilson interval |
| `compare.py` | the measurement: the kit's priors against the head's, on the same Stim shots |
| `candidate.py` | the head's answer beside the hierarchy's members (R, P1, P2, T), read from their files |
| `plugin.py` | the decoder for the team's evaluator (`evaluation.evaluate`) |
| `plugin_check.py` | checks that the plugin, through the evaluator's loop, gives `compare.py`'s answers |
| `files.py` | small file helpers without torch, so the decoding worker processes stay small |
| `secondary.py` | the test run's secondary rows: the head's answers against two other decoders' files, same shots |
| `fallback.py` | the fallback rule: the head's relay-BP where it converges, TCBP-JOBS's answer where it does not |
| `pair.py` | the kit's pair score (an X shot and a Z shot) for the head, the rule, TCBP-JOBS and relay-BP |
| `oracle.py` | the ceiling: relay-BP told the true faults (a diagnostic, never a result) |
| `test_predecoder.py` | tests |

Run every command from `challenge-1-ml-decoder-kit/`.

## How it works, plainly

**Two fault lists.** The network was trained on shots drawn from the full error model of the
memory circuit: 1,728 detectors and 67,824 kinds of fault. The decoders solve a smaller
problem: X memory keeps only the 936 detectors on X checks, and faults that look the same on
those detectors are merged into one column, 8,784 columns in all. The kit builds that
problem (`circuit.dem_matrices`). `mapping.py` follows each of the 67,824 faults to the column
it becomes (or to none: a fault that touches only Z checks is invisible to this problem). A
column "fired" in a shot when an odd number of its faults fired, because two of them cancel;
that is exactly how the kit merged their probabilities. Checked three ways: the merged
probabilities equal the kit's priors (to 1e-16, at every start of training); and on the first
batch of every training report, in the oracle and in the tests, H times the fired columns
equals the shot's 936 detector bits and L times them equals its 12 logical bits.

**The head** (`head.py`). For each column, it looks at the network's vectors on the 2 to 6
detectors the column would light up, and averages them separately over the detectors that
did fire in this shot and those that did not. It adds what kind of column it is (the 8,784
columns are 122 shapes, each repeated at the 72 positions of the torus) and the kit's prior,
and a small network shared by all columns turns that into a correction to the kit's
log-odds. Before training the correction is zero, so an untrained head gives exactly the
kit's priors, and training can only move away from them where the data says so. The head
has the code's symmetry: shifting the shot on the torus shifts its output the same way
(tested to 1e-9).

**Training** (`train.py`). Fresh shots from the full error model every step (no shot is seen
twice) at p = 0.003, 0.004 and 0.005 in turn; the labels are the fired columns. The loss is the
plain cross-entropy, which rewards honest probabilities: a column the head puts at 2% should
fire about 2% of the time. That matters because BP reads the numbers as probabilities. The
network itself stays fixed (frozen); only the head (17,617 numbers) learns.

**Two knobs** (`priors.py`). The head has already read the syndrome, and BP reads it again; fed
the head's numbers as they are, BP counts the same evidence twice and may become overconfident.
`alpha` mixes the head with the kit (1 = the head as it is, 0 = the kit), in probabilities or
in log-odds; `eps` keeps every prior inside [eps, 1 - eps]. They are fitted on train shots
only, never on the shots they are scored on. Failures are too rare to rank settings on a plain
slice of train shots (about 1 in 200), so the fit decodes two train files made by `subset.py`:
every train shot that relay-BP with the kit's priors gets wrong, and a sample of the ones it
gets right. A setting's score is the failures it leaves plus the right shots it breaks (scaled
to the whole file); ties go to fewer "hard" shots (relay-BP's first 80-iteration leg alone
does not converge) and then fewer iterations. This rule was fixed before a trained head was
decoded.

**The measurement** (`compare.py`). Three versions of relay-BP at the kit's settings, on the
same shots in the same order:

| arm | priors | decoder |
|---|---|---|
| kit | the kit's | built once, decodes the shots in order (the evaluator's "relay-bp") |
| fresh | the kit's | built anew for every shot |
| head | the head's, for this shot | built anew for every shot |

relay-bp takes priors only when a decoder is built, so the head's priors need a new decoder
per shot (about 8 ms, counted in the time). A decoder built once also carries its random state
from shot to shot, which a new one does not. So **head against fresh** changes the priors and
nothing else: that pair decides the question. Head against kit is what a user would see.

**The kill lines** (fixed in the brief before any number): the head must beat the kit's priors
in log-loss on held-out error-model shots and be calibrated; and on the 5,000 shots of
`data/p0.004/validation_X.npz`, relay-BP with the head's priors must fail fewer shots than with
the kit's priors, paired, with p < 0.05 (head against fresh; reported also against kit).

## Commands

These are the commands that made the numbers below, in order (placeholders in angle brackets).
The environment is the kit's `requirements.txt` plus torch (on a machine without a GPU, from the
CPU index: `pip install torch --index-url https://download.pytorch.org/whl/cpu`), as for `neural/`.

```sh
# tests (about 4 minutes on 4 CPU threads)
python -m pytest -p no:cacheprovider test_kit.py neural predecoder

# the ceiling: relay-BP told the true faults, on DEM shots
python -m predecoder.oracle --p 0.004 --shots 300 --seed 11 --workers 4

# 1. train the head (a GPU: 0.072 s per step on the laptop's RTX 4060, 36 minutes for 30,000 steps)
python -m predecoder.train --trunk <final X checkpoint> --out <run folder> --seed 1 \
    --steps 30000 --batch 64 --p-train 0.003 0.004 0.005 --report-every 2000 --report-shots 500

# 2. check it on held-out error-model shots, against the kit's priors and BP's own beliefs
python -m predecoder.calibration --head <run folder>/head.pt --trunk <final X checkpoint> --p 0.004 --shots 1000 --seed 77

# 3. the train shots relay-BP fails with the kit's priors, and a sample of the ones it gets right
python -m predecoder.compare --data data/p0.004/train_X.npz --arms fresh --workers 6 --out <train compare folder>
python -m predecoder.subset --data data/p0.004/train_X.npz --per-shot <train compare folder>/per_shot.npz \
    --arm fresh --right 2000 --seed 5 --out <subset folder>

# 4. the head's logits on those two files and on the validation file (and the time of one shot)
python -m predecoder.evaluate --head <head.pt> --trunk <final X checkpoint> --data <subset folder>/train_X_fresh_failed.npz \
    --out <logits_failed> --time-shots 0
python -m predecoder.evaluate --head <head.pt> --trunk <final X checkpoint> --data <subset folder>/train_X_fresh_right.npz \
    --out <logits_right> --time-shots 0
python -m predecoder.evaluate --head <head.pt> --trunk <final X checkpoint> --data data/p0.004/validation_X.npz \
    --out <logits_validation> --time-shots 200

# 5. fit the two knobs on the train files only. The default grid (alpha 0.3 to 0.7) chose its
#    edge, so the rule extended it; this is the command that made logs/fit_X_p0.004.json:
python -m predecoder.priors --data <subset folder>/train_X_fresh_failed.npz <subset folder>/train_X_fresh_right.npz \
    --weights 1 9.9555 --logits <logits_failed> <logits_right> --alphas 0.1 0.2 0.3 0.4 0.5 0.6 0.7 --workers 4 \
    --out <fit.json>

# 6. the measurement on the validation shots
python -m predecoder.compare --data data/p0.004/validation_X.npz --logits <logits_validation> \
    --settings predecoder/logs/fit_X_p0.004.json --arms kit fresh head --kit-chunk 500 --workers 4 --out <compare folder>

# 7. the head's answer beside the hierarchy's members (read from their files)
python -m predecoder.candidate --ours <compare folder>/per_shot.npz \
    --members <laptop-artifacts: priortilt/results_8b/selector_validation_X.npz> \
    --tcbp <server-logs: files/tcbp_p0.004/validation_X.tcbp.npz>

# 8. the plugin gives compare.py's answers (the first 200 validation shots through the evaluator's loop)
python -m predecoder.plugin_check --data data/p0.004/validation_X.npz --per-shot <compare folder>/per_shot.npz \
    --head <head.pt> --trunk <final X checkpoint> --settings predecoder/logs/fit_X_p0.004.json --shots 200

# the team's evaluator
python -m evaluation.evaluate --test-x <X shots> --test-z <Z shots> --decoder predecoder.plugin:make_decoder \
    --plugin-options <options.json> --out <folder>
```

`options.json` of the plugin (X memory gets the head; Z memory, which has no head, runs the kit's
relay-BP). `settings` is required with a head: without the fitted knobs the head's numbers go to
BP as they are, which made BP worse on train shots.

```json
{"heads": {"X": "<head.pt>"}, "trunks": {"X": "<the trunk checkpoint>"}, "settings": "predecoder/logs/fit_X_p0.004.json",
 "device": "cpu", "torch_threads": 4}
```

**The one run on the held-out test shots** (written before any test result existed; the
declared gate at p = 0.004 did not pass, so it was Sal's call: he said yes at 13:34 UTC on
2026-11-22, and it ran once, 13:34:51 to 14:17:34 UTC, with lanes A, B and C, 12 threads and 12
processes in place of the 16 below: `logs/test-run.sh.txt` is the script as run). From commit `1bf0e7b`,
the commit the validation results came from, with nothing changed; the logits on a CPU in 32-bit
floats, the setting of the headline numbers, which a second machine reproduces byte for byte
(the reviewer's logits at p = 0.004 equal ours):

```sh
python -m predecoder.evaluate --head <head.pt, sha256 38ab667e...> --trunk <final X checkpoint, sha256 d8d1c70f...> \
    --data data/p0.005/test_X.npz --out <logits_test> --device cpu --threads 4 --final
python -m predecoder.compare --data data/p0.005/test_X.npz --logits <logits_test> --settings predecoder/logs/fit_X_p0.004.json \
    --arms kit fresh head --kit-chunk 500 --workers <cores> --out <test compare folder> --final
```

The declared pair is head against fresh. Use `compare.py` for this, not the plugin through the
evaluator: on a GPU the plugin's batch-one logits differ from the batched ones in their last
digits (up to 1.6e-2 on the laptop), so its answers would not be exactly `compare.py`'s.

## Numbers

### The ceiling (oracle, DEM shots: a diagnostic, never a decoder result)

`python -m predecoder.oracle --p 0.004 --shots 300 --seed 11 --workers 4` on this PC (lane A,
4 processes), log `logs/oracle_p0.004_300_s11.log`. X memory, 300 shots drawn from the full
error model, each mapped onto the kit's columns (the map held on every shot). relay-BP at the
kit's settings, built per shot. "true at q": the kit's priors with the shot's truly fired
columns raised to q; "and as many false": as many columns that did not fire are raised too.

| priors | failures of 300 | unconverged | mean iterations |
|---|---|---|---|
| the kit's | 1 | 1 | 499 |
| true at 0.05 | 0 | 0 | 44 |
| true at 0.2 | 0 | 0 | 19 |
| true at 0.5 | 0 | 0 | 5 |
| true and as many false, at 0.05 | 0 | 0 | 45 |
| true and as many false, at 0.2 | 0 | 0 | 20 |
| true and as many false, at 0.5 | 1 | 4 | 310 |

What it says: BP reads per-shot priors and they steer it strongly. Raising the true faults
even to 5% cuts the work tenfold, and half wrong guesses at 5% or 20% cost almost nothing.
Confident wrong guesses (50%) start to confuse it. 300 shots hold too few failures (the kit's
priors fail 1) to say anything about failures.

### Smoke runs on this PC (CPU only; the head trained for 300 steps of 8 shots: a plumbing check, not the head we use)

This PC: Ryzen 7 7730U, lane A (4 threads), niced, torch 2.15.0+cpu. Trunk: the X network at
step 90,000 (`cascade_X_s1/checkpoint_step90000.pt` on `origin/laptop-artifacts` at `f210ebd`,
sha256 `26a78b77...6f90df89`). Logs in `logs/`.

**Training** (`logs/train_smoke_cpu_step90000_300steps.log`):
`python -m predecoder.train --trunk <step 90,000> --out <folder> --seed 1 --steps 300 --batch 8
--report-every 100 --report-shots 100 --p-train 0.004 --device cpu --threads 4 --warmup-steps 50`.
Held-out error-model shots, 100 at p = 0.004, the same in every report:

| step | shots seen | log-loss per column, head | the kit's priors | fired faults per shot | of them above 0.5 | not fired but above 0.5 |
|---|---|---|---|---|---|---|
| 0 | 0 | 0.031571 | 0.031571 | 47.0 | 0.0 | 0.00 |
| 100 | 800 | 0.010128 | 0.031571 | 47.0 | 27.5 | 12.93 |
| 300 | 2,400 | 0.008403 | 0.031571 | 47.0 | 30.6 | 10.82 |

**Against BP's own beliefs** (`logs/calibration_smoke_p0.004_300.log`; `python -m predecoder.calibration
--head <smoke head> --trunk <step 90,000> --p 0.004 --shots 300 --seed 77 --device cpu --threads 4`;
300 fresh error-model shots, 47.4 fired faults per shot):

| probabilities | log-loss per column | fired faults above 0.5, per shot | not fired but above 0.5, per shot |
|---|---|---|---|
| the kit's priors | 0.031654 | 0.0 | 0.00 |
| BP's belief after 80 iterations (min-sum, memory 0.15) | 0.035037 | 40.2 | 6.91 |
| the smoke head | 0.008377 | 30.9 | 11.43 |

BP's own beliefs are badly overconfident (of the faults BP puts above 0.97, 87% fired), which
is why their log-loss is worse than the fixed priors, but they find more of the fired faults
than the smoke head does. The head has to beat this on the shots where BP struggles.

**The whole chain on 300 train shots** (Stim shots of `data/p0.004/train_X.npz`, exploration;
`logs/evaluate_smoke_train300.log`, `logs/fit_smoke_train300_grid.log`). The network and the
head take 50.8 ms median, 63.6 ms mean per shot, one shot at a time on 4 CPU threads. Relay-BP
at the kit's settings, built per shot:

| priors | failures of 300 | hard shots (first leg alone does not converge) | mean iterations |
|---|---|---|---|
| the kit's | 0 | 51 | 349 |
| log-odds mix, alpha 0.5, eps 1e-4 | 0 | 11 | 158 |
| log-odds mix, alpha 0.8, eps 1e-4 | 2 | 15 | 326 |
| the head as it is (alpha 1), eps 1e-4 | 3 | 18 | 390 |
| linear mix, alpha 0.5, eps 1e-4 | 2 | 12 | 257 |

Even this weak head steers relay-BP: the hard shots fall from 51 to 11 with a half-way mix in
log-odds. Taken at face value (alpha 1) the head makes BP worse: the double-counting the knobs
are there for. 300 shots are too few for failures. These numbers set the default grid of the
fit (log-odds, alpha 0.3 to 0.7); they are train shots, and no knob was chosen on them.

### The baselines on the validation shots, and the train failures of the fit (no head yet)

**Validation** (`data/p0.004/validation_X.npz`, 5,000 Stim shots, seed 201; this PC, lane A,
4 processes on 2 physical cores, both threads busy, so the times are inflated; log
`logs/compare_validation_X_kit_fresh.log`; the first attempt, stopped by hand to use smaller
kit chunks, is `logs/compare_validation_X_kit_fresh_part1_stopped.log`, and its finished fresh
chunks were reused):

    python -m predecoder.compare --data data/p0.004/validation_X.npz --arms kit fresh --kit-chunk 500 --workers 4 --out <folder>

| arm | failures, 95% interval | unconverged | hard shots | mean iterations | build + decode, mean |
|---|---|---|---|---|---|
| kit (built once per 500 shots, in order) | 31, 0.0062 [0.0044, 0.0088] | 17 | 691 | 565 | 538 ms |
| fresh (built per shot, seed 0) | 20, 0.0040 [0.0026, 0.0062] | 11 | 691 | 554 | 442 ms |

Paired, fresh against kit: both fail 15, only fresh 5, only kit 16, exact two-sided p = 0.027.
The kit arm equals the laptop's relay-BP on the same file shot for shot (same failures, same
predicted bits, same iteration counts: `selector_validation_X.npz`, R, on
`origin/laptop-artifacts`), a check across two machines. The two arms differ only in relay-BP's
random draws (gate G2: another draw changes about 44% of relay-BP's failures), so the gap
between them is the size of the noise a single seed carries. The kill line is read against
"fresh", the stronger of the two here.

**Train** (`data/p0.004/train_X.npz`, 20,000 shots, seed 101; run by the orchestrator on the
teammate's server, job ctl/070, 6 processes, 666 s; `logs/server_job_ctl_070_fresh_train.out`,
`logs/server_fresh_train_p0.004_summary.json`): the fresh arm fails 89, 0.0045 [0.0036, 0.0055],
40 unconverged, 2,759 hard, mean 534 iterations, 168 ms per shot on that machine. `subset.py`
(seed 5, `logs/subset_train_X.log`) wrote the 89 failed shots and 2,000 of the 19,911 right ones
for the fit; the right file's weight is 9.9555.

### The trained head (kill line 1: PASSED)

**Training** on the laptop's GPU (RTX 4060 Laptop, torch 2.14.1+cu130; run by the orchestrator at
commit `c694bcd`, 09:35 to 10:14 UTC; `logs/train_gpu_predecoder_X_s1.log`):

    python -m predecoder.train --trunk <final X checkpoint> --out runs\predecoder_X_s1 --seed 1 --steps 30000 --batch 64 \
        --p-train 0.003 0.004 0.005 --report-every 2000 --report-shots 500 --max-hours 1.25

- Trunk: the final X network, step 120,000, `cascade_X_s1/checkpoint_final_step120000.pt` on
  `origin/laptop-artifacts` (commit `2f752e9`), sha256 `d8d1c70f552233300f1c10997b69292092042eb8517f5b6526b607c1f6dc4933`. Frozen.
- The head: `predecoder_X_s1/head.pt` on `origin/laptop-artifacts` (commit `ed20439`), 269,971 bytes,
  sha256 `38ab667e251c4e7b991e26e9b660c12e80a2d1206589e638c5b372806a648a0d`. Not in this branch (a checkpoint).
- 30,000 steps of 64 fresh error-model shots (1.9 million shots), 0.072 s per step, 1.19 GB of GPU memory.
- `logs/train_gpu_predecoder_X_s1_SHA256SUMS` lists `train.log` as `3808aff4...`: the hash of the
  laptop's file, with Windows line endings. The copies in git (here and on `laptop-artifacts`)
  have Unix line endings and hash to `98619458011ad437abe0e07bb5e9db45e374808b52b5577137aad1e21d4f97f7`;
  the content is the same.

Held-out error-model shots in the trainer's reports (500 per p, the same in every report; not Stim shots):

| step | p = 0.003: head / kit log-loss | p = 0.004 | p = 0.005 | p = 0.004: fired per shot, of them above 0.5, false above 0.5 | calibration error at 0.004 |
|---|---|---|---|---|---|
| 0 | 0.02485 / 0.02485 | 0.03170 / 0.03170 | 0.03816 / 0.03816 | 47.5, 0.0, 0.00 | 7.9e-05 |
| 4,000 | 0.00241 (-90.3%) | 0.00460 (-85.5%) | 0.00722 (-81.1%) | 47.5, 40.4, 4.82 | 1.3e-04 |
| 16,000 | 0.00212 (-91.5%) | 0.00417 (-86.9%) | 0.00661 (-82.7%) | 47.5, 40.8, 4.18 | 7.6e-05 |
| 30,000 | 0.00207 (-91.7%) | 0.00410 (-87.1%) | 0.00652 (-82.9%) | 47.5, 41.0, 4.15 | 6.6e-05 |

The head learned almost everything in the first 4,000 steps and was flat from about 16,000.

**The check here, on this PC's CPU in float32** (the laptop's trunk ran in float16;
`logs/calibration_p0.004_1000_s77.log`): 1,000 fresh error-model shots at p = 0.004, seed 77,
split by whether BP's own first 80 iterations converge:

    python -m predecoder.calibration --head <head.pt> --trunk <final X checkpoint> --p 0.004 --shots 1000 --seed 77 --device cpu --threads 4

| shots | probabilities | log-loss per column | fired faults per shot | of them above 0.5 | not fired but above 0.5 |
|---|---|---|---|---|---|
| all 1,000 | the kit's priors | 0.03161 | 47.3 | 0.0 | 0.00 |
| | BP after 80 iterations | 0.03377 | | 40.4 | 6.55 |
| | the head | **0.00381** | | **41.2** | **4.00** |
| BP's first leg converges (889) | BP after 80 iterations | 0.03126 | 46.6 | 40.5 | 5.79 |
| | the head | 0.00361 | | 41.0 | 3.86 |
| BP's first leg does not converge (111) | BP after 80 iterations | 0.05391 | 52.6 | 39.9 | 12.64 |
| | the head | **0.00547** | | **42.9** | **5.19** |

The head is calibrated in every band (for example, columns it puts between 0.03 and 0.1 have
mean 0.054 and fired 0.054; the full tables are in the log), it agrees with the laptop's
reports (0.0038 here against 0.0041 there, on other shots), and on the shots where BP struggles
it knows clearly more than BP does: more of the fired faults, and fewer than half the false ones.
Kill line 1 (beat the kit's log-loss clearly, calibrated) is passed.

### The measurement on the validation shots (kill line 2: NOT PASSED)

`data/p0.004/validation_X.npz` (5,000 Stim shots, seed 201). The knobs are the ones fitted on
train shots and frozen before this run (`logs/fit_X_p0.004.json`: log-odds mix, alpha 0.2,
eps 1e-4). This PC, lane A, 4 processes on 2 physical cores with both threads busy (times
inflated); the network and head were run one shot at a time on 4 CPU threads in
`predecoder.evaluate` (`logs/evaluate_head_final_p0.004.log`: 170 ms mean, 81 ms median; on the
laptop's GPU the same step takes 22 ms). Logs: `logs/compare_validation_X_head.log` (the head
arm's run), `logs/compare_validation_X_all_arms.log` (all three arms with the shot-by-shot
iteration counts), `logs/compare_validation_X_summary.json`.

    python -m predecoder.compare --data data/p0.004/validation_X.npz --logits <logits of validation_X> \
        --settings predecoder/logs/fit_X_p0.004.json --arms kit fresh head --kit-chunk 500 --workers 4 --out <folder>

| arm | failures | share, 95% interval | unconverged | hard shots | mean iterations | p99 iterations | build + decode, mean | build + decode, p99 | adding the network and head's mean time |
|---|---|---|---|---|---|---|---|---|---|
| kit's priors, built once | 31 | 0.0062 [0.0044, 0.0088] | 17 | 691 | 565 | 13,982 | 538 ms | 12.8 s | (no network) |
| kit's priors, built per shot ("fresh") | 20 | 0.0040 [0.0026, 0.0062] | 11 | 691 | 554 | 13,940 | 442 ms | 10.5 s | (no network) |
| **the head's priors, built per shot** | **14** | 0.0028 [0.0017, 0.0047] | 8 | 163 | 189 | 2,012 | 189 ms | 1.95 s | 359 ms mean |

The last column adds the network and head's mean time per shot (170 ms here) to every shot; it
is not a per-shot tail. The network's own per-shot p99 on this busy CPU is 1.47 s
(`logs/evaluate_head_final_p0.004.log`), so the head arm's whole per-shot tail here was not
measured; the tail is stated in iterations, which do not depend on the machine. The reviewer
re-decoded this file on the same PC: the head arm (logits byte-identical to ours) and a 532-shot
part of the fresh arm give the same rows (`reports/predecoder-review.md`, section 0).

| pair (same shots) | both fail | only the first | only the second | exact two-sided p | fewer iterations: first / second / equal |
|---|---|---|---|---|---|
| head against fresh (**the kill line**) | 8 | 6 | 12 | **0.238** | 4,758 / 228 / 14 |
| head against kit | 11 | 3 | 20 | 0.0005 | 4,694 / 286 / 20 |
| fresh against kit | 15 | 5 | 16 | 0.027 | 2,533 / 2,415 / 52 |

How to read it:

- **The kill line is not passed.** It compares the head's priors with the kit's priors in the
  same per-shot decoder, so that only the priors differ, and it needs p < 0.05. 14 against 20
  could be chance (p = 0.24).
- Head against the kit's decoder built once looks significant (3 against 20), but that is not
  the comparison we fixed in advance, and the "fresh" against "kit" line shows why: the two
  differ only in relay-BP's random draws, and that alone gave 5 against 16 here. A single seed
  of relay-BP carries that much noise, so the honest bar is the per-shot decoder.
- The work is a clear effect, not noise: the head's priors need fewer iterations on 4,758 of
  5,000 shots, the hard shots fall by a factor 4.2, and the slowest 1% need 2,012 iterations or more against 13,940. In
  time per shot this PC's busy CPU eats most of that in the network (170 ms mean), so the
  latency gain is for a GPU (22 ms per shot for the network and head there) and is stated here
  only in iterations, which do not depend on the machine.
- On the train subsets the same knobs left 21 of 89 failures wrong while breaking none of
  2,000 right shots (`logs/fit_X_p0.004.log`). That pointed to a larger gain than validation
  shows (14 of 20, not about 5 of 20). The train figure is biased upwards: the 89 shots were
  chosen as the ones relay-BP with seed 0 got wrong, and any other draw of relay-BP alone fixes
  about 44% of such shots (gate G2). So the fit ranks settings but overstates the gain; the
  validation count (12 fixed, 6 broken) is the unbiased reading. Nothing was tuned on
  validation to close the gap.
- One wrinkle of the fit: the first grid (alpha 0.3 to 0.7) chose its edge, 0.3, by 1 broken
  right shot against 4 for the others (each counts 10 times). That difference is itself mostly
  chance. The rule then extended the grid to 0.1 and 0.2 on train shots and chose 0.2, which
  broke none (`logs/fit_X_p0.004_grid1.log`, `logs/fit_X_p0.004.log`).

### Beside the hierarchy's members (a diagnostic on the same validation shots, not a decoder)

`python -m predecoder.candidate` (`logs/candidate_validation_X.log`) puts H, relay-BP with the
head's priors, beside the members of `hierarchy/`, read from their files: R (relay-BP) and P1,
P2 (PriorTilt, seeds 1 and 1001) from the laptop's `priortilt/results_8b/selector_validation_X.npz`,
and T (TCBP-JOBS) from the server's `files/tcbp_p0.004/validation_X.tcbp.npz`.

| | R | P1 | P2 | T | H |
|---|---|---|---|---|---|
| failures of 5,000 | 31 | 26 | 18 | 16 | 14 |
| H against it: only H, only it, p | 3, 20, 0.0005 | 7, 19, 0.029 | 6, 10, 0.45 | 7, 9, 0.80 | |
| right on the 27 shots where P1, P2 and T do not all agree | 12 | 5 | 13 | 15 | 21 |

- R, P1, P2 and T all fail 9 shots; H is right on 3 of them. A perfect choice among the members
  would fail 9 shots; with H among them, 6.
- P1, P2 and T agree on a wrong answer on 4 shots; H is right on 2 of them.
- H cannot be told apart from the best single member, TCBP-JOBS (7 against 9, p = 0.80), and
  fails less than PriorTilt under one seed and than relay-BP. None of that was the
  pre-registered question, and the member counts here come from other runs on the same file.
  It suggests H as a member of the ensemble; that is not built here (`hierarchy/` is not ours
  to change).
- At p = 0.005 the reviewer made the same comparison (`logs/review_full_cpu_p0.005.log`,
  `logs/review_paired_p0.005.log`). With this CPU's logits H fails 79 against TCBP-JOBS's 91:
  41 against 53, p = 0.26, **no claim**. Against the hierarchy's converged rule (108 or 109,
  by its two readings) H is ahead on both devices (p = 0.003 with the CPU's logits). These are
  the reviewer's numbers, a diagnostic beside the hierarchy, not a result of this folder.

### At p = 0.005 and p = 0.003 (secondary results, knobs fitted at p = 0.004)

The same head, trunk and frozen knobs (`logs/fit_X_p0.004.json`) on the team's validation
files, at commit `1bf0e7b`, by two runs:

- **the reference: this PC's CPU, the network in 32-bit floats.** The reviewer's full decode of
  the p = 0.005 file (`reports/predecoder-review.md`, section 3; `logs/review_full_cpu_p0.005.log`):
  `predecoder.evaluate --device cpu`, then the head arm of `predecoder.compare`, 4 processes.
  Its fresh arm was re-decoded on 516 shots, which hold all 206 failures of the laptop's fresh
  arm, and equals the laptop's on every row (relay-BP is bit-exact across the two machines), so
  the fresh count is the laptop's.
- **the laptop: the network on its GPU (RTX 4060 Laptop, autocast to 16-bit floats), relay-BP
  on its CPU** (Windows), run by the orchestrator. Per-shot files on `origin/laptop-artifacts` at
  `7bd45b9`, `predecoder/{fresh,head}_validation_p0.00{5,3}/`; summaries in
  `logs/laptop_*_summary.json`; the counts re-derived here from the per-shot predictions against
  the data files' observables. Its logits (88 MB per file) were not committed.

    python -m predecoder.evaluate --head <head.pt> --trunk <final X checkpoint> --data data\p0.005\validation_X.npz --out <logits> --device cuda
    python -m predecoder.compare --data data\p0.005\validation_X.npz --arms fresh --workers 3 --out <folder>          # at 372d050
    python -m predecoder.compare --data data\p0.005\validation_X.npz --logits <logits> --settings predecoder\logs\fit_X_p0.004.json \
        --arms fresh head --workers 12 --out <the same folder>                                                         # at 1bf0e7b

| p | arm | the network on | failures of 5,000 | share, 95% interval | unconverged | hard shots | mean iterations |
|---|---|---|---|---|---|---|---|
| 0.005 | the kit's priors, per shot | (no network) | 206 | 0.0412 [0.0360, 0.0471] | 131 | 1,800 | 2,367 |
| 0.005 | **the head's priors** | **this PC's CPU, 32-bit (the reference)** | **79** | 0.0158 [0.0127, 0.0196] | 44 | 629 | 732 |
| 0.005 | the head's priors | the laptop's GPU, 16-bit | 70 | 0.0140 [0.0111, 0.0176] | 42 | 648 | 721 |
| 0.003 | the kit's priors, per shot | (no network) | 0 | 0 [0, 0.0008] | 0 | 163 | 133 |
| 0.003 | the head's priors | the laptop's GPU, 16-bit | 2 | 0.0004 [0.0001, 0.0015] | 1 | 20 | 58 |

Paired at p = 0.005, the head against the kit's priors: with the CPU's logits both fail 62, only
the head 17, only the kit's priors 144, exact two-sided p = 3.0e-26; with the laptop's, 60, 10,
146, p = 4.1e-32. At p = 0.003 (the laptop's run only) only the head fails, on 2 shots (p = 0.5):
one does not converge (18,080 iterations) and one converges to a wrong answer; the kit's priors
solved both.

**Why the two p = 0.005 runs differ.** The two sets of logits differ only in the arithmetic of
the network (16-bit on the GPU, 32-bit on the CPU), yet 33 of the 5,000 shots change outcome
(21 newly wrong on the CPU, 12 newly right; p = 0.16 for the lean) and iteration counts differ
on 4,810. 25 of the 33 shots run past 16,900 iterations in one of the two runs, 21 of them to
relay-BP's cap of 18,080: these are shots on the edge, where relay-BP's long search goes either
way on the smallest change of its priors. The network in 16-bit on this CPU (the reviewer's
check, `logs/review_fp16_changed_p0.005.log`) fails 16 of the 33 and agrees with the laptop on
only 13, so no device is "the right one". This is the same size of effect as relay-BP's own
random draws (31 against 20 at p = 0.004). The CPU number is the headline because a second
machine gets it back.

- The gain grows with the noise: none shown at 0.003 (both near zero), 20 to 14 at 0.004 (not
  significant), 206 to 79 at 0.005 (70 on the laptop). At higher noise BP's fixed priors are furthest from the
  truth of a shot, and the head has most to tell it.
- The network and head take 22.1 ms per shot one at a time on the laptop's GPU (20.6 ms at
  p = 0.003). The decode times of the two arms come from runs with different numbers of
  processes (3 and 12) on a shared machine and are not compared here; the iterations are.
- These knobs were never fitted at 0.005 or 0.003, and nothing was changed after these
  numbers were seen.

### The one run on the held-out test shots (p = 0.005, X memory)

**Secondary, chosen after the p = 0.004 gate failed, X only.** Sal said yes at 13:34 UTC; the
orchestrator passed it on. One run, `--final`, from the detached checkout of `1bf0e7b`, with
the head (`38ab667e...`), the final X trunk (`d8d1c70f...`) and the frozen knobs
(`logs/fit_X_p0.004.json`, `b45a0d75...`), each checked by sha256 before it started; the network
on this PC's CPU in 32-bit floats; relay-BP at the kit's settings, built per shot, 12 processes
(lanes A, B and C: `flock` on their locks in that order, `nice -n 15`, `taskset -c 4-15`).
Shots: `data/p0.005/test_X.npz`, 10,000, seed 301, sha256 `1b13521c...`. Logs:
`logs/test_run_p0.005_X.log`, `logs/test_run_p0.005_X_summary.json`,
`logs/test_run_p0.005_X_logits_meta.json`; the script, `logs/test-run.sh.txt`. The per-shot file
stays in the orchestration scratch (it holds the test shots' truth).

| arm | failures of 10,000 | share, 95% interval | unconverged | hard shots | mean iterations |
|---|---|---|---|---|---|
| the kit's priors, per shot ("fresh") | 433 | 0.0433 [0.0395, 0.0475] | 258 | 3,641 | 2,436 |
| **the head's priors, per shot** | **151** | 0.0151 [0.0129, 0.0177] | 90 | 1,316 | 711 |

**The declared pair, head against fresh:** both fail 125, only the head 26, only the kit's
priors 308, exact two-sided p = 2.4e-62. 2.9 times fewer failures and 3.4 times fewer
iterations, as on the validation file (79 against 206 there, on the same device).

Time (a shared 12-worker pool on this PC, not a clean timing: 12 processes on 6 physical
cores with both threads busy, the machine's load about 14 from other work): the network and
head took 19.1 ms per shot one at a time on 12 CPU threads; building and decoding took 599 ms
per shot on average with the head's priors against 2,041 ms with the kit's. The two arms ran
one after the other in that pool, so the load may have differed between them; the iterations
are the machine-free measure.

**Beside two other decoders on the same test shots** (chosen after the p = 0.004 gate failed,
X only, accuracy only;
`python -m predecoder.secondary`, `logs/test_run_p0.005_X_secondary.log`). The two files were
named by the orchestrator, from `origin/server-logs` at `e91760d`: TCBP-JOBS,
`files/clean/tcbp_p0.005/test_X.tcbp.npz` (sha256 `ba3b9ab7...`), and the team evaluator's
relay-bp row, `files/clean/eval_p0.005_relay-bp/01_relay-bp_shots.npz` (sha256 `c42661e5...`,
its X column). Both were checked against the test file: its sha256, its truth, and their own
stored failures. Nothing else of those folders was read.

| decoder on the same 10,000 test X shots | failures | share, 95% interval | H against it: both, only H, only it, p |
|---|---|---|---|
| H, relay-BP with the head's priors | 151 | 0.0151 [0.0129, 0.0177] | |
| T, TCBP-JOBS | 186 | 0.0186 [0.0161, 0.0214] | 90, 61, 96, p = 0.0065 |
| R, the evaluator's relay-bp (built once) | 432 | 0.0432 [0.0394, 0.0474] | 124, 27, 308, p = 1.5e-61 |

These rows are X memory alone: the kit's score is per pair of an X and a Z shot, and Z has no
head here.

## The extension, after the p = 0.005 test result (Sal, 14:19 to 14:27 UTC: the bar is TCBP-JOBS)

Everything in this section was run after the p = 0.005 X test result and is labelled so. Its
plan was registered before any of its numbers existed (`logs/extension_registered_spec.md`,
copied from the orchestrator's log; `logs/z_plan_declared.md`; `logs/fallback_rule_frozen.md`).
Every test file is decoded once.

### X at p = 0.004 and p = 0.003 on the held-out test shots (the registered extension)

The same frozen head, trunk and knobs as the p = 0.005 test run, from `1bf0e7b`, the network on this
PC's CPU in 32-bit floats; one run per file (`logs/test-run-x.3lanes.sh.txt` for p = 0.004 on lanes A,
B and C with 12 processes; `logs/test-run-x.sh.txt` for p = 0.003 on lanes A and C with 8). Declared
pair: head against fresh. Secondaries (accuracy only, X only, not a pair score): TCBP-JOBS and the team
evaluator's relay-bp row, from the files the orchestrator named on `origin/server-logs` at `99a93336`
(each checked against the test file: sha256, truth, stored failures; `logs/test_run_p0.00{4,3}_X_secondary.log`).

| test_X, 10,000 shots | H (head's priors) | fresh (kit's priors) | H against fresh: both, only H, only fresh, p | T, TCBP-JOBS | H against T | R, evaluator's relay-bp | H against R | mean iterations, H / fresh |
|---|---|---|---|---|---|---|---|---|
| p = 0.004 (sha256 `53a7cee3...`; `logs/test_run_p0.004_X.log`) | **15** | 39 | 12, 3, 27, **p = 8.4e-6** | 20 | 8, 7, 12, p = 0.36 | 36 | 10, 5, 26, p = 1.9e-4 | 167 / 514 |
| p = 0.003 (sha256 `f7bc1813...`; `logs/test_run_p0.003_X.log`) | **2** | 7 | 1, 1, 6, p = 0.125 | 1 | 1, 1, 0, p = 1 | 2 | 1, 1, 1, p = 1 | 61 / 144 |

At p = 0.004 the head's priors cut relay-BP's failures 2.6 times on the test shots, significantly,
where the 5,000 validation shots had not shown it (14 against 20, p = 0.24); against TCBP-JOBS the
difference is not significant. At p = 0.003 all decoders fail almost never and none differs.

### Z memory, and the kit's pair score at p = 0.005 (the registered extension)

**The Z head** was trained on the laptop's GPU from `ee359d7f` (14:25:51 to 14:51:08 UTC) on the
final Z network (`cascade_Z_s1_final_step100000.pt`, sha256 `fad24e80...`, light logical operators),
with the X head's recipe: seed 1, 20,000 steps, batch 64, p = 0.003, 0.004, 0.005
(`logs/train_gpu_predecoder_Z_s1.log`). The head: `predecoder_Z_s1/head.pt` on `laptop-artifacts`
at `a2c3d5d`, sha256 `083147d2...`. Held-out error-model shots: log-loss 87.5% below the kit's
priors at p = 0.004; on this PC's CPU (`logs/calibration_Z_p0.004_1000_s77.log`; its header says
"X memory" from a fixed label, corrected later in `calibration.py`; the run used the Z network)
0.0041 against the kit's 0.0314 and BP's own 0.0338; where BP's first leg fails it finds 42.0 of
52.3 fired faults with 5.7 false (BP: 39.4 and 12.5).

**Its knobs**, by X's procedure (`logs/z_plan_declared.md`): the fresh arm on train_Z p = 0.004
fails 95 of 20,000 (`logs/fresh_train_Z_p0.004.log`); the fit on those 95 and 2,000 right shots
(weight 9.9525) chose log-odds, alpha 0.2, eps 1e-4, the same as X, inside its grid: 29 of the 95
left wrong, 1 of the 2,000 broken, mean iterations 593 to 182 (`logs/fit_Z_p0.004.log`,
`logs/fit_Z_p0.004.json`, sha256 `35bc0be9...`).

**Validation first** (validation_Z p = 0.005, 5,000 shots; `logs/validation_Z_p0.005.log`): the
head's priors 74 failures against the kit's 206 (only head 15, only fresh 147, p = 2.1e-28).

**The Z test** (test_Z p = 0.005, 10,000 shots, sha256 `b86ae320...`; one run from the detached
checkout at `87b484a6` with the frozen knobs, the head arm first; `logs/test_run_p0.005_Z_head_arm.log`):
relay-BP with the head's priors fails **132** (0.0132 [0.0111, 0.0156]). The declared pair, with its
fresh arm (`logs/test_run_p0.005_Z.log`, ended 16:29:13 UTC): the kit's priors fail 398 (0.0398
[0.0361, 0.0438]); both 108, only the head 24, only the kit's priors 290, exact two-sided p = 3.6e-59;
mean iterations 695 against 2,364. Secondaries (accuracy only; `logs/test_run_p0.005_Z_secondary.log`):
TCBP-JOBS 177 (only H 56, only T 101, p = 0.0004), the evaluator's relay-bp row 427 (15 / 310). The
frozen fallback rule 100 (it falls back on 73 shots; against T 76 / 24 / 101, p = 2.0e-12;
`logs/fallback_test_Z_p0.005.log`).

**The kit's score: pairs of an X and a Z shot** (shot k of each; a pair fails when either fails;
`python -m predecoder.pair`, `logs/pair_test_p0.005.log`; every file checked against its test
file; TCBP-JOBS and relay-BP from the team's files named above):

| on the 10,000 test pairs at p = 0.005 | X failures | Z failures | failed pairs | 95% interval | per cycle |
|---|---|---|---|---|---|
| **the fallback rule** (H; TCBP-JOBS where H does not converge) | 107 | 100 | **206** | [0.0180, 0.0236] | 1.73e-3 +- 1.2e-4 |
| **H: relay-BP with the head's priors** | 151 | 132 | **281** | [0.0250, 0.0315] | 2.37e-3 +- 1.4e-4 |
| TCBP-JOBS | 186 | 177 | 360 | [0.0325, 0.0398] | 3.05e-3 +- 1.6e-4 |
| relay-BP (the team evaluator's row) | 432 | 427 | 845 | [0.0792, 0.0901] | 7.33e-3 +- 2.5e-4 |

Paired on the pairs: H against TCBP-JOBS, both fail 167, only H 114, only TCBP-JOBS 193, exact
two-sided p = 7.6e-6; the rule against TCBP-JOBS 167 / 39 / 193, p = 1.1e-25; H against relay-BP
242 / 39 / 603; the rule against relay-BP 179 / 27 / 666.

How to read it: on the kit's own score at p = 0.005, relay-BP with the network's per-shot priors
fails 22% fewer pairs than TCBP-JOBS and 67% fewer than relay-BP, and the fallback rule 43% fewer
than TCBP-JOBS. H is the registered extension (every knob fitted on train, Z checked on validation
first); the rule was designed after the first test result (its text frozen at 14:32:15 UTC, before
test_Z was opened at 15:44). One test run per file. These are p = 0.005 only: at p = 0.004 the X
shots show H at 15 against TCBP-JOBS's 20, not a significant difference, and no Z test at 0.004 or
0.003 was run.

### Latency, clean: one process, one thread (the team's server)

Run by the orchestrator as server job cpu/050 at predecoder `e96cc020` (head `38ab667e...`, trunk
`d8d1c70f...`), with `OMP_NUM_THREADS=1` and one worker, on the first 1,000 shots of
`data/p0.005/validation_X.npz` (a validation sample: no test shot decoded twice); files on
`origin/server-logs` under `files/predecoder/latency_validation_X_p0.005/`:

| per X shot, p = 0.005 | mean | median | p99 |
|---|---|---|---|
| network + head (batch one, CPU, 32-bit) | 26.8 ms | 26.7 ms | 28.0 ms |
| relay-BP with the head's priors (build + decode) | 184.8 ms | 30.1 ms | 5,467 ms |

992 of the 1,000 shots converge (141.6 ms mean on those); the 8 that do not run to relay-BP's cap
and take 5,544 ms mean. Those are the shots the fallback rule hands to TCBP-JOBS, so under the rule
they cost the capped relay-BP plus TCBP-JOBS.

### The fallback to TCBP-JOBS (Sal's idea, designed after the first test result)

**The rule:** relay-BP with the head's priors (H) where that relay-BP converges; on the shots
where it does not, TCBP-JOBS's answer (T). It has no knob. Unconverged shots announce
themselves (relay-BP found no correction that explains the syndrome): at p = 0.005, 88 of H's
151 test failures were among the 90 unconverged shots (`python -m predecoder.fallback`).

| shots (p = 0.005, X memory) | H | T, TCBP-JOBS | **the rule** | falls back on | the rule against T: both, only the rule, only T, p |
|---|---|---|---|---|---|
| validation_X, 5,000 (scored first; `logs/fallback_validation_X_p0.005.log`) | 79 | 91 | **50** | 44 | 38, 12, 53, p = 2.8e-7 |
| test_X, 10,000 (post hoc; `logs/fallback_test_X_p0.005.log`) | 151 | 186 | **107** | 90 | 91, 16, 95, p = 7.5e-15 |
| test_X p = 0.004, 10,000 (the registered extension; `logs/fallback_test_X_p0.004.log`) | 15 | 20 | **10** | 12 | 8, 2, 12, p = 0.013 |
| test_X p = 0.003, 10,000 (the registered extension; `logs/fallback_test_X_p0.003.log`) | 2 | 1 | **1** | 1 | 1, 0, 0, p = 1 |

The rule's text was frozen (14:32:15 UTC) after its validation number and before it was applied
to any test file. test_X at p = 0.004 had been opened by `predecoder.evaluate` about 25 s before
the freeze (its run started 14:31:33; no result of it existed before 14:54), so for p = 0.004 the
rule was frozen before it was applied, not before the file was opened; test_X at p = 0.003
(opened 14:55:52) and test_Z (later) were opened after the freeze. On the test shots it cuts TCBP-JOBS's failures by 42%, and H's by 29%
(1 shot only the rule gets wrong, 45 only H). The TCBP-JOBS files are the team's (validation:
`files/tcbp_p0.005/validation_X.tcbp.npz` on `origin/server-logs`, sha256 `876c47c6...`; test:
the file named above). Costs: the rule runs TCBP-JOBS on the 0.9% of shots that fall back, on a
GPU in the team's runs; it was not timed here.

## What is tested, what is not, and what is left

`test_predecoder.py`, 21 tests (`python -m pytest -p no:cacheprovider test_kit.py neural predecoder`;
the full run of `logs/pytest_kit_neural_predecoder.log` had 93 passing, before the Z-trunk and pair-score
tests were added, each of which passes on its own):

- the mapping: on 200 sampled shots H x fired columns = the basis detectors and L x them = the
  observables, the merged priors equal the kit's, a flipped column breaks the check, and the
  map is the same at p = 0.003 and 0.005;
- the head: 122 shapes of 72 columns; untrained, it gives exactly the kit's priors; a torus shift
  of the shot moves its output with the columns (to 1e-9), and the kit's priors have that symmetry;
- the priors: mixing and clipping; relay-BP really reads them (told the true faults it converges
  in its first leg, with fewer iterations); a decoder built per shot repeats itself;
- the plugin: the truth-shuffle test (the observables permuted between shots change no
  prediction), the priors relay-BP is built with are the head's mixed and clipped ones, a head
  without fitted knobs is refused, a basis
  without a head runs the kit's relay-BP;
- the scripts end to end on a few shots: train, evaluate, compare (three arms; the counts add
  up), subset and the weighted fit, calibration, the candidate counts, and the refusal of
  held-out files.

Not tested, or not done:

- **The held-out test files:** X at p = 0.005 once, on Sal's yes, after the declared gate at p = 0.004
  failed (above); then, under the registered extension written after that result
  (`logs/extension_registered_spec.md`), X at p = 0.004 once (`logs/test_run_p0.004_X.log`) and X at
  p = 0.003 once (`logs/test_run_p0.003_X.log`), and Z at p = 0.005 once (the head arm,
  `logs/test_run_p0.005_Z_head_arm.log`, then the fresh arm, `logs/test_run_p0.005_Z.log`, which reuses
  the head arm's saved chunks). The head, its knobs and
  relay-BP were not changed after any result. The fallback rule was designed after the first test
  result and is labelled so wherever it appears.
- **Z memory has a head** (above; from `ee359d7f`, which lets a head train on a Z network with light
  logical operators). The plugin takes it as `{"heads": {"X": ..., "Z": ...}}`; without a Z head it
  decodes Z with the kit's relay-BP. The pair score above was computed from `compare.py`'s per-shot
  files, not through the team's evaluator.
- The GPU path of `train.py` has no test here (this PC has no GPU); it ran on the laptop.
- BP-OSD with the head's priors (the brief's optional variant) was not tried.
- The head as a member of `hierarchy/`'s ensemble is only a diagnostic table above; no rule
  was built or fitted with it.
- Latencies: the per-shot times above come from busy machines with different process counts;
  only the iteration counts compare cleanly.

**The plugin gives compare.py's answers** (`logs/plugin_check_validation_X_200.log`, this PC's CPU,
4 threads): the first 200 validation shots through `evaluation.adapters.create_session` and
`evaluation.evaluate.run_basis` give the same 12 bits as the head arm on 200 of 200 (1 failure in
both). Everything inside `decode_shot` (network, head, building relay-BP, decoding) takes 108 ms
median, 198 ms mean, 1.7 s p99 per shot here.
