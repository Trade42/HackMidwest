# predecoder: the network tells BP, shot by shot, which faults to suspect

**The idea in one paragraph.** Relay-BP, the kit's strongest decoder, starts every shot with
the same fixed guess of how likely each possible fault is (the kit's "priors"). Our trained
neural network has already read the whole shot. This folder gives the network a second output
(a "head") that says, for each of the 8,784 possible faults BP considers, how likely it is that
this fault happened in this shot. Those numbers replace the fixed priors for that one shot,
and relay-BP then decodes as usual. If the network is right about where the faults are, BP
should find the answer sooner and fail less.

Status: built and tested on this PC; the head's training runs on a GPU (commands below).
The measurement is not in yet; every number that is in is listed under "Numbers".

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

```sh
# tests (about 4 minutes on 4 CPU threads)
python -m pytest -p no:cacheprovider predecoder

# the ceiling: relay-BP told the true faults, on DEM shots
python -m predecoder.oracle --p 0.004 --shots 300 --seed 11 --workers 4

# train the head (a GPU; about 0.06 s per step expected on an RTX 4060, not measured yet)
python -m predecoder.train --trunk <cascade_X_s1 checkpoint> --out <run folder> --seed 1 \
    --steps 30000 --batch 64 --p-train 0.003 0.004 0.005 --report-every 2000 --report-shots 500

# the head's logits on a shot file, and the time of one shot
python -m predecoder.evaluate --head <run folder>/head.pt --trunk <cascade_X_s1 checkpoint> \
    --data data/p0.004/train_X.npz --limit 1000 --out <logits_train>
python -m predecoder.evaluate --head <run folder>/head.pt --trunk <cascade_X_s1 checkpoint> \
    --data data/p0.004/validation_X.npz --out <logits_validation>

# fit the two knobs on train shots, then the measurement on validation shots
python -m predecoder.priors --data data/p0.004/train_X.npz --logits <logits_train> --shots 1000 --out <fit.json>
python -m predecoder.compare --data data/p0.004/validation_X.npz --logits <logits_validation> \
    --settings <fit.json> --arms fresh head kit --workers 4 --out <compare folder>

# the head's answer beside the hierarchy's members (read from their files), on validation_X
python -m predecoder.candidate --ours <compare folder>/per_shot.npz \
    --members <laptop-artifacts: priortilt/results_8b/selector_validation_X.npz> \
    --tcbp <server-logs: files/tcbp_p0.004/validation_X.tcbp.npz>

# the team's evaluator
python -m evaluation.evaluate --test-x <X shots> --test-z <Z shots> --decoder predecoder.plugin:make_decoder \
    --plugin-options <options.json> --out <folder>
```

The fit's two train files come from a run of the "fresh" arm on the train file:

```sh
python -m predecoder.compare --data data/p0.004/train_X.npz --arms fresh --workers 4 --out <train compare folder>
python -m predecoder.subset --data data/p0.004/train_X.npz --per-shot <train compare folder>/per_shot.npz \
    --arm fresh --right 2000 --seed 5 --out <subset folder>
# then predecoder.evaluate on each of the two files, and the fit:
python -m predecoder.priors --data <subset folder>/train_X_fresh_failed.npz <subset folder>/train_X_fresh_right.npz \
    --weights 1 <the right-file weight subset.py prints> --logits <logits of the failed file> <logits of the right file> \
    --workers 4 --out <fit.json>
```

`options.json` of the plugin (X memory gets the head; Z memory, which has no head, runs the kit's
relay-BP):

```json
{"heads": {"X": "<head.pt>"}, "trunks": {"X": "<the trunk checkpoint>"}, "settings": "<fit.json>",
 "device": "cpu", "torch_threads": 4}
```

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
