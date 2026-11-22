# selector: a learned choice among our decoders' answers

**Status (2026-11-22, 09:20 UTC): frozen for the run on the test shots.** The frozen p = 0.005 model is
`models/p0.005.json`, sha256 `696bf5449a5468177dc2c00664e8557e7d019419c858a44052eea05521698ce6`. The
commit, and the exact command, are in "The run on the test shots" below. This is the branch
`learned-selector`, merged nowhere. Every number in this file is from training or validation shots.
Nothing here was run on a `test_*` or `go_*` file.

## The idea in plain words

Our ensemble runs several decoders on every shot, and each one returns a guess of the 12 logical bits.
On most shots they all agree, and that shared answer stands. On a few shots they disagree (3.6% at
p = 0.005), and something must pick one answer. Today a hand-written rule does it: rule f of
`hierarchy/` (trust joint relay-BP if it converged, else TCBP-JOBS). At p = 0.005 joint relay-BP is
not run, so rule f is just "take TCBP-JOBS's answer".

The selector replaces that rule with a small learned model. Whenever PriorTilt (two seeds, P1 and P2)
and TCBP-JOBS (T) do not all agree, it looks at each distinct answer the decoders gave. That includes
relay-BP's (R), which runs only on those shots. For each answer it reads a few facts the decoders
reported about how they got there; **it never reads the truth.** It gives each answer a score and
takes the highest. The facts are:

- which decoders gave this answer;
- whether their corrections really reproduce the measured syndrome ("converged");
- how heavy their corrections are compared with TCBP-JOBS's (a lighter correction is a more probable
  set of faults);
- how many iterations PriorTilt and relay-BP needed (a decoder that struggled is less trustworthy).

It costs nothing extra to decode, because it only reads answers that are already computed. Its own
work is about 0.1 ms per disagreeing shot on this PC.

**The model** is a *conditional logit*, the simplest logistic model for "which of these is the right
one":

- Each answer gets a score: a weighted sum of its facts, 11 weights in all.
- The chance an answer is right is `exp(score) / sum of exp(score)` over that shot's answers.
- The weights are fitted on training shots.
- One more number, the strength lambda of the penalty on large weights, is chosen by 5-fold
  cross-validation over the training shots (folds by shot, fixed seed).
- The fitted model is a small JSON file, applied with numpy alone, with no scikit-learn.

**What it learned at p = 0.005** (`logs/fit_p0.005.log`; weights on standardized facts). The largest
weights go to:

- "a PriorTilt seed that gives this answer converged" (+1.34);
- "relay-BP gives it and converged" (+1.04);
- "PriorTilt needed few iterations" (-0.96 on the log of the iterations);
- TCBP-JOBS's own vote (+0.47);
- "this correction is lighter than TCBP-JOBS's" (-0.36 on the weight difference).

In words: TCBP-JOBS is right more often than any other decoder, but when PriorTilt or relay-BP reached
a *different* answer easily, with a correction that reproduces the syndrome, that answer often wins.

## What was seen before fitting

Per-member counts on the p = 0.005 validation files were seen during orientation, before any model
existed: on the disagreeing shots, how often each member is right, and the "lowest converged weight"
rule. Every fit uses training files only. Every scoring of a validation file is listed, in order, in
`logs/validation_runs.md`, each with its log. The p = 0.005 procedure, the kill line included, was
committed (ab38a66, 08:39 UTC) before the p = 0.005 model existed. The model was committed (cdeef70)
before it was scored.

## Results (validation shots, 5,000 per basis; not test shots)

### p = 0.005: the selector beats TCBP-JOBS alone, and passes the kill line

Fitted on p = 0.005 `train_X` (20,000 shots, 785 disagreeing) and p = 0.004 `train_X` and `train_Z`
(202 disagreeing), log `logs/fit_p0.005.log`. Scored once on p = 0.005 `validation_X` (Stim seed 201,
data sha256 `25e9ce8e...`) and `validation_Z` (seed 202, `24f12efa...`), log `logs/validation_p0.005.log`.

| row | failures [X Z] | failed pairs of 5,000 | 95% Wilson interval (pairs) | rate per cycle (the kit's) |
|---|---|---|---|---|
| **the selector** | **[83 93]** | **172** | 2.97% to 3.98% | **2.91e-3 +- 0.22e-3** |
| T: TCBP-JOBS alone (= rule f here) | [91 97] | 187 | 3.25% to 4.30% | 3.17e-3 +- 0.23e-3 |
| gw: the vote of T, P1, P2, R (ties to T) | [90 92] | 179 | 3.10% to 4.13% | 3.03e-3 +- 0.23e-3 |
| ac: part 1's best rule without learning | [108 104] | 207 | | 3.52e-3 |
| bw: the lowest weight of P1, P2, T, R | [107 113] | 218 | | 3.71e-3 |
| R: relay-BP alone | [194 189] | 373 | | 6.44e-3 |
| P1 / P2: PriorTilt, one seed | [137 144] / [152 145] | 276 / 289 | | |
| ORACLE: a perfect choice among P1, P2, T, R (a ceiling; needs the truth) | [50 55] | 105 | | 1.77e-3 |

**Paired with TCBP-JOBS alone, on the same shots.** The table gives: both fail, only the selector
fails, only TCBP-JOBS fails, and the exact two-sided binomial test on the shots only one of the two
gets wrong.

| comparison | both | only the selector | only T | p |
|---|---|---|---|---|
| **pairs (the kill line)** | 167 | 5 | 20 | **0.004** |
| shots, X and Z together (10,000) | 168 | 8 | 20 | 0.036 |
| X shots alone | 79 | 4 | 12 | 0.077 |
| Z shots alone | 89 | 4 | 8 | 0.39 |

How to read it:

- **The kill line passes:** 172 failed pairs against 187, paired p = 0.004. The decision written down
  before the fit was "fewer than 187, paired p < 0.05".
- **The pair figure is the stronger of the two.** On pairs, 3 of the selector's 8 lone failures fall
  in pairs where TCBP-JOBS also fails the other basis's shot. The selector has 4 pairs where both
  shots fail, against TCBP-JOBS's 1. X and Z shots are independent draws, so this is luck, not skill.
  Shot by shot the gain is 8 against 20, p = 0.036: still below 0.05, but less strongly.
- **Z is the weaker basis** (4 against 8). Every p = 0.005 training shot is an X shot. The only Z
  training shots are those of p = 0.004.
- **The train cross-validation predicted this size of gain.** On the held-out training folds it was
  wrong alone on 19 shots and TCBP-JOBS alone on 50, out of 814. That is +31 on 814 shots, or about
  +11 on the 364 disagreeing validation shots; validation gave +12 (20 against 8).
- **Not separated from the plain vote.** Against gw: 172 against 179 pairs; both 162, only the
  selector 10, only gw 17, p = 0.25. The vote alone is not separated from TCBP-JOBS either (179
  against 187; 20 against 28, p = 0.31). So the learned model is the only row here that beats
  TCBP-JOBS with p < 0.05, but it is not shown to beat the vote.
- **The gain is small next to the room.** It wins back 15 of the 82 pairs between TCBP-JOBS and the
  ORACLE. The tripwire written down before the fit was a gain of more than half of that, which would
  have meant fewer than about 146 pairs and a check for leakage first. 172 is far from it.

**No truth reaches a choice** (`logs/shuffle_check.log`). Every truth-derived array was moved between
shots: the truth, every `failed_*`, and J's rows by their `shot`. The truth moved on about 4,999 of
5,000 shots, in both p = 0.005 validation files and in p = 0.004 `validation_X`. No feature, choice or
answer of the selector changed.

### p = 0.004, with joint relay-BP (J) in the pool: equal to rule f

Fitted on `train_X` and `train_Z` at p = 0.004 (40,000 shots, 202 disagreeing), log
`logs/fit_p0.004.log`. Scored on `validation_X` and `validation_Z`, log `logs/validation_p0.004.log`.

| row | failures [X Z] | failed pairs of 5,000 | rate per cycle |
|---|---|---|---|
| the selector | [8 3] | 11 | 1.84e-4 +- 0.55e-4 |
| T: TCBP-JOBS alone | [16 9] | 25 | 4.18e-4 +- 0.84e-4 |
| f: the frozen ensemble rule | [8 3] | 11 | 1.84e-4 +- 0.55e-4 |
| gw: the vote | [10 6] | 16 | 2.67e-4 +- 0.67e-4 |
| ORACLE among P1, P2, T, R, J (a ceiling) | [6 2] | 8 | |

Paired on pairs:

- against T: 11 both, 0 only the selector, 14 only T (p = 1.2e-4);
- against f: 11, 0, 0. It makes rule f's choice on every validation shot;
- against gw: 11, 0, 5 (p = 0.06).

At p = 0.004 the model learned rule f: "joint relay-BP converged" gets the largest weight. That is no
gain over rule f, and there is little room for one (a ceiling of 8 against 11). Without J (cross-
validation on the p = 0.004 train files only), it ties TCBP-JOBS: 147 against 146 right.

### p = 0.003, the p = 0.004 model unchanged

The p = 0.003 train files have no TCBP-JOBS answers, so no p = 0.003 model was fitted. Applied
unchanged to the p = 0.003 validation files (`logs/validation_p0.003.log`), the p = 0.004 model gives
selector [1 0], 1 failed pair of 5,000; T [0 1], 1; f [1 0], 1. Every decoder fails 0 to 5 times
here, so this says nothing either way.

### The time of the row (combined offline)

**This row is combined offline from per-shot files; it is not one decoder timed in one process.** Its
time is the sum of the members' decode calls it needs: P1, P2 and T on every shot, and R on the
disagreeing shots. Each is timed on its own machine (`evaluate.py` prints them; p = 0.005 validation):

| member | runs on | X (the laptop, Ryzen 9 8945HS, beside a GPU training) | Z (the server, AMD EPYC-Milan, 8 processes) |
|---|---|---|---|
| P1 | every shot | 193.5 ms per shot | 135.3 ms |
| P2 | every shot | 189.5 ms | 135.3 ms |
| T | every shot | unmeasured: batch mode on the server's A100, no time per shot | unmeasured |
| R | the 3.6% disagreeing shots | 196.2 ms per shot of the file (5.3 s per disagreeing shot) | 166.4 ms |
| the selector itself | the disagreeing shots | 0.1 ms per disagreeing shot (this PC) | 0.1 ms |

- **The row's time is unmeasured**, because TCBP-JOBS has no time per shot here. The known parts sum
  to 579 ms per shot on X and 437 ms on Z, on different machines.
- **R's time is a counterfactual.** R's answers come from chunk mode, where relay-BP decoded every
  shot of the file in order. "R on the disagreeing shots only" is that decoder's time on those shots,
  not a run that decoded only them.

## The run on the test shots

**Frozen:** the code of `selector/` at the commit named in the orchestrator's message (the commit
that adds this section), and the model file `models/p0.005.json`, sha256
`696bf5449a5468177dc2c00664e8557e7d019419c858a44052eea05521698ce6`.

With C = a checkout of `files/clean/` on `origin/server-logs`, made by the run on the test shots, run
from `challenge-1-ml-decoder-kit/`:

```sh
python -m selector.evaluate --model selector/models/p0.005.json \
    --expect-sha256 696bf5449a5468177dc2c00664e8557e7d019419c858a44052eea05521698ce6 \
    --label "test, p = 0.005" \
    --x $C/cand_p0.005/test_X.chunks.npz $C/tcbp_p0.005/test_X.tcbp.npz \
    --z $C/cand_p0.005/test_Z.chunks.npz $C/tcbp_p0.005/test_Z.tcbp.npz \
    --save <an output folder>/selector_test_p0.005
```

- It prints the failures per basis, the pair score and the paired comparisons, as above. It saves
  per shot `failed` and `predicted` (`selector_test_p0.005_X.npz`, `_Z.npz`) and the numbers
  (`selector_test_p0.005.json`).
- It refuses to start on a held-out file without `--expect-sha256`, and stops if the model file has
  another sha256. It fits nothing. `selector.fit` refuses any `test_*` or `go_*` file, or anything
  under `clean/`.
- Give no joint file at p = 0.005. The model never chooses J, but rule f would then use J, and the
  row f would no longer be the TCBP-JOBS row that was validated.
- **p = 0.004, optional and not a claimed row.** The selector equals rule f on every validation shot,
  so it claims nothing over rule f. As a check, the same command with `models/p0.004.json` (sha256
  `164990fdf389d0362218deb2aca6754f4e088dcc6fa4daf333f23292af9b5b31`) and, per basis, the
  `cand_p0.004`, `tcbp_p0.004` and `joint_p0.004` files.

## Physics checks

- **Scored by the kit's rule.** A shot fails when any of its 12 predicted logical bits differs from
  the file's observables. A pair (shot k of X with shot k of Z) fails when either fails. The rate per
  cycle is `decoders.per_cycle_rate` (12 cycles). TCBP-JOBS, rule f, gw and the ORACLE rows are
  scored by `hierarchy.ensemble.score` itself, the same code as the ensemble's tables. Its counts here
  equal the hierarchy README's: T [91 97] and [16 9], f [8 3], ORACLE [50 55] and [6 2].
- **Every member decoded the same shots.** `hierarchy.ensemble.load_members` stops if two member
  files name different data files (sha256) or hold another truth. `logs/check_data.log` shows each of
  the 28 member files naming a committed `data/p*/train_*` or `validation_*` file by its sha256. Only
  those were hashed; no test file was opened.
- **No truth reaches a feature.** `features.py` builds the features from the members' predictions,
  convergence, weights and iteration counts. `labels()` is the only reader of the truth, for fitting
  and scoring. The truth-shuffle test (`test_selector.py`) and the real-file check
  (`shuffle_check.py`, above) move every truth-derived array between shots, and no choice changes.
- **Fit on train, choose on train, report on validation, nothing on test.** `fit.py` refuses
  validation files, `test_*`, `go_*` and `clean/` (by path and by the data name a file records).
  lambda comes from cross-validation over the training shots. The validation scorings are all
  listed in `logs/validation_runs.md`.
- **The members' files are read only to fit and score.** No decoder was re-run here, so the
  decoders' physics (circuits, priors, convergence) is that of `hierarchy/` and its README.

## Commands (every number above)

Run from `challenge-1-ml-decoder-kit/`, in a venv with the kit's `requirements.txt` (`requirements.txt`
here adds only pytest):

```sh
export PYTHONDONTWRITEBYTECODE=1
sh selector/get_members.sh <a checkout with origin/laptop-artifacts and origin/server-logs fetched> $M
A=$M/laptop-artifacts/priortilt; F=$M/server-logs/files
# p = 0.005
python -m selector.fit --pool T P1 P2 R --train $A/results_p005main/selector_train_X.npz $F/tcbp_p0.005/train_X.tcbp.npz \
    --train $A/results_8b/selector_train_X.npz $F/tcbp_p0.004/train_X.tcbp.npz \
    --train $A/results_8b/selector_train_Z.npz $F/tcbp_p0.004/train_Z.tcbp.npz --out selector/models/p0.005.json
python -m selector.evaluate --model selector/models/p0.005.json \
    --x $A/results_p005main/selector_validation_X.npz $F/tcbp_p0.005/validation_X.tcbp.npz \
    --z $F/hier/cand_p0.005/validation_Z.chunks.npz $F/tcbp_p0.005/validation_Z.tcbp.npz --save $O/validation_p0.005
# p = 0.004 (and the same model on p = 0.003's files: cand_p0.003, tcbp_p0.003, joint_p0.003)
python -m selector.fit --pool T P1 P2 R J \
    --train $A/results_8b/selector_train_X.npz $F/tcbp_p0.004/train_X.tcbp.npz $F/joint_p0.004/train_X.joint.npz \
    --train $A/results_8b/selector_train_Z.npz $F/tcbp_p0.004/train_Z.tcbp.npz $F/joint_p0.004/train_Z.joint.npz \
    --out selector/models/p0.004.json
python -m selector.evaluate --model selector/models/p0.004.json \
    --x $A/results_8b/selector_validation_X.npz $F/tcbp_p0.004/validation_X.tcbp.npz $F/joint_p0.004/validation_X.joint.npz \
    --z $A/results_8b/selector_validation_Z.npz $F/tcbp_p0.004/validation_Z.tcbp.npz $F/joint_p0.004/validation_Z.joint.npz \
    --save $O/validation_p0.004
# the checks
python -m selector.check_data $M
python -m selector.shuffle_check --model selector/models/p0.005.json \
    --members $A/results_p005main/selector_validation_X.npz $F/tcbp_p0.005/validation_X.tcbp.npz --work $O/shuffled
python -m pytest -p no:cacheprovider test_kit.py hierarchy selector     # 71 passed, 9 skipped (PriorTilt, torch) on this PC
```

- **A refit reproduces the weights, not the sha256.** The model file records the local paths of the
  files it was fitted on, so a refit elsewhere writes another file. Compare the weights and the
  choices, and never rewrite the frozen file.
- **The logs** begin with the time, the machine and the commit, and write the members' folder as `$M`.
- **Seeds.** The folds' seed is 20261122 and the shuffle check's is the same. The fit itself has no
  randomness.

| file | what it does |
|---|---|
| `features.py` | reads the members' files (through `hierarchy.ensemble.load_members`) and builds the choice table: one row per distinct answer of a disagreeing shot, with its 11 features. No truth. |
| `model.py` | the conditional logit: fit, cross-validation, pick, save and load the JSON model |
| `fit.py` | fits on training files; refuses validation files, `test_*`, `go_*` and `clean/` |
| `evaluate.py` | applies a frozen model to an X and a Z file; prints every row, the paired tests and the pair score; saves per-shot `failed` and `predicted` |
| `shuffle_check.py` | the leakage check on real files |
| `check_data.py` | every member file names a committed train or validation data file |
| `get_members.sh` | copies the exploration members' files out of the data branches |
| `test_selector.py` | 25 tests: the choice table, the truth shuffle, fitting, the refusals, evaluate and the pair score |
| `models/` | `p0.005.json` (frozen), `p0.004.json` |
| `logs/` | every fit and scoring, and `validation_runs.md`, the list of validation scorings |

## Limits, and what is left

- **The network is not used.** The final networks' logits that the test run uses did not exist on
  any training file before the freeze, so a network feature fitted now would not match them.
- **Few training shots.** 814 disagreeing training shots had a right answer. p = 0.005 `train_Z` has
  no PriorTilt or relay-BP answers, so the Z side learned from p = 0.004 alone.
- **Not shown to beat the vote gw.** More training shots, above all Z at p = 0.005, would settle it.
- **The test numbers** go here once the orchestrator has run the command above.
