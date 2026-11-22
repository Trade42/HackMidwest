# selector: a learned choice among our decoders' answers

**Status: work in progress (branch `learned-selector`).** p = 0.004 is fitted and validated below;
p = 0.005 waits for its training files. Nothing here was run on a `test_*` or `go_*` file.

## The idea in plain words

Our ensemble runs several decoders on every shot. Each one returns a guess of the 12 logical bits.
On most shots they all agree, and that shared answer stands. On a few shots they disagree, and
something must pick one answer. Today a hand-written rule does it: rule f of `hierarchy/` (trust joint
relay-BP if it converged, else TCBP-JOBS).

The selector replaces that rule with a small learned model. On every shot where PriorTilt (two
seeds, P1 and P2) and TCBP-JOBS (T) do not all agree, it looks at each distinct answer the decoders
gave. For each answer it reads a few facts the decoders reported about how they got there. It never
reads the truth. It gives each answer a score and picks the highest. The facts are:

- which decoders gave this answer;
- whether their corrections really reproduce the syndrome ("converged");
- how heavy their corrections are, compared with TCBP-JOBS's;
- how many iterations PriorTilt and relay-BP needed.

It costs nothing extra to decode: it only reads answers that are already computed.

The model is a **conditional logit**, the simplest logistic model for "which of these is the right
one". The chance that answer a is right is `exp(s(a)) / sum over the shot's answers of exp(s(b))`,
with `s(a)` a weighted sum of the answer's facts. It has 11 weights, fitted on the training files.
One more number, the strength lambda of the penalty on large weights, is chosen by 5-fold
cross-validation over the training shots. The fitted model is a small JSON file, and applying it
needs numpy only.

## What was seen before fitting

Per-member counts on the p = 0.005 validation files were seen during orientation: on the shots where
P1, P2 and T disagree, how often each member is right, and the "lowest converged weight" rule. Every
fit uses training files only. Every scoring of a validation file is listed in
`logs/validation_runs.md` with its log.

## Results so far (validation shots, not test shots)

### p = 0.004, with joint relay-BP (J) in the pool

Fitted on `train_X` and `train_Z` at p = 0.004 (40,000 shots, 202 choice shots), log
`logs/fit_p0.004.log`. Scored on `validation_X` and `validation_Z` (5,000 shots each), log
`logs/validation_p0.004.log`.

| row | failures [X Z] | failed pairs of 5,000 | rate per cycle |
|---|---|---|---|
| the selector | [8 3] | 11 | 1.84e-4 +- 0.55e-4 |
| T: TCBP-JOBS alone | [16 9] | 25 | 4.18e-4 +- 0.84e-4 |
| f: the frozen ensemble rule | [8 3] | 11 | 1.84e-4 +- 0.55e-4 |
| gw: the vote | [10 6] | 16 | 2.67e-4 +- 0.67e-4 |
| ORACLE among P1, P2, T, R, J (a ceiling; needs the truth) | [6 2] | 8 | |

Paired on pairs (both fail, only the selector fails, only the other fails, exact two-sided p):
against T 11, 0, 14 (p = 1.2e-4); against f 11, 0, 0 (it makes the same choice as rule f on every
validation shot); against gw 11, 0, 5 (p = 0.06).

**What it means:** at p = 0.004 the model learned rule f: "trust joint relay-BP where it converged"
gets the largest weight. That is no gain over rule f, and there is little room for one (the ceiling
is 8 failed pairs against 11).

### p = 0.003, the p = 0.004 model unchanged

There are no TCBP-JOBS answers on the p = 0.003 train files, so no p = 0.003 model was fitted. The
p = 0.004 model, applied unchanged to the p = 0.003 validation files (log `logs/validation_p0.003.log`):
selector [1 0], 1 failed pair of 5,000; T [0 1], 1; f [1 0], 1. Every decoder fails 0 to 5 times
here, so this says nothing either way.

## Commands

Run from `challenge-1-ml-decoder-kit/`, in a venv with the kit's `requirements.txt` (see
`requirements.txt` here; no scikit-learn).

```sh
export PYTHONDONTWRITEBYTECODE=1
sh selector/get_members.sh <a checkout with origin/laptop-artifacts and origin/server-logs> $M   # copies the members' files
A=$M/laptop-artifacts/priortilt; F=$M/server-logs/files
python -m selector.fit --pool T P1 P2 R J \
    --train $A/results_8b/selector_train_X.npz $F/tcbp_p0.004/train_X.tcbp.npz $F/joint_p0.004/train_X.joint.npz \
    --train $A/results_8b/selector_train_Z.npz $F/tcbp_p0.004/train_Z.tcbp.npz $F/joint_p0.004/train_Z.joint.npz \
    --out selector/models/p0.004.json
python -m selector.evaluate --model selector/models/p0.004.json \
    --x $A/results_8b/selector_validation_X.npz $F/tcbp_p0.004/validation_X.tcbp.npz $F/joint_p0.004/validation_X.joint.npz \
    --z $A/results_8b/selector_validation_Z.npz $F/tcbp_p0.004/validation_Z.tcbp.npz $F/joint_p0.004/validation_Z.joint.npz \
    --save $O/validation_p0.004
python -m pytest -p no:cacheprovider test_kit.py hierarchy selector
```

| file | what it does |
|---|---|
| `features.py` | reads the members' files (through `hierarchy.ensemble.load_members`) and builds the choice table: one row per distinct answer of a choice shot, with its features. No truth. |
| `model.py` | the conditional logit: fit, cross-validation, pick, save and load the JSON model |
| `fit.py` | fits on training files; refuses `test_*`, `go_*`, files under `clean/`, and validation files |
| `evaluate.py` | applies a frozen model to an X and a Z file; prints every row, the paired tests and the pair score; saves per-shot `failed` and `predicted` |
| `get_members.sh` | copies the exploration members' files out of the data branches |
| `test_selector.py` | tests, among them the truth-shuffle test and the refusals |
