# A loss-time tracker for the Gross code: when was each lost data atom lost?

Task `loss-tracker` (idea T2 of `orchestration/PLAN.md`). It builds on `loss/` (task `loss-gates`): the same loss
model, the same Clifft simulation and the same map from a loss to the decoder's columns. These are exploration
numbers on shots that Clifft simulates. No file of the team's `data/` folder is involved.

This file is written by `python -m loss_tracker.assemble` from `RESULTS.template.md`: every table below is a file
of `results/`, copied in unchanged except that its headings move down two levels. Edit the template, not this file.

{{include results/summary.md shift 0}}

## What the tracker does, in plain words

A data atom that is lost stays lost, and the machine only learns of it at the final readout. The published
loss-aware decoders therefore spread the loss evenly over all 12 rounds (the "uniform guess"). `loss-gates`
measured what the true loss round would be worth: told the truth, relay-BP fails 1.64 times less often.

The tracker estimates the loss round from the syndrome history. After an atom is lost, the checks next to it
measure an operator with one atom missing, and their outcomes become random: their detectors start to "flicker".
For every atom flagged at the final readout, the tracker compares two stories for each round, "still there"
and "already gone", over the 3 X checks and the 3 Z checks next to the atom, and returns a probability for
each of the 72 possible loss times. Relay-BP then gets priors built from that probability instead of the
uniform guess. Nothing else in the decoder changes.

## What the tracker assumes

| What | Our choice |
|---|---|
| Loss model, simulator, noise | Exactly those of `loss/RESULTS.md`: Clifft 0.13.0 `noncomp`, the kit's X-memory circuit at p = 0.004, each atom lost with probability s = 1e-3 after each of its CX, perfect heralds, check atoms restored at their reset, data atoms flagged only at the end. |
| Inputs | The 1,728 detector bits of a shot (X-check and Z-check detectors, although the decoder of X memory uses only the X-check ones) and the heralds. Nothing else: `tracker.features` takes these two arrays, and `track.py`, the only script that runs the tracker on a dataset, opens only `detectors` and `heralds` of the file. |
| Hidden variable | For each flagged data atom, its loss slot j = 1..72 (lost after its j-th CX). Prior: geometric, P(j) ∝ (1 − s)^(j−1). The herald says that a loss happened, so there is no "never lost" value. |
| Observations | Per detector round, the 3 detector bits of the atom's X checks as one pattern of 8, and the 3 bits of its Z checks as another. |
| Emission tables | P(pattern given the pattern of the round before and the class), counted on training shots. The class says how far the loss is from the round: still present; lost in this cycle, in the cycle before, or two cycles before, after CX k (k = 1..6 each); lost three or more cycles before. Separate tables for left and right data atoms (their CX orders differ), for X and Z checks, and for the first two rounds and the last. Counted only on "clean" triples (next rows). Smoothing: half a count per pattern, and 4 counts of the single-round table as the prior of each conditional row. |
| Why a round is conditioned on the round before | In this CX schedule a lost atom's detectors are not independent between rounds: one detector repeats, one round later, what its two partners showed (step 1). A model with independent rounds counts that random bit twice. |
| Unreliable ("dirty") bits | (a) The detector of a check whose own atom was flagged in this cycle or the one before is a fair coin. (b) So is the detector of a check that ANOTHER lost data atom has already left. (c) All 3 bits of a triple are set aside when a check atom of the other type next to the atom was flagged in this cycle or the one before: its hook error can flip the triple like a loss does (step 1). (a) and (c) are known from the heralds. |
| Atoms that share checks | (b) depends on the unknown loss times of the other flagged atoms. The tracker starts from their priors and then recomputes every atom's posterior from the others' posteriors of the sweep before, all atoms at once (synchronous message passing). We chose this over alternating atom by atom because the result does not depend on any atom order and it vectorises; one sweep already reaches the fixed point to three digits (step 2). Different atoms are treated as independent given their posteriors. |
| Explaining a shared bit | On a shared check the bit is (this atom's part) XOR (the other lost atom's part) XOR noise. The first versions replaced such a bit by a fair coin. The final version reads the other atom's part off that atom's own other detectors (a lost atom fires its checks together, or repeats them a round later) and flips the bit back with probability P(the other atom has left the check) × P(it fired there). With probability 1/2 this is the fair coin again. |
| What is NOT modelled | Correlations between the X triple and the Z triple of an atom. The exact time course of a hook error (the triple is set aside for two rounds instead). Three-way effects between lost atoms: the other atoms on a check are treated as independent. |
| Use in the decoder | The posterior over the slot is summed to the 12 rounds and spread over the 6 slots of each round as the ORACLE `oracle-round` does; then `loss.lossmap`'s `data_activation` and the rest of the condition `exact`, unchanged (`priors.py`). Relay-BP at the kit's settings, a fresh decoder with seed 1 per shot and condition, as in `loss/decode.py`. |

## Splits

| Split | Seed | Shots | sha256 of the file | Lost data atoms per shot | Used for |
|---|---|---|---|---|---|
| training | 7100 (chunks 7100–7109) | 20,000 | `5d32f279bbfd4619bc2b45f9483d1344f44c4ef02476628db88162367751563e` | 10.02 | the emission tables (`fit.py`), the signal tables of step 1, the learned tracker of step 4 |
| validation | 7200 (7200–7201) | 4,000 | `8b09d471d063650d5740c8bc2ad7bbfe24a784c7b7e16d8d82124a471dc454e8` | 10.04 | the tracker's accuracy (step 2); the choice of the decoder variant on its first 1,500 shots (step 3) |
| test | 7300 (7300–7301) | 4,000 | `4ee65bb56e69fcb51fd638b17f3c766178dece67c4a990fe0e93ec80f1225583` | 9.98 | one run of the frozen decoder (step 3) |

All are X memory, p = 0.004, s = 1e-3, sampled on this PC (`runs/sample.sh`, log `logs/sample.log`). The files
are in `/home/sal/hackMidwest/orchestration/scratch/loss-tracker/data/` and are not committed.

The splits of step 4 (tracking only; `runs/extra.sh`, logs `logs/extra_*.log`), 20,000 training and 4,000
validation shots each:

| Point | Training seed, sha256 | Validation seed, sha256 | Lost data atoms per shot |
|---|---|---|---|
| X, s = 5e-4 | 7160, `7a9bba703cd1ff7a1f397fc0164e13fad61d590a0ebd38499ebedf9ebc57b921` | 7260, `63d06b61cdc3ec9bd37dab3e4d6b405dd3fe9a5f446cc345c8fefa8674ae35e7` | 5.1 |
| X, s = 2e-3 | 7140, `33cf6440bd56f238158aa079f9e970a18c7c29679f4431da75c09d86b02d1f50` | 7240, `f8ed976b101da032fc5059430cb16eebb0425ce115de794f3bcad968a0cc03b4` | 19.3 |
| Z, s = 1e-3 | 7120, `47acaded1a668a36c667874b280bad5c31c7bffb22c3a76cbbb953c837cb397e` | 7220, `ff75557b4d3dda94e3cbf541f3e1215f5a67b2484bed7163eba824dd41cca61d` | 10.0 |

The replication's split (seed 7310, 4,000 shots) was sampled on the team's server: sha256
`07c2f43737bdd07f612669432094b7bd2614fc13614ccabff640c28f4dd29c2e`.

The truth (the status probes) is read in five places only: `fit.py` and `learned.py train` (training shots),
`signal.py` (training shots), `accuracy.py` (scoring), and the ORACLE rows and the scoring in `decode.py`.

## Step 1: the signal

Command: `loss_tracker/runs/step1_step2.sh` (`python -m loss_tracker.signal` on the training split). Files:
`results/signal_train.{md,json}`. Log: `logs/step1_step2.log`. Machine: this PC, lane B.

{{include results/signal_train.md}}

What this says (`loss/flicker.py`, the isolated-atom flicker table L5 that `loss-gates` handed on, was not run:
`signal.py` measures the same rates by round relative to the loss, on clean triples instead of fully isolated
atoms, which keeps several times more data; step 2's table by sharing gives the isolated-atom view of the tracker):

- **Most lost atoms are not alone.** 78% share a check with another lost atom, and 35% have a flagged check atom
  next to them; 14% are alone in every sense. (The brief said each atom shares its checks with 30 others. The
  measured number is 21: 15 through X checks, 15 through Z checks, 9 through both.) Half of the lost atoms
  (48%) already had a randomised check when they were lost. From the loss round on, 46 to 48% of the triples
  are fully clean, 37% have one dirty bit, 12% two and 5% three. A tracker for isolated atoms would cover little.
- **The flicker is there, and it starts within one round.** Before the loss a detector fires 14% of the time.
  In the loss round the rate depends on where in the cycle the atom was lost (0.14 to 0.44); from the next round
  on it is 0.50.
- **The joint pattern is not "all three or none" everywhere.** For a left atom's Z checks and a right atom's X
  checks (three CX in a row) the triple fires together: none 0.34, all three 0.34. For a left atom's X checks
  (CX positions 1, 5, 6) and a right atom's Z checks (1, 2, 6) the detector that is apart from the other two in
  the cycle fires on its own: four patterns at 0.20 each.
- **The rounds of a lost atom are not independent.** In those two cases the lone detector and the pair are tied
  across rounds. For a left atom's X checks, the lone first detector repeats, one round later, what the pair
  showed; for a right atom's Z checks, the pair repeats what the lone last detector showed a round earlier. They
  agree 77.5% of the time (0.5 = unrelated; the rest is noise). The checks that meet the atom at the end of one
  cycle and at the start of the next have no check of the other type between them, so they see the same broken
  state. So a lost atom makes one fresh random bit per check type and round, not two. Present atoms show no such
  relation (0.015 at most).
- **A flagged check atom** makes its own two detectors fair coins (0.50 in rounds c and c + 1). On the data
  atoms next to it, the triple of the OTHER check type fires together more often (all three: 0.8–1.5% → 1.8–5.6%;
  exactly two: 5% → 10–13%): the hook error. That is the pattern a loss makes, so the tracker sets those two
  rounds aside.
- **A check that another lost atom has left is a fair coin** for this atom (0.500), unrelated to the other two
  bits of the triple (largest difference 0.002). So summing that bit out of the tables is exact.

## Step 2: the tracker's accuracy on validation shots, before any decoding

Commands: `loss_tracker/runs/fit.sh` (the tables, from the training split; `tables/X_p0.004_s1e-3.npz`, 78 kB,
sha256 `d915ce275e58caa08d4b8e569502c894fb18a2b0223b534344a8be1154c30597`), `loss_tracker/runs/track_val.sh` (every
version on the 4,000 validation shots), `loss_tracker/runs/step1_step2.sh` (`python -m loss_tracker.accuracy`).
Files: `results/accuracy_val.{md,json}`. Logs: `logs/fit.log`, `logs/track_val.log`, `logs/step1_step2.log`.

Versions: `order0` = independent rounds, `order1` = each round conditioned on the round before; `naive` =
ignores the other lost atoms; `drop` = ignores every check shared with another flagged atom; `coupledN` = the
other atoms' posteriors say from when a shared check is ignored, N sweeps; `explainN` = the same, with shared
bits explained instead of ignored.

{{include results/accuracy_val.md}}

What this says:

- **The kill line is passed.** The line was: under half of the losses of rounds 1 to 9 within one round of the
  truth. The tracker puts 78% there (77% if "rounds 1 to 9" means cycles 1 to 9).
- **The exact round is found half of the time** (51%), and the posterior gives the true round 0.38 on average
  against 0.083 for the uniform guess.
- **How the accuracy changes with the loss round.** It does not fall for late losses, as the brief expected:
  it is lowest in the middle rounds (45% exact, 73 to 77% within one round) and highest at both ends (63% in
  round 1, 73% in round 12). The ends have fewer neighbours to confuse with. And for the last round the herald
  helps: an atom that is known to be lost and whose checks never flickered must have been lost at the end. The
  simplest version (`order0_naive`) shows the same shape (59% and 54% at the ends, 35 to 43% in between), so
  this is not a property of one model. What does change with the round is the direction of the error: early
  losses are placed too late (+0.8 round in round 1), late ones too early (−0.6 in round 12).
- **A single round says little; the timing comes from several.** An atom lost earlier still shows a present
  atom's most common pattern (no detector of the triple fires) in 34% of rounds on its in-a-row triple and 20%
  on its split triple, while noise makes a present atom's triple fire in 35% of rounds (step 1). Atoms that share
  no check with another flagged atom reach 53% exact and 81% within one round. The learned tracker of step 4 does
  slightly better on exactly those atoms (−1.234 against −1.271), so the model leaves a little on the table there.
- **Each piece of the model is worth something** (mean log P of the true round; uniform guess −2.485):
  independent rounds, ignoring the other atoms −1.676; conditioning on the round before −1.536; coupling the
  atoms −1.368; explaining shared bits −1.325. For atoms that share X and Z checks with other flagged atoms
  (54% of them) explaining moves the score from −1.430 to −1.362, most of the way to atoms that share nothing
  (−1.271).
- **The posterior is calibrated**: expected calibration error 0.003 over all (atom, round) pairs; where it says
  0.9 to 1.0 it is right 92% of the time. Temperature 1 gives the best log score, so no temperature is needed
  for calibration. (Whether relay-BP wants a different temperature is a separate question, step 3.)
- **Reruns give the same tables.** Rerunning `track_val.sh` (4 tracker runs side by side instead of one at a time)
  gave bit-identical outputs for the five versions written before, array by array. A second run of
  `step1_step2.sh` after the test (`logs/step1_step2.log`) rewrote `signal_train` and both `accuracy_val` tables
  byte for byte the same (git shows no change).

{{include results/choice_rules.md shift 0}}

## Step 3: the decoder

### The plumbing check

Command: `loss_tracker/runs/plumbing.sh`. This folder's pipeline (`decode.py`, `priors.py`) with the TRUE loss
round put in by hand, on the first 500 shots of `loss-gates`' dataset (seed 2300), against `loss-gates`' per-shot
results for its ORACLE `oracle-round`. File: `results/plumbing.json`. Log: `logs/plumbing.log`.

```json
{{include results/plumbing.json}}
```

Array-wise, the priors are bit-identical to `loss.lossmap.PriorBuilder`'s for `oracle-round`, and with the
uniform guess to its `exact` (`test_loss_tracker.py::test_plumbing_true_round_equals_oracle`).

### Validation: the variants, and the choice

Commands: `loss_tracker/runs/val_references.sh` (`exact` and the ORACLE, with `loss.decode`),
`loss_tracker/runs/val_variants.sh` and `runs/val_part.sh` (the five variants), `loss_tracker/runs/val_explain.sh`
(the later tracker versions), `loss_tracker/runs/val_summary.sh` (the tables and the choice). Logs:
`logs/val_references.log`, `logs/val_variants.log`, `logs/val_variants_laneB.log`, `logs/val_explain.log`,
`logs/val_summary.log`. Machine: this PC, lanes B and D, 4 processes each.

{{include results/val_variants.md}}

{{include results/choice.md}}

{{include results/val_frozen.md}}

{{include results/val_notes.md}}

### Test: one run of the frozen decoder

The frozen pair (`runs/frozen.sh`: tracker version `order1_explain1`, soft posterior), with `starter`, `exact` and
the ORACLE `oracle-round`, every condition on every one of the 4,000 test shots (seed 7300). Commands:
`loss_tracker/runs/test.sh 0 0` (the tracker's output for the split; it reads detectors and heralds only), then
two streams, `LANE=D loss_tracker/runs/test.sh 0 2000` and `LANE=B loss_tracker/runs/test.sh 2000 4000`, at
commit `61d4e92` (the log's first line names it), and `loss_tracker/runs/test_summary.sh` for the tables. Logs:
`logs/test_track.log`, `logs/test_laneD.log`, `logs/test_laneB.log`, `logs/test_summary.log`. Machine: this PC,
lanes D and B, 4 processes each. Rows with `>` are second attempts computed from the per-shot results (`A>B`: A,
and B where A did not converge); they are extra rows, not the frozen decoder.

{{include results/test.md}}

{{include results/test_notes.md}}

The frozen tracker against the probes of the test split (`loss_tracker/runs/diagnose.sh`, `python -m
loss_tracker.accuracy`; scoring only, after the decoding; the full tables are in `results/accuracy_test.md`):

{{include results/accuracy_test.md from 3 to 7}}

Where the tracker loses against the ORACLE, shot by shot (`loss_tracker/runs/diagnose.sh`, `python -m
loss_tracker.diagnose`; `results/diagnose_test.md`, and `results/diagnose_val.md` for the validation shots):

{{include results/diagnose_test.md}}

#### A secondary row: the tracker version before the frozen one

`order1_coupled2` (shared bits dropped instead of explained) had fewer failures on the validation shots than the
frozen version (45 against 54, p = 0.18). It was decoded on the same test shots after the frozen run
(`loss_tracker/runs/test_secondary.sh`, `loss_tracker/runs/test_summary.sh secondary`; log
`logs/test_secondary.log`). It is not the frozen decoder, and the primary number does not use it.

{{include results/test_secondary.md}}

#### The replication on the team's server (secondary)

The same job on a second seed (7310), sampled and decoded on the team's server by one command
(`python -m loss_tracker.job --seed 7310 --shots 4000 --workers 4 --mode coupled --sweeps 1 --order 1 --explain
--out ... --tables loss_tracker/tables/X_p0.004_s1e-3.npz --conditions starter,exact,tracker,oracle-round`, ctl job
060 of the orchestrator, from `integration` at `650f7ae`). Its printout is `job_ctl_060-loss-tracker-replication.out`
on the branch `server-logs` (commit `d13f0df`), copied to `logs/server_job_ctl_060.out`. Its files were copied back and
its tables rebuilt here from its per-shot parts, with the same rows as the test's (`loss_tracker/runs/replication.sh`;
log `logs/replication.log`). The rebuilt rows equal the server's own `tables.md`
(`results/replication_seed7310_server_tables.md`). The dataset's sha256 on the server is `07c2f437…`. Machine: the
team's server, 4 worker processes.

{{include results/replication_seed7310.md}}

The tracker alone on the replication's probes (`results/replication_seed7310_accuracy.md`):

{{include results/replication_seed7310_accuracy.md from 3 to 7}}

The secondary row on the replication shots (the earlier tracker version, `rep_seed7310_coupled2`):

{{include results/replication_seed7310_secondary.md from 26 to 35}}

Pooled with the test (8,000 shots: the test's shots 0–3,999, the replication's numbered 4,000–7,999;
`python -m loss_tracker.pool`):

{{include results/pooled_7300_7310.md}}

## Step 4

{{include results/step4_plan.md}}

### A learned tracker

`learned.py`: a bidirectional GRU (64 units, 34,950 weights) over the 13 detector rounds of one flagged atom. Its
inputs are the 3 X-check and 3 Z-check detector bits, which of them are reliable, the heralds of its 6 check atoms
and of the other-type check atoms next to it, its side, and how many other flagged data atoms sit on each of its
checks. Its output is a softmax over the 72 loss slots. Trained on the training split (seed 7100, 200,376 lost
atoms) with the probes' slot as the label: 20 epochs, Adam with a one-cycle schedule (peak 3e-3), batches of 512,
seed 0, CPU, 4 threads. Model file `tables/learned_X_p0.004_s1e-3.pt`. Command `loss_tracker/runs/learned.sh`;
log `logs/learned.log`; table `results/accuracy_val_learned.md` (validation split, seed 7200).

{{include results/accuracy_val_learned.md from 1 to 10}}

{{include results/step4_learned.md}}

### Other loss rates: s = 2e-3 (about today's hardware) and s = 5e-4

`loss_tracker/runs/extra.sh`: a training split (20,000 shots) and a validation split (4,000 shots) at the new
rate, the tables fitted on that training split, the frozen tracker version run on the validation split (`tracker`),
and the same shots with the tables fitted at s = 1e-3 (`tables_of_1e-3`). Logs `logs/extra_X_s2e-3.log`,
`logs/extra_X_s5e-4.log`.

s = 2e-3 (training seed 7140, validation seed 7240):

{{include results/accuracy_val_X_p0.004_s2e-3.md from 1 to 25}}

s = 5e-4 (training seed 7160, validation seed 7260):

{{include results/accuracy_val_X_p0.004_s5e-4.md from 1 to 25}}

### Z memory (s = 1e-3)

Training seed 7120, validation seed 7220, tables fitted on Z-memory shots (`tables/Z_p0.004_s1e-3.npz`). Log
`logs/extra_Z_s1e-3.log`.

{{include results/accuracy_val_Z_p0.004_s1e-3.md from 1 to 25}}

### The tracker's cost per shot

`loss_tracker/runs/timing.sh`: one process pinned to CPU 0 under the lane-D lock, numpy on one thread, the first
300 validation shots. Log `logs/timing.log`.

{{include results/timing.md}}

{{include results/step4.md}}

## What this says

{{include results/what_this_says.md}}

## What is not tested, and limits

- **Decoding at one point only**: X memory, p = 0.004, s = 1e-3, perfect heralds, independent loss, no leakage.
  The tracker's accuracy alone was also measured at s = 5e-4 and 2e-3 and in Z memory (step 4); decoding there
  was not run.
- **Not simulated**: other Pauli strengths p, correlated (pair) loss, imperfect heralds, leakage.
- **The emission tables were fitted at the test's p.** They carry over between loss rates (step 4); whether they
  carry over to another p is not known.
- **The decoder's map is `loss-gates`' own**, not changed or re-tuned here: the Pauli envelope, the cap at 0.5,
  and relay-BP at the kit's settings.
- **The share of the oracle's gain is known only to about ±0.2** with 4,000 shots (0.58, 0.38–0.78 on the test).
- **The cost** is measured in numpy on one thread of a laptop-class CPU, not in an implementation built for speed.

## Disclosures

{{include results/disclosures.md}}
