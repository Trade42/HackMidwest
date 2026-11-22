# A loss-time tracker for the Gross code: when was each lost data atom lost?

Task `loss-tracker` (idea T2 of `orchestration/PLAN.md`). It builds on `loss/` (task `loss-gates`): the same loss
model, the same Clifft simulation and the same map from a loss to the decoder's columns. These are exploration
numbers on shots that Clifft simulates. No file of the team's `data/` folder is involved.

**DRAFT: the tables are filled in as the runs finish. A row that is not measured yet says so.**

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
| Hidden variable | For each flagged data atom, its loss slot j = 1..72 (lost after its j-th CX). Prior: geometric, P(j) ∝ (1 − s)^(j−1). |
| Observations | Per detector round, the 3 detector bits of the atom's X checks as one pattern of 8, and the 3 bits of its Z checks as another. |
| Emission tables | P(pattern | pattern of the round before, class), counted on training shots, where the class says how far the loss is from the round: still present; lost in this cycle, in the cycle before, or two cycles before, after CX k (k = 1..6 each); lost three or more cycles before. Separate tables for left and right data atoms (their CX orders differ), for X and Z checks, and for the first two rounds and the last. Counted only on "clean" triples (see below). |
| Why a round is conditioned on the round before | In this CX schedule a lost atom's detectors are not independent between rounds: one detector repeats, one round later, what its two partners showed (step 1). A model with independent rounds counts that random bit twice. |
| Unreliable ("dirty") bits | (a) The detector of a check whose own atom was flagged in this cycle or the one before is a fair coin. (b) So is the detector of a check that ANOTHER lost data atom has already left. (c) All 3 bits of a triple are set aside when a check atom of the other type next to the atom was flagged in this cycle or the one before (its hook error flips the triple like a loss does). (a) and (c) are known from the heralds. A dirty bit is summed out of the tables. |
| Atoms that share checks | (b) depends on the unknown loss times of the other flagged atoms. The tracker starts from their priors and then recomputes every atom's posterior from the others' posteriors of the sweep before, all at once, twice (synchronous message passing). We chose this over alternating atom by atom because the result does not depend on any atom order and it vectorises; one sweep is already at the fixed point to three digits (step 2). Different atoms are treated as independent given their posteriors. |
| What is NOT modelled | The value of a bit another lost atom has randomised is thrown away, although that atom's other checks say something about it. Correlations between the X triple and the Z triple of an atom are ignored. |
| Use in the decoder | The posterior over the slot is summed to the 12 rounds and spread over the 6 slots of each round as the ORACLE `oracle-round` does; then `loss.lossmap`'s `data_activation` and the rest of the condition `exact`, unchanged (`priors.py`). |

## Splits

| Split | Seed | Shots | sha256 of the file | Lost data atoms per shot | Used for |
|---|---|---|---|---|---|
| training | 7100 (chunks 7100–7109) | 20,000 | `5d32f279bbfd4619bc2b45f9483d1344f44c4ef02476628db88162367751563e` | 10.02 | the emission tables (`fit.py`), the signal tables of step 1 |
| validation | 7200 (7200–7201) | 4,000 | `8b09d471d063650d5740c8bc2ad7bbfe24a784c7b7e16d8d82124a471dc454e8` | 10.04 | the tracker's accuracy (step 2); the choice of the decoder variant on its first 1,500 shots (step 3) |
| test | 7300 (7300–7301) | 4,000 | `4ee65bb56e69fcb51fd638b17f3c766178dece67c4a990fe0e93ec80f1225583` | 9.98 | one run of the frozen decoder (step 3) |

All are X memory, p = 0.004, s = 1e-3, sampled on this PC (`runs/sample.sh`, log `logs/sample.log`). The files
are in `/home/sal/hackMidwest/orchestration/scratch/loss-tracker/data/` and are not committed.

The truth (the status probes) is read in four places only: `fit.py` (training shots), `signal.py` (training
shots), `accuracy.py` (scoring), and the ORACLE rows and the scoring in `decode.py`.

## How the frozen decoder is chosen (written at 06:53 UTC, before the variant runs finished)

1. **The tracker version** is the one with the best mean log P(true round) on the 4,000 validation shots (step 2).
2. **The decoder variant** is the one with the fewest failures on the first 1,500 validation shots, among: the
   soft posterior (`tracker`), two tempered posteriors (`tracker-T2` flatter, `tracker-T0.5` sharper), the single
   most probable round where its posterior is at least 0.6 (`tracker-hard0.6`), and the posterior over the 72
   slots used as it is (`tracker-slot`). Ties: fewer unconverged shots, then fewer mean iterations.
   - These five were decoded with the tracker version `order1_coupled2`, the best one when the runs started. If
     rule 1 picks a later version, the winning variant is decoded once more with that version on the same
     validation shots, for the record; the variant is not re-chosen.
3. A second attempt for unconverged shots (the tracker's priors first, then the uniform guess where relay-BP did
   not converge) is reported as an extra row computed from the per-shot results. It is not the frozen decoder.
4. The test split is decoded once, with the frozen pair, together with `starter`, `exact` and the ORACLE.
