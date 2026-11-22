# Gates G1 to G3: how relay-BP fails, how much accuracy sits above it, where its time goes

**These are exploration numbers on train, validation and exploration shots. Nothing here is a final result.** No test or go file was opened: the loader (`analysis/common.py`, `refuse_held_out`) refuses any file whose name starts with `test_` or `go_`.

## In short

Only what the tables below support. Relay-BP is the kit's baseline (80 / 60 / stop after 5, 300 legs); a pair is shot k of X with shot k of Z.

- **G1, how relay-BP fails.** It is both: its failures split between "did not converge" and "converged to a wrong answer", and the mix moves with p. The unconverged share is 36% at p = 0.003 (44 failures), 41% at 0.004 (245) and 63% at 0.005 (404). At every p, 84% to 98% of the failures come from the shots whose first 80-iteration leg fails (3.5%, 14% and 36% of shots). Where the first leg converges, its answer is almost always right.
- **G2, how much accuracy sits above relay-BP.** A lot, but most of it is not reached by simple means. At p = 0.004:
  - decoding a failed shot again with other random draws fixes 44% of the failures (the re-roll floor);
  - a larger budget fixes 53%;
  - BP-OSD (order 10 or 60) and Tesseract (beam 5) fix 32% to 33%, below the floor;
  - among 16 relay-BP candidates the right logical class is present for 79% of the converged-wrong failures, but the lowest weight picks it for only 49%, no better than one re-roll.
  - Joint decoding on all 1,728 detectors fixes 43 of 44 failures at p = 0.003 and 89 of 100 sampled at 0.004, at about 60 to 70 times the time on a typical shot. It broke 0 of 300 and 1 of 200 controls.
  - The relay-BP authors' own runs put joint decoding overall 2.5 times below separate decoding at p = 0.003, 1.3 times at 0.004, and above it at 0.005.
  - At p = 0.005 the same pattern holds with lower fixes: a re-roll fixes 27%, a larger budget 37%, BP-OSD about 15% (breaking about 10% of controls), and the right class is among the 16 candidates for 79% of converged-wrong failures while the lowest weight finds it for 42%.
- **G3, where the time goes.** The slowest 10% of shots take 48% to 71% of all iterations (p = 0.003 to 0.004).
  - At p = 0.004 the later legs must stay at 60 iterations.
  - Stopping after 3 solutions (80 / 60 / 3) cuts mean iterations 1.44 times with no measurable cost: 42 failed pairs against 51 of 5,000.
  - Stopping at the first solution (80 / 60 / 1) cuts them 3.4 times for about 40% more failures.

## Setup

- **Machine:** luminaty, AMD Ryzen 7 7730U (a 15 W laptop CPU, 8 cores, 16 threads), inside the dq fence. Every heavy run was niced (`nice -n 15`) and pinned to cores 8 to 15, which are 4 physical cores with 2 threads each, under the lane B and C locks. Other workers loaded cores 0 to 7 at the same time.
- **Processes:** 8 side by side, unless a row says otherwise. Under all-core load this CPU throttles: the kit's relay-BP ran at about 0.76 ms per iteration per process, against 0.28 ms on one idle core.
- **Milliseconds** in every table are the time inside the decoder call on this machine with these 8 processes, and are labelled so. **Iterations** do not depend on the machine. Compare schedules by iterations.
- **Versions:** Python 3.12.15, stim 1.16.0, ldpc 2.4.1, relay-bp 0.2.2, numpy 2.5.5, scipy 1.18.1, and tesseract-decoder 0.1.1.dev20261117025421 for the probe. The venv was built from `requirements.txt` outside the worktree.
- **The commands** are the steps of `analysis/steps.sh`, run from `challenge-1-ml-decoder-kit/` with `V`, `S` and `D004` set:

  ```sh
  export V=/home/sal/hackMidwest/orchestration/scratch/gates/venv S=/home/sal/hackMidwest/orchestration/scratch/gates \
         D004=/home/sal/hackMidwest/repo/challenge-1-ml-decoder-kit/data/p0.004
  analysis/steps.sh datasets      # the exploration sets and the manifests
  analysis/steps.sh step1         # and so on: each section names its steps
  ```

  On `main` the p = 0.004 files are `challenge-1-ml-decoder-kit/data/p0.004/`, so `D004=data/p0.004` works there.

## Datasets

A Stim seed does not give the same shots on every machine, so every dataset is named by its sha256 (`results/datasets_p0.004.json`, `results/datasets_exploration.json`).

| p | file | basis | seed | shots in file | shots used | sha256 |
|---|---|---|---|---|---|---|
| 0.003 | `explore_X.npz` (scratch) | X | 9103 | 100,000 | first 50,000 | `76efa8a04c9b857fff81e6f62869d6cfeb9caf30be5a816aa25e43878b2b01a1` |
| 0.003 | `explore_Z.npz` (scratch) | Z | 9104 | 100,000 | first 50,000 | `9355d3f722c412abba43cdafa3947314d7da4a625108198f9412ed50e7e801b6` |
| 0.004 | `train_X.npz` (main) | X | 101 | 20,000 | all | `98c0afeb2a0b6b27f3b689e832160a060a4c0aeb37d229f84daa1995a9a0f7cc` |
| 0.004 | `train_Z.npz` (main) | Z | 102 | 20,000 | all | `14dce11586fa51256f476224d0fe3965e0562f2d8f68cba6192883a65fb6b24a` |
| 0.004 | `validation_X.npz` (main) | X | 201 | 5,000 | all | `b00ee09c35ed088a12bcba60b8b704b528c47fb156c7d09cf0f171f90a181f3d` |
| 0.004 | `validation_Z.npz` (main) | Z | 202 | 5,000 | all | `843c4daf24e2b56c5988670550bbcdbac8f50b18dbb9d24beb64969146cb8e20` |
| 0.005 | `explore_X.npz` (scratch) | X | 9105 | 20,000 | first 5,000 | `8fcd2650edbaec915e2d52ef73549d1aac69345ed7500738fb129a46a3f55ed4` |
| 0.005 | `explore_Z.npz` (scratch) | Z | 9106 | 20,000 | first 5,000 | `43346c307e995fbb08e81f5cb807a438a668010e8afaf45d5c85ffadc065102a` |

The exploration sets were made with the kit's `make_dataset.py` (`analysis/steps.sh datasets`, stim 1.16.0). They are not committed: they sit in `/home/sal/hackMidwest/orchestration/scratch/gates/data/`.

Since this task was briefed, `main` gained the team's own shot files at the other two noise levels (commit `80b9b48`): `data/p0.003/` and `data/p0.005/`, each with train (20,000), validation (5,000) and held-out test (10,000) files for X and Z. They are not the exploration sets used here, which came from other seeds (9103 to 9106) and are named by their sha256 above. Nothing here opened them.

**Shot cuts, against the brief's aims:** the brief aimed at 100,000 shots per basis at p = 0.003 and 20,000 at 0.005. Its lanes were cut from 12 to 8 processes, and this CPU turned out about 2.7 times slower per process under full load than on one core. So p = 0.003 uses the first 50,000 shots of each file (the orchestrator's suggested cut) and p = 0.005 the first 5,000 (about 200 failures per basis, still above the brief's 100).

## What was cut or changed against the brief

- **Lanes.** The brief granted lanes B, C and D (12 processes). The orchestrator cut that to B and C (8 processes) at 01:43 UTC. Only the first smoke run (2 processes, 2 chunks of validation_X) ran on cores 0 to 3 as well.
- **Shots.** At p = 0.003 G1 used 50,000 shots per basis, not 100,000. At p = 0.005 it used 5,000, not 20,000 (see Datasets).
- **G2 samples.** These are seeded subsamples, stated in each table and set in `analysis/steps.sh`:
  - at p = 0.005, 100 of the failures and 100 controls per basis;
  - for the 16-candidate runs, at most 10 (p = 0.004) or 5 (p = 0.005) of G1's unconverged failures per basis;
  - for joint decoding at p = 0.004, 50 failures and 100 controls per basis;
  - at p = 0.003 the joint control was 150 shots per basis.
- **G2 item 5, joint decoding,** was added by the orchestrator (2026-11-22 01:47 UTC), with setting (b) and p = 0.003 as its main point. Joint decoding at p = 0.005 (step 7b) was planned and then dropped by the orchestrator for the deadline: it was not run.
- **On another machine.** G3 item 2 (the frontier) and G2 items 1 and 2 at p = 0.004 ran on the team's ICRN server, by the orchestrator's decision. That machine was shown to give luminaty's iterations and corrections exactly. Its milliseconds are its own.
- **Not done.** G3 item 3 (four schedules at p = 0.003 and 0.005 on 3,000 shots) was not run. Tesseract was tried at its default beam only, inside the 30-minute box.
- **Choices made here.** BP-OSD's "higher order" is OSD-CS order 60. On 5 failed shots it cost 1.9 times order 10, and order 20 or 40 differed little in cost.

## How every run is made reproducible

Relay-BP keeps its random state from one decode to the next: its memory strengths for legs 2 onwards are drawn from one generator, seeded once when the decoder is built (`relay_bp/src/bp/relay.rs`, `init_next_set`). So a decoder's answers depend on which shots it decoded before.

- **G1 and G3:** every dataset is cut into fixed 500-shot chunks in file order. Each chunk is decoded in order on a fresh decoder with seed 0, which is the constructor's default and what the kit's `relay_decoder` gets. Each chunk's output is its own file, and the outputs are put back together in file order (`analysis/common.py`).
- **G2:** every shot gets its own fresh decoder (one per seed for the 16 candidates). No G2 result depends on which other shots were decoded.
- **Scoring is the kit's.** Every decode goes through `decoders.score`: a shot fails when any of the 12 predicted observables is wrong. An unconverged shot is scored on the correction relay-BP returned, which is the first leg's hard decision. The per-shot predictions are checked against `score`'s failure flags.
- **The truth reaches no decoder.** The observables are used only for scoring and, in G2, to pick the failed and control sets (from G1's scores) and to score the candidates.

### The process-count check (done-when 4): passed

`analysis/check_processes.py` (run inside `analysis/steps.sh step2`; output `results/check_processes.json`, log `logs/check_processes.log`), on `validation_X.npz` at p = 0.004:

| check | shots | result |
|---|---|---|
| one process decodes chunks 2 to 5 again, against G1's 8-process run | 2,000 | **identical**: 0 shots differ in failed, converged, iterations, predicted bits, weight (count and log-odds), validity or correction (hash); 11 failures in both |
| the kit's plain `decode()` against `decode_detailed()`, chunk 0, fresh decoders | 500 | identical corrections on all 500 |
| control: one decoder for chunks 0 and 1 without the fresh start at shot 500 | 1,000 | chunk 0 identical; in chunk 1, **309 of 500 corrections and 492 of 500 iteration counts differ** |

So the outputs do not depend on the process count, and the check can see the decoder's carried-over state when it is there.

## G1: how relay-BP fails

Decoder: the kit's baseline, `decoders.relay_decoder(H, priors)` with `RELAY_SETTINGS` unchanged (first leg 80 iterations, up to 300 more legs of 60, stop after 5 solutions), seed 0, through `decode_detailed`. A second pass decodes the same chunks with the first leg alone (`num_sets=0`, which uses no random draws). It says whether the first 80 iterations found a solution: at stop-after-5 the kit's decoder goes on after that, so its own iteration count cannot say.

Commands: `analysis/steps.sh step1` (p = 0.004), `step3` (p = 0.003), `step6` (p = 0.005), then `analysis/steps.sh summarize-g1`. Logs: `logs/g1_run_p*.log`, `logs/g1_summary.log`. Summary: `results/g1_summary.json` and `.csv`. Per shot: `results/pershot/g1_p<p>_<basis>_<split>.npz`, with failed, converged, iterations, the 12 predicted bits (`pred`, packed into a uint16, bit j for observable j), the truth (packed the same way), the correction's count (`wcount`) and its sum of log((1 - prior) / prior) (`wllr`), valid (the correction reproduces the syndrome), the first leg's converged, iterations, failed and pred, and the time (`ms`, float16).

### p = 0.004 (train and validation, 25,000 shots per basis)

| basis | shots | failures | did not converge | converged to a wrong answer | rate per basis [95% Wilson] | iterations mean / median / p90 / p99 / max | first leg converged |
|---|---|---|---|---|---|---|---|
| X | 25,000 | 122 | 55 | 67 | 4.88e-3 [4.09e-3, 5.82e-3] | 539 / 135 / 762 / 11,355 / 18,080 | 86.2% [85.8, 86.6] |
| Z | 25,000 | 123 | 45 | 78 | 4.92e-3 [4.13e-3, 5.87e-3] | 547 / 137 / 760 / 12,086 / 18,080 | 85.7% [85.3, 86.1] |

Per file: validation_X 31 of 5,000, validation_Z 20 of 5,000, train_X 91 of 20,000, train_Z 103 of 20,000 (all in `results/g1_summary.csv`).

| basis | failures whose first leg converged | the first leg's own answer wrong, where it converged | failures if a decoder kept the first leg's answer whenever it converged | converged-wrong failures that ran at least 296 legs (found 1 to 4 solutions) | iterations of the failed shots, median | unconverged shots scored right |
|---|---|---|---|---|---|---|
| X | 12 of 122 | 15 of 21,550 | 125 (kit: 122) | 24 of 67 | 18,053 | 2 of 57 |
| Z | 9 of 123 | 14 of 21,430 | 128 (kit: 123) | 29 of 78 | 18,029 | 0 of 45 |

The returned correction's weight: median 46 faults (sum of log-odds 225.5) on the shots relay-BP got right, against 55 faults (272 and 267) on the converged-wrong failures (`results/g1_summary.json`). Time per shot, 8 processes on this machine: mean 404 ms (X) and 466 ms (Z), median 102 and 117 ms, p99 8.1 s and 10.3 s.

**Sanity line (done-when 5).** On `validation_X.npz` relay-BP failed 31 of 5,000 shots, 6.2e-3 [4.4e-3, 8.8e-3]. The kit's reference, 45 of 10,000 (4.5e-3), is inside that interval. On `validation_Z.npz` the counts are 20 of 5,000, 4.0e-3 [2.6e-3, 6.2e-3], and the reference, 60 of 10,000 (6.0e-3), is inside too.

**What this says at p = 0.004:**

- Relay-BP's failures split about evenly between "did not converge" (100 of 245) and "converged to a wrong answer" (145 of 245).
- The failures sit almost all in the 14% of shots whose first leg fails: 224 of 245.
- Where the first leg converges, its answer is as good as the kit's 5-solution pick. Keeping it would give 125 + 128 failures against the kit's 122 + 123 on the same shots.
- Most failed shots use (nearly) the whole budget: the median failed shot ran 18,000 iterations.
- Converged-wrong corrections are heavier than right ones (median 55 faults against 46). But weight alone does not say which shots are wrong: no threshold was tested here.

### p = 0.003 (the first 50,000 shots of each exploration file)

| basis | shots | failures | did not converge | converged to a wrong answer | rate per basis [95% Wilson] | iterations mean / median / p90 / p99 / max | first leg converged |
|---|---|---|---|---|---|---|---|
| X | 50,000 | 23 | 7 | 16 | 4.60e-4 [3.07e-4, 6.90e-4] | 146 / 76 / 199 / 1,088 / 18,080 | 96.6% [96.4, 96.7] |
| Z | 50,000 | 21 | 9 | 12 | 4.20e-4 [2.75e-4, 6.42e-4] | 143 / 77 / 203 / 1,038 / 18,080 | 96.5% [96.3, 96.6] |

| basis | failures whose first leg converged | the first leg's own answer wrong, where it converged | failures if a decoder kept the first leg's answer whenever it converged | converged-wrong failures that ran at least 296 legs | iterations of the failed shots, median | unconverged shots scored right |
|---|---|---|---|---|---|---|
| X | 6 of 23 | 7 of 48,290 | 24 (kit: 23) | 2 of 16 | 7,822 | 2 of 9 |
| Z | 1 of 21 | 3 of 48,234 | 23 (kit: 21) | 1 of 12 | 15,858 | 0 of 9 |

The kit's README reports 43 and 54 failures in 100,000 shots per basis at p = 0.003 (4.3e-4 and 5.4e-4), and the relay-BP authors 32 and 40 (3.2e-4 and 4.0e-4). Our 4.6e-4 and 4.2e-4 agree with both within the intervals. Correction weight: median 35 faults on right shots against 43 to 44 on converged-wrong failures. Time per shot, 8 processes: mean 134 and 128 ms, median 70 ms, p99 1.0 s and 0.93 s.

**What this says at p = 0.003:** about 44 failures in 100,000 shots, too few for fine splits.

- Two thirds of them converged to a wrong answer (28 of 44).
- The first leg converges on 96.5% of shots. Keeping its answer there would cost 3 more failures than the kit's 44 (47 in all).
- As at p = 0.004, the failures sit mostly where the first leg fails: 37 of 44.

### p = 0.005 (the first 5,000 shots of each exploration file)

| basis | shots | failures | did not converge | converged to a wrong answer | rate per basis [95% Wilson] | iterations mean / median / p90 / p99 / max | first leg converged |
|---|---|---|---|---|---|---|---|
| X | 5,000 | 207 | 125 | 82 | 4.14e-2 [3.62e-2, 4.73e-2] | 2,464 / 368 / 9,181 / 18,080 / 18,080 | 63.7% [62.4, 65.0] |
| Z | 5,000 | 197 | 128 | 69 | 3.94e-2 [3.44e-2, 4.52e-2] | 2,291 / 372 / 7,101 / 18,080 / 18,080 | 64.1% [62.8, 65.4] |

| basis | failures whose first leg converged | the first leg's own answer wrong, where it converged | failures if a decoder kept the first leg's answer whenever it converged | converged-wrong failures that ran at least 296 legs | iterations of the failed shots, median | unconverged shots scored right |
|---|---|---|---|---|---|---|
| X | 6 of 207 | 6 of 3,185 | 207 (kit: 207) | 45 of 82 | 18,080 | 1 of 126 |
| Z | 3 of 197 | 5 of 3,206 | 199 (kit: 197) | 34 of 69 | 18,080 | 1 of 129 |

The kit's README reports 411 and 414 failures in 10,000 shots per basis at p = 0.005 (4.1e-2), and the relay-BP authors 216 and 205 in 5,000 (4.3e-2 and 4.1e-2). Ours agree. Correction weight: median 57 faults on right shots against 66 on converged-wrong failures. Time per shot, 8 processes: mean 2.3 s (X) and 1.8 s (Z), median 0.35 and 0.28 s, p99 18 and 17 s.

**What this says at p = 0.005:**

- Here most failures did not converge: 253 of 404 (63%).
- The first leg converges on only 64% of shots, and its answer there is almost always right (11 wrong in 6,391).
- Of the 151 converged-wrong failures, 79 ran at least 296 legs, so they found only 1 to 4 solutions.
- Relay-BP's failures at p = 0.005 are mostly shots where it runs out of budget, not shots where it is fooled.

### Across p

| p | failures X + Z (shots per basis) | did not converge | converged wrong | first leg converged | failures where the first leg did not converge |
|---|---|---|---|---|---|
| 0.003 | 44 (50,000) | 16 (36%) | 28 (64%) | 96.5% | 37 of 44 |
| 0.004 | 245 (25,000) | 100 (41%) | 145 (59%) | 86.0% | 224 of 245 |
| 0.005 | 404 (5,000) | 253 (63%) | 151 (37%) | 63.9% | 395 of 404 |

**G1's answer: it is both, and the mix moves with p.**

- At p = 0.003 and 0.004 most failures converged to a wrong answer: those need a better choice among solutions.
- At p = 0.005 most did not converge: those need a solution found at all.
- At every p, nearly all failures (84% to 98%) come from the shots whose first leg fails. Where the first leg converges, keeping its answer costs 0 to 5 failures per basis against the kit's 5-solution pick.

## G2: how much accuracy sits above relay-BP

**The sets.** Per p and basis:

- **F** is every shot the kit's relay-BP failed in G1.
- **C**, the control, is the same number of shots it got right, drawn without replacement with numpy `default_rng([20261122, round(p * 1e4), ord(basis)])`. The selections are `results/g2_selection_p*.npz`.

Every G2 decode runs on a fresh decoder per shot, so no result depends on the other shots. Commands: `analysis/steps.sh step2`, `step2b`, `step4`, `step5` and `step7`, then `analysis/steps.sh summarize-g2`. Code: `analysis/g2_headroom.py`. Summary: `results/g2_summary.json` and `results/g2_tables.csv`. Log: `logs/g2_summary.log`.

**How to read the tables.**

- The 2 x 2 counts are relay-BP (G1) against the other decoder on F + C, and the exact paired test is the two-sided binomial test of the discordant shots.
- Because C is a sample of about 0.5% of the shots relay-BP got right, the raw test compares the other decoder's fix rate on failures with its break rate on a small sample. It says nothing about which decoder fails less overall: BP-OSD, which fails about 5 times as often as relay-BP (the kit's README), "wins" it at p = 1e-10.
- The column "other's failures on all G1 shots" therefore scales up both parts: relay-BP's failures times (1 - fix rate), plus all of relay-BP's right shots times the control's break rate. Its range uses the exact (Clopper-Pearson) 95% intervals of both rates. With 122 controls, a break rate can only be bounded at about 3%, so the range is wide. **Read the fix rate, the cost per shot and the range, never the raw p-value, as the answer.**
- **The re-roll floor.** Relay-BP is randomized: legs 2 onwards draw their memory strengths at random. `kit_fresh` decodes each shot again with the kit's settings and seed 0 on a fresh decoder, so with other random draws than in G1. It fixes failures by luck alone, and every other row's fixes should be read against it.

### Items 3 and 4 at p = 0.004, and the re-roll floor

| decoder | basis | F | fixed: all / of unconverged / of converged-wrong | C | broken | 2 x 2: both wrong / only relay-BP / only other / both right | exact paired p | break rate [95% CP] | other's failures on all 25,000 shots, estimate [range], against relay-BP's | iterations mean F / C | ms mean F / C |
|---|---|---|---|---|---|---|---|---|---|---|---|
| re-roll (kit settings, fresh decoder) | X | 122 | 56 / 24 of 55 / 32 of 67 | 122 | 0 | 66 / 56 / 0 / 122 | 2.8e-17 | 0.000 [0.000, 0.030] | 66 [55, 818] vs 122 | 13,205 / 373 | 9,953 / 292 |
| re-roll | Z | 123 | 52 / 16 of 45 / 36 of 78 | 123 | 2 | 71 / 52 / 2 / 121 | 1.6e-13 | 0.016 [0.002, 0.058] | 476 [109, 1,512] vs 123 | 12,626 / 752 | 8,460 / 540 |
| BP-OSD, OSD-CS order 10 (kit) | X | 122 | 43 / 21 of 55 / 22 of 67 | 122 | 2 | 79 / 43 / 2 / 120 | 5.9e-11 | 0.016 [0.002, 0.058] | 487 [117, 1,531] vs 122 | 79 / 53 | 1,148 / 464 |
| BP-OSD, order 10 | Z | 123 | 36 / 11 of 45 / 25 of 78 | 123 | 4 | 87 / 36 / 4 / 119 | 1.9e-7 | 0.033 [0.009, 0.081] | 896 [298, 2,116] vs 123 | 79 / 52 | 1,214 / 419 |
| BP-OSD, OSD-CS order 60 | X | 122 | 43 / 21 of 55 / 22 of 67 | 122 | 2 | 79 / 43 / 2 / 120 | 5.9e-11 | 0.016 [0.002, 0.058] | 487 [117, 1,531] vs 122 | 79 / 53 | 1,331 / 519 |
| BP-OSD, order 60 | Z | 123 | 38 / 12 of 45 / 26 of 78 | 123 | 4 | 85 / 38 / 4 / 119 | 5.7e-8 | 0.033 [0.009, 0.081] | 894 [296, 2,114] vs 123 | 79 / 52 | 1,245 / 441 |
| Tesseract, det_beam 5 | X | 122 | 43 / 21 of 55 / 22 of 67 | 122 | 2 | 79 / 43 / 2 / 120 | 5.9e-11 | 0.016 [0.002, 0.058] | 487 [117, 1,531] vs 122 | – | 3,743 / 571 |
| Tesseract, det_beam 5 | Z | 123 | 37 / 11 of 45 / 26 of 78 | 123 | 4 | 86 / 37 / 4 / 119 | 1.0e-7 | 0.033 [0.009, 0.081] | 895 [297, 2,115] vs 123 | – | 3,676 / 577 |

The ms are per shot inside the decode call, a fresh decoder per shot (its construction not timed), with 8 processes on this machine. BP-OSD's iterations are its BP stage's.

Tesseract's X row equals BP-OSD's in every count, which was checked shot by shot because it looked like a bug. It is not one: the two agree on the failed flag on 204 of the 244 X shots and differ on the other 40 (`agreement` in `results/g2_summary.json`). Tesseract's correction is lighter than BP-OSD's on 193 of the 244 shots, yet it fixes no more of relay-BP's failures.

**Item 4, the Tesseract probe** (`analysis/tesseract_probe.py`, `results/tesseract_probe.json`). `pip install tesseract-decoder` installed version 0.1.1.dev20261117025421 here. It took the kit's 936-detector problem as a DEM built from H, L and the priors, with one error per column; building the decoder took 0.3 s. It decoded the first 3 failed validation_X shots in 3.9 to 5.7 s each (under the 60 s limit) and got all 3 wrong. Its corrections there were heavier than relay-BP's (sum of log-odds 322 against 314, 280 against 251, 302 against 299). Its settings were the package defaults (`det_beam=5`, `pqlimit=200000`, `no_revisit_dets=True`); no larger beam was tried within the 30-minute box. The first probe run crashed: numpy 2 wrote `np.float64(...)` into the DEM text. It was fixed and run again alone, as the 8th process beside step 2's 7.

### Items 1 and 2 at p = 0.004 (run on another machine)

**Where it ran.** These ran on the ICRN server (`jupyter-blakew3`, AMD EPYC-Milan, 10 processes), right after the G3 frontier:

- the commit: `ed3fd35`;
- the command: `PYTHONDONTWRITEBYTECODE=1 python -m analysis.remote g2 --data-dir <the p = 0.004 folder of main> --out ~/mlBranchMidwest/runs/gates_g2 --processes 10`.

Its machine check was again identical to luminaty's. Its stdout is `logs/server_g2_p0.004.log` (from `server-logs`, commit `2e883a3`). Its summaries are `results/g2_server_p0.004_summary.json` and `results/g2_server_p0.004_tables.csv`. Their `ms_label` field wrongly says "8 processes on luminaty": that text was hard-coded at `ed3fd35`, and is a parameter now. The milliseconds in those two files and below are the server's, with 10 processes. The shots are the committed selection, all 245 failures and 245 controls. The candidates covered all 145 converged-wrong failures and a seeded 10 of the unconverged failures per basis.

**Item 1, a larger budget:** relay-BP with `stop_nconv=20` and `num_sets=600`, seed 0, a fresh decoder per shot.

| decoder | basis | F | fixed: all / of unconverged / of converged-wrong | C | broken | 2 x 2: both wrong / only relay-BP / only other / both right | exact paired p | break rate [95% CP] | iterations mean on F / C | ms mean on F / C (server) |
|---|---|---|---|---|---|---|---|---|---|---|
| stop after 20, 600 legs | X | 122 | 65 / 27 of 55 / 38 of 67 | 122 | 0 | 57 / 65 / 0 / 122 | 5.4e-20 | 0.000 [0.000, 0.030] | 29,220 / 1,654 | 9,611 / 530 |
| stop after 20, 600 legs | Z | 123 | 64 / 23 of 45 / 41 of 78 | 123 | 1 | 59 / 64 / 1 / 122 | 3.6e-18 | 0.008 [0.000, 0.044] | 27,075 / 2,340 | 8,769 / 747 |

- It fixes 129 of 245 failures (53%, 95% interval 46% to 59%), against the re-roll floor's 108 (44%, 38% to 51%) on the same shots.
- On the controls it takes 1,654 and 2,340 mean iterations, against the kit settings' 373 and 752 in the re-roll. That is 3 to 4 times the work on a typical shot.
- On the failures its median is 35,960 of the 36,080-iteration budget: most of them never reach 20 solutions.

**Item 2, several candidates:** relay-BP at `stop_nconv=1` under seeds 1 to 16, each on a fresh decoder. A candidate is a converged correction; distinct candidates are told apart by their correction. The lowest-weight pick and the vote by class (each distinct candidate adds exp(-weight), with weight the sum of log((1 - prior) / prior)) use no truth. The truth only scores them.

| basis | set | shots | right class among the candidates | lowest weight right | vote right | no candidate | one distinct candidate | first leg converged: shots, any / lowest / vote right | first leg unconverged: shots, any / lowest / vote right |
|---|---|---|---|---|---|---|---|---|---|
| X | converged-wrong failures | 67 | 50 | 34 | 33 | 0 | 12 | 12: 3 / 3 / 3 | 55: 47 / 31 / 30 |
| X | unconverged failures (seeded 10) | 10 | 9 | 8 | 8 | 0 | 0 | 0: – | 10: 9 / 8 / 8 |
| X | control | 122 | 122 | 122 | 122 | 0 | 109 | 109: 109 / 109 / 109 | 13: 13 / 13 / 13 |
| Z | converged-wrong failures | 78 | 64 | 37 | 36 | 0 | 9 | 9: 2 / 2 / 2 | 69: 62 / 35 / 34 |
| Z | unconverged failures (seeded 10) | 10 | 7 | 6 | 6 | 0 | 2 | 0: – | 10: 7 / 6 / 6 |
| Z | control | 123 | 123 | 122 | 122 | 0 | 105 | 105: 105 / 105 / 105 | 18: 18 / 17 / 17 |

Both bases, on the 145 converged-wrong failures:

- **The right class is among the 16 candidates in 114 (79%, 71% to 85%).**
- **The lowest-weight candidate has it in 71 (49%, 41% to 57%), and the vote in 69 (48%).** A single re-roll on a fresh decoder fixes 68 of the same 145 (47%, 39% to 55%).
- The median shot has 16 distinct candidates in 2 classes.
- Where the first leg converged (21 shots), all 16 seeds return the first leg's solution, because the first leg draws no random numbers. So those shots have one candidate each, and it is right in only 5.
- On the controls, the lowest-weight pick broke 0 of 122 (X) and 1 of 123 (Z).
- All 20 sampled unconverged failures had at least one candidate (median 6 distinct), and the right class was among them in 16.

**What items 1 and 2 say at p = 0.004:**

- More of the same budget fixes about 10 percentage points more than a free re-roll (53% against 44%), at 3 to 4 times the work on typical shots.
- I3 passes its kill line: the right class is among relay-BP's candidates in 79% of the converged-wrong failures, far above 30%.
- But weight does not find it. The lowest-weight and vote picks get 48% to 49%, no better than one re-roll (47%).
- So the information is in the candidate set and the weight is the wrong score. A learned score over a few candidates is an opening this measures directly; no such score was trained or tested here.

### Items 1 to 3 at p = 0.005, and the re-roll floor

`analysis/steps.sh step7` (log `logs/g2_run_step7.log`), on luminaty, 8 processes, from 23:50 CST to 00:28 CST (05:50 to 06:28 UTC). F is a seeded 100 of relay-BP's 207 (X) and 197 (Z) failures (`--max-failed 0.005:100`), and C is 100 controls per basis. The 16-candidate runs used every converged-wrong failure in F and a seeded 5 of its unconverged ones per basis. "Other's failures on all shots" scales to the 5,000 G1 shots per basis.

| decoder | basis | F | fixed: all / of unconverged / of converged-wrong | C | broken | 2 x 2: both wrong / only relay-BP / only other / both right | exact paired p | break rate [95% CP] | other's failures on all 5,000 shots, estimate [range], against relay-BP's | iterations mean F / C | ms mean F / C |
|---|---|---|---|---|---|---|---|---|---|---|---|
| re-roll (kit settings, fresh decoder) | X | 100 | 23 / 13 of 65 / 10 of 35 | 100 | 3 | 77 / 23 / 3 / 97 | 8.8e-5 | 0.030 [0.006, 0.085] | 303 [170, 584] vs 207 | 15,551 / 1,378 | 12,848 / 1,139 |
| re-roll | Z | 100 | 30 / 13 of 64 / 17 of 36 | 100 | 1 | 70 / 30 / 1 / 99 | 3.0e-8 | 0.010 [0.000, 0.054] | 186 [119, 417] vs 197 | 15,401 / 1,800 | 13,385 / 1,554 |
| stop after 20, 600 legs | X | 100 | 35 / 22 of 65 / 13 of 35 | 100 | 1 | 65 / 35 / 1 / 99 | 1.1e-9 | 0.010 [0.000, 0.054] | 182 [115, 415] vs 207 | 32,924 / 4,807 | 28,781 / 4,312 |
| stop after 20, 600 legs | Z | 100 | 39 / 20 of 64 / 19 of 36 | 100 | 1 | 61 / 39 / 1 / 99 | 7.5e-11 | 0.010 [0.000, 0.054] | 168 [101, 401] vs 197 | 33,081 / 6,057 | 27,200 / 4,830 |
| BP-OSD, order 10 (kit) | X | 100 | 14 / 8 of 65 / 6 of 35 | 100 | 9 | 86 / 14 / 9 / 91 | 0.40 | 0.090 [0.042, 0.164] | 609 [362, 977] vs 207 | 80 / 67 | 1,208 / 623 |
| BP-OSD, order 10 | Z | 100 | 17 / 11 of 64 / 6 of 36 | 100 | 10 | 83 / 17 / 10 / 90 | 0.25 | 0.100 [0.049, 0.176] | 644 [382, 1,023] vs 197 | 80 / 70 | 1,182 / 759 |
| BP-OSD, order 60 | X | 100 | 12 / 7 of 65 / 5 of 35 | 100 | 9 | 88 / 12 / 9 / 91 | 0.66 | 0.090 [0.042, 0.164] | 614 [367, 980] vs 207 | 80 / 67 | 1,203 / 633 |
| BP-OSD, order 60 | Z | 100 | 17 / 11 of 64 / 6 of 36 | 100 | 10 | 83 / 17 / 10 / 90 | 0.25 | 0.100 [0.049, 0.176] | 644 [382, 1,023] vs 197 | 80 / 70 | 1,229 / 809 |

| basis | set | shots | right class among the candidates | lowest weight right | vote right | no candidate | one distinct candidate | first leg converged: shots, any / lowest / vote right | first leg unconverged: shots, any / lowest / vote right |
|---|---|---|---|---|---|---|---|---|---|
| X | converged-wrong failures | 35 | 24 | 16 | 16 | 0 | 5 | 5: 0 / 0 / 0 | 30: 24 / 16 / 16 |
| X | unconverged failures (seeded 5) | 5 | 4 | 4 | 4 | 1 | 0 | 0: – | 5: 4 / 4 / 4 |
| X | control | 100 | 100 | 100 | 100 | 0 | 73 | 73: 73 / 73 / 73 | 27: 27 / 27 / 27 |
| Z | converged-wrong failures | 36 | 32 | 14 | 14 | 0 | 2 | 2: 0 / 0 / 0 | 34: 32 / 14 / 14 |
| Z | unconverged failures (seeded 5) | 5 | 3 | 3 | 3 | 1 | 1 | 0: – | 5: 3 / 3 / 3 |
| Z | control | 100 | 100 | 100 | 100 | 0 | 70 | 70: 70 / 70 / 70 | 30: 30 / 30 / 30 |

**What this says at p = 0.005:**

- The re-roll floor is lower here: 53 of 200 failures (27%). Most failures are unconverged, and another draw rarely rescues them (26 of 129).
- The larger budget fixes 74 of 200 (37%), 10 points above the floor again, at 3 to 4 times the work on a typical shot.
- BP-OSD fixes 29 to 31 of 200 and breaks 19 of 200 controls (about 10%): it is the weaker decoder here, as the kit's README shows (it fails about 3 times as often at p = 0.005).
- The 16 candidates hold the right class for 56 of the 71 converged-wrong failures (79%, 66% to 87%), exactly as at p = 0.004. The lowest weight picks it for only 30 (42%, 31% to 55%), and the vote for 30.

### Item 5: joint decoding of both check types

The kit's baselines decode X memory from its 936 X-check detectors. `circuit.dem_matrices(circuit)` with no basis gives the full problem: all 1,728 detectors and 67,824 mechanisms, with 391,464 nonzeros in H (13 times the basis problem's 30,672). It keeps the correlations between X and Z errors. Here relay-BP decodes the full 1,728-bit detector row of each shot with the same 12 observables, at two settings:

- **(a) `joint`:** the kit's `RELAY_SETTINGS`, seed 0. The relay-BP authors' joint runs below use exactly these settings.
- **(b) `joint600`:** the same with `num_sets=600`. On a fresh decoder with the same seed, (b)'s first 300 legs are (a)'s, draw for draw (`relay.rs`: one generator, seeded at construction). So (b) can differ from (a) only where (a) ran all 300 legs without finding 5 solutions, which takes at least 296 x 60 = 17,760 iterations. Elsewhere (b) equals (a) by construction. (b) is decoded only on the shots where (a) reached that floor, plus 2 seeded shots per p and basis below it, as a check that the two agree there.

**The relay-BP authors' own runs** (theirs, not ours). These are from `examples/data/bicycle_bivariate_144_12_12_fine_ler_sweep.csv` at https://github.com/trmue/relay, fetched on 2026-11-22 (sha256 `cb7e2d8bb6a7fd3be6b2cfd86b7e18be91158166ce8c844d6bfbee1549cca271`), read with `python -m analysis.authors_joint bb144.csv` (log `logs/authors_joint.log`). In their notebook, `xyz=False` keeps only the detectors of the memory basis, which is what the kit decodes, and `xyz=True` decodes all detectors. Both use the kit's `RELAY_SETTINGS`. Their seconds are sinter's decoder time on their machine.

| p | decoding | shots per basis | failures X, Z | rate per shot X [95% Wilson] | Z | seconds per shot X, Z |
|---|---|---|---|---|---|---|
| 0.003 | separate | 100,000 | 32, 40 | 3.20e-4 [2.3e-4, 4.5e-4] | 4.00e-4 [2.9e-4, 5.4e-4] | 0.096, 0.098 |
| 0.003 | joint (xyz) | 100,000 | 15, 14 | 1.50e-4 [9.1e-5, 2.5e-4] | 1.40e-4 [8.3e-5, 2.4e-4] | 12.6, 12.5 |
| 0.004 | separate | 10,000 | 53, 44 | 5.30e-3 [4.1e-3, 6.9e-3] | 4.40e-3 [3.3e-3, 5.9e-3] | 0.36, 0.36 |
| 0.004 | joint (xyz) | 10,000 | 35, 39 | 3.50e-3 [2.5e-3, 4.9e-3] | 3.90e-3 [2.9e-3, 5.3e-3] | 35.5, 35.2 |
| 0.005 | separate | 5,000 | 216, 205 | 4.32e-2 [3.8e-2, 4.9e-2] | 4.10e-2 [3.6e-2, 4.7e-2] | 1.5, 1.6 |
| 0.005 | joint (xyz) | 5,000 | 376, 359 | 7.52e-2 [6.8e-2, 8.3e-2] | 7.18e-2 [6.5e-2, 7.9e-2] | 141, 137 |

In their data, joint decoding fails about 2.5 times less at p = 0.003, about 1.3 times less at 0.004, and about 1.8 times more at 0.005, at about 100 times the time per shot.

**Rates per shot pair.** The kit's rate is per shot pair: shot k of X memory is paired with shot k of Z memory, and the pair fails when either basis fails. Published numbers summed over X and Z (failures X + Z over shots) are close to the kit's pair rate when failures are rare. Per-basis numbers are about half of it. Compare like with like.

**Our joint runs at p = 0.003** (`analysis/steps.sh step4`, log `logs/g2_run_step4.log`): every relay-BP failure from G1, and 150 control shots per basis (`--control-min 0.003:150`), each on a fresh decoder.

| setting | basis | F | fixed: all / of unconverged / of converged-wrong | C | broken | 2 x 2: both wrong / only relay-BP / only joint / both right | exact paired p | break rate, 95% upper bound | iterations on F: mean / median / max | on C: mean / median / max | ms on F: mean / median | on C: mean / median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) kit settings | X | 23 | 23 / 7 of 7 / 16 of 16 | 150 | 0 | 0 / 23 / 0 / 150 | 2.4e-7 | 0 of 150, at most 2.4% | 2,411 / 975 / 18,080 | 658 / 392 / 18,044 | 31,765 / 14,243 | 9,006 / 5,794 |
| (a) kit settings | Z | 21 | 20 / 9 of 9 / 11 of 12 | 150 | 0 | 1 / 20 / 0 / 150 | 1.9e-6 | 0 of 150, at most 2.4% | 4,065 / 1,801 / 18,080 | 519 / 373 / 3,896 | 60,706 / 26,827 | 7,681 / 5,547 |
| (b) 600 legs | X | 23 | 23 / 7 of 7 / 16 of 16 | 150 | 0 | 0 / 23 / 0 / 150 | 2.4e-7 | at most 2.4% | 3,193 / 975 / 36,080 | 756 / 392 / 32,695 | 34,812 / 14,243 | 9,593 / 5,794 |
| (b) 600 legs | Z | 21 | 20 / 9 of 9 / 11 of 12 | 150 | 0 | 1 / 20 / 0 / 150 | 1.9e-6 | at most 2.4% | 4,922 / 1,801 / 36,080 | 519 / 373 / 3,896 | 61,899 / 26,827 | 7,675 / 5,547 |

Both bases together: joint decoding fixed **43 of relay-BP's 44 failures** (98%, exact 95% interval 88% to 99.9%) and broke **0 of 300** control shots (break rate at most 1.2%, exact 95% upper bound).

- (b) was decoded on 7 shots: 3 where (a) reached the 17,760-iteration floor, and 4 seeded check shots below it. On all 4 check shots (b) gave (a)'s correction and iteration count exactly (`joint600_check` in `results/g2_summary.json`). On the other 3 shots, 300 more legs changed no answer: the one remaining failure (Z) stays.
- In (b)'s rows, the shots below the floor carry (a)'s outputs and times, which are the same by construction.
- Times are per shot inside the decode call, a fresh decoder per shot, 8 processes on this machine. In G1 at p = 0.003, the kit's separate relay-BP took a mean of 134 ms (X) and 128 ms (Z) per shot on the same machine, so joint decoding costs about 60 times as much on a typical control shot.

**Is it too good?** A gain this large was checked for leakage:

- The decoder gets the shot's 1,728 detector bits and nothing else. Stim keeps the observables apart from the detectors, and the kit's README allows a decoder all 1,728.
- The scoring is the kit's, with L from the same full DEM.
- It agrees with the authors' runs. Their joint decoder still fails about 29 times per 100,000 shots per basis at p = 0.003. Our numbers say those failures fall almost entirely on other shots than separate relay-BP's: on our 100,000 shots that would be about 29 breaks among relay-BP's 99,956 right shots (0.03%), about 0.1 expected in a 300-shot control. The control can neither see nor exclude that rate. So the 98% is the fix rate, not a claim that joint decoding fails 44 times less.

**What this says at p = 0.003:**

- Joint decoding fixes almost every shot separate relay-BP fails: all 16 unconverged failures and 27 of the 28 converged-wrong ones.
- Joint decoding on every shot costs about 60 times the time and, by the authors' rows, fails about 2.5 times less than separate decoding.
- A cheaper design runs separate relay-BP first and joint decoding only on the shots it flags. Every unconverged failure is flagged for free, and joint decoding fixed all 16 of them; these are 36% of the failures. The converged-wrong failures (64%) would need a router to find them, and that router is not measured here.

**Our joint runs at p = 0.004** (`analysis/steps.sh step5`, log `logs/g2_run_step5.log`). This is a seeded subsample of 50 of relay-BP's failures and 100 of the control shots per basis (`--caps joint@0.004=50/100/all`): a joint shot here takes 1 to 5 minutes, too long for all 490.

| setting | basis | F | fixed: all / of unconverged / of converged-wrong | C | broken | 2 x 2: both wrong / only relay-BP / only joint / both right | exact paired p | break rate, 95% upper bound | iterations on F: mean / median | on C: mean / median | ms on F: mean / median | on C: mean / median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) kit settings | X | 50 | 45 / 15 of 16 / 30 of 34 | 100 | 0 | 5 / 45 / 0 / 100 | 5.7e-14 | 0 of 100, at most 3.6% | 8,987 / 6,224 | 1,893 / 798 | 134,028 / 91,670 | 28,221 / 11,684 |
| (a) kit settings | Z | 50 | 44 / 12 of 16 / 32 of 34 | 100 | 1 | 6 / 44 / 1 / 99 | 2.6e-12 | 1 of 100, at most 5.4% | 9,307 / 7,658 | 2,310 / 924 | 135,460 / 115,024 | 33,470 / 14,064 |
| (b) 600 legs | X | 50 | 46 / 15 of 16 / 31 of 34 | 100 | 0 | 4 / 46 / 0 / 100 | 2.8e-14 | 0 of 100, at most 3.6% | 13,871 / 6,224 | 2,124 / 798 | 211,403 / 91,670 | 32,047 / 11,684 |
| (b) 600 legs | Z | 50 | 45 / 13 of 16 / 32 of 34 | 100 | 0 | 5 / 45 / 0 / 100 | 5.7e-14 | 0 of 100, at most 3.6% | 13,187 / 7,658 | 2,591 / 924 | 192,631 / 115,024 | 38,041 / 14,064 |

Both bases together:

- **(a) fixed 89 of 100 failures** (exact 95% interval 81% to 94%) and broke 1 of 200 controls (at most 2.8%).
- **(b) fixed 91 of 100** (84% to 96%) and broke none (at most 1.8%). The extra legs fixed 2 more failures and undid (a)'s one break.
- Of the unconverged failures, (a) fixed 27 of 32 (84%, 67% to 95%).
- (b) was decoded on 47 shots: 39 at or above (a)'s 17,760-iteration floor, and 8 check shots below it. All 8 check shots gave (a)'s correction and iteration count exactly.

In G1 at p = 0.004 the kit's separate relay-BP took a mean of 404 and 466 ms per shot here (8 processes). Joint decoding takes about 30 s on a control shot (70 times as long) and over 2 minutes on a failed one.

**What this says at p = 0.004:**

- Joint decoding still fixes nearly 9 in 10 of separate relay-BP's failures, but its own failures now show: 1 break in 200 controls.
- The authors' rows put joint decoding's overall rate only about 1.3 times below separate decoding here. That fits a joint decoder failing about 0.6% of the shots separate relay-BP gets right, which a 200-shot control can neither show nor exclude.
- As a fallback after separate decoding, it would fix most of the flagged failures: 84% of the unconverged ones.

## G3: where relay-BP's time goes

### Item 1: where the iterations go (from G1)

`python -m analysis.g3_schedules tail` (in `analysis/steps.sh summarize-g3`; output `results/g3_tail.json`, log `logs/g3_tail.log`), from G1's per-shot iterations with the kit's schedule:

| p | basis | shots | mean iterations | share of all iterations in the slowest 1% of shots | in the slowest 10% | in the unconverged shots (count) | in the failed shots (count) | derived: first leg's answer where it converged, else the kit's run: mean iterations (speed-up) | its failures against the kit's |
|---|---|---|---|---|---|---|---|---|---|
| 0.003 | X | 50,000 | 146 | 27.0% | 49.2% | 2.2% (9) | 2.8% (23) | 64 (2.27x) | 24 vs 23 |
| 0.003 | Z | 50,000 | 143 | 24.6% | 47.7% | 2.3% (9) | 3.0% (21) | 61 (2.36x) | 23 vs 21 |
| 0.004 | X | 25,000 | 539 | 31.4% | 70.9% | 7.6% (57) | 12.1% (122) | 396 (1.36x) | 125 vs 122 |
| 0.004 | Z | 25,000 | 547 | 31.5% | 71.0% | 6.0% (45) | 11.2% (123) | 407 (1.34x) | 128 vs 123 |
| 0.005 | X | 5,000 | 2464 | 7.3% | 65.9% | 18.5% (126) | 26.7% (207) | 2225 (1.11x) | 207 vs 207 |
| 0.005 | Z | 5,000 | 2291 | 7.9% | 67.4% | 20.4% (129) | 27.4% (197) | 2060 (1.11x) | 199 vs 197 |

The derived column is computed from G1's two passes (the first leg alone, and the kit's run). A real run of that rule would draw other random memory strengths on the shots it passes on, so its failures would differ shot by shot. As G2 shows, that changes the count only within noise.

**What this says:**

- At p = 0.003 and 0.004 the slowest 10% of shots take 48% to 71% of all iterations, and the slowest 1% take 25% to 32%.
- At p = 0.005, 2.5% of shots run the whole 18,080-iteration budget without converging. They alone take a fifth of the iterations, and the slowest 10% take two thirds.
- The failed shots take 3% (p = 0.003), 12% (0.004) and 27% (0.005) of all iterations, a share far above their number.
- Stopping at the first leg where it converges cuts mean iterations 2.3 times at p = 0.003, 1.35 times at 0.004 and 1.1 times at 0.005, at 0 to 5 more failures per basis.
- So what a faster first stage can save is the easy shots' extra legs; the tail of hard shots stays, and it holds most of the work at p = 0.004 and 0.005.

### Item 2: the speed against accuracy frontier at p = 0.004 (run on another machine)

**Where it ran.** This job ran on the team's ICRN server (host `jupyter-blakew3`, AMD EPYC-Milan, 10 processes), started by hand by Sal:

- the commit: `ed3fd353fec15895958742f73e33d0f8f3db610d` of this branch;
- the command, from `challenge-1-ml-decoder-kit/`: `PYTHONDONTWRITEBYTECODE=1 python -m analysis.remote g3 --data-dir <the p = 0.004 folder of main> --out ~/mlBranchMidwest/runs/gates_g3 --processes 10`;
- its versions: Python 3.12.16, numpy 2.5.5, stim 1.16.0, ldpc 2.4.1, relay-bp 0.2.2.

Its stdout is `logs/server_g3_frontier_p0.004.log`, copied from the repository's `server-logs` branch (`gates_g3_g2.out`, lines 1 to 343, commit `617e64502fe8`). Its summaries are `results/g3_frontier_p0.004.json` and `.csv`, from `files/gates_g3/` on that branch. The table below is `python -m analysis.g3_schedules table --json analysis/results/g3_frontier_p0.004.json` (log `logs/g3_frontier_table.log`). The same run can be made here with `analysis/steps.sh step8`, then `summarize-g3`.

**The two machines agree shot for shot.**

- The job's machine check decoded the first 10 validation_X shots with the kit's relay-BP on a fresh decoder, seed 0. Its iterations, corrections and predicted bits are identical to luminaty's.
- Its 20 chunks of the kit's schedule have the same mean iterations and failures as G1's 20 chunks on luminaty (`logs/g1_run_p0.004.log`, and the first two chunks in `logs/g1_run_p0.004_first_chunks.log`, decoded by the first smoke run with 2 processes).
- So its iterations and failures are luminaty's. Its milliseconds are the server's (10 processes), not luminaty's.

**Shots:** the validation files, 5,000 shots per basis, each schedule in 500-shot chunks on fresh decoders with seed 0, every other setting the kit's (300 legs). Each schedule is paired with the kit's 80 / 60 / 5 on the same shots. A pair is shot k of X with shot k of Z.

| schedule (first leg / later legs / stop after) | mean iterations X, Z | median X, Z | p99 X, Z | speed-up (mean iterations) | failures [X Z] | unconverged [X Z] | paired vs kit, X: only kit wrong / only this wrong (p) | Z | pairs failed of 5,000 [95% Wilson] | ms mean X, Z |
|---|---|---|---|---|---|---|---|---|---|---|
| 80/60/1 | 167, 154 | 21, 21 | 2,681, 3,413 | 3.42 | [41 32] | [12 7] | 10 / 20 (0.099) | 6 / 18 (0.023) | 73 [58, 92] | 62, 55 |
| 40/60/1 | 171, 161 | 21, 21 | 2,702, 2,894 | 3.32 | [42 38] | [12 9] | 10 / 21 (0.071) | 4 / 22 (0.00053) | 80 [64, 99] | 63, 58 |
| 80/30/1 | 183, 172 | 21, 21 | 6,223, 4,782 | 3.10 | [56 45] | [38 31] | 5 / 30 (2.2e-05) | 5 / 30 (2.2e-05) | 101 [83, 122] | 68, 62 |
| 40/30/1 | 194, 177 | 21, 21 | 6,881, 4,717 | 2.97 | [59 49] | [42 35] | 5 / 33 (4.3e-06) | 7 / 36 (9e-06) | 108 [90, 130] | 72, 60 |
| 80/60/2 | 280, 262 | 45, 45 | 4,626, 4,707 | 2.03 | [33 30] | [15 11] | 11 / 13 (0.84) | 6 / 16 (0.052) | 63 [49, 80] | 87, 80 |
| 80/30/2 | 288, 282 | 45, 46 | 9,075, 9,073 | 1.93 | [53 45] | [43 38] | 5 / 27 (0.00011) | 6 / 31 (4.1e-05) | 97 [80, 118] | 90, 88 |
| 40/60/2 | 294, 278 | 45, 45 | 5,889, 5,773 | 1.92 | [43 26] | [18 9] | 8 / 20 (0.036) | 7 / 13 (0.26) | 69 [55, 87] | 95, 91 |
| 40/30/2 | 293, 282 | 45, 46 | 9,032, 9,032 | 1.91 | [49 44] | [41 35] | 8 / 26 (0.0029) | 7 / 31 (0.00012) | 93 [76, 114] | 101, 98 |
| 80/30/3 | 377, 363 | 73, 75 | 9,079, 9,078 | 1.49 | [53 52] | [46 41] | 6 / 28 (0.0002) | 4 / 36 (1.9e-07) | 105 [87, 127] | 119, 112 |
| 40/60/3 | 376, 378 | 74, 74 | 7,475, 7,903 | 1.46 | [36 24] | [17 9] | 7 / 12 (0.36) | 8 / 12 (0.5) | 60 [47, 77] | 118, 116 |
| 40/30/3 | 378, 379 | 74, 76 | 9,036, 9,038 | 1.45 | [50 45] | [41 40] | 5 / 24 (0.00055) | 5 / 30 (2.2e-05) | 95 [78, 116] | 119, 116 |
| 80/60/3 | 386, 376 | 72, 74 | 7,875, 7,722 | 1.44 | [22 20] | [13 7] | 14 / 5 (0.064) | 11 / 11 (1) | 42 [31, 57] | 120, 115 |
| 80/30/5 | 510, 516 | 129, 132 | 9,078, 9,073 | 1.07 | [53 45] | [45 39] | 9 / 31 (0.00068) | 7 / 32 (7e-05) | 97 [80, 118] | 158, 156 |
| 40/30/5 | 526, 512 | 128, 129 | 9,039, 9,035 | 1.06 | [53 40] | [44 34] | 5 / 27 (0.00011) | 7 / 27 (0.00082) | 93 [76, 114] | 163, 155 |
| 40/60/5 | 562, 532 | 134, 136 | 14,669, 10,643 | 1.00 | [28 21] | [10 9] | 11 / 8 (0.65) | 10 / 11 (1) | 49 [37, 65] | 174, 161 |
| 80/60/5 (kit) | 565, 535 | 135, 136 | 13,982, 10,777 | 1.00 | [31 20] | [17 7] | 0 / 0 (1) | 0 / 0 (1) | 51 [39, 67] | 174, 161 |
The ms column: ms per shot inside decode_detailed, 10 processes side by side on jupyter-blakew3 (10 processes).

**How to read the paired counts.** Changing the schedule also changes which random memory strengths each shot gets: a stop rule that ends sooner draws fewer numbers, so later shots in a chunk see other draws. G2 showed that a fresh draw alone fixes about 44% of relay-BP's failures. So even two schedules of the same strength disagree on about 10 shots per basis here: 40 / 60 / 5 against the kit's 80 / 60 / 5 has 11 / 8 and 10 / 11. The p-values are not corrected for the 15 comparisons.

**What this says at p = 0.004:**

- **Later legs must stay at 60 iterations.** All 8 schedules with 30-iteration legs fail more than the kit's (93 to 108 failed pairs against 51; paired p at or below 0.003 in every basis). Their unconverged shots grow 2 to 6 times. Short legs are what made 40 / 30 / 1 lose in the review.
- **The first leg matters little.** At each k, 40 / 60 / k and 80 / 60 / k take nearly the same mean iterations (171 and 167 at k = 1, 376 and 386 at k = 3 in X). Their failed pairs differ by at most 18 (60 against 42 at k = 3). No test between the two was made, and neither 40 / 60 / 3 nor 80 / 60 / 3 differs significantly from the kit.
- **Stopping after 3 solutions costs nothing measurable and saves 31% of the iterations.** 80 / 60 / 3 runs 1.44 times fewer mean iterations (386 and 376 against 565 and 535) with 42 failed pairs against the kit's 51 (X 14 / 5, p = 0.064; Z 11 / 11). 40 / 60 / 3 runs 1.46 times fewer with 60 pairs (X 7 / 12, Z 8 / 12, p at or above 0.36).
- **Stopping after 2 saves half (2.0 times) for about 12 more failed pairs.** 80 / 60 / 2 has 63 against 51 (Z 6 / 16, p = 0.052).
- **Stopping after the first solution saves 3.4 times, at a measurable cost.** 80 / 60 / 1 has 73 pairs against 51, about 40% more failures (Z 6 / 18, p = 0.023; X 10 / 20, p = 0.099).
- On 5,000 pairs a difference of about 30% in failures is at the edge of what these counts can see. A schedule chosen from this table should be confirmed on train shots or a larger validation set, never on test shots.
