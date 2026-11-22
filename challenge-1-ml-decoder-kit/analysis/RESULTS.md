# Gates G1 to G3: how relay-BP fails, how much accuracy sits above it, where its time goes

**These are exploration numbers on train, validation and exploration shots. Nothing here is a final result.** No test or go file was opened: the loader (`analysis/common.py`, `refuse_held_out`) refuses any file whose name starts with `test_` or `go_`.

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

**Shot cuts, against the brief's aims:** the brief aimed at 100,000 shots per basis at p = 0.003 and 20,000 at 0.005. Its lanes were cut from 12 to 8 processes, and this CPU turned out about 2.7 times slower per process under full load than on one core. So p = 0.003 uses the first 50,000 shots of each file (the orchestrator's suggested cut) and p = 0.005 the first 5,000 (about 200 failures per basis, still above the brief's 100).

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

### Items 1, 3 and 4 at p = 0.004, and the re-roll floor

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
