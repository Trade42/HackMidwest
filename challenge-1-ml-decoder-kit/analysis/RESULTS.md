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
- **The truth reaches no decoder.** The observables are used only for scoring and, in G2, to pick the control shots and to score the candidates.

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
