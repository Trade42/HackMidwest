# TCBP pipeline with the head retrained on the neutral-atom-shaped noise: declared before any number

**Label: exploration, after the main test results; a separate branch (`tcbp-retrain`).** Written 2026-11-22 before
18:10 UTC (pushed in commit `7d8c80e7` at 18:10; this line corrected afterwards, nothing else changed) by the
implementer of task `tcbp-retrain`. At that time the head was still training on the laptop: no decode
with it, and no decode of the new shot file by any decoder, existed. Every number below is reported whatever it shows.

## Why

In phase 1 (`tcbp_check_declared.md`, RESULTS "Exploration: the TCBP pipeline on this noise"), TCBP-JOBS on question
1's factor-0.8 X file (seed 4100, 2,000 shots) failed 25 shots with the matched priors and 209 with the kit-trained
head's priors. That head had been trained only on the kit's uniform noise at p = 0.003 to 0.005. This task retrains
the head alone on the noise of the shots: the same trunk, the same architecture and the same knobs.

## The retrained head (`predecoder.train --noise-factor 0.8`, commit `f4948928`)

- **Trunk:** the final X network, `cascade_X_s1/checkpoint_final_step120000.pt`, sha256 `d8d1c70f...`. It is frozen,
  as for the kit head.
- **Training shots:** fresh shots every step from the full detector error model of
  `model_circuit("X", neutral_atom(0.8))`: 35,064 mechanisms, mapped onto the kit's 8,784 columns. These are
  error-model shots drawn by torch from stream seeds of `--seed 1`, never Stim shots, so they are disjoint from every
  test file.
- **The head's base prior:** the matched vector, `matched_priors("X", neutral_atom(0.8))`, sha256 of its float64
  values `181d3c74...`. The merged mechanism priors equal it to 4.9e-17. The untrained head is therefore arm m below,
  up to the float16 rounding of the saved logits.
- **Recipe:** the X head's recipe (seed 1, batch 64, lr 1e-3, warmup 500, weight decay 1e-4, project 32, hidden 64,
  shape width 16), 20,000 steps, with a cap of 0.75 hours on the laptop's RTX 4060.
- **The checkpoint used:** the last one the run writes (`head.pt`), at step 20,000 or wherever the cap stops it.
  It is fixed now. No checkpoint is chosen by any decode.

## The shots

1. **Primary: a fresh test file**, `X_f0.8_seed4800.npz`. It holds 4,000 shots from Stim's detector sampler on
   `model_circuit("X", neutral_atom(0.8))`, seed 4800, made on luminaty (Stim 1.16.0) with `make_shots.py`, sha256
   `b2c023bbf24108ee771d6ebea4405971888164732aad0ab0da53be4848bc1162`, 323.3 detection events per shot.
   - Converted to the kit's format by `tcbp_check convert --p 0.0326371`:
     `X_f0.8_seed4800_kitformat_p0.0326371.npz`, sha256 `32fc7703d19ba4e7...`.
   - The seed was never used before. No decoder has read this file, and nothing was tuned on it.
2. **Secondary: phase 1's file**, `X_f0.8_seed4100.npz` (2,000 shots, sha256 `4e6e2867...`; converted `a251dfe0...`).
   - Its m, g and h numbers are known (25, 957 and 209), and they motivated this task, so it is not a clean test.
   - It is reported for continuity with phase 1 and with this branch's BP-OSD and relay-BP rows.

## The arms

The decoder in every arm is TCBP-JOBS (cudaq-qec preset `tcbp-jobs`, one decoder built per shot, one CPU thread,
the server's A100). Every arm decodes the kit's X-memory H and L (936 x 8,784); only the prior vector differs.

| Arm | Base vector | Head | The head's logits (`predecoder.evaluate`, fp32 on the A100) |
|---|---|---|---|
| **m** matched | `priors_matched.npy` | none | |
| **g** generic | `priors_generic.npy`: the kit's uniform priors at p = 0.0326371 (BP-OSD's fitted p, as in phase 1) | none | |
| **h** kit head, generic base | generic | kit-trained, `38ab667e...` | no `--base-priors`: the base is the kit's priors at the file's p, which equal the generic vector |
| **hm** kit head, matched base | matched | kit-trained, `38ab667e...` | `--base-priors priors_matched.npy` |
| **r** retrained head, matched base | matched | retrained (above) | `--base-priors priors_matched.npy` (the vector it was trained with, checked) |

- **Knobs.** The three head arms use the frozen TCBP knobs of `predecoder/logs/fit_tcbp_X_p0.005.json` (sha256
  `7697010678a3...`): log-odds mixing with that arm's base vector, alpha 0.4, eps 1e-4, every prior capped at 0.5.
  **No refit.**
- **Two pairings are clean.** In each, the head's input and its mixing base are the same vector, so alpha 0 gives
  that arm's base arm exactly: m for hm and r, g for h.
- **The arm hm** separates the two changes between h and r. r against hm changes the head's training and nothing
  else. hm against h changes the base and nothing else.

## The pairs

Each pair gets an exact two-sided sign test on the shots where exactly one of the two arms fails (binomial, one half),
and every failure count gets a 95% Wilson interval. A failure is a shot with at least one of the 12 logical bits wrong
(X memory only, so these are not the kit's pair score).

- **Primary, on the seed-4800 file:**
  - **r vs m:** does a head trained on this noise add to the true error model?
  - **r vs h:** does retraining help, against phase 1's pipeline as it was?
- **Secondary:** r vs hm, hm vs m, h vs m, m vs g, r vs g, h vs g. On the seed-4100 file every pair among m, h, hm
  and r is secondary.

## The stopping rule

There is one server job and one TCBP-JOBS model, fitted once because both files have the same H.
- Blocks of 100 shots are taken in turn: block 0 of seed 4800, then block 0 of seed 4100, then block 1 of each, and so
  on.
- Inside each block the arms run in the order m, g, h, hm, r. The seed-4100 task has no g arm: phase 1 measured it
  (957), and it is the slowest arm.
- No new block starts after the deadline the orchestrator writes into the job.
- **A task's scored set is the longest prefix of whole blocks that all of its arms finished.** A shot outside it is
  not scored for any arm. If every block finishes, all 4,000 and 2,000 shots are scored.

## The expectation, stated before any number

- **r against h:** r should beat h by a wide margin. h failed 209 of 2,000 on the seed-4100 file: the kit head moved
  the priors only part of the way from generic towards this noise. r starts from the matched vector and has learned
  this noise.
- **r against m:** we do not know.
  - On the kit's own noise the head's priors beat TCBP-JOBS with the true priors. That was 72 against 186 failures on
    10,000 test shots at p = 0.005, and 24 against 68 on the wrong-p task of phase 2.
  - Factor 0.8 is a stronger noise than the trunk ever saw: the fitted uniform p is about 6.5 times its training
    range. The trunk's features may carry less here, and the alpha of 0.4 was fitted on another noise.
- **The claim this task can make:** the retrained head adds to the true error model only if r fails fewer shots
  than m with p < 0.05 on the seed-4800 file. Otherwise we report that it does not, with the numbers.
- **What is not done:** no refit of the knobs, no other factor, no Z memory, no atom loss and no kit pair score.
  These go under "what is left".
