# TCBP pipeline, phase 2b: SI1000 at two more strengths, declared before any number of them

**Label: exploration, after the main test results; a separate branch (`infleqtion-noise`).** Written 2026-11-22
about 16:46 UTC by the implementer of task `tcbp-noise`, with the orchestrator's yes (dq-orch-6, 16:45 UTC). No
shot of these two strengths had been decoded when it was written.

## Why, stated plainly

Phase 2 (`tcbp_check2_declared.md`) declared SI1000 at p = 0.004. While its job ran (server job `cpu/100`), that
task **failed 0 of the first 600 shots in every arm** (matched, generic and head alike). It will be reported as
declared: 0 failures, which says nothing about the arms. **The two strengths below were chosen after seeing that
p = 0.004 fails nothing.** They bracket the strength upward, to find failures to compare; they are not tuned to
favour any arm. Nothing about the arms or the head was looked at to choose them, beyond the fact that all three were at 0.

## The tasks (`tcbp_models.py prepare-si1000`)

SI1000 exactly as in phase 2 (Gidney, Newman, Fowler, Broughton, Quantum 5, 605 (2021), Table 2; on the kit's
circuit: two-qubit depolarizing p after each CX, readout flip 5p, reset flip 2p, every idle slot p/10; the resonator
idle 2p and the one-qubit-gate noise left out), at:

- **p = 0.006**: 3,000 shots, Stim seed 7101;
- **p = 0.008**: 3,000 shots, Stim seed 7102.

The arms are those of phase 2. (g) is the kit's uniform priors at the model's two-qubit error, which is p, with nothing
fitted. (m) is the circuit's own error model on the kit's columns. (h) is the network and head at the frozen TCBP knobs,
told p (its p input and the mixing base). The pairs are the same: h vs g, h vs m, m vs g, each with an exact two-sided
sign test and a 95% Wilson interval for every failure count. The network was trained on p = 0.003 to 0.005 kit noise, so both
strengths are above its range.

## The stopping rule

One server job (`gpu/090`), one TCBP model shared by both tasks. Blocks of 100 taken in turn (p = 0.006, then
0.008), arms m, g, h inside each block, no new block after **17:35 UTC**. A task's scored set is the longest
prefix of whole blocks that all three of its arms finished.
