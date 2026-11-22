**The learned predecoder on the test shots: secondary (chosen after the declared gate at p = 0.004 failed), one basis at a time, accuracy only, not a pair score; its times are not clean (a shared 12-worker pool on the team's PC), so iterations are shown instead.** Relay-BP at the kit's settings, given for each shot the priors that a second output (a "head") of our trained network computes from that shot's detectors; truth enters only the scoring. Per noise level and basis: the saved test file of that basis alone, so a row counts shots, not pairs, and cannot sit in the pair-scored tables. H against the others: both fail / only H fails / only the other fails (exact p), on the same shots. Counts from the run's own files (`predecoder.compare`, `predecoder.secondary`); TCBP-JOBS's and relay-BP's totals are checked against their rows in this report.

| p | Basis | Decoder | Failed shots | Share (95% interval) | H against it: both / only H / only it (exact p) | Mean BP iterations per shot; shots not converged |
|---|---|---|---|---|---|---|
| 0.005 | X | relay-BP with the head's priors, a decoder built per shot (H; ours) | 151 of 10,000 | 1.51% (1.29% to 1.77%) |  | 711; 90 not converged |
| 0.005 | X | relay-BP with the kit's priors, a decoder built per shot (the declared comparison; the kit) | 433 of 10,000 | 4.33% (3.95% to 4.75%) | 125 / 26 / 308 (p = 2e-62) | 2,436; 258 not converged |
| 0.005 | X | TCBP-JOBS (NVIDIA, GPU; its row of this report) | 186 of 10,000 | 1.86% (1.61% to 2.14%) | 90 / 61 / 96 (p = 0.006) |  |
| 0.005 | X | relay-BP, the evaluator's timed row (one decoder through all shots; its row of this report) | 432 of 10,000 | 4.32% (3.94% to 4.74%) | 124 / 27 / 308 (p = 1e-61) |  |
| 0.004 | X | relay-BP with the head's priors, a decoder built per shot (H; ours) [run after the p = 0.005 X test result (the registered extension)] | 15 of 10,000 | 0.15% (0.09% to 0.25%) |  | 167; 12 not converged |
| 0.004 | X | relay-BP with the kit's priors, a decoder built per shot (the declared comparison; the kit) | 39 of 10,000 | 0.39% (0.29% to 0.53%) | 12 / 3 / 27 (p = 8e-06) | 514; 13 not converged |
| 0.004 | X | TCBP-JOBS (NVIDIA, GPU; its row of this report) | 20 of 10,000 | 0.20% (0.13% to 0.31%) | 8 / 7 / 12 (p = 0.36) |  |
| 0.004 | X | relay-BP, the evaluator's timed row (one decoder through all shots; its row of this report) | 36 of 10,000 | 0.36% (0.26% to 0.50%) | 10 / 5 / 26 (p = 2e-04) |  |
| 0.003 | X | relay-BP with the head's priors, a decoder built per shot (H; ours) [run after the p = 0.005 X test result (the registered extension)] | 2 of 10,000 | 0.02% (0.01% to 0.07%) |  | 61; 1 not converged |
| 0.003 | X | relay-BP with the kit's priors, a decoder built per shot (the declared comparison; the kit) | 7 of 10,000 | 0.07% (0.03% to 0.14%) | 1 / 1 / 6 (p = 0.12) | 144; 2 not converged |
| 0.003 | X | TCBP-JOBS (NVIDIA, GPU; its row of this report) | 1 of 10,000 | 0.01% (0.00% to 0.06%) | 1 / 1 / 0 (p = 1.00) |  |
| 0.003 | X | relay-BP, the evaluator's timed row (one decoder through all shots; its row of this report) | 2 of 10,000 | 0.02% (0.01% to 0.07%) | 1 / 1 / 1 (p = 1.00) |  |
| 0.005 | Z | relay-BP with the head's priors (H) [run after the p = 0.005 X test result (the registered extension)] | not measured | not measured | not measured | not measured |

Source: `report/data/test/predecoder/test_run_p<p>_<basis>_{summary,secondary}.json` (copied from `predecoder/logs/`).

**Not held out: the fallback rule, designed after the first test result; frozen on the validation file before this application; the rule was prompted by the X test shots at p = 0.005, so its X number there is post hoc, not held out.** The rule: for every shot, the answer of relay-BP with the head's priors (H) where that relay-BP converged (its own success flag, no truth), else TCBP-JOBS's answer (T). No knob, no threshold (`predecoder/logs/fallback_rule_frozen.md`). It is not an equal of the rows above.

| p | Basis | What this number is | Shots that fall back to TCBP-JOBS | The rule fails (share, 95% interval) | H fails | TCBP-JOBS fails | Rule against TCBP-JOBS: both / only the rule / only TCBP-JOBS (exact p) | Rule against H: both / only the rule / only H (exact p) |
|---|---|---|---|---|---|---|---|---|
| 0.005 | X | post hoc: the rule was prompted by these same test shots, so this number is not held out | 90 of 10,000 | 107 (1.07%, 0.89% to 1.29%) | 151 | 186 | 91 / 16 / 95 (p = 8e-15) | 106 / 1 / 45 (p = 1e-12) |
| 0.005 | Z | its clean test, run after the p = 0.005 X test result (the registered extension); the rule was frozen before the Z test file was opened | not measured yet | not measured yet | not measured yet | not measured yet | not measured yet | not measured yet |
| 0.004 | X | post hoc: the rule was designed after the first test result, and frozen before it was applied here; run after the p = 0.005 X test result (the registered extension) | 12 of 10,000 | 10 (0.10%, 0.05% to 0.18%) | 15 | 20 | 8 / 2 / 12 (p = 0.013) | 10 / 0 / 5 (p = 0.062) |
| 0.003 | X | post hoc: the rule was designed after the first test result, and frozen before it was applied here; run after the p = 0.005 X test result (the registered extension) | 1 of 10,000 | 1 (0.01%, 0.00% to 0.06%) | 2 | 1 | 1 / 0 / 0 (p = 1.00) | 1 / 0 / 1 (p = 1.00) |

Source: `report/data/test/predecoder/fallback_test_<basis>_p<p>.{json,log}` (`predecoder.fallback`).

**The pair score at p = 0.005: run after the p = 0.005 X test result (the registered extension); the fallback rule's half on X is post hoc.** The kit's score: a pair of an X shot and a Z shot fails when either fails. Not a held-out result, and not in the pair-scored tables of our decoder. TCBP-JOBS's and relay-BP's counts are checked against their rows in this report.

| Decoder | Failures X, Z | Failed pairs | Rate per cycle ± error | Against TCBP-JOBS: both / only this / only TCBP-JOBS (exact p) | Against relay-BP's timed row: both / only this / only relay-BP (exact p) |
|---|---|---|---|---|---|
| relay-BP with the head's priors in both bases, H: run after the p = 0.005 X test result (the registered extension) | not measured yet | not measured yet | not measured yet | not measured yet | not measured yet |

Source: `report/data/test/predecoder/pair_test_p0.005.{json,log}` (`predecoder.pair`).
