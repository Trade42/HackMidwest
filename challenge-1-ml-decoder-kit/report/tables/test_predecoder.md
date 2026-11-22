**The learned predecoder on the test shots: secondary (chosen after the declared gate at p = 0.004 failed), one basis at a time, accuracy only, not a pair score; its times are not clean (a shared 12-worker pool on the team's PC), so iterations are shown instead.** Relay-BP at the kit's settings, given for each shot the priors that a second output (a "head") of our trained network computes from that shot's detectors; truth enters only the scoring. Per noise level and basis: the saved test file of that basis alone, so a row counts shots, not pairs, and cannot sit in the pair-scored tables. H against the others: both fail / only H fails / only the other fails (exact p), on the same shots. Counts from the run's own files (`predecoder.compare`, `predecoder.secondary`); TCBP-JOBS's and relay-BP's totals are checked against their rows in this report.

| p | Basis | Decoder | Failed shots | Share (95% interval) | H against it: both / only H / only it (exact p) | Mean BP iterations per shot; shots not converged |
|---|---|---|---|---|---|---|
| 0.005 | X | relay-BP with the head's priors, a decoder built per shot (H; ours) | 151 of 10,000 | 1.51% (1.29% to 1.77%) |  | 711; 90 not converged |
| 0.005 | X | relay-BP with the kit's priors, a decoder built per shot (the declared comparison; the kit) | 433 of 10,000 | 4.33% (3.95% to 4.75%) | 125 / 26 / 308 (p = 2e-62) | 2,436; 258 not converged |
| 0.005 | X | TCBP-JOBS (NVIDIA, GPU; its row of this report) | 186 of 10,000 | 1.86% (1.61% to 2.14%) | 90 / 61 / 96 (p = 0.006) |  |
| 0.005 | X | relay-BP, the evaluator's timed row (one decoder through all shots; its row of this report) | 432 of 10,000 | 4.32% (3.94% to 4.74%) | 124 / 27 / 308 (p = 1e-61) |  |
| 0.004 | X | relay-BP with the head's priors (H) | not measured | not measured | not measured | not measured |
| 0.003 | X | relay-BP with the head's priors (H) | not measured | not measured | not measured | not measured |

Source: `report/data/test/predecoder/test_run_p<p>_<basis>_{summary,secondary}.json` (copied from `predecoder/logs/`).

**Not held out: the fallback rule, designed after the first test result; frozen on the validation file before this application; the rule was prompted by the X test shots at p = 0.005, so its X number there is post hoc, not held out.** The rule: for every shot, the answer of relay-BP with the head's priors (H) where that relay-BP converged (its own success flag, no truth), else TCBP-JOBS's answer (T). No knob, no threshold (`predecoder/logs/fallback_rule_frozen.md`). It is not an equal of the rows above.

| p | Basis | What this number is | Shots that fall back to TCBP-JOBS | The rule fails (share, 95% interval) | H fails | TCBP-JOBS fails | Rule against TCBP-JOBS: both / only the rule / only TCBP-JOBS (exact p) | Rule against H: both / only the rule / only H (exact p) |
|---|---|---|---|---|---|---|---|---|
| 0.005 | X | post hoc: the rule was prompted by these same test shots, so this number is not held out | 90 of 10,000 | 107 (1.07%, 0.89% to 1.29%) | 151 | 186 | 91 / 16 / 95 (p = 8e-15) | 106 / 1 / 45 (p = 1e-12) |
| 0.005 | Z | its clean test: the rule was frozen before the Z test file was opened | not measured yet | not measured yet | not measured yet | not measured yet | not measured yet | not measured yet |

Source: `report/data/test/predecoder/fallback_test_<basis>_p<p>.{json,log}` (`predecoder.fallback`).
