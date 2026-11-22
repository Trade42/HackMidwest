**TCBP-JOBS with the head's priors: designed after the test results were seen; formerly a separate branch (predecoder-tcbp).** TCBP-JOBS (NVIDIA's GPU decoder) given, for each shot, the priors our network's head computes from that shot's detectors, in place of the kit's fixed priors.

**Exploration shots, not test shots:** the 5,000 shots of `data/p0.005/validation_X.npz`. TH against each row: both fail / only TH / only that row (exact p), on the same shots.

| Decoder | Failed shots | Share (95% interval) | TH against it: both / only TH / only it (exact p) |
|---|---|---|---|
| TCBP-JOBS with the head's priors, per shot (TH) | 33 of 5,000 | 0.66% (0.47% to 0.93%) |  |
| TCBP-JOBS with the kit's priors, per shot, the same run (the like-for-like control) | 100 of 5,000 | 2.00% (1.65% to 2.43%) | 25 / 8 / 75 (p = 9e-15) |
| TCBP-JOBS with the kit's priors, the stored file (another fit) | 91 of 5,000 | 1.82% (1.48% to 2.23%) | 25 / 8 / 66 (p = 2e-12) |
| relay-BP with the head's priors (H, the predecoder) | 79 of 5,000 | 1.58% (1.27% to 1.96%) | 29 / 4 / 50 (p = 4e-11) |
| the frozen fallback rule (post hoc on test X at p = 0.005) | 50 of 5,000 | 1.00% (0.76% to 1.32%) | 28 / 5 / 22 (p = 0.002) |
| H where H converged, else TH (designed after the test results) | 44 of 5,000 | 0.88% (0.66% to 1.18%) | 29 / 4 / 15 (p = 0.019) |

**Test shots: designed after the test results were seen; formerly a separate branch (predecoder-tcbp).** One basis at a time (X until the Z run lands), accuracy only. Its knobs were fitted at p = 0.005 and applied unchanged at p = 0.004 and 0.003, with no validation there. The declared pair is TH against TCBP-JOBS's stored test file; at p = 0.005 the same run's kit-priors arm is the like-for-like control.

| p | Basis | TH fails | TCBP-JOBS, its stored test file (its row in this report), fails | TH against it, the declared pair: both / only TH / only TCBP-JOBS (exact p) | TCBP-JOBS with the kit's priors in the same run (the like-for-like control) fails | TH against that control (exact p) |
|---|---|---|---|---|---|---|
| 0.005 | X | 72 of 10,000 | 186 | 50 / 22 / 136 (p = 3e-21) | 186 | 50 / 22 / 136 (p = 3e-21) |
| 0.004 | X | 4 of 10,000 | 20 | 2 / 2 / 18 (p = 4e-04) | not run (X at p = 0.005 only) |  |
| 0.003 | X | 0 of 10,000 | 1 | 0 / 0 / 1 (p = 1.00) | not run (X at p = 0.005 only) |  |
| 0.005 | Z | not measured yet | not measured yet | not measured yet | not measured yet | not measured yet |

Source: `report/data/exploration/predecoder_tcbp/` (branch `predecoder-tcbp` at `dad4fd50`) and `report/data/test/predecoder_tcbp/`.
