**Whose gain is whose, on the test shots.** Every rung is scored on the same per-shot answers (relay-BP and PriorTilt under two seeds on every shot, a fresh decoder per 500 shots), so each step is a paired comparison with the rung before it. These rows are combined offline from per-shot files; they are not the timed rows of the table above. Relay-BP keeps random state from shot to shot, so this decode (a fresh decoder every 500 shots) and the timed row (one decoder through all 10,000 shots) differ: `checks.md` gives both counts and the paired test between them. Compare rungs with rungs, and timed rows with timed rows.

| p | Rung | Whose work | Failures X, Z | Failed pairs | Rate per cycle ± error | Against the rung before: both fail / only this / only the one before (exact p) | Against relay-BP: both fail / only this / only relay-BP (exact p) |
|---|---|---|---|---|---|---|---|
| 0.003 | relay-BP (kit baseline) | the kit | 3, 2 | 5 of 10,000 | 4.17e-05 ± 1.9e-05 | the first rung | the reference |
| 0.003 | PriorTilt, one run (seed A) | IonQ | 3, 3 | 6 of 10,000 | 5.00e-05 ± 2.0e-05 | 4 / 2 / 1 (p = 1.00) | 4 / 2 / 1 (p = 1.00) |
| 0.003 | our hierarchy, lowest-weight tie-break (nothing learned by us) | ours | 2, 1 | 3 of 10,000 | 2.50e-05 ± 1.4e-05 | 3 / 0 / 3 (p = 0.25) | 3 / 0 / 2 (p = 0.50) |
| 0.003 | our hierarchy, lowest weight among the converged candidates (nothing learned by us) | ours | 2, 1 | 3 of 10,000 | 2.50e-05 ± 1.4e-05 | 3 / 0 / 0 (p = 1.00) | 3 / 0 / 2 (p = 0.50) |
| 0.003 | our hierarchy, neural tie-break | ours | 2, 2 | 4 of 10,000 | 3.33e-05 ± 1.7e-05 | 3 / 1 / 0 (p = 1.00) | 3 / 1 / 2 (p = 1.00) |
| 0.003 | ORACLE: a perfect tie-break on the same candidates (a ceiling, not a decoder) | nobody: needs the truth | 2, 1 | 3 of 10,000 | 2.50e-05 ± 1.4e-05 | 3 / 0 / 1 (p = 1.00) | 3 / 0 / 2 (p = 0.50) |
| 0.003 | TCBP-JOBS (NVIDIA, GPU), for reference | NVIDIA; run by teammates | 1, 2 | 3 of 10,000 | 2.50e-05 ± 1.4e-05 | not a rung | 2 / 1 / 3 (p = 0.62) |
| 0.004 | relay-BP (kit baseline) | the kit | 42, 48 | 90 of 10,000 | 7.53e-04 ± 7.9e-05 | the first rung | the reference |
| 0.004 | PriorTilt, one run (seed A) | IonQ | 37, 34 | 71 of 10,000 | 5.94e-04 ± 7.0e-05 | 46 / 25 / 44 (p = 0.029) | 46 / 25 / 44 (p = 0.029) |
| 0.004 | our hierarchy, lowest-weight tie-break (nothing learned by us) | ours | 26, 31 | 57 of 10,000 | 4.76e-04 ± 6.3e-05 | 51 / 6 / 20 (p = 0.009) | 44 / 13 / 46 (p = 2e-05) |
| 0.004 | our hierarchy, lowest weight among the converged candidates (nothing learned by us) | ours | 24, 29 | 53 of 10,000 | 4.43e-04 ± 6.1e-05 | 53 / 0 / 4 (p = 0.12) | 40 / 13 / 50 (p = 3e-06) |
| 0.004 | our hierarchy, neural tie-break | ours | 25, 25 | 50 of 10,000 | 4.18e-04 ± 5.9e-05 | 42 / 8 / 11 (p = 0.65) | 38 / 12 / 52 (p = 5e-07) |
| 0.004 | ORACLE: a perfect tie-break on the same candidates (a ceiling, not a decoder) | nobody: needs the truth | 19, 21 | 40 of 10,000 | 3.34e-04 ± 5.3e-05 | 40 / 0 / 10 (p = 0.002) | 33 / 7 / 57 (p = 8e-11) |
| 0.004 | TCBP-JOBS (NVIDIA, GPU), for reference | NVIDIA; run by teammates | 20, 21 | 41 of 10,000 | 3.42e-04 ± 5.3e-05 | not a rung | 27 / 14 / 63 (p = 1e-08) |
| 0.005 | relay-BP (kit baseline) | the kit | 419, 385 | 794 of 10,000 | 6.87e-03 ± 2.4e-04 | the first rung | the reference |
| 0.005 | PriorTilt, one run (seed A) | IonQ | 303, 296 | 595 of 10,000 | 5.10e-03 ± 2.1e-04 | 413 / 182 / 381 (p = 3e-17) | 413 / 182 / 381 (p = 3e-17) |
| 0.005 | our hierarchy, lowest-weight tie-break (nothing learned by us) | ours | 269, 262 | 527 of 10,000 | 4.50e-03 ± 2.0e-04 | 461 / 66 / 134 (p = 2e-06) | 443 / 84 / 351 (p = 7e-40) |
| 0.005 | our hierarchy, lowest weight among the converged candidates (nothing learned by us) | ours | 241, 231 | 469 of 10,000 | 3.99e-03 ± 1.8e-04 | 469 / 0 / 58 (p = 7e-18) | 391 / 78 / 403 (p = 8e-54) |
| 0.005 | our hierarchy, neural tie-break | ours | 250, 241 | 490 of 10,000 | 4.18e-03 ± 1.9e-04 | 412 / 78 / 57 (p = 0.085) | 393 / 97 / 401 (p = 6e-45) |
| 0.005 | ORACLE: a perfect tie-break on the same candidates (a ceiling, not a decoder) | nobody: needs the truth | 195, 192 | 386 of 10,000 | 3.28e-03 ± 1.7e-04 | 386 / 0 / 104 (p = 1e-31) | 332 / 54 / 462 (p = 8e-82) |
| 0.005 | TCBP-JOBS (NVIDIA, GPU), for reference | NVIDIA; run by teammates | 186, 177 | 360 of 10,000 | 3.05e-03 ± 1.6e-04 | not a rung | 300 / 60 / 494 (p = 7e-86) |
