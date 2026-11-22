**Whose gain is whose, on the test shots.** Every rung is scored on the same per-shot answers (relay-BP and PriorTilt under two seeds on every shot, a fresh decoder per 500 shots), so each step is a paired comparison with the rung before it. These rows are combined offline from per-shot files; they are not the timed rows of the table above, and relay-BP's count here differs from its timed row by a few pairs because its random draws differ.

| p | Rung | Whose work | Failures X, Z | Failed pairs | Rate per cycle ± error | Against the rung before: both fail / only this / only the one before (exact p) | Against relay-BP: both fail / only this / only relay-BP (exact p) |
|---|---|---|---|---|---|---|---|
| 0.003 | relay-BP (kit baseline) | the kit | 3, 2 | 5 of 10,000 | 4.17e-05 ± 1.9e-05 | the first rung | the reference |
| 0.003 | PriorTilt, one run (seed A) | IonQ | 3, 3 | 6 of 10,000 | 5.00e-05 ± 2.0e-05 | 4 / 2 / 1 (p = 1.00) | 4 / 2 / 1 (p = 1.00) |
| 0.003 | our hierarchy, lowest-weight tie-break (nothing learned by us) | ours | 2, 1 | 3 of 10,000 | 2.50e-05 ± 1.4e-05 | 3 / 0 / 3 (p = 0.25) | 3 / 0 / 2 (p = 0.50) |
| 0.003 | our hierarchy, lowest weight among the converged candidates (nothing learned by us) | ours | 2, 1 | 3 of 10,000 | 2.50e-05 ± 1.4e-05 | 3 / 0 / 0 (p = 1.00) | 3 / 0 / 2 (p = 0.50) |
| 0.003 | our hierarchy, neural tie-break | ours | 2, 2 | 4 of 10,000 | 3.33e-05 ± 1.7e-05 | 3 / 1 / 0 (p = 1.00) | 3 / 1 / 2 (p = 1.00) |
| 0.003 | ORACLE: a perfect tie-break on the same candidates (a ceiling, not a decoder) | nobody: needs the truth | 2, 1 | 3 of 10,000 | 2.50e-05 ± 1.4e-05 | 3 / 0 / 1 (p = 1.00) | 3 / 0 / 2 (p = 0.50) |
| 0.003 | TCBP-JOBS (NVIDIA, GPU), for reference | NVIDIA; run by teammates | 1, 2 | 3 of 10,000 | 2.50e-05 ± 1.4e-05 | not a rung | 2 / 1 / 3 (p = 0.62) |
| 0.004 | every rung | not measured | not measured | not measured | not measured | not measured | not measured |
| 0.005 | every rung | not measured | not measured | not measured | not measured | not measured | not measured |
