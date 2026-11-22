Paired comparisons on the test shots: two decoders on the same 10,000 pairs, and how many pairs only one of the two fails. The exact p is the two-sided binomial test on those pairs (under "no difference" each is a fair coin). This is where each gain is attributed: PriorTilt is IonQ's, the hierarchy and its tie-break are ours.

| p | First decoder | Second decoder | Failed pairs: first against second | Both fail | Only the first fails / only the second fails | Exact p |
|---|---|---|---|---|---|---|
| 0.003 | relay-BP with a learned router | relay-BP (kit baseline) | 6 against 5 | 4 | 2 / 1 | 1.00 |
| 0.003 | PriorTilt, one run (from per-shot files, not a timed evaluator row) | relay-BP (kit baseline) | 6 against 5 | 3 | 3 / 2 | 1.00 |
| 0.003 | our hierarchy, lowest-weight tie-break (from per-shot files, not a timed evaluator row) | PriorTilt, one run (from per-shot files, not a timed evaluator row) | 3 against 6 | 3 | 0 / 3 | 0.25 |
| 0.003 | our hierarchy, lowest-weight tie-break (from per-shot files, not a timed evaluator row) | relay-BP (kit baseline) | 3 against 5 | 2 | 1 / 3 | 0.62 |
| 0.003 | our hierarchy, neural tie-break | our hierarchy, classical tie-break (lowest weight among the converged candidates) (from per-shot files, not a timed evaluator row) | 3 against 3 | 2 | 1 / 1 | 1.00 |
| 0.003 | our hierarchy, classical tie-break (lowest weight among the converged candidates) (from per-shot files, not a timed evaluator row) | PriorTilt, one run (from per-shot files, not a timed evaluator row) | 3 against 6 | 3 | 0 / 3 | 0.25 |
| 0.003 | our hierarchy, classical tie-break (lowest weight among the converged candidates) (from per-shot files, not a timed evaluator row) | relay-BP (kit baseline) | 3 against 5 | 2 | 1 / 3 | 0.62 |
| 0.003 | our hierarchy, neural tie-break | our hierarchy, lowest-weight tie-break (from per-shot files, not a timed evaluator row) | 3 against 3 | 2 | 1 / 1 | 1.00 |
| 0.003 | our hierarchy, neural tie-break | relay-BP (kit baseline) | 3 against 5 | 3 | 0 / 2 | 0.50 |
| 0.003 | our hierarchy, neural tie-break | TCBP-JOBS (NVIDIA, GPU) | 3 against 3 | 2 | 1 / 1 | 1.00 |
| 0.003 | our ensemble (combined offline) (rule f: P1 where P1 = P2 = T; else J where it converged, else T; frozen on the training shots) | our hierarchy, neural tie-break | 2 against 3 | 2 | 0 / 1 | 1.00 |
| 0.003 | our ensemble (combined offline) (rule f: P1 where P1 = P2 = T; else J where it converged, else T; frozen on the training shots) | TCBP-JOBS (NVIDIA, GPU) | 2 against 3 | 2 | 0 / 1 | 1.00 |
| 0.003 | TCBP-JOBS (NVIDIA, GPU) | relay-BP (kit baseline) | 3 against 5 | 2 | 1 / 3 | 0.62 |
| 0.004 | relay-BP with a learned router | relay-BP (kit baseline) | 87 against 83 | 51 | 36 / 32 | 0.72 |
| 0.004 | PriorTilt, one run | relay-BP (kit baseline) | not measured | not measured | not measured | not measured |
| 0.004 | our hierarchy, lowest-weight tie-break | PriorTilt, one run | not measured | not measured | not measured | not measured |
| 0.004 | our hierarchy, lowest-weight tie-break | relay-BP (kit baseline) | not measured | not measured | not measured | not measured |
| 0.004 | our hierarchy, neural tie-break | our hierarchy, classical tie-break (lowest weight among the converged candidates) | not measured | not measured | not measured | not measured |
| 0.004 | our hierarchy, classical tie-break (lowest weight among the converged candidates) | PriorTilt, one run | not measured | not measured | not measured | not measured |
| 0.004 | our hierarchy, classical tie-break (lowest weight among the converged candidates) | relay-BP (kit baseline) | not measured | not measured | not measured | not measured |
| 0.004 | our hierarchy, neural tie-break | our hierarchy, lowest-weight tie-break | not measured | not measured | not measured | not measured |
| 0.004 | our hierarchy, neural tie-break | relay-BP (kit baseline) | 44 against 83 | 29 | 15 / 54 | 3e-06 |
| 0.004 | our hierarchy, neural tie-break | TCBP-JOBS (NVIDIA, GPU) | 44 against 41 | 24 | 20 / 17 | 0.74 |
| 0.004 | our ensemble (combined offline) | our hierarchy, neural tie-break | not measured | not measured | not measured | not measured |
| 0.004 | our ensemble (combined offline) | TCBP-JOBS (NVIDIA, GPU) | not measured | not measured | not measured | not measured |
| 0.004 | TCBP-JOBS (NVIDIA, GPU) | relay-BP (kit baseline) | 41 against 83 | 27 | 14 / 56 | 4e-07 |
| 0.005 | relay-BP with a learned router | relay-BP (kit baseline) | not measured | not measured | not measured | not measured |
| 0.005 | PriorTilt, one run | relay-BP (kit baseline) | not measured | not measured | not measured | not measured |
| 0.005 | our hierarchy, lowest-weight tie-break | PriorTilt, one run | not measured | not measured | not measured | not measured |
| 0.005 | our hierarchy, lowest-weight tie-break | relay-BP (kit baseline) | not measured | not measured | not measured | not measured |
| 0.005 | our hierarchy, neural tie-break | our hierarchy, classical tie-break (lowest weight among the converged candidates) | not measured | not measured | not measured | not measured |
| 0.005 | our hierarchy, classical tie-break (lowest weight among the converged candidates) | PriorTilt, one run | not measured | not measured | not measured | not measured |
| 0.005 | our hierarchy, classical tie-break (lowest weight among the converged candidates) | relay-BP (kit baseline) | not measured | not measured | not measured | not measured |
| 0.005 | our hierarchy, neural tie-break | our hierarchy, lowest-weight tie-break | not measured | not measured | not measured | not measured |
| 0.005 | our hierarchy, neural tie-break | relay-BP (kit baseline) | not measured | not measured | not measured | not measured |
| 0.005 | our hierarchy, neural tie-break | TCBP-JOBS (NVIDIA, GPU) | not measured | not measured | not measured | not measured |
| 0.005 | our ensemble (combined offline) | our hierarchy, neural tie-break | not measured | not measured | not measured | not measured |
| 0.005 | our ensemble (combined offline) | TCBP-JOBS (NVIDIA, GPU) | not measured | not measured | not measured | not measured |
| 0.005 | TCBP-JOBS (NVIDIA, GPU) | relay-BP (kit baseline) | 360 against 845 | 306 | 54 / 539 | 1e-101 |
