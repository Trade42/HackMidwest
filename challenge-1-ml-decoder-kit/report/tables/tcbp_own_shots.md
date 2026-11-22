TCBP-JOBS as the teammates first ran it (`tcbp_jobs_baseline.py`, an NVIDIA A100 slice on the ICRN server). **These are its own freshly sampled shots, not the saved test files**, so there is no paired count against the other rows, and the log prints the rate per cycle but not the number of failed pairs.

| p | Failures X, Z | Shots | Rate per cycle ± error (as logged) | Time per pair, ms: mean / p99 | File |
|---|---|---|---|---|---|
| 0.003 | 2, 3 | 10,000 per basis, its own sample (seed 1) | 4.17e-05 ± 1.9e-05 | 4.3 / 15.0 | `tcbp_own_shots/tcbp-jobs_p0.003_shots10000_seed1_20261122_005215_437826+0000.log` |
| 0.004 | 18, 23 | 10,000 per basis, its own sample (seed 1) | 3.42e-04 ± 5.3e-05 | not usable: the GPU was shared with a network training of ours (logged: 42.9 / 109) | `tcbp_own_shots/tcbp-jobs_p0.004_shots10000_seed1_20261122_043426_147901+0000.log` |
| 0.005 | 192, 188 | 10,000 per basis, its own sample (seed 1) | 3.20e-03 ± 1.6e-04 | not usable: the GPU was shared with a network training of ours (logged: 81.7 / 201) | `tcbp_own_shots/tcbp-jobs_p0.005_shots10000_seed1_20261122_050203_539870+0000.log` |
