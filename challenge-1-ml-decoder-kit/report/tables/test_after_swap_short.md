**Run after the test results were seen, at the team's request; an ablation, not our decoder.** Failed pairs of 10,000 test pairs, every row scored offline by the ensemble on the same saved answers (no time); against T and a: the pairs only one of the two fails, exact p. `APPENDIX.md` J has the full table.

| Row | p = 0.003 | p = 0.004 | p = 0.005 |
|---|---|---|---|
| TCBP-JOBS in place of relay-BP in our hierarchy (the ensemble's rule d; run after the test results were seen, at the team's request; an ablation, not our decoder) | **2** · against T: only d 0, only T 1 (p = 1.00) · against a: only d 0, only a 2 (p = 0.50) | **39** · against T: only d 10, only T 12 (p = 0.83) · against a: only d 4, only a 15 (p = 0.019) | **403** · against T: only d 112, only T 69 (p = 0.002) · against a: only d 25, only a 112 (p = 2e-14) |
| TCBP-JOBS alone (T; NVIDIA, GPU) | 3 | 41 | 360 |
| our frozen decoder, the offline decode of the same shots (rule a) | 4 | 50 | 490 |
| our ensemble's frozen rule f (TCBP-JOBS plus joint decoding) | 2 | 16 | 360 |
