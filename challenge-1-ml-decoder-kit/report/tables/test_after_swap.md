**Run after the test results were seen, at the team's request; an ablation, not our decoder.** TCBP-JOBS in place of relay-BP in our hierarchy, as the team asked after the test results: no new rule is computed, the rows are rules the ensemble scored on the test shots among its pre-registered rules (`hierarchy.ensemble`; P1 and P2 PriorTilt under two seeds, T TCBP-JOBS, R relay-BP, all from the same saved answers). Rule d is the closest to that swap. Both comparisons stay in one decode mode: TCBP-JOBS alone (T) and our frozen decoder's offline decode (rule a: P1 where the two runs agree, else the network among R, P1, P2) come from the same files. Offline rows: no time.

| p | Rule (scored offline by the ensemble on the test shots) | Failures X, Z | Failed pairs | Rate per cycle ± error | Against TCBP-JOBS alone: failed pairs; both fail / only this rule / only TCBP-JOBS (exact p) | Against our frozen decoder's offline decode (rule a): failed pairs; both fail / only this rule / only rule a (exact p) |
|---|---|---|---|---|---|---|
| 0.003 | rule d: TCBP-JOBS in place of relay-BP (where at least two of the three, the two PriorTilt runs and TCBP-JOBS, agree, that answer; where all three differ, the network among them and relay-BP) | 1, 1 | 2 of 10,000 | 1.67e-05 ± 1.2e-05 | 2 against 3; 2 / 0 / 1 (p = 1.00) | 2 against 4; 2 / 0 / 2 (p = 0.50) |
| 0.003 | rule f, the ensemble's frozen rule, for reference (where the two PriorTilt runs and TCBP-JOBS all agree, that answer; elsewhere joint decoding where it converged, else TCBP-JOBS) | 1, 1 | 2 of 10,000 | 1.67e-05 ± 1.2e-05 | 2 against 3; 2 / 0 / 1 (p = 1.00) | 2 against 4; 2 / 0 / 2 (p = 0.50) |
| 0.004 | rule d: TCBP-JOBS in place of relay-BP (where at least two of the three, the two PriorTilt runs and TCBP-JOBS, agree, that answer; where all three differ, the network among them and relay-BP) | 19, 20 | 39 of 10,000 | 3.26e-04 ± 5.2e-05 | 39 against 41; 29 / 10 / 12 (p = 0.83) | 39 against 50; 35 / 4 / 15 (p = 0.019) |
| 0.004 | rule f, the ensemble's frozen rule, for reference (where the two PriorTilt runs and TCBP-JOBS all agree, that answer; elsewhere joint decoding where it converged, else TCBP-JOBS) | 8, 8 | 16 of 10,000 | 1.33e-04 ± 3.3e-05 | 16 against 41; 15 / 1 / 26 (p = 4e-07) | 16 against 50; 14 / 2 / 36 (p = 5e-09) |
| 0.005 | rule d: TCBP-JOBS in place of relay-BP (where at least two of the three, the two PriorTilt runs and TCBP-JOBS, agree, that answer; where all three differ, the network among them and relay-BP) | 204, 202 | 403 of 10,000 | 3.42e-03 ± 1.7e-04 | 403 against 360; 291 / 112 / 69 (p = 0.002) | 403 against 490; 378 / 25 / 112 (p = 2e-14) |
| 0.005 | rule f, the ensemble's frozen rule, for reference (where the two PriorTilt runs and TCBP-JOBS all agree, that answer; elsewhere joint decoding where it converged, else TCBP-JOBS) | 186, 177 | 360 of 10,000 | 3.05e-03 ± 1.6e-04 | 360 against 360; 360 / 0 / 0 (p = 1.00) | 360 against 490; 266 / 94 / 224 (p = 2e-13) |

**Where rule d and TCBP-JOBS part**, in shots (not pairs): per basis, shots only rule d fails / shots only TCBP-JOBS fails, split by what rule d did on the shot.

| p | On the shots where | X memory: only rule d / only TCBP-JOBS | Z memory: only rule d / only TCBP-JOBS |
|---|---|---|---|
| 0.003 | the two PriorTilt runs agree: rule d keeps their answer | 0 / 0 | 0 / 1 |
| 0.003 | all three differ: rule d takes the network's pick among P1, P2, T and R | 0 / 0 | 0 / 0 |
| 0.003 | the two PriorTilt runs disagree and TCBP-JOBS sides with one of them: rule d's answer is TCBP-JOBS's (checked on every shot) | 0 / 0 | 0 / 0 |
| 0.004 | the two PriorTilt runs agree: rule d keeps their answer | 4 / 6 | 4 / 4 |
| 0.004 | all three differ: rule d takes the network's pick among P1, P2, T and R | 1 / 0 | 1 / 2 |
| 0.004 | the two PriorTilt runs disagree and TCBP-JOBS sides with one of them: rule d's answer is TCBP-JOBS's (checked on every shot) | 0 / 0 | 0 / 0 |
| 0.005 | the two PriorTilt runs agree: rule d keeps their answer | 36 / 31 | 42 / 26 |
| 0.005 | all three differ: rule d takes the network's pick among P1, P2, T and R | 19 / 6 | 16 / 7 |
| 0.005 | the two PriorTilt runs disagree and TCBP-JOBS sides with one of them: rule d's answer is TCBP-JOBS's (checked on every shot) | 0 / 0 | 0 / 0 |

Source: `report/data/test/ensemble_p<p>/test_{X,Z}.npz` and `.json`.
