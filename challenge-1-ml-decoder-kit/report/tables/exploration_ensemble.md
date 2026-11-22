**Exploration shots, not test shots**: the team's validation files (5,000 shots per basis), each member's saved answers combined offline by `hierarchy.ensemble`. Members: R relay-BP, P1 and P2 PriorTilt under two seeds, T TCBP-JOBS (GPU), J joint relay-BP on the shots where P1, P2 and T are not unanimous. A rule that names "the network" uses the learned tie-break; the others need no learning of ours. An ORACLE is given the truth: a ceiling, not a decoder. A pair is shot k of X with shot k of Z.

| p | Row | Failures X, Z | Failed pairs | Rate per cycle ± error | Against relay-BP alone: both fail / only this / only relay-BP (exact p) | Against TCBP-JOBS alone: both fail / only this / only TCBP-JOBS (exact p) |
|---|---|---|---|---|---|---|
| 0.003 | R alone | 2, 2 | 4 of 5,000 | 6.67e-05 ± 3.3e-05 |  | 0 / 4 / 1 (p = 0.38) |
| 0.003 | P1 alone | 2, 2 | 4 of 5,000 | 6.67e-05 ± 3.3e-05 | 3 / 1 / 1 (p = 1.00) | 0 / 4 / 1 (p = 0.38) |
| 0.003 | P2 alone | 3, 2 | 5 of 5,000 | 8.34e-05 ± 3.7e-05 | 3 / 2 / 1 (p = 1.00) | 0 / 5 / 1 (p = 0.22) |
| 0.003 | T alone | 0, 1 | 1 of 5,000 | 1.67e-05 ± 1.7e-05 | 0 / 1 / 4 (p = 0.38) |  |
| 0.003 | ac: P1 where P1 = P2; else the lowest weight among the converged of R, P1, P2 | 2, 2 | 4 of 5,000 | 6.67e-05 ± 3.3e-05 | 4 / 0 / 0 (p = 1.00) | 0 / 4 / 1 (p = 0.38) |
| 0.003 | aw: P1 where P1 = P2; else the lowest weight of R, P1, P2 | 2, 2 | 4 of 5,000 | 6.67e-05 ± 3.3e-05 | 4 / 0 / 0 (p = 1.00) | 0 / 4 / 1 (p = 0.38) |
| 0.003 | bw: P1 where P1 = P2 = T; else the lowest weight of P1, P2, T, R | 0, 0 | 0 of 5,000 | 0.00e+00 ± 0.0e+00 | 0 / 0 / 4 (p = 0.12) | 0 / 0 / 1 (p = 1.00) |
| 0.003 | f: P1 where P1 = P2 = T; else J where it converged, else T **(the rule chosen on the training shots)** | 1, 0 | 1 of 5,000 | 1.67e-05 ± 1.7e-05 | 0 / 1 / 4 (p = 0.38) | 0 / 1 / 1 (p = 1.00) |
| 0.003 | gw: P1 where P1 = P2 = T; else most of J, T, P1, P2, R; a tie: that order | 1, 0 | 1 of 5,000 | 1.67e-05 ± 1.7e-05 | 1 / 0 / 3 (p = 0.25) | 0 / 1 / 1 (p = 1.00) |
| 0.003 | ORACLE (needs the truth): a perfect choice among P1, P2, T | 0, 0 | 0 of 5,000 | 0.00e+00 ± 0.0e+00 | 0 / 0 / 4 (p = 0.12) | 0 / 0 / 1 (p = 1.00) |
| 0.003 | ORACLE (needs the truth): a perfect choice among P1, P2, T, R | 0, 0 | 0 of 5,000 | 0.00e+00 ± 0.0e+00 | 0 / 0 / 4 (p = 0.12) | 0 / 0 / 1 (p = 1.00) |
| 0.003 | ORACLE (needs the truth): a perfect choice among P1, P2, T, R, J | 0, 0 | 0 of 5,000 | 0.00e+00 ± 0.0e+00 | 0 / 0 / 4 (p = 0.12) | 0 / 0 / 1 (p = 1.00) |
| 0.003 | ORACLE (needs the truth): a perfect choice among R, P1, P2 | 1, 1 | 2 of 5,000 | 3.33e-05 ± 2.4e-05 | 2 / 0 / 2 (p = 0.50) | 0 / 2 / 1 (p = 1.00) |
| 0.005 | R alone | 194, 189 | 373 of 5,000 | 6.44e-03 ± 3.3e-04 |  | 151 / 222 / 36 (p = 7e-34) |
| 0.005 | P1 alone | 137, 144 | 276 of 5,000 | 4.72e-03 ± 2.8e-04 | 188 / 88 / 185 (p = 4e-09) | 137 / 139 / 50 (p = 7e-11) |
| 0.005 | P2 alone | 152, 145 | 289 of 5,000 | 4.95e-03 ± 2.9e-04 | 198 / 91 / 175 (p = 3e-07) | 138 / 151 / 49 (p = 3e-13) |
| 0.005 | T alone | 91, 97 | 187 of 5,000 | 3.17e-03 ± 2.3e-04 | 151 / 36 / 222 (p = 7e-34) |  |
| 0.005 | ac: P1 where P1 = P2; else the lowest weight among the converged of R, P1, P2 | 108, 104 | 207 of 5,000 | 3.52e-03 ± 2.4e-04 | 177 / 30 / 196 (p = 5e-31) | 126 / 81 / 61 (p = 0.11) |
| 0.005 | aw: P1 where P1 = P2; else the lowest weight of R, P1, P2 | 129, 118 | 241 of 5,000 | 4.11e-03 ± 2.6e-04 | 206 / 35 / 167 (p = 8e-22) | 137 / 104 / 50 (p = 2e-05) |
| 0.005 | bw: P1 where P1 = P2 = T; else the lowest weight of P1, P2, T, R | 107, 113 | 218 of 5,000 | 3.71e-03 ± 2.5e-04 | 184 / 34 / 189 (p = 3e-27) | 182 / 36 / 5 (p = 8e-07) |
| 0.005 | f: P1 where P1 = P2 = T; else J where it converged, else T **(the rule chosen on the training shots)** | 91, 97 | 187 of 5,000 | 3.17e-03 ± 2.3e-04 | 151 / 36 / 222 (p = 7e-34) | 187 / 0 / 0 (p = 1.00) |
| 0.005 | gw: P1 where P1 = P2 = T; else most of J, T, P1, P2, R; a tie: that order | 90, 92 | 179 of 5,000 | 3.03e-03 ± 2.3e-04 | 162 / 17 / 211 (p = 9e-44) | 159 / 20 / 28 (p = 0.31) |
| 0.005 | ORACLE (needs the truth): a perfect choice among P1, P2, T | 53, 63 | 116 of 5,000 | 1.95e-03 ± 1.8e-04 | 105 / 11 / 268 (p = 4e-65) | 116 / 0 / 71 (p = 8e-22) |
| 0.005 | ORACLE (needs the truth): a perfect choice among P1, P2, T, R | 50, 55 | 105 of 5,000 | 1.77e-03 ± 1.7e-04 | 105 / 0 / 268 (p = 4e-81) | 105 / 0 / 82 (p = 4e-25) |
| 0.005 | ORACLE (needs the truth): a perfect choice among P1, P2, T, R, J | 50, 55 | 105 of 5,000 | 1.77e-03 ± 1.7e-04 | 105 / 0 / 268 (p = 4e-81) | 105 / 0 / 82 (p = 4e-25) |
| 0.005 | ORACLE (needs the truth): a perfect choice among R, P1, P2 | 75, 82 | 155 of 5,000 | 2.62e-03 ± 2.1e-04 | 155 / 0 / 218 (p = 5e-66) | 106 / 49 / 81 (p = 0.006) |

Source: `report/data/exploration/ensemble/ens_p<p>/validation_{X,Z}.npz` and `.json`.
