The teammates' own evaluator runs on the same test files, made earlier on other machines or under other load. They are a cross-check of the counts. **Their times are not comparable with the timed table** and are shown only with their machine.

| p | Decoder | Failures X, Z | Failed pairs | Rate per cycle ± error | Time per pair, ms: mean / median / p99 | Router against relay-BP: both / only router / only relay-BP (exact p) | Machine |
|---|---|---|---|---|---|---|---|
| 0.003 | BP-OSD | 26, 34 | 60 of 10,000 | 5.01e-04 ± 6.5e-05 | 131 / 18.8 / 803 |  | ICRN server, before the quiet window |
| 0.003 | relay-BP | 2, 3 | 5 of 10,000 | 4.17e-05 ± 1.9e-05 | 85.7 / 52.6 / 556 |  | ICRN server, before the quiet window |
| 0.003 | relay-BP, first solution only (a control) | 7, 6 | 13 of 10,000 | 1.08e-04 ± 3.0e-05 | 18.1 / 9.0 / 165 |  | ICRN server, before the quiet window |
| 0.003 | relay-BP, first leg or full (a control) | 3, 4 | 7 of 10,000 | 5.84e-05 ± 2.2e-05 | 36.5 / 8.9 / 574 |  | ICRN server, before the quiet window |
| 0.003 | relay-BP with the learned router | 3, 3 | 6 of 10,000 | 5.00e-05 ± 2.0e-05 | 47.2 / 9.7 / 698 | 4 / 2 / 1 (p = 1.00) | ICRN server, before the quiet window |
| 0.004 | BP-OSD | 231, 250 | 480 of 10,000 | 4.09e-03 ± 1.9e-04 | 144 / 159 / 439 |  | a teammate's Mac (arm64) |
| 0.004 | relay-BP | 41, 44 | 84 of 10,000 | 7.03e-04 ± 7.7e-05 | 131 / 42.9 / 2,215 |  | a teammate's Mac (arm64) |
| 0.004 | relay-BP, first solution only (a control) | 53, 71 | 122 of 10,000 | 1.02e-03 ± 9.3e-05 | 36.5 / 6.1 / 632 |  | a teammate's Mac (arm64) |
| 0.004 | relay-BP, first leg or full (a control) | 47, 51 | 98 of 10,000 | 8.20e-04 ± 8.3e-05 | 96.3 / 6.0 / 2,204 |  | a teammate's Mac (arm64) |
| 0.004 | relay-BP with the learned router | 37, 47 | 84 of 10,000 | 7.03e-04 ± 7.7e-05 | 124 / 6.1 / 2,663 | 48 / 36 / 36 (p = 1.00) | a teammate's Mac (arm64) |
