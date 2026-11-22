Times on the test shots, each beside the row's failures, so that a time is never read without its accuracy. "Mean time ÷ relay-BP's" below 1 means faster than relay-BP on average. One process per row on the team's server; a row on two cores or on a GPU says so. A time is marked "not clean" when the server's CPU quota throttled it for more than 5% of the row's wall time; it is then given no ratio and is not drawn in the time figures.

| p | Decoder | Failed pairs | Mean time per pair, ms | Mean time ÷ relay-BP's | Median, ms | 99th percentile, ms | Runs on |
|---|---|---|---|---|---|---|---|
| 0.003 | BP-OSD (kit baseline) | 60 of 10,000 | 129 (not clean) | time not clean | 18.2 | 792 | CPU, 1 core |
| 0.003 | relay-BP (kit baseline) | 5 of 10,000 | 88.0 (not clean) | time not clean | 54.0 | 579 | CPU, 1 core |
| 0.003 | relay-BP with a learned router | 6 of 10,000 | 42.5 (not clean) | time not clean | 9.5 | 627 | CPU, 1 core |
| 0.003 | PriorTilt, one run | not measured | not measured | not measured | not measured | not measured | not measured |
| 0.003 | our hierarchy, lowest-weight tie-break | not measured | not measured | not measured | not measured | not measured | not measured |
| 0.003 | our hierarchy, neural tie-break | not measured | not measured | not measured | not measured | not measured | not measured |
| 0.003 | our hierarchy, neural tie-break, the two fast decodes side by side on two cores | not measured | not measured | not measured | not measured | not measured | not measured |
| 0.003 | our ensemble (combined offline) | not measured | not measured | not measured | not measured | not measured | not measured |
| 0.003 | TCBP-JOBS (NVIDIA, GPU) | not measured | not measured | not measured | not measured | not measured | not measured |
| 0.004 | BP-OSD (kit baseline) | 472 of 10,000 | 302 | 0.94 | 350 | 863 | CPU, 1 core |
| 0.004 | relay-BP (kit baseline) | 83 of 10,000 | 319 | the reference | 106 | 5,514 | CPU, 1 core |
| 0.004 | relay-BP with a learned router | 87 of 10,000 | 315 | 0.99 | 15.0 | 6,865 | CPU, 1 core |
| 0.004 | PriorTilt, one run | not measured | not measured | not measured | not measured | not measured | not measured |
| 0.004 | our hierarchy, lowest-weight tie-break | not measured | not measured | not measured | not measured | not measured | not measured |
| 0.004 | our hierarchy, neural tie-break | not measured | not measured | not measured | not measured | not measured | not measured |
| 0.004 | our hierarchy, neural tie-break, the two fast decodes side by side on two cores | not measured | not measured | not measured | not measured | not measured | not measured |
| 0.004 | our ensemble (combined offline) | not measured | not measured | not measured | not measured | not measured | not measured |
| 0.004 | TCBP-JOBS (NVIDIA, GPU) | not measured | not measured | not measured | not measured | not measured | not measured |
| 0.005 | BP-OSD (kit baseline) | 2433 of 10,000 | 581 | relay-BP not measured yet | 546 | 1,093 | CPU, 1 core |
| 0.005 | relay-BP (kit baseline) | not measured | not measured | not measured | not measured | not measured | not measured |
| 0.005 | relay-BP with a learned router | not measured | not measured | not measured | not measured | not measured | not measured |
| 0.005 | PriorTilt, one run | not measured | not measured | not measured | not measured | not measured | not measured |
| 0.005 | our hierarchy, lowest-weight tie-break | not measured | not measured | not measured | not measured | not measured | not measured |
| 0.005 | our hierarchy, neural tie-break | not measured | not measured | not measured | not measured | not measured | not measured |
| 0.005 | our hierarchy, neural tie-break, the two fast decodes side by side on two cores | not measured | not measured | not measured | not measured | not measured | not measured |
| 0.005 | our ensemble (combined offline) | not measured | not measured | not measured | not measured | not measured | not measured |
| 0.005 | TCBP-JOBS (NVIDIA, GPU) | not measured | not measured | not measured | not measured | not measured | not measured |
