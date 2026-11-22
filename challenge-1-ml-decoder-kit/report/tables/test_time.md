Times on the test shots, each beside the row's failures, so that a time is never read without its accuracy. "Mean time ÷ relay-BP's" below 1 means faster than relay-BP on average. One process per row on the team's server; a row on two cores or on a GPU says so. A time is marked "not clean" when the server's CPU quota throttled it for more than 5% of its wall time, estimated as the kernel's throttled seconds (summed over the server's CPUs) divided by the quota of 12 CPUs times the row's wall time; it is then given no ratio and is not drawn in the time figures.

| p | Decoder | Failed pairs | Mean time per pair, ms | Mean time ÷ relay-BP's | Median, ms | 99th percentile, ms | Runs on |
|---|---|---|---|---|---|---|---|
| 0.003 | BP-OSD (kit baseline) | 60 of 10,000 | 131 | 1.54 | 18.7 | 800 | CPU, 1 core |
| 0.003 | relay-BP (kit baseline) | 5 of 10,000 | 85.3 | the reference | 52.5 | 557 | CPU, 1 core |
| 0.003 | relay-BP with a learned router | 6 of 10,000 | 41.0 | 0.48 | 9.3 | 604 | CPU, 1 core |
| 0.003 | PriorTilt, one run | 4 of 10,000 | 22.7 | 0.27 | 7.2 | 188 | CPU, 1 core |
| 0.003 | our hierarchy, lowest-weight tie-break (run after the test results were seen, at the team's request; an ablation, not our decoder) | 4 of 10,000 | 46.9 | 0.55 | 14.6 | 404 | CPU, 1 core |
| 0.003 | our hierarchy, classical tie-break (lowest weight among the converged candidates) | 4 of 10,000 | 46.3 | 0.54 | 14.4 | 399 | CPU, 1 core |
| 0.003 | our hierarchy, neural tie-break | 3 of 10,000 | 45.7 | 0.54 | 14.2 | 395 | CPU, 1 core |
| 0.003 | our hierarchy, neural tie-break, the two fast decodes side by side on two cores | 3 of 10,000 | 28.9 | 0.34 | 7.8 | 245 | CPU, 2 cores |
| 0.003 | our ensemble (combined offline) (rule f, frozen on the training shots: where PriorTilt twice and TCBP-JOBS agree, that answer; elsewhere joint decoding where it converged, else TCBP-JOBS; joint decoding ran on 5 of the 5 disputed shots) | 2 of 10,000 | not timed in one process |  |  |  |  |
| 0.003 | TCBP-JOBS (NVIDIA, GPU) | 3 of 10,000 | 3.9 | not comparable: another device and timer | 3.8 | 15.0 | GPU |
| 0.003 | our hierarchy, classical tie-break (lowest weight among the converged candidates), the two fast decodes side by side on two cores (run after the test results were seen, at the team's request; an ablation, not our decoder) | 4 of 10,000 | 29.0 | 0.34 | 7.8 | 245 | CPU, 2 cores |
| 0.004 | BP-OSD (kit baseline) | 472 of 10,000 | 302 | 0.94 | 350 | 863 | CPU, 1 core |
| 0.004 | relay-BP (kit baseline) | 83 of 10,000 | 319 | the reference | 106 | 5,514 | CPU, 1 core |
| 0.004 | relay-BP with a learned router | 87 of 10,000 | 315 | 0.99 | 15.0 | 6,865 | CPU, 1 core |
| 0.004 | PriorTilt, one run | 74 of 10,000 | 87.3 | 0.27 | 34.2 | 884 | CPU, 1 core |
| 0.004 | our hierarchy, lowest-weight tie-break (run after the test results were seen, at the team's request; an ablation, not our decoder) | 60 of 10,000 | 215 | 0.67 | 81.9 | 2,054 | CPU, 1 core |
| 0.004 | our hierarchy, classical tie-break (lowest weight among the converged candidates) | 52 of 10,000 | 218 | 0.68 | 83.4 | 2,090 | CPU, 1 core |
| 0.004 | our hierarchy, neural tie-break | 44 of 10,000 | 216 | 0.68 | 83.1 | 2,064 | CPU, 1 core |
| 0.004 | our hierarchy, neural tie-break, the two fast decodes side by side on two cores | 44 of 10,000 | 150 | 0.47 | 55.5 | 1,292 | CPU, 2 cores |
| 0.004 | our ensemble (combined offline) (rule f, frozen on the training shots: where PriorTilt twice and TCBP-JOBS agree, that answer; elsewhere joint decoding where it converged, else TCBP-JOBS; joint decoding ran on 93 of the 93 disputed shots) | 16 of 10,000 | not timed in one process |  |  |  |  |
| 0.004 | TCBP-JOBS (NVIDIA, GPU) | 41 of 10,000 | 7.8 | not comparable: another device and timer | 5.8 | 35.3 | GPU |
| 0.004 | our hierarchy, classical tie-break (lowest weight among the converged candidates), the two fast decodes side by side on two cores (run after the test results were seen, at the team's request; an ablation, not our decoder) | 52 of 10,000 | 150 | 0.47 | 55.6 | 1,302 | CPU, 2 cores |
| 0.005 | BP-OSD (kit baseline) | 2433 of 10,000 | 581 | 0.40 | 546 | 1,093 | CPU, 1 core |
| 0.005 | relay-BP (kit baseline) | 845 of 10,000 | 1,471 | the reference | 424 | 8,270 | CPU, 1 core |
| 0.005 | relay-BP with a learned router | 807 of 10,000 | 1,928 | 1.31 | 303 | 12,708 | CPU, 1 core |
| 0.005 | PriorTilt, one run | 601 of 10,000 | 264 | 0.18 | 120 | 2,167 | CPU, 1 core |
| 0.005 | our hierarchy, lowest-weight tie-break (from per-shot files, not a timed evaluator row) | 527 of 10,000 | not timed in one process |  |  |  |  |
| 0.005 | our hierarchy, classical tie-break (lowest weight among the converged candidates) (from per-shot files, not a timed evaluator row) | 469 of 10,000 | not timed in one process |  |  |  |  |
| 0.005 | our hierarchy, neural tie-break | 479 of 10,000 | 825 | 0.56 | 246 | 9,885 | CPU, 1 core |
| 0.005 | our hierarchy, neural tie-break, the two fast decodes side by side on two cores | 479 of 10,000 | 608 | 0.41 | 138 | 7,794 | CPU, 2 cores |
| 0.005 | our ensemble (combined offline) (rule f, frozen on the training shots: where PriorTilt twice and TCBP-JOBS agree, that answer; elsewhere joint decoding where it converged, else TCBP-JOBS; joint decoding ran on 0 of the 735 disputed shots; so it fails exactly the shots TCBP-JOBS fails) | 360 of 10,000 | not timed in one process |  |  |  |  |
| 0.005 | TCBP-JOBS (NVIDIA, GPU) | 360 of 10,000 | 16.9 | not comparable: another device and timer | 13.3 | 56.4 | GPU |
| 0.005 | our hierarchy, classical tie-break (lowest weight among the converged candidates), the two fast decodes side by side on two cores | not measured | not measured | not measured | not measured | not measured | not measured |
