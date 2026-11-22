| Decoder | p = 0.003 | p = 0.004 | p = 0.005 |
|---|---|---|---|
| BP-OSD (kit baseline) | 5.01e-04 ± 6.5e-05 · 60 pairs · 129 ms | 4.02e-03 ± 1.8e-04 · 472 pairs · 302 ms | not measured |
| relay-BP (kit baseline) | 4.17e-05 ± 1.9e-05 · 5 pairs · 88.0 ms | 6.94e-04 ± 7.6e-05 · 83 pairs · 319 ms | not measured |
| relay-BP with a learned router | 5.00e-05 ± 2.0e-05 · 6 pairs · 42.5 ms | 7.28e-04 ± 7.8e-05 · 87 pairs · 315 ms | not measured |
| PriorTilt, one run | not measured | not measured | not measured |
| our hierarchy, lowest-weight tie-break | not measured | not measured | not measured |
| our hierarchy, neural tie-break | not measured | not measured | not measured |
| our hierarchy, neural tie-break, the two fast decodes side by side on two cores | not measured | not measured | not measured |
| our ensemble (combined offline) | not measured | not measured | not measured |
| TCBP-JOBS (NVIDIA, GPU) | not measured on these shots; on its own shots: 4.17e-05 ± 1.9e-05 | not measured on these shots; on its own shots: 3.42e-04 ± 5.3e-05 | not measured on these shots; on its own shots: 3.20e-03 ± 1.6e-04 |

Each cell: logical error rate per cycle ± the kit's error · failed pairs of 10,000 · mean time per pair. Test shots; `report/tables/test_p<p>.md` has the intervals, the paired counts and where each row ran.
