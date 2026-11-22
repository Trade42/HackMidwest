| Decoder | p = 0.003 | p = 0.004 | p = 0.005 |
|---|---|---|---|
| BP-OSD (kit baseline) | 5.01e-04 ± 6.5e-05 · 60 pairs · 129 ms | 4.02e-03 ± 1.8e-04 · 472 pairs · 302 ms | 2.30e-02 ± 4.6e-04 · 2433 pairs · 581 ms |
| relay-BP (kit baseline) | 4.17e-05 ± 1.9e-05 · 5 pairs · 88.0 ms | 6.94e-04 ± 7.6e-05 · 83 pairs · 319 ms | 7.33e-03 ± 2.5e-04 · 845 pairs · 1,471 ms |
| relay-BP with a learned router | 5.00e-05 ± 2.0e-05 · 6 pairs · 42.5 ms | 7.28e-04 ± 7.8e-05 · 87 pairs · 315 ms | not measured |
| PriorTilt, one run | not measured | not measured | not measured |
| our hierarchy, lowest-weight tie-break | not measured | not measured | not measured |
| our hierarchy, classical tie-break (lowest weight among the converged candidates) | not measured | not measured | not measured |
| our hierarchy, neural tie-break | 2.50e-05 ± 1.4e-05 · 3 pairs · 45.7 ms | 3.67e-04 ± 5.5e-05 · 44 pairs · 216 ms | not measured |
| our hierarchy, neural tie-break, the two fast decodes side by side on two cores | 2.50e-05 ± 1.4e-05 · 3 pairs · 28.9 ms | 3.67e-04 ± 5.5e-05 · 44 pairs · 150 ms | 4.08e-03 ± 1.9e-04 · 479 pairs · 608 ms |
| our ensemble (combined offline) | not measured | not measured | not measured |
| TCBP-JOBS (NVIDIA, GPU) | 2.50e-05 ± 1.4e-05 · 3 pairs · 3.9 ms | 3.42e-04 ± 5.3e-05 · 41 pairs · 7.8 ms | 3.05e-03 ± 1.6e-04 · 360 pairs · 16.9 ms |

Each cell: logical error rate per cycle ± the kit's error · failed pairs of 10,000 · mean time per pair. Test shots; `report/tables/test_p<p>.md` has the intervals, the paired counts and where each row ran.
