## Our part: a hierarchical decoder with a learned tie-breaker, and a study of atom loss

Two runs of a fast decoder (IonQ's PriorTilt) settle the shots they agree on; where they disagree, relay-BP joins and a neural
network picks the most likely answer (`challenge-1-ml-decoder-kit/hierarchy/`, `neural/`). The hierarchy is the result (5 failed pairs for relay-BP, 3 for the hierarchy, over p = 0.003); the neural tie-break did not lower its failures on the test shots. The classical tie-break (lowest weight among the converged candidates) fails 3 pairs and the network 4; only the network rule fails 1 pair, only the classical rule 0, exact p = 1.00.
Under atom loss (our own simplified model), relay-BP with the loss flags fails 4.4% of shots where a loss-blind one
fails 57.7%, and an ORACLE told when each atom was lost fails 1.64 times fewer still: the ceiling for a loss-time tracker (`loss/`).

| Decoder | p = 0.003 | p = 0.004 | p = 0.005 |
|---|---|---|---|
| BP-OSD (kit baseline) | 5.01e-04 ± 6.5e-05, 129 ms | 4.02e-03 ± 1.8e-04, 302 ms | 2.30e-02 ± 4.6e-04, 581 ms |
| relay-BP (kit baseline) | 4.17e-05 ± 1.9e-05, 88.0 ms | 6.94e-04 ± 7.6e-05, 319 ms | 7.33e-03 ± 2.5e-04, 1,471 ms |
| relay-BP with a learned router | 5.00e-05 ± 2.0e-05, 42.5 ms | 7.28e-04 ± 7.8e-05, 315 ms | not measured |
| PriorTilt, one run | 5.00e-05 ± 2.0e-05, not timed | not measured | not measured |
| our hierarchy, lowest-weight tie-break | 2.50e-05 ± 1.4e-05, not timed | not measured | not measured |
| our hierarchy, classical tie-break (lowest weight among the converged candidates) | 2.50e-05 ± 1.4e-05, not timed | not measured | not measured |
| our hierarchy, neural tie-break | 2.50e-05 ± 1.4e-05, 45.7 ms | 3.67e-04 ± 5.5e-05, 216 ms | not measured |
| our hierarchy, neural tie-break, the two fast decodes side by side on two cores | 2.50e-05 ± 1.4e-05, 28.9 ms | 3.67e-04 ± 5.5e-05, 150 ms | 4.08e-03 ± 1.9e-04, 608 ms |
| our ensemble (combined offline) | 1.67e-05 ± 1.2e-05, not timed | not measured | not measured |
| TCBP-JOBS (NVIDIA, GPU) | 2.50e-05 ± 1.4e-05, 3.9 ms | 3.42e-04 ± 5.3e-05, 7.8 ms | 3.05e-03 ± 1.6e-04, 16.9 ms |

Test shots (10,000 pairs per p): logical error rate per cycle ± the kit's error, mean time per pair. Against TCBP-JOBS, NVIDIA's GPU decoder, on the same test pairs: at p = 0.003 TCBP-JOBS and our decoder fail as many pairs (3 against 3; only TCBP-JOBS fails 1, only ours 1, exact p = 1.00, not a significant difference); at p = 0.004 TCBP-JOBS fails fewer pairs than our decoder (41 against 44; only TCBP-JOBS fails 17, only ours 20, exact p = 0.74, not a significant difference).

![Logical error rate per cycle against p](challenge-1-ml-decoder-kit/report/figures/ler_vs_p.png)

Write-up, with every table made by a script from committed files: [report/REPORT.md](challenge-1-ml-decoder-kit/report/REPORT.md). Everything we built and why: [information.md](information.md).
