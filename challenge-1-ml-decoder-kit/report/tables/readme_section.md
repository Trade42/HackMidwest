## Our part: a hierarchical decoder with a learned tie-breaker, and a study of atom loss

Two runs of a fast decoder (IonQ's PriorTilt) settle the shots they agree on; where they disagree, relay-BP joins and a neural
network picks the most likely answer (`challenge-1-ml-decoder-kit/hierarchy/`, `neural/`). On the test shots (one offline decode of each shot, the same candidates for every rung; p = 0.003, 0.004, 0.005 together), relay-BP fails 889 pairs, PriorTilt run once (IonQ's decoder) 672, and our decoder as frozen (two PriorTilt runs, relay-BP where they disagree, the neural tie-break) 544; a perfect tie-break would fail 429. Of the drop from relay-BP to our decoder, 63% is already there with PriorTilt run once; the hierarchy around it gives the rest. As an ablation, with the best hand-written tie-break (lowest weight among the converged candidates) in place of the network, the hierarchy fails 525: the network is no better than the hand-written rule (no significant difference). The classical tie-break (lowest weight among the converged candidates) fails 525 pairs and the network 544; only the network rule fails 87 pairs, only the classical rule 68, exact p = 0.15.
Under atom loss (our own simplified model), relay-BP with the loss flags fails 4.4% of shots where a loss-blind one
fails 57.7%, and an ORACLE told when each atom was lost fails 1.64 times fewer still: the ceiling for a loss-time tracker (`loss/`).

| Decoder | p = 0.003 | p = 0.004 | p = 0.005 |
|---|---|---|---|
| BP-OSD (kit baseline) | 5.01e-04 ± 6.5e-05, 129 ms | 4.02e-03 ± 1.8e-04, 302 ms | 2.30e-02 ± 4.6e-04, 581 ms |
| relay-BP (kit baseline) | 4.17e-05 ± 1.9e-05, 88.0 ms | 6.94e-04 ± 7.6e-05, 319 ms | 7.33e-03 ± 2.5e-04, 1,471 ms |
| relay-BP with a learned router | 5.00e-05 ± 2.0e-05, 42.5 ms | 7.28e-04 ± 7.8e-05, 315 ms | not measured |
| PriorTilt, one run | 3.33e-05 ± 1.7e-05, 22.7 ms | 5.94e-04 ± 7.0e-05, not timed | 5.10e-03 ± 2.1e-04, not timed |
| our hierarchy, lowest-weight tie-break | 2.50e-05 ± 1.4e-05, not timed | 4.76e-04 ± 6.3e-05, not timed | 4.50e-03 ± 2.0e-04, not timed |
| our hierarchy, classical tie-break (lowest weight among the converged candidates) | 2.50e-05 ± 1.4e-05, not timed | 4.43e-04 ± 6.1e-05, not timed | 3.99e-03 ± 1.8e-04, not timed |
| our hierarchy, neural tie-break | 2.50e-05 ± 1.4e-05, 45.7 ms | 3.67e-04 ± 5.5e-05, 216 ms | not measured |
| our hierarchy, neural tie-break, the two fast decodes side by side on two cores | 2.50e-05 ± 1.4e-05, 28.9 ms | 3.67e-04 ± 5.5e-05, 150 ms | 4.08e-03 ± 1.9e-04, 608 ms |
| our ensemble (combined offline) | 1.67e-05 ± 1.2e-05, not timed | 1.33e-04 ± 3.3e-05, not timed | 3.05e-03 ± 1.6e-04, not timed |
| TCBP-JOBS (NVIDIA, GPU) | 2.50e-05 ± 1.4e-05, 3.9 ms | 3.42e-04 ± 5.3e-05, 7.8 ms | 3.05e-03 ± 1.6e-04, 16.9 ms |

Test shots (10,000 pairs per p): logical error rate per cycle ± the kit's error, mean time per pair. Against TCBP-JOBS, NVIDIA's GPU decoder, on the same test pairs: at p = 0.003 TCBP-JOBS and our decoder fail as many pairs (3 against 3; only TCBP-JOBS fails 1, only ours 1, exact p = 1.00, not a significant difference); at p = 0.004 TCBP-JOBS fails fewer pairs than our decoder (41 against 44; only TCBP-JOBS fails 17, only ours 20, exact p = 0.74, not a significant difference).

![Logical error rate per cycle against p](challenge-1-ml-decoder-kit/report/figures/ler_vs_p.png)

Write-up, with every table made by a script from committed files: [report/REPORT.md](challenge-1-ml-decoder-kit/report/REPORT.md). Everything we built and why: [information.md](information.md).
