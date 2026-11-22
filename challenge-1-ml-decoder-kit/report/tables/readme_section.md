## Our part: a hierarchical decoder with a learned tie-breaker, and a study of atom loss

Two runs of a fast decoder (IonQ's PriorTilt) settle the shots they agree on; where they disagree, relay-BP joins and a neural
network picks the most likely answer (`challenge-1-ml-decoder-kit/hierarchy/`, `neural/`). On the test shots our decoder fails fewer pairs than relay-BP at p = 0.004 and 0.005, and as many within the error at p = 0.003; it is level with TCBP-JOBS at p = 0.003 and 0.004, and behind it at p = 0.005. As a tie-break, the neural network is no better than the best hand-written rule, so the gain over relay-BP is PriorTilt's and the hierarchy's.
Under atom loss (our own simplified model), relay-BP with the loss flags fails 4.4% of shots where a loss-blind one
fails 57.7%; our loss-time tracker, which estimates from the syndrome when each atom was lost, recovers 58% (38% to 78%) of what knowing the true loss time would add (`loss/`, `loss_tracker/`).

| Decoder | p = 0.003 | p = 0.004 | p = 0.005 |
|---|---|---|---|
| BP-OSD (kit baseline) | 5.01e-04 ± 6.5e-05, 129 ms | 4.02e-03 ± 1.8e-04, 302 ms | 2.30e-02 ± 4.6e-04, 581 ms |
| relay-BP (kit baseline) | 4.17e-05 ± 1.9e-05, 88.0 ms | 6.94e-04 ± 7.6e-05, 319 ms | 7.33e-03 ± 2.5e-04, 1,471 ms |
| PriorTilt, one run | 3.33e-05 ± 1.7e-05, 22.7 ms | 5.94e-04 ± 7.0e-05, offline, no time | 5.10e-03 ± 2.1e-04, offline, no time |
| our hierarchy, neural tie-break | 2.50e-05 ± 1.4e-05, 45.7 ms | 3.67e-04 ± 5.5e-05, 216 ms | 4.08e-03 ± 1.9e-04, 608 ms (2 cores) |
| TCBP-JOBS (NVIDIA, GPU) | 2.50e-05 ± 1.4e-05, 3.9 ms | 3.42e-04 ± 5.3e-05, 7.8 ms | 3.05e-03 ± 1.6e-04, 16.9 ms |

Test shots (10,000 pairs per p): logical error rate per cycle ± the kit's error, mean time per pair. Against TCBP-JOBS, NVIDIA's GPU decoder, on the same test pairs: at p = 0.003 TCBP-JOBS and our decoder fail as many pairs (3 against 3; only TCBP-JOBS fails 1, only ours 1, exact p = 1.00, not a significant difference); at p = 0.004 TCBP-JOBS fails fewer pairs than our decoder (41 against 44; only TCBP-JOBS fails 17, only ours 20, exact p = 0.74, not a significant difference).

![Logical error rate per cycle against p](challenge-1-ml-decoder-kit/report/figures/ler_vs_p.png)

Write-up, with every table made by a script from committed files: [report/REPORT.md](challenge-1-ml-decoder-kit/report/REPORT.md). Everything we built and why: [information.md](information.md).
