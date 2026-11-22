## Our part: a hierarchical decoder with a learned tie-breaker, and a study of atom loss

Two runs of a fast decoder (IonQ's PriorTilt) settle the shots they agree on; where they disagree, relay-BP joins and a neural
network picks the most likely answer (`challenge-1-ml-decoder-kit/hierarchy/`, `neural/`). not measured
Under atom loss (our own simplified model), relay-BP with the loss flags fails 4.4% of shots where a loss-blind one
fails 57.7%, and an ORACLE told when each atom was lost fails 1.64 times fewer still: the ceiling for a loss-time tracker (`loss/`).

| Decoder | p = 0.003 | p = 0.004 | p = 0.005 |
|---|---|---|---|
| BP-OSD (kit baseline) | not measured | not measured | not measured |
| relay-BP (kit baseline) | not measured | not measured | not measured |
| relay-BP with a learned router | not measured | not measured | not measured |
| PriorTilt, one run | not measured | not measured | not measured |
| our hierarchy, lowest-weight tie-break | not measured | not measured | not measured |
| our hierarchy, neural tie-break | not measured | not measured | not measured |
| our ensemble (combined offline) | not measured | not measured | not measured |
| TCBP-JOBS (NVIDIA, GPU) | not measured | not measured | not measured |

Test shots (10,000 pairs per p): logical error rate per cycle ± the kit's error, mean time per pair. TCBP-JOBS, NVIDIA's GPU decoder, is not measured on the test shots yet; on the exploration shots at p = 0.004 it fails fewer pairs than our hierarchy (113 against 162 of 25,000).

![Logical error rate per cycle against p](challenge-1-ml-decoder-kit/report/figures/ler_vs_p.png)

Write-up, with every table made by a script from committed files: [report/REPORT.md](challenge-1-ml-decoder-kit/report/REPORT.md). Everything we built and why: [information.md](information.md).
