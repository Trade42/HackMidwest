Test shots, p = 0.003: `data/p0.003/test_X.npz` and `test_Z.npz` (10,000 shots per basis, Stim seeds 301 and 302; sha256 `f7bc181399a0…` and `f5ac5b336f41…`). Every measured row decoded these two files.

| Decoder | Built by | Failures X, Z | Failed pairs | Rate per cycle ± error | 95% interval | Time per pair, ms: mean / median / p99 | Mean time ÷ relay-BP's | Against relay-BP, same pairs: both fail / only this row / only relay-BP (exact p) | Where it ran |
|---|---|---|---|---|---|---|---|---|---|
| BP-OSD (kit baseline) | the kit | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| relay-BP (kit baseline) | the kit | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| relay-BP with a learned router | teammates | 3, 3 | 6 of 10,000 | 5.00e-05 ± 2.0e-05 | 2.3e-05 to 1.1e-04 | 42.5 / 9.5 / 627 | relay-BP not measured yet | relay-BP not measured yet | CPU, 1 process, ICRN server (AMD EPYC-Milan, 12 CPUs by quota); while it ran: 11.8 CPUs busy on average (at most 12.0), up to 6 evaluator rows at once, other work of ours at most: tcbp member 1, the CPU quota held the notebook back for 129 s in all |
| PriorTilt, one run | IonQ (arXiv:2611.13373) | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| our hierarchy, lowest-weight tie-break | ours | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| our hierarchy, neural tie-break | ours | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| our hierarchy, neural tie-break, the two fast decodes side by side on two cores | ours | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| our ensemble (combined offline) | ours | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| TCBP-JOBS (NVIDIA, GPU) | NVIDIA; run by teammates | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
