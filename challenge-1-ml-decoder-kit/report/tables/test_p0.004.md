Test shots, p = 0.004: `data/p0.004/test_X.npz` and `test_Z.npz` (10,000 shots per basis, Stim seeds 301 and 302; sha256 `53a7cee30160…` and `b61c1325de88…`). Every measured row decoded these two files.

| Decoder | Built by | Failures X, Z | Failed pairs | Rate per cycle ± error | 95% interval | Time per pair, ms: mean / median / p99 | Against relay-BP, same pairs: both fail / only this row / only relay-BP (exact p) | Where it ran |
|---|---|---|---|---|---|---|---|---|
| BP-OSD (kit baseline) | the kit | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| relay-BP (kit baseline) | the kit | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| relay-BP with a learned router | teammates | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| PriorTilt, one run | IonQ (arXiv:2611.13373) | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| our hierarchy, lowest-weight tie-break | ours | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| our hierarchy, neural tie-break | ours | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| our ensemble (combined offline) | ours | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| TCBP-JOBS (NVIDIA, GPU) | NVIDIA; run by teammates | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
