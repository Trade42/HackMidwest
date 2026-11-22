Test shots, p = 0.005: `data/p0.005/test_X.npz` and `test_Z.npz` (10,000 shots per basis, Stim seeds 301 and 302; sha256 `1b13521cddb2…` and `b86ae320d13f…`). Every measured row decoded these two files.

| Decoder | Built by | Failures X, Z | Failed pairs | Rate per cycle ± error | 95% interval | Time per pair, ms: mean / median / p99 | Mean time ÷ relay-BP's | Against relay-BP, same pairs: both fail / only this row / only relay-BP (exact p) | Where it ran |
|---|---|---|---|---|---|---|---|---|---|
| BP-OSD (kit baseline) | the kit | 1307, 1320 | 2433 of 10,000 | 2.30e-02 ± 4.6e-04 | 2.2e-02 to 2.4e-02 | 581 / 546 / 1,093 | relay-BP not measured yet | relay-BP not measured yet | CPU, 1 process, ICRN server (AMD EPYC-Milan, 12 CPUs by quota); while it ran: 9.1 CPUs busy on average (at most 12.0), up to 9 evaluator rows at once, other work of ours at most: tcbp member 1, evaluate stim 1, the CPU quota held the notebook back for 259 s of the row's 5,820 s (4.5%); server jobs started in that time: cpu/030-candidates-explore, ctl/050-preflight-dry-run, ctl/060-loss-tracker-replication, gpu/045-test-run-dry, gpu/047-tcbp-train-p0.005 |
| relay-BP (kit baseline) | the kit | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| relay-BP with a learned router | teammates | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| PriorTilt, one run | IonQ (arXiv:2611.13373) | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| our hierarchy, lowest-weight tie-break | ours | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| our hierarchy, neural tie-break | ours | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| our hierarchy, neural tie-break, the two fast decodes side by side on two cores | ours | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| our ensemble (combined offline) | ours | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| TCBP-JOBS (NVIDIA, GPU) | NVIDIA; run by teammates | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
