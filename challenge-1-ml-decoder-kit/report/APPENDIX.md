# Appendix to REPORT.md: the full tables

Written by `python -m report.make_tables` from the files under `report/data/`. Each table is also in `report/tables/`
as Markdown and as CSV.

## A. The test shots: every row, per noise level

Every method in one table first (each row's label says what kind of number it is), then each noise level in full.

<!-- BEGIN table:master_comparison -->
**Every method we ran on the test shots, in one table.** Each cell is that method's own measured row: failed pairs of 10,000 (the kit's score, with its X failures beside it) and the mean time per pair, or, for a method run on one basis only, its failures of 10,000 shots of that basis. The **Label** says what kind of number it is; rows of different labels are not equals, so this table picks no best method. Times: one process on the team's server unless the cell says otherwise; TCBP-JOBS's is a GPU decode call; the predecoder's is a clean one-thread sample on 1,000 validation shots (exploration shots). The paired tests are in the tables each row comes from.

| Method | Folder | Label | p = 0.003 | p = 0.004 | p = 0.005 |
|---|---|---|---|---|---|
| BP-OSD (kit baseline) | `decoders.py` (the kit) | pre-registered (the test run) | 60 pairs (X 26) · 5.01e-04 ± 6.5e-05 per cycle · 131 ms per pair | 472 pairs (X 230) · 4.02e-03 ± 1.8e-04 per cycle · 302 ms per pair | 2,433 pairs (X 1,307) · 2.30e-02 ± 4.6e-04 per cycle · 581 ms per pair |
| relay-BP (kit baseline) | `decoders.py` (the kit) | pre-registered (the test run) | 5 pairs (X 2) · 4.17e-05 ± 1.9e-05 per cycle · 85.3 ms per pair | 83 pairs (X 36) · 6.94e-04 ± 7.6e-05 per cycle · 319 ms per pair | 845 pairs (X 432) · 7.33e-03 ± 2.5e-04 per cycle · 1,471 ms per pair |
| relay-BP with a learned router | `evaluation/router.py` (teammates) | pre-registered (the test run) | 6 pairs (X 3) · 5.00e-05 ± 2.0e-05 per cycle · 41.0 ms per pair | 87 pairs (X 42) · 7.28e-04 ± 7.8e-05 per cycle · 315 ms per pair | 807 pairs (X 415) · 6.99e-03 ± 2.5e-04 per cycle · 1,928 ms per pair |
| PriorTilt, one run | `hierarchy/priortilt_plugin.py` (IonQ's decoder) | pre-registered (the test run) | 4 pairs (X 2) · 3.33e-05 ± 1.7e-05 per cycle · 22.7 ms per pair | 74 pairs (X 35) · 6.19e-04 ± 7.2e-05 per cycle · 87.3 ms per pair | 601 pairs (X 303) · 5.15e-03 ± 2.1e-04 per cycle · 264 ms per pair |
| our hierarchy, lowest-weight tie-break | `hierarchy/` | after-test ablation | 4 pairs (X 2) · 3.33e-05 ± 1.7e-05 per cycle · 46.9 ms per pair | 60 pairs (X 27) · 5.01e-04 ± 6.5e-05 per cycle · 215 ms per pair | 553 pairs (X 279) · 4.73e-03 ± 2.0e-04 per cycle · 817 ms per pair |
| our hierarchy, lowest-weight tie-break | `hierarchy/` | pre-registered (the test run), offline decode, no time | 3 pairs (X 2) · 2.50e-05 ± 1.4e-05 per cycle · offline | 57 pairs (X 26) · 4.76e-04 ± 6.3e-05 per cycle · offline | 527 pairs (X 269) · 4.50e-03 ± 2.0e-04 per cycle · offline |
| our hierarchy, classical tie-break (lowest weight among the converged candidates) | `hierarchy/` | pre-registered (the test run) | 4 pairs (X 2) · 3.33e-05 ± 1.7e-05 per cycle · 46.3 ms per pair | 52 pairs (X 25) · 4.34e-04 ± 6.0e-05 per cycle · 218 ms per pair |  |
| our hierarchy, classical tie-break (lowest weight among the converged candidates) | `hierarchy/` | after-test ablation |  |  | 481 pairs (X 239) · 4.10e-03 ± 1.9e-04 per cycle · 815 ms per pair |
| our hierarchy, classical tie-break (lowest weight among the converged candidates) | `hierarchy/` | pre-registered (the test run), offline decode, no time | 3 pairs (X 2) · 2.50e-05 ± 1.4e-05 per cycle · offline | 53 pairs (X 24) · 4.43e-04 ± 6.1e-05 per cycle · offline | 469 pairs (X 241) · 3.99e-03 ± 1.8e-04 per cycle · offline |
| our hierarchy, neural tie-break | `hierarchy/`, `neural/` | pre-registered (the test run) | 3 pairs (X 1) · 2.50e-05 ± 1.4e-05 per cycle · 45.7 ms per pair | 44 pairs (X 18) · 3.67e-04 ± 5.5e-05 per cycle · 216 ms per pair | 479 pairs (X 245) · 4.08e-03 ± 1.9e-04 per cycle · 825 ms per pair |
| our hierarchy, neural tie-break, the two fast decodes side by side on two cores | `hierarchy/`, `neural/` | pre-registered (the test run) | 3 pairs (X 1) · 2.50e-05 ± 1.4e-05 per cycle · 28.9 ms per pair | 44 pairs (X 18) · 3.67e-04 ± 5.5e-05 per cycle · 150 ms per pair | 479 pairs (X 245) · 4.08e-03 ± 1.9e-04 per cycle · 608 ms per pair |
| our ensemble (combined offline) | `hierarchy/ensemble.py` | pre-registered (the test run), offline decode, no time | 2 pairs (X 1) · 1.67e-05 ± 1.2e-05 per cycle · offline | 16 pairs (X 8) · 1.33e-04 ± 3.3e-05 per cycle · offline | 360 pairs (X 186) · 3.05e-03 ± 1.6e-04 per cycle · offline |
| TCBP-JOBS (NVIDIA, GPU) | `hierarchy/tcbp_member.py` (NVIDIA's decoder) | pre-registered (the test run) | 3 pairs (X 1) · 2.50e-05 ± 1.4e-05 per cycle · 3.9 ms per pair (GPU call) | 41 pairs (X 20) · 3.42e-04 ± 5.3e-05 per cycle · 7.8 ms per pair (GPU call) | 360 pairs (X 186) · 3.05e-03 ± 1.6e-04 per cycle · 16.9 ms per pair (GPU call) |
| our hierarchy, classical tie-break (lowest weight among the converged candidates), the two fast decodes side by side on two cores | `hierarchy/` | after-test ablation | 4 pairs (X 2) · 3.33e-05 ± 1.7e-05 per cycle · 29.0 ms per pair | 52 pairs (X 25) · 4.34e-04 ± 6.0e-05 per cycle · 150 ms per pair | 481 pairs (X 239) · 4.10e-03 ± 1.9e-04 per cycle · 604 ms per pair |
| TCBP-JOBS in place of relay-BP in our hierarchy (the ensemble's rule d) | `hierarchy/ensemble.py` | after-test ablation, offline from saved answers, no time | 2 pairs (X 1) · 1.67e-05 ± 1.2e-05 per cycle · offline | 39 pairs (X 19) · 3.26e-04 ± 5.2e-05 per cycle · offline | 403 pairs (X 204) · 3.42e-03 ± 1.7e-04 per cycle · offline |
| the predecoder: relay-BP with the head's priors (X memory only) | `predecoder/` | secondary: chosen after the predecoder's p = 0.004 gate failed |  |  | X: 151 shots · 212 ms per X shot (validation sample) |
| the predecoder: relay-BP with the head's priors (X memory only) | `predecoder/` | registered extension, run after the X p = 0.005 test result | X: 2 shots · 52.9 ms per X shot (validation sample) | X: 15 shots · 94.0 ms per X shot (validation sample) |  |
| the predecoder: relay-BP with the head's priors (Z memory only) | `predecoder/` | registered extension, run after the X p = 0.005 test result |  |  | Z: 132 shots |
| the predecoder in both bases, the kit's pair score | `predecoder/` | registered extension, run after the X p = 0.005 test result |  |  | 281 pairs · 2.37e-03 ± 1.4e-04 per cycle |
| the predecoder with the fallback to TCBP-JOBS (X memory only) | `predecoder/fallback.py` | post hoc: designed after the first test result |  | X: 10 shots | X: 107 shots |
| the predecoder with the fallback to TCBP-JOBS (Z memory only) | `predecoder/fallback.py` | designed after the first test result; frozen before this file was opened |  |  | Z: 100 shots |
| the predecoder with the fallback to TCBP-JOBS (X memory only) | `predecoder/fallback.py` | designed after the first test result; frozen before this file was opened | X: 1 shot |  |  |
| the predecoder with the fallback, the kit's pair score | `predecoder/fallback.py` | post hoc: designed after the first test result |  |  | 206 pairs · 1.73e-03 ± 1.2e-04 per cycle (post hoc) |
| TCBP-JOBS with the head's priors (X memory only) | `predecoder/` (formerly branch `predecoder-tcbp`) | designed after the test results were seen; formerly a separate branch | X: 0 shots; X only, no pair rate | X: 4 shots; X only, no pair rate | X: 72 shots (same-run control, the kit's priors: 186) |
| TCBP-JOBS with the head's priors (Z memory only) | `predecoder/` (formerly branch `predecoder-tcbp`) | designed after the test results were seen; formerly a separate branch |  |  | Z: 61 shots |
| TCBP-JOBS with the head's priors, the kit's pair score | `predecoder/` (formerly branch `predecoder-tcbp`) | designed after the test results were seen; formerly a separate branch |  |  | 132 pairs · 1.11e-03 ± 9.6e-05 per cycle (TCBP-JOBS's stored file: 360 pairs, 3.05e-03 ± 1.6e-04) |
| a learned selector among the ensemble's members | `selector/` | frozen on validation shots, run once on the test shots; its declared line (fewer failed pairs than TCBP-JOBS, paired) not met |  |  | 353 pairs (against TCBP-JOBS's 360: only the selector 23, only TCBP-JOBS 30, exact p = 0.41) |

Not in this table, because it decodes its own simulated shots, not the saved test files: `neutral_atom_noise/` (exploration; a noise model built from Infleqtion's published figures, not Infleqtion's noise model, decoded with the kit's relay-BP and BP-OSD; `neutral_atom_noise/RESULTS.md`).
<!-- END table:master_comparison -->

<!-- BEGIN table:test_p0.003 -->
Test shots, p = 0.003: `data/p0.003/test_X.npz` and `test_Z.npz` (10,000 shots per basis, Stim seeds 301 and 302; sha256 `f7bc181399a0…` and `f5ac5b336f41…`). Every measured row decoded these two files.

| Decoder | Built by | Failures X, Z | Failed pairs | Rate per cycle ± error | 95% interval | Time per pair, ms: mean / median / p99 | Mean time ÷ relay-BP's | Against relay-BP, same pairs: both fail / only this row / only relay-BP (exact p) | Where it ran |
|---|---|---|---|---|---|---|---|---|---|
| BP-OSD (kit baseline) | the kit | 26, 34 | 60 of 10,000 | 5.01e-04 ± 6.5e-05 | 3.9e-04 to 6.5e-04 | 131 / 18.7 / 800 | 1.54 | 5 / 55 / 0 (p = 6e-17) | CPU, 1 process, ICRN server (AMD EPYC-Milan, 12 CPUs by quota); **re-timed: a second run in a quiet window confirms the time** (`test/eval_p0.003_bposd_retime`); its answers equal the first run's on every pair, the time shown is the second run's, and the first run's mean was 129 ms, with the CPU quota throttling at most 1.2% of one process's time; while it ran: 2.1 CPUs busy on average (at most 3.2), up to 3 evaluator rows at once, the CPU quota throttled the notebook for 0 s summed over its CPUs during the row's 1,317 s, at most 0.0% of one process's time |
| relay-BP (kit baseline) | the kit | 2, 3 | 5 of 10,000 | 4.17e-05 ± 1.9e-05 | 1.8e-05 to 9.8e-05 | 85.3 / 52.5 / 557 | the reference | the reference | CPU, 1 process, ICRN server (AMD EPYC-Milan, 12 CPUs by quota); **re-timed: a second run in a quiet window confirms the time** (`test/eval_p0.003_relay-bp_retime`); its answers equal the first run's on every pair, the time shown is the second run's, and the first run's mean was 88.0 ms, with the CPU quota throttling at most 1.7% of one process's time; while it ran: 2.6 CPUs busy on average (at most 3.2), up to 3 evaluator rows at once, the CPU quota throttled the notebook for 0 s summed over its CPUs during the row's 859 s, at most 0.0% of one process's time |
| relay-BP with a learned router | teammates | 3, 3 | 6 of 10,000 | 5.00e-05 ± 2.0e-05 | 2.3e-05 to 1.1e-04 | 41.0 / 9.3 / 604 | 0.48 | 4 / 2 / 1 (p = 1.00) | CPU, 1 process, ICRN server (AMD EPYC-Milan, 12 CPUs by quota); **re-timed: a second run in a quiet window confirms the time** (`test/eval_p0.003_router_retime`); its answers equal the first run's on every pair, the time shown is the second run's, and the first run's mean was 42.5 ms, with the CPU quota throttling at most 2.5% of one process's time; while it ran: 2.9 CPUs busy on average (at most 3.2), up to 3 evaluator rows at once, the CPU quota throttled the notebook for 0 s summed over its CPUs during the row's 417 s, at most 0.0% of one process's time |
| PriorTilt, one run | IonQ (arXiv:2611.13373) | 2, 2 | 4 of 10,000 | 3.33e-05 ± 1.7e-05 | 1.3e-05 to 8.6e-05 | 22.7 / 7.2 / 188 | 0.27 | 3 / 1 / 2 (p = 1.00) | CPU, 1 process, ICRN server (AMD EPYC-Milan, 12 CPUs by quota); while it ran: 9.9 CPUs busy on average (at most 10.2), up to 6 evaluator rows at once, other work of ours at most: joint 1, the CPU quota throttled the notebook for 0 s summed over its CPUs during the row's 238 s, at most 0.0% of one process's time |
| our hierarchy, lowest-weight tie-break (run after the test results were seen, at the team's request; an ablation, not our decoder) | ours | 2, 2 | 4 of 10,000 | 3.33e-05 ± 1.7e-05 | 1.3e-05 to 8.6e-05 | 46.9 / 14.6 / 404 | 0.55 | 3 / 1 / 2 (p = 1.00) | CPU, 1 process, ICRN server (AMD EPYC-Milan, 12 CPUs by quota); while it ran: 7.9 CPUs busy on average (at most 8.7), up to 7 evaluator rows at once, the CPU quota throttled the notebook for 0 s summed over its CPUs during the row's 483 s, at most 0.0% of one process's time |
| our hierarchy, classical tie-break (lowest weight among the converged candidates) | ours | 2, 2 | 4 of 10,000 | 3.33e-05 ± 1.7e-05 | 1.3e-05 to 8.6e-05 | 46.3 / 14.4 / 399 | 0.54 | 3 / 1 / 2 (p = 1.00) | CPU, 1 process, ICRN server (AMD EPYC-Milan, 12 CPUs by quota); while it ran: 6.9 CPUs busy on average (at most 9.7), up to 6 evaluator rows at once, other work of ours at most: joint 1, the CPU quota throttled the notebook for 0 s summed over its CPUs during the row's 477 s, at most 0.0% of one process's time |
| our hierarchy, neural tie-break | ours | 1, 2 | 3 of 10,000 | 2.50e-05 ± 1.4e-05 | 8.5e-06 to 7.4e-05 | 45.7 / 14.2 / 395 | 0.54 | 3 / 0 / 2 (p = 0.50) | CPU, 1 process, ICRN server (AMD EPYC-Milan, 12 CPUs by quota); while it ran: 8.5 CPUs busy on average (at most 9.5), up to 8 evaluator rows at once, other work of ours at most: tcbp member 1, the CPU quota throttled the notebook for 1 s summed over its CPUs during the row's 477 s, at most 0.0% of one process's time |
| our hierarchy, neural tie-break, the two fast decodes side by side on two cores | ours | 1, 2 | 3 of 10,000 | 2.50e-05 ± 1.4e-05 | 8.5e-06 to 7.4e-05 | 28.9 / 7.8 / 245 | 0.34 | 3 / 0 / 2 (p = 0.50) | CPU, 1 process on 2 cores, ICRN server (AMD EPYC-Milan, 12 CPUs by quota); while it ran: 9.4 CPUs busy on average (at most 9.7), up to 8 evaluator rows at once, other work of ours at most: candidates 1, tcbp member 1, evaluate stim 1, the CPU quota throttled the notebook for 1 s summed over its CPUs during the row's 315 s, at most 0.0% of one process's time; server jobs started in that time: ctl/100-test-logs-to-logs |
| our ensemble (combined offline) (rule f, frozen on the training shots: where PriorTilt twice and TCBP-JOBS agree, that answer; elsewhere joint decoding where it converged, else TCBP-JOBS; joint decoding ran on 5 of the 5 disputed shots) | ours | 1, 1 | 2 of 10,000 | 1.67e-05 ± 1.2e-05 | 4.6e-06 to 6.1e-05 | 50.2 / not timed per pair | not timed in one process | 2 / 0 / 3 (p = 0.25) | combined offline from per-shot files (CPU and GPU decoders, ICRN server); its time is the sum of the decode calls the row makes, each timed alone |
| TCBP-JOBS (NVIDIA, GPU) | NVIDIA; run by teammates | 1, 2 | 3 of 10,000 | 2.50e-05 ± 1.4e-05 | 8.5e-06 to 7.4e-05 | 3.9 / 3.8 / 15.0 | not comparable: another device and timer | 2 / 1 / 3 (p = 0.62) | GPU, 1 process, ICRN server, A100 slice (20 GB); while it ran: 9.6 CPUs busy on average (at most 9.7), up to 8 evaluator rows at once, other work of ours at most: tcbp member 1, the CPU quota throttled the notebook for 1 s summed over its CPUs during the row's 166 s, at most 0.1% of one process's time; no other process on the GPU at its start; its time is the decode call alone (hierarchy.tcbp_member), not the evaluator's whole decode |
| our hierarchy, classical tie-break (lowest weight among the converged candidates), the two fast decodes side by side on two cores (run after the test results were seen, at the team's request; an ablation, not our decoder) | ours | 2, 2 | 4 of 10,000 | 3.33e-05 ± 1.7e-05 | 1.3e-05 to 8.6e-05 | 29.0 / 7.8 / 245 | 0.34 | 3 / 1 / 2 (p = 1.00) | CPU, 1 process on 2 cores, ICRN server (AMD EPYC-Milan, 12 CPUs by quota); while it ran: 8.1 CPUs busy on average (at most 8.7), up to 7 evaluator rows at once, the CPU quota throttled the notebook for 0 s summed over its CPUs during the row's 309 s, at most 0.0% of one process's time |
<!-- END table:test_p0.003 -->

<!-- BEGIN table:test_p0.004 -->
Test shots, p = 0.004: `data/p0.004/test_X.npz` and `test_Z.npz` (10,000 shots per basis, Stim seeds 301 and 302; sha256 `53a7cee30160…` and `b61c1325de88…`). Every measured row decoded these two files.

| Decoder | Built by | Failures X, Z | Failed pairs | Rate per cycle ± error | 95% interval | Time per pair, ms: mean / median / p99 | Mean time ÷ relay-BP's | Against relay-BP, same pairs: both fail / only this row / only relay-BP (exact p) | Where it ran |
|---|---|---|---|---|---|---|---|---|---|
| BP-OSD (kit baseline) | the kit | 230, 243 | 472 of 10,000 | 4.02e-03 ± 1.8e-04 | 3.7e-03 to 4.4e-03 | 302 / 350 / 863 | 0.94 | 60 / 412 / 23 (p = 2e-93) | CPU, 1 process, ICRN server (AMD EPYC-Milan, 12 CPUs by quota); while it ran: 10.3 CPUs busy on average (at most 12.0), up to 9 evaluator rows at once, other work of ours at most: tcbp member 1, evaluate stim 1, the CPU quota throttled the notebook for 69 s summed over its CPUs during the row's 3,023 s, at most 0.2% of one process's time; server jobs started in that time: cpu/030-candidates-explore, ctl/050-preflight-dry-run, ctl/060-loss-tracker-replication, gpu/045-test-run-dry |
| relay-BP (kit baseline) | the kit | 36, 47 | 83 of 10,000 | 6.94e-04 ± 7.6e-05 | 5.6e-04 to 8.6e-04 | 319 / 106 / 5,514 | the reference | the reference | CPU, 1 process, ICRN server (AMD EPYC-Milan, 12 CPUs by quota); while it ran: 10.3 CPUs busy on average (at most 12.0), up to 9 evaluator rows at once, other work of ours at most: tcbp member 1, evaluate stim 1, the CPU quota throttled the notebook for 75 s summed over its CPUs during the row's 3,200 s, at most 0.2% of one process's time; server jobs started in that time: cpu/030-candidates-explore, ctl/050-preflight-dry-run, ctl/060-loss-tracker-replication, gpu/045-test-run-dry, gpu/047-tcbp-train-p0.005 |
| relay-BP with a learned router | teammates | 42, 45 | 87 of 10,000 | 7.28e-04 ± 7.8e-05 | 5.9e-04 to 9.0e-04 | 315 / 15.0 / 6,865 | 0.99 | 51 / 36 / 32 (p = 0.72) | CPU, 1 process, ICRN server (AMD EPYC-Milan, 12 CPUs by quota); while it ran: 10.3 CPUs busy on average (at most 12.0), up to 9 evaluator rows at once, other work of ours at most: tcbp member 1, evaluate stim 1, the CPU quota throttled the notebook for 75 s summed over its CPUs during the row's 3,162 s, at most 0.2% of one process's time; server jobs started in that time: cpu/030-candidates-explore, ctl/050-preflight-dry-run, ctl/060-loss-tracker-replication, gpu/045-test-run-dry, gpu/047-tcbp-train-p0.005 |
| PriorTilt, one run | IonQ (arXiv:2611.13373) | 35, 39 | 74 of 10,000 | 6.19e-04 ± 7.2e-05 | 4.9e-04 to 7.8e-04 | 87.3 / 34.2 / 884 | 0.27 | 41 / 33 / 42 (p = 0.36) | CPU, 1 process, ICRN server (AMD EPYC-Milan, 12 CPUs by quota); while it ran: 7.5 CPUs busy on average (at most 10.2), up to 6 evaluator rows at once, other work of ours at most: joint 1, the CPU quota throttled the notebook for 0 s summed over its CPUs during the row's 884 s, at most 0.0% of one process's time |
| our hierarchy, lowest-weight tie-break (run after the test results were seen, at the team's request; an ablation, not our decoder) | ours | 27, 33 | 60 of 10,000 | 5.01e-04 ± 6.5e-05 | 3.9e-04 to 6.5e-04 | 215 / 81.9 / 2,054 | 0.67 | 40 / 20 / 43 (p = 0.005) | CPU, 1 process, ICRN server (AMD EPYC-Milan, 12 CPUs by quota); while it ran: 6.0 CPUs busy on average (at most 8.7), up to 7 evaluator rows at once, the CPU quota throttled the notebook for 0 s summed over its CPUs during the row's 2,162 s, at most 0.0% of one process's time |
| our hierarchy, classical tie-break (lowest weight among the converged candidates) | ours | 25, 27 | 52 of 10,000 | 4.34e-04 ± 6.0e-05 | 3.3e-04 to 5.7e-04 | 218 / 83.4 / 2,090 | 0.68 | 34 / 18 / 49 (p = 2e-04) | CPU, 1 process, ICRN server (AMD EPYC-Milan, 12 CPUs by quota); while it ran: 4.0 CPUs busy on average (at most 10.2), up to 6 evaluator rows at once, other work of ours at most: joint 1, the CPU quota throttled the notebook for 0 s summed over its CPUs during the row's 2,197 s, at most 0.0% of one process's time |
| our hierarchy, neural tie-break | ours | 18, 26 | 44 of 10,000 | 3.67e-04 ± 5.5e-05 | 2.7e-04 to 4.9e-04 | 216 / 83.1 / 2,064 | 0.68 | 29 / 15 / 54 (p = 3e-06) | CPU, 1 process, ICRN server (AMD EPYC-Milan, 12 CPUs by quota); while it ran: 9.1 CPUs busy on average (at most 10.1), up to 8 evaluator rows at once, other work of ours at most: candidates 2, tcbp member 1, evaluate stim 1, the CPU quota throttled the notebook for 3 s summed over its CPUs during the row's 2,183 s, at most 0.0% of one process's time; server jobs started in that time: ctl/100-test-logs-to-logs |
| our hierarchy, neural tie-break, the two fast decodes side by side on two cores | ours | 18, 26 | 44 of 10,000 | 3.67e-04 ± 5.5e-05 | 2.7e-04 to 4.9e-04 | 150 / 55.5 / 1,292 | 0.47 | 29 / 15 / 54 (p = 3e-06) | CPU, 1 process on 2 cores, ICRN server (AMD EPYC-Milan, 12 CPUs by quota); while it ran: 9.6 CPUs busy on average (at most 10.1), up to 8 evaluator rows at once, other work of ours at most: candidates 2, tcbp member 1, evaluate stim 1, the CPU quota throttled the notebook for 3 s summed over its CPUs during the row's 1,527 s, at most 0.0% of one process's time; server jobs started in that time: ctl/100-test-logs-to-logs |
| our ensemble (combined offline) (rule f, frozen on the training shots: where PriorTilt twice and TCBP-JOBS agree, that answer; elsewhere joint decoding where it converged, else TCBP-JOBS; joint decoding ran on 93 of the 93 disputed shots) | ours | 8, 8 | 16 of 10,000 | 1.33e-04 ± 3.3e-05 | 8.2e-05 to 2.2e-04 | 413 / not timed per pair | not timed in one process | 12 / 4 / 71 (p = 7e-17) | combined offline from per-shot files (CPU and GPU decoders, ICRN server); its time is the sum of the decode calls the row makes, each timed alone |
| TCBP-JOBS (NVIDIA, GPU) | NVIDIA; run by teammates | 20, 21 | 41 of 10,000 | 3.42e-04 ± 5.3e-05 | 2.5e-04 to 4.6e-04 | 7.8 / 5.8 / 35.3 | not comparable: another device and timer | 27 / 14 / 56 (p = 4e-07) | GPU, 1 process, ICRN server, A100 slice (20 GB); while it ran: 9.0 CPUs busy on average (at most 9.1), up to 7 evaluator rows at once, other work of ours at most: tcbp member 1, the CPU quota throttled the notebook for 0 s summed over its CPUs during the row's 206 s, at most 0.0% of one process's time; no other process on the GPU at its start; its time is the decode call alone (hierarchy.tcbp_member), not the evaluator's whole decode |
| our hierarchy, classical tie-break (lowest weight among the converged candidates), the two fast decodes side by side on two cores (run after the test results were seen, at the team's request; an ablation, not our decoder) | ours | 25, 27 | 52 of 10,000 | 4.34e-04 ± 6.0e-05 | 3.3e-04 to 5.7e-04 | 150 / 55.6 / 1,302 | 0.47 | 34 / 18 / 49 (p = 2e-04) | CPU, 1 process on 2 cores, ICRN server (AMD EPYC-Milan, 12 CPUs by quota); while it ran: 6.6 CPUs busy on average (at most 8.7), up to 7 evaluator rows at once, the CPU quota throttled the notebook for 0 s summed over its CPUs during the row's 1,518 s, at most 0.0% of one process's time |
<!-- END table:test_p0.004 -->

<!-- BEGIN table:test_p0.005 -->
Test shots, p = 0.005: `data/p0.005/test_X.npz` and `test_Z.npz` (10,000 shots per basis, Stim seeds 301 and 302; sha256 `1b13521cddb2…` and `b86ae320d13f…`). Every measured row decoded these two files.

| Decoder | Built by | Failures X, Z | Failed pairs | Rate per cycle ± error | 95% interval | Time per pair, ms: mean / median / p99 | Mean time ÷ relay-BP's | Against relay-BP, same pairs: both fail / only this row / only relay-BP (exact p) | Where it ran |
|---|---|---|---|---|---|---|---|---|---|
| BP-OSD (kit baseline) | the kit | 1307, 1320 | 2433 of 10,000 | 2.30e-02 ± 4.6e-04 | 2.2e-02 to 2.4e-02 | 581 / 546 / 1,093 | 0.40 | 714 / 1719 / 131 (p = 0e+00) | CPU, 1 process, ICRN server (AMD EPYC-Milan, 12 CPUs by quota); while it ran: 9.1 CPUs busy on average (at most 12.0), up to 9 evaluator rows at once, other work of ours at most: tcbp member 1, evaluate stim 1, the CPU quota throttled the notebook for 259 s summed over its CPUs during the row's 5,820 s, at most 0.4% of one process's time; server jobs started in that time: cpu/030-candidates-explore, ctl/050-preflight-dry-run, ctl/060-loss-tracker-replication, gpu/045-test-run-dry, gpu/047-tcbp-train-p0.005 |
| relay-BP (kit baseline) | the kit | 432, 427 | 845 of 10,000 | 7.33e-03 ± 2.5e-04 | 6.9e-03 to 7.8e-03 | 1,471 / 424 / 8,270 | the reference | the reference | CPU, 1 process, ICRN server (AMD EPYC-Milan, 12 CPUs by quota); while it ran: 7.7 CPUs busy on average (at most 12.0), up to 9 evaluator rows at once, other work of ours at most: candidates 2, tcbp member 1, evaluate stim 1, the CPU quota throttled the notebook for 267 s summed over its CPUs during the row's 14,715 s, at most 0.2% of one process's time; server jobs started in that time: cpu/030-candidates-explore, ctl/050-preflight-dry-run, ctl/060-loss-tracker-replication, ctl/070-predecoder-fresh-train, ctl/080-predecoder-fresh-validation, ctl/090-z-final-checkpoint-to-logs, ctl/100-test-logs-to-logs, gpu/045-test-run-dry, gpu/047-tcbp-train-p0.005, gpu/050-test-run |
| relay-BP with a learned router | teammates | 415, 404 | 807 of 10,000 | 6.99e-03 ± 2.5e-04 | 6.5e-03 to 7.5e-03 | 1,928 / 303 / 12,708 | 1.31 | 604 / 203 / 241 (p = 0.079) | CPU, 1 process, ICRN server (AMD EPYC-Milan, 12 CPUs by quota); while it ran: 7.6 CPUs busy on average (at most 12.0), up to 9 evaluator rows at once, other work of ours at most: joint 1, candidates 2, tcbp member 1, evaluate stim 1, the CPU quota throttled the notebook for 267 s summed over its CPUs during the row's 19,286 s, at most 0.1% of one process's time; server jobs started in that time: cpu/030-candidates-explore, cpu/040-retime-p0.003-baselines, ctl/050-preflight-dry-run, ctl/060-loss-tracker-replication, ctl/070-predecoder-fresh-train, ctl/080-predecoder-fresh-validation, ctl/090-z-final-checkpoint-to-logs, ctl/100-test-logs-to-logs, ctl/110-selector-test, gpu/045-test-run-dry, gpu/047-tcbp-train-p0.005, gpu/050-test-run |
| PriorTilt, one run | IonQ (arXiv:2611.13373) | 303, 307 | 601 of 10,000 | 5.15e-03 ± 2.1e-04 | 4.8e-03 to 5.6e-03 | 264 / 120 / 2,167 | 0.18 | 431 / 170 / 414 (p = 2e-24) | CPU, 1 process, ICRN server (AMD EPYC-Milan, 12 CPUs by quota); while it ran: 6.4 CPUs busy on average (at most 10.2), up to 6 evaluator rows at once, other work of ours at most: joint 1, candidates 1, the CPU quota throttled the notebook for 0 s summed over its CPUs during the row's 2,647 s, at most 0.0% of one process's time; server jobs started in that time: cpu/040-retime-p0.003-baselines, ctl/110-selector-test |
| our hierarchy, lowest-weight tie-break (run after the test results were seen, at the team's request; an ablation, not our decoder) | ours | 279, 282 | 553 of 10,000 | 4.73e-03 ± 2.0e-04 | 4.4e-03 to 5.1e-03 | 817 / 244 / 9,776 | 0.56 | 436 / 117 / 409 (p = 6e-39) | CPU, 1 process, ICRN server (AMD EPYC-Milan, 12 CPUs by quota); while it ran: 4.6 CPUs busy on average (at most 8.7), up to 7 evaluator rows at once, the CPU quota throttled the notebook for 0 s summed over its CPUs during the row's 8,181 s, at most 0.0% of one process's time; server jobs started in that time: cpu/050-predecoder-latency, cpu/060-predecoder-latency-p0.003-p0.004, cpu/070-predecoder-tcbp-4-gpu-network, cpu/080-predecoder-tcbp-6-test-X-p0.004-p0.003, cpu/090-tcbp-noise-infleqtion, ctl/130-predecoder-tcbp-3-validation, ctl/140-predecoder-tcbp-7-Z-p0.005, gpu/060-predecoder-tcbp-1, gpu/070-predecoder-tcbp-2-fit, gpu/075-predecoder-tcbp-2b-fit, gpu/080-predecoder-tcbp-5-test-X-p0.005 |
| our hierarchy, classical tie-break (lowest weight among the converged candidates) (run after the test results were seen, at the team's request; an ablation, not our decoder) | ours | 239, 248 | 481 of 10,000 | 4.10e-03 ± 1.9e-04 | 3.7e-03 to 4.5e-03 | 815 / 244 / 9,755 | 0.55 | 382 / 99 / 463 (p = 3e-57) | CPU, 1 process, ICRN server (AMD EPYC-Milan, 12 CPUs by quota); while it ran: 4.6 CPUs busy on average (at most 8.7), up to 7 evaluator rows at once, the CPU quota throttled the notebook for 0 s summed over its CPUs during the row's 8,159 s, at most 0.0% of one process's time; server jobs started in that time: cpu/050-predecoder-latency, cpu/060-predecoder-latency-p0.003-p0.004, cpu/070-predecoder-tcbp-4-gpu-network, cpu/080-predecoder-tcbp-6-test-X-p0.004-p0.003, cpu/090-tcbp-noise-infleqtion, ctl/130-predecoder-tcbp-3-validation, ctl/140-predecoder-tcbp-7-Z-p0.005, gpu/060-predecoder-tcbp-1, gpu/070-predecoder-tcbp-2-fit, gpu/075-predecoder-tcbp-2b-fit, gpu/080-predecoder-tcbp-5-test-X-p0.005 |
| our hierarchy, neural tie-break | ours | 245, 238 | 479 of 10,000 | 4.08e-03 ± 1.9e-04 | 3.7e-03 to 4.5e-03 | 825 / 246 / 9,885 | 0.56 | 386 / 93 / 459 (p = 4e-59) | CPU, 1 process, ICRN server (AMD EPYC-Milan, 12 CPUs by quota); while it ran: 8.4 CPUs busy on average (at most 10.7), up to 8 evaluator rows at once, other work of ours at most: joint 1, candidates 2, tcbp member 1, evaluate stim 1, the CPU quota throttled the notebook for 7 s summed over its CPUs during the row's 8,266 s, at most 0.0% of one process's time; server jobs started in that time: cpu/040-retime-p0.003-baselines, ctl/100-test-logs-to-logs, ctl/110-selector-test |
| our hierarchy, neural tie-break, the two fast decodes side by side on two cores | ours | 245, 238 | 479 of 10,000 | 4.08e-03 ± 1.9e-04 | 3.7e-03 to 4.5e-03 | 608 / 138 / 7,794 | 0.41 | 386 / 93 / 459 (p = 4e-59) | CPU, 1 process on 2 cores, ICRN server (AMD EPYC-Milan, 12 CPUs by quota); while it ran: 8.7 CPUs busy on average (at most 10.7), up to 8 evaluator rows at once, other work of ours at most: candidates 2, tcbp member 1, evaluate stim 1, the CPU quota throttled the notebook for 7 s summed over its CPUs during the row's 6,100 s, at most 0.0% of one process's time; server jobs started in that time: ctl/100-test-logs-to-logs |
| our ensemble (combined offline) (rule f, frozen on the training shots: where PriorTilt twice and TCBP-JOBS agree, that answer; elsewhere joint decoding where it converged, else TCBP-JOBS; joint decoding ran on 0 of the 735 disputed shots; so it fails exactly the shots TCBP-JOBS fails) | ours | 186, 177 | 360 of 10,000 | 3.05e-03 ± 1.6e-04 | 2.8e-03 to 3.4e-03 | 551 / not timed per pair | not timed in one process | 306 / 54 / 539 (p = 1e-101) | combined offline from per-shot files (CPU and GPU decoders, ICRN server); its time is the sum of the decode calls the row makes, each timed alone |
| TCBP-JOBS (NVIDIA, GPU) | NVIDIA; run by teammates | 186, 177 | 360 of 10,000 | 3.05e-03 ± 1.6e-04 | 2.8e-03 to 3.4e-03 | 16.9 / 13.3 / 56.4 | not comparable: another device and timer | 306 / 54 / 539 (p = 1e-101) | GPU, 1 process, ICRN server, A100 slice (20 GB); while it ran: 6.0 CPUs busy on average (at most 9.1), up to 7 evaluator rows at once, other work of ours at most: tcbp member 1, the CPU quota throttled the notebook for 0 s summed over its CPUs during the row's 300 s, at most 0.0% of one process's time; no other process on the GPU at its start; its time is the decode call alone (hierarchy.tcbp_member), not the evaluator's whole decode |
| our hierarchy, classical tie-break (lowest weight among the converged candidates), the two fast decodes side by side on two cores (run after the test results were seen, at the team's request; an ablation, not our decoder) | ours | 239, 248 | 481 of 10,000 | 4.10e-03 ± 1.9e-04 | 3.7e-03 to 4.5e-03 | 604 / 138 / 7,747 | 0.41 | 382 / 99 / 463 (p = 3e-57) | CPU, 1 process on 2 cores, ICRN server (AMD EPYC-Milan, 12 CPUs by quota); while it ran: 5.0 CPUs busy on average (at most 8.7), up to 7 evaluator rows at once, the CPU quota throttled the notebook for 0 s summed over its CPUs during the row's 6,061 s, at most 0.0% of one process's time; server jobs started in that time: cpu/050-predecoder-latency, cpu/060-predecoder-latency-p0.003-p0.004, cpu/070-predecoder-tcbp-4-gpu-network, ctl/130-predecoder-tcbp-3-validation, gpu/060-predecoder-tcbp-1, gpu/070-predecoder-tcbp-2-fit, gpu/075-predecoder-tcbp-2b-fit |
<!-- END table:test_p0.005 -->

<!-- BEGIN table:test_time -->
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
| 0.005 | our hierarchy, lowest-weight tie-break (run after the test results were seen, at the team's request; an ablation, not our decoder) | 553 of 10,000 | 817 | 0.56 | 244 | 9,776 | CPU, 1 core |
| 0.005 | our hierarchy, classical tie-break (lowest weight among the converged candidates) (run after the test results were seen, at the team's request; an ablation, not our decoder) | 481 of 10,000 | 815 | 0.55 | 244 | 9,755 | CPU, 1 core |
| 0.005 | our hierarchy, neural tie-break | 479 of 10,000 | 825 | 0.56 | 246 | 9,885 | CPU, 1 core |
| 0.005 | our hierarchy, neural tie-break, the two fast decodes side by side on two cores | 479 of 10,000 | 608 | 0.41 | 138 | 7,794 | CPU, 2 cores |
| 0.005 | our ensemble (combined offline) (rule f, frozen on the training shots: where PriorTilt twice and TCBP-JOBS agree, that answer; elsewhere joint decoding where it converged, else TCBP-JOBS; joint decoding ran on 0 of the 735 disputed shots; so it fails exactly the shots TCBP-JOBS fails) | 360 of 10,000 | not timed in one process |  |  |  |  |
| 0.005 | TCBP-JOBS (NVIDIA, GPU) | 360 of 10,000 | 16.9 | not comparable: another device and timer | 13.3 | 56.4 | GPU |
| 0.005 | our hierarchy, classical tie-break (lowest weight among the converged candidates), the two fast decodes side by side on two cores (run after the test results were seen, at the team's request; an ablation, not our decoder) | 481 of 10,000 | 604 | 0.41 | 138 | 7,747 | CPU, 2 cores |
<!-- END table:test_time -->

The learned predecoder's time per X shot, measured cleanly on exploration shots: its own table, since test and
exploration shots never share one.

<!-- BEGIN table:predecoder_latency -->
**Exploration shots, not test shots: the predecoder's time per X shot.** The team's server, one process and one thread (server jobs cpu/050 and cpu/060), the first 1,000 X shots of `data/p<p>/validation_X.npz` (so no test file was decoded twice); the head and the final X network pinned by sha256 as in the test run. Per X shot, not per pair; ms: mean / median / 99th percentile.

| p | The network and the head, on the CPU | Relay-BP with the head's priors, built and run per shot | Relay-BP converged (mean ms) | The two together, mean ms per X shot |
|---|---|---|---|---|
| 0.003 | 26.4 / 26.3 / 27.5 | 26.5 / 19.0 / 56.7 | 999 of 1,000 (21.0 ms); not converged 1 (5,515 ms) | 52.9 |
| 0.004 | 26.7 / 26.6 / 27.3 | 67.3 / 23.1 / 631 | 997 of 1,000 (51.0 ms); not converged 3 (5,512 ms) | 94.0 |
| 0.005 | 26.8 / 26.7 / 28.0 | 185 / 30.1 / 5,467 | 992 of 1,000 (142 ms); not converged 8 (5,544 ms) | 212 |

For scale only, other shots and so no paired comparison, on the test X shots: p = 0.003: relay-BP's timed row 44.3 ms per X shot (its 85.3 ms per pair is an X shot plus a Z shot), TCBP-JOBS 2.0 ms per X shot; p = 0.004: relay-BP's timed row 156 ms per X shot (its 319 ms per pair is an X shot plus a Z shot), TCBP-JOBS 3.9 ms per X shot; p = 0.005: relay-BP's timed row 760 ms per X shot (its 1,471 ms per pair is an X shot plus a Z shot), TCBP-JOBS 8.7 ms per X shot (TCBP-JOBS: a GPU decode call alone, not comparable with a CPU time).

Source: `report/data/exploration/predecoder/latency_validation_X_p<p>/` (`summary.json`, `per_shot.npz`, `latency.txt`) and the jobs' printouts `job_cpu_050-predecoder-latency.out`, `job_cpu_060-predecoder-latency-p0.003-p0.004.out`.
<!-- END table:predecoder_latency -->

<!-- BEGIN table:time_split -->
Where our decoder's time goes on the test shots, per pair (an X shot and a Z shot), averaged over all 10,000 pairs. The two PriorTilt runs decode every shot; relay-BP and the tie-break run only on the shots where the two runs disagree. From the plugin's own per-shot timers; the last column is the evaluator's timer, which wraps everything.

| p | Row | Shots where the two runs disagree | Two PriorTilt runs, ms | relay-BP, ms | the network, ms | Whole decode, ms (the evaluator) |
|---|---|---|---|---|---|---|
| 0.003 | our hierarchy, neural tie-break | 2 of 20,000 | 45.0 | 0.5 | 0.0 | 45.7 |
| 0.004 | our hierarchy, neural tie-break | 90 of 20,000 | 178 | 37.6 | 0.2 | 216 |
| 0.005 | our hierarchy, neural tie-break | 609 of 20,000 | 540 | 282 | 1.7 | 825 |
<!-- END table:time_split -->

![Accuracy against time](figures/ler_vs_time.png)

## B. Paired comparisons on the test shots

<!-- BEGIN table:test_comparisons -->
Paired comparisons on the test shots: two decoders on the same 10,000 pairs, and how many pairs only one of the two fails. The exact p is the two-sided binomial test on those pairs (under "no difference" each is a fair coin). This is where each gain is attributed: PriorTilt is IonQ's, the hierarchy and its tie-break are ours. A pair marked "mixed" compares a timed row with an offline combination; relay-BP's random state makes those two decodes differ (`checks.md`), so read it with care.

| p | Decodes | First decoder | Second decoder | Failed pairs: first against second | Both fail | Only the first fails / only the second fails | Exact p |
|---|---|---|---|---|---|---|---|
| 0.003 | both timed | relay-BP with a learned router | relay-BP (kit baseline) | 6 against 5 | 4 | 2 / 1 | 1.00 |
| 0.003 | both timed | PriorTilt, one run | relay-BP (kit baseline) | 4 against 5 | 3 | 1 / 2 | 1.00 |
| 0.003 | both timed | our hierarchy, lowest-weight tie-break (run after the test results were seen, at the team's request; an ablation, not our decoder) | PriorTilt, one run | 4 against 4 | 3 | 1 / 1 | 1.00 |
| 0.003 | both timed | our hierarchy, lowest-weight tie-break (run after the test results were seen, at the team's request; an ablation, not our decoder) | relay-BP (kit baseline) | 4 against 5 | 3 | 1 / 2 | 1.00 |
| 0.003 | both timed | our hierarchy, neural tie-break | PriorTilt, one run | 3 against 4 | 3 | 0 / 1 | 1.00 |
| 0.003 | both timed | our hierarchy, neural tie-break | our hierarchy, classical tie-break (lowest weight among the converged candidates) | 3 against 4 | 3 | 0 / 1 | 1.00 |
| 0.003 | both timed | our hierarchy, classical tie-break (lowest weight among the converged candidates) | PriorTilt, one run | 4 against 4 | 3 | 1 / 1 | 1.00 |
| 0.003 | both timed | our hierarchy, classical tie-break (lowest weight among the converged candidates) | relay-BP (kit baseline) | 4 against 5 | 3 | 1 / 2 | 1.00 |
| 0.003 | both timed | our hierarchy, neural tie-break | our hierarchy, lowest-weight tie-break (run after the test results were seen, at the team's request; an ablation, not our decoder) | 3 against 4 | 3 | 0 / 1 | 1.00 |
| 0.003 | both timed | our hierarchy, neural tie-break | relay-BP (kit baseline) | 3 against 5 | 3 | 0 / 2 | 0.50 |
| 0.003 | both timed | our hierarchy, neural tie-break | TCBP-JOBS (NVIDIA, GPU) | 3 against 3 | 2 | 1 / 1 | 1.00 |
| 0.003 | **mixed**: a timed row against an offline combination, with other random draws | our ensemble (combined offline) (rule f, frozen on the training shots: where PriorTilt twice and TCBP-JOBS agree, that answer; elsewhere joint decoding where it converged, else TCBP-JOBS; joint decoding ran on 5 of the 5 disputed shots) | our hierarchy, neural tie-break | 2 against 3 | 2 | 0 / 1 | 1.00 |
| 0.003 | **mixed**: a timed row against an offline combination, with other random draws | our ensemble (combined offline) (rule f, frozen on the training shots: where PriorTilt twice and TCBP-JOBS agree, that answer; elsewhere joint decoding where it converged, else TCBP-JOBS; joint decoding ran on 5 of the 5 disputed shots) | TCBP-JOBS (NVIDIA, GPU) | 2 against 3 | 2 | 0 / 1 | 1.00 |
| 0.003 | both timed | TCBP-JOBS (NVIDIA, GPU) | relay-BP (kit baseline) | 3 against 5 | 2 | 1 / 3 | 0.62 |
| 0.004 | both timed | relay-BP with a learned router | relay-BP (kit baseline) | 87 against 83 | 51 | 36 / 32 | 0.72 |
| 0.004 | both timed | PriorTilt, one run | relay-BP (kit baseline) | 74 against 83 | 41 | 33 / 42 | 0.36 |
| 0.004 | both timed | our hierarchy, lowest-weight tie-break (run after the test results were seen, at the team's request; an ablation, not our decoder) | PriorTilt, one run | 60 against 74 | 48 | 12 / 26 | 0.034 |
| 0.004 | both timed | our hierarchy, lowest-weight tie-break (run after the test results were seen, at the team's request; an ablation, not our decoder) | relay-BP (kit baseline) | 60 against 83 | 40 | 20 / 43 | 0.005 |
| 0.004 | both timed | our hierarchy, neural tie-break | PriorTilt, one run | 44 against 74 | 39 | 5 / 35 | 1e-06 |
| 0.004 | both timed | our hierarchy, neural tie-break | our hierarchy, classical tie-break (lowest weight among the converged candidates) | 44 against 52 | 36 | 8 / 16 | 0.15 |
| 0.004 | both timed | our hierarchy, classical tie-break (lowest weight among the converged candidates) | PriorTilt, one run | 52 against 74 | 44 | 8 / 30 | 5e-04 |
| 0.004 | both timed | our hierarchy, classical tie-break (lowest weight among the converged candidates) | relay-BP (kit baseline) | 52 against 83 | 34 | 18 / 49 | 2e-04 |
| 0.004 | both timed | our hierarchy, neural tie-break | our hierarchy, lowest-weight tie-break (run after the test results were seen, at the team's request; an ablation, not our decoder) | 44 against 60 | 39 | 5 / 21 | 0.002 |
| 0.004 | both timed | our hierarchy, neural tie-break | relay-BP (kit baseline) | 44 against 83 | 29 | 15 / 54 | 3e-06 |
| 0.004 | both timed | our hierarchy, neural tie-break | TCBP-JOBS (NVIDIA, GPU) | 44 against 41 | 24 | 20 / 17 | 0.74 |
| 0.004 | **mixed**: a timed row against an offline combination, with other random draws | our ensemble (combined offline) (rule f, frozen on the training shots: where PriorTilt twice and TCBP-JOBS agree, that answer; elsewhere joint decoding where it converged, else TCBP-JOBS; joint decoding ran on 93 of the 93 disputed shots) | our hierarchy, neural tie-break | 16 against 44 | 13 | 3 / 31 | 8e-07 |
| 0.004 | **mixed**: a timed row against an offline combination, with other random draws | our ensemble (combined offline) (rule f, frozen on the training shots: where PriorTilt twice and TCBP-JOBS agree, that answer; elsewhere joint decoding where it converged, else TCBP-JOBS; joint decoding ran on 93 of the 93 disputed shots) | TCBP-JOBS (NVIDIA, GPU) | 16 against 41 | 15 | 1 / 26 | 4e-07 |
| 0.004 | both timed | TCBP-JOBS (NVIDIA, GPU) | relay-BP (kit baseline) | 41 against 83 | 27 | 14 / 56 | 4e-07 |
| 0.005 | both timed | relay-BP with a learned router | relay-BP (kit baseline) | 807 against 845 | 604 | 203 / 241 | 0.079 |
| 0.005 | both timed | PriorTilt, one run | relay-BP (kit baseline) | 601 against 845 | 431 | 170 / 414 | 2e-24 |
| 0.005 | both timed | our hierarchy, lowest-weight tie-break (run after the test results were seen, at the team's request; an ablation, not our decoder) | PriorTilt, one run | 553 against 601 | 477 | 76 / 124 | 8e-04 |
| 0.005 | both timed | our hierarchy, lowest-weight tie-break (run after the test results were seen, at the team's request; an ablation, not our decoder) | relay-BP (kit baseline) | 553 against 845 | 436 | 117 / 409 | 6e-39 |
| 0.005 | both timed | our hierarchy, neural tie-break | PriorTilt, one run | 479 against 601 | 440 | 39 / 161 | 8e-19 |
| 0.005 | both timed | our hierarchy, neural tie-break | our hierarchy, classical tie-break (lowest weight among the converged candidates) (run after the test results were seen, at the team's request; an ablation, not our decoder) | 479 against 481 | 414 | 65 / 67 | 0.93 |
| 0.005 | both timed | our hierarchy, classical tie-break (lowest weight among the converged candidates) (run after the test results were seen, at the team's request; an ablation, not our decoder) | PriorTilt, one run | 481 against 601 | 437 | 44 / 164 | 2e-17 |
| 0.005 | both timed | our hierarchy, classical tie-break (lowest weight among the converged candidates) (run after the test results were seen, at the team's request; an ablation, not our decoder) | relay-BP (kit baseline) | 481 against 845 | 382 | 99 / 463 | 3e-57 |
| 0.005 | both timed | our hierarchy, neural tie-break | our hierarchy, lowest-weight tie-break (run after the test results were seen, at the team's request; an ablation, not our decoder) | 479 against 553 | 434 | 45 / 119 | 7e-09 |
| 0.005 | both timed | our hierarchy, neural tie-break | relay-BP (kit baseline) | 479 against 845 | 386 | 93 / 459 | 4e-59 |
| 0.005 | both timed | our hierarchy, neural tie-break | TCBP-JOBS (NVIDIA, GPU) | 479 against 360 | 265 | 214 / 95 | 1e-11 |
| 0.005 | **mixed**: a timed row against an offline combination, with other random draws | our ensemble (combined offline) (rule f, frozen on the training shots: where PriorTilt twice and TCBP-JOBS agree, that answer; elsewhere joint decoding where it converged, else TCBP-JOBS; joint decoding ran on 0 of the 735 disputed shots; so it fails exactly the shots TCBP-JOBS fails) | our hierarchy, neural tie-break | 360 against 479 | 265 | 95 / 214 | 1e-11 |
| 0.005 | **mixed**: a timed row against an offline combination, with other random draws | our ensemble (combined offline) (rule f, frozen on the training shots: where PriorTilt twice and TCBP-JOBS agree, that answer; elsewhere joint decoding where it converged, else TCBP-JOBS; joint decoding ran on 0 of the 735 disputed shots; so it fails exactly the shots TCBP-JOBS fails) | TCBP-JOBS (NVIDIA, GPU) | 360 against 360 | 360 | 0 / 0 | 1.00 |
| 0.005 | both timed | TCBP-JOBS (NVIDIA, GPU) | relay-BP (kit baseline) | 360 against 845 | 306 | 54 / 539 | 1e-101 |
<!-- END table:test_comparisons -->

### Whose gain is whose, rung by rung

<!-- BEGIN table:test_ladder -->
**Whose gain is whose, on the test shots.** Every rung is scored on the same per-shot answers (relay-BP and PriorTilt under two seeds on every shot, a fresh decoder per 500 shots), so each step is a paired comparison with the rung before it. These rows are combined offline from per-shot files; they are not the timed rows of the table above. Relay-BP keeps random state from shot to shot, so this decode (a fresh decoder every 500 shots) and the timed row (one decoder through all 10,000 shots) differ: `checks.md` gives both counts and the paired test between them. Compare rungs with rungs, and timed rows with timed rows.

| p | Rung | Whose work | Failures X, Z | Failed pairs | Rate per cycle ± error | Against the rung before: both fail / only this / only the one before (exact p) | Against relay-BP: both fail / only this / only relay-BP (exact p) |
|---|---|---|---|---|---|---|---|
| 0.003 | relay-BP (kit baseline), a fresh decoder per 500 shots (the ensemble's member) | the kit | 3, 2 | 5 of 10,000 | 4.17e-05 ± 1.9e-05 | the first rung | the reference |
| 0.003 | PriorTilt, one run (seed A) | IonQ | 3, 3 | 6 of 10,000 | 5.00e-05 ± 2.0e-05 | 4 / 2 / 1 (p = 1.00) | 4 / 2 / 1 (p = 1.00) |
| 0.003 | our hierarchy, lowest-weight tie-break (nothing learned by us) | ours | 2, 1 | 3 of 10,000 | 2.50e-05 ± 1.4e-05 | 3 / 0 / 3 (p = 0.25) | 3 / 0 / 2 (p = 0.50) |
| 0.003 | our hierarchy, lowest weight among the converged candidates (nothing learned by us) | ours | 2, 1 | 3 of 10,000 | 2.50e-05 ± 1.4e-05 | 3 / 0 / 0 (p = 1.00) | 3 / 0 / 2 (p = 0.50) |
| 0.003 | our hierarchy, neural tie-break | ours | 2, 2 | 4 of 10,000 | 3.33e-05 ± 1.7e-05 | 3 / 1 / 0 (p = 1.00) | 3 / 1 / 2 (p = 1.00) |
| 0.003 | ORACLE: a perfect tie-break on the same candidates (a ceiling, not a decoder) | nobody: needs the truth | 2, 1 | 3 of 10,000 | 2.50e-05 ± 1.4e-05 | 3 / 0 / 1 (p = 1.00) | 3 / 0 / 2 (p = 0.50) |
| 0.003 | TCBP-JOBS (NVIDIA, GPU), for reference | NVIDIA; run by teammates | 1, 2 | 3 of 10,000 | 2.50e-05 ± 1.4e-05 | not a rung | 2 / 1 / 3 (p = 0.62) |
| 0.004 | relay-BP (kit baseline), a fresh decoder per 500 shots (the ensemble's member) | the kit | 42, 48 | 90 of 10,000 | 7.53e-04 ± 7.9e-05 | the first rung | the reference |
| 0.004 | PriorTilt, one run (seed A) | IonQ | 37, 34 | 71 of 10,000 | 5.94e-04 ± 7.0e-05 | 46 / 25 / 44 (p = 0.029) | 46 / 25 / 44 (p = 0.029) |
| 0.004 | our hierarchy, lowest-weight tie-break (nothing learned by us) | ours | 26, 31 | 57 of 10,000 | 4.76e-04 ± 6.3e-05 | 51 / 6 / 20 (p = 0.009) | 44 / 13 / 46 (p = 2e-05) |
| 0.004 | our hierarchy, lowest weight among the converged candidates (nothing learned by us) | ours | 24, 29 | 53 of 10,000 | 4.43e-04 ± 6.1e-05 | 53 / 0 / 4 (p = 0.12) | 40 / 13 / 50 (p = 3e-06) |
| 0.004 | our hierarchy, neural tie-break | ours | 25, 25 | 50 of 10,000 | 4.18e-04 ± 5.9e-05 | 42 / 8 / 11 (p = 0.65) | 38 / 12 / 52 (p = 5e-07) |
| 0.004 | ORACLE: a perfect tie-break on the same candidates (a ceiling, not a decoder) | nobody: needs the truth | 19, 21 | 40 of 10,000 | 3.34e-04 ± 5.3e-05 | 40 / 0 / 10 (p = 0.002) | 33 / 7 / 57 (p = 8e-11) |
| 0.004 | TCBP-JOBS (NVIDIA, GPU), for reference | NVIDIA; run by teammates | 20, 21 | 41 of 10,000 | 3.42e-04 ± 5.3e-05 | not a rung | 27 / 14 / 63 (p = 1e-08) |
| 0.005 | relay-BP (kit baseline), a fresh decoder per 500 shots (the ensemble's member) | the kit | 419, 385 | 794 of 10,000 | 6.87e-03 ± 2.4e-04 | the first rung | the reference |
| 0.005 | PriorTilt, one run (seed A) | IonQ | 303, 296 | 595 of 10,000 | 5.10e-03 ± 2.1e-04 | 413 / 182 / 381 (p = 3e-17) | 413 / 182 / 381 (p = 3e-17) |
| 0.005 | our hierarchy, lowest-weight tie-break (nothing learned by us) | ours | 269, 262 | 527 of 10,000 | 4.50e-03 ± 2.0e-04 | 461 / 66 / 134 (p = 2e-06) | 443 / 84 / 351 (p = 7e-40) |
| 0.005 | our hierarchy, lowest weight among the converged candidates (nothing learned by us) | ours | 241, 231 | 469 of 10,000 | 3.99e-03 ± 1.8e-04 | 469 / 0 / 58 (p = 7e-18) | 391 / 78 / 403 (p = 8e-54) |
| 0.005 | our hierarchy, neural tie-break | ours | 250, 241 | 490 of 10,000 | 4.18e-03 ± 1.9e-04 | 412 / 78 / 57 (p = 0.085) | 393 / 97 / 401 (p = 6e-45) |
| 0.005 | ORACLE: a perfect tie-break on the same candidates (a ceiling, not a decoder) | nobody: needs the truth | 195, 192 | 386 of 10,000 | 3.28e-03 ± 1.7e-04 | 386 / 0 / 104 (p = 1e-31) | 332 / 54 / 462 (p = 8e-82) |
| 0.005 | TCBP-JOBS (NVIDIA, GPU), for reference | NVIDIA; run by teammates | 186, 177 | 360 of 10,000 | 3.05e-03 ± 1.6e-04 | not a rung | 300 / 60 / 494 (p = 7e-86) |
<!-- END table:test_ladder -->

## C. The teammates' earlier runs on the same test files, as a cross-check

<!-- BEGIN table:earlier_runs -->
The teammates' own evaluator runs on the same test files, made earlier on other machines or under other load. They are a cross-check of the counts. **Their times are not comparable with the timed table** and are shown only with their machine.

| p | Decoder | Failures X, Z | Failed pairs | Rate per cycle ± error | Time per pair, ms: mean / median / p99 | Router against relay-BP: both / only router / only relay-BP (exact p) | Machine |
|---|---|---|---|---|---|---|---|
| 0.003 | BP-OSD | 26, 34 | 60 of 10,000 | 5.01e-04 ± 6.5e-05 | 131 / 18.8 / 803 |  | ICRN server, before the quiet window |
| 0.003 | relay-BP | 2, 3 | 5 of 10,000 | 4.17e-05 ± 1.9e-05 | 85.7 / 52.6 / 556 |  | ICRN server, before the quiet window |
| 0.003 | relay-BP, first solution only (a control) | 7, 6 | 13 of 10,000 | 1.08e-04 ± 3.0e-05 | 18.1 / 9.0 / 165 |  | ICRN server, before the quiet window |
| 0.003 | relay-BP, first leg or full (a control) | 3, 4 | 7 of 10,000 | 5.84e-05 ± 2.2e-05 | 36.5 / 8.9 / 574 |  | ICRN server, before the quiet window |
| 0.003 | relay-BP with the learned router | 3, 3 | 6 of 10,000 | 5.00e-05 ± 2.0e-05 | 47.2 / 9.7 / 698 | 4 / 2 / 1 (p = 1.00) | ICRN server, before the quiet window |
| 0.004 | BP-OSD | 231, 250 | 480 of 10,000 | 4.09e-03 ± 1.9e-04 | 144 / 159 / 439 |  | a teammate's Mac (arm64) |
| 0.004 | relay-BP | 41, 44 | 84 of 10,000 | 7.03e-04 ± 7.7e-05 | 131 / 42.9 / 2,215 |  | a teammate's Mac (arm64) |
| 0.004 | relay-BP, first solution only (a control) | 53, 71 | 122 of 10,000 | 1.02e-03 ± 9.3e-05 | 36.5 / 6.1 / 632 |  | a teammate's Mac (arm64) |
| 0.004 | relay-BP, first leg or full (a control) | 47, 51 | 98 of 10,000 | 8.20e-04 ± 8.3e-05 | 96.3 / 6.0 / 2,204 |  | a teammate's Mac (arm64) |
| 0.004 | relay-BP with the learned router | 37, 47 | 84 of 10,000 | 7.03e-04 ± 7.7e-05 | 124 / 6.1 / 2,663 | 48 / 36 / 36 (p = 1.00) | a teammate's Mac (arm64) |
<!-- END table:earlier_runs -->

## D. TCBP-JOBS on its own sampled shots

<!-- BEGIN table:tcbp_own_shots -->
TCBP-JOBS as the teammates first ran it (`tcbp_jobs_baseline.py`, an NVIDIA A100 slice on the ICRN server). **These are its own freshly sampled shots, not the saved test files**, so there is no paired count against the other rows, and the log prints the rate per cycle but not the number of failed pairs.

| p | Failures X, Z | Shots | Rate per cycle ± error (as logged) | Time per pair, ms: mean / p99 | File |
|---|---|---|---|---|---|
| 0.003 | 2, 3 | 10,000 per basis, its own sample (seed 1) | 4.17e-05 ± 1.9e-05 | 4.3 / 15.0 | `tcbp_own_shots/tcbp-jobs_p0.003_shots10000_seed1_20261122_005215_437826+0000.log` |
| 0.004 | 18, 23 | 10,000 per basis, its own sample (seed 1) | 3.42e-04 ± 5.3e-05 | not usable: the GPU was shared with a network training of ours (logged: 42.9 / 109) | `tcbp_own_shots/tcbp-jobs_p0.004_shots10000_seed1_20261122_043426_147901+0000.log` |
| 0.005 | 192, 188 | 10,000 per basis, its own sample (seed 1) | 3.20e-03 ± 1.6e-04 | not usable: the GPU was shared with a network training of ours (logged: 81.7 / 201) | `tcbp_own_shots/tcbp-jobs_p0.005_shots10000_seed1_20261122_050203_539870+0000.log` |
<!-- END table:tcbp_own_shots -->

## E. Exploration shots

<!-- BEGIN table:exploration_p0.004 -->
**Exploration shots, not test shots**: the team's train and validation files at p = 0.004 (`data/p0.004/train_*.npz`, `validation_*.npz`; 25,000 shots per basis). The rules were chosen on these same shots, which flatters them. Decoded on a laptop (Ryzen 9 8945HS) beside a GPU training: compare the times with each other only. In this first table a time is per shot of one basis.

| Shots | Decoder | Built by | Failures | Share of shots (95% interval) | Against relay-BP, same shots: both fail / only this row / only relay-BP (exact p) | Mean time per shot, ms |
|---|---|---|---|---|---|---|
| X memory, 25,000 shots | relay-BP (kit baseline) | the kit | 122 of 25,000 | 0.488% (0.409% to 0.582%) | the reference | 167 |
| X memory, 25,000 shots | PriorTilt, seed A | IonQ | 102 of 25,000 | 0.408% (0.336% to 0.495%) | 55 / 47 / 67 (p = 0.075) | 52.0 |
| X memory, 25,000 shots | PriorTilt, seed B | IonQ | 82 of 25,000 | 0.328% (0.264% to 0.407%) | 52 / 30 / 70 (p = 8e-05) | 52.4 |
| X memory, 25,000 shots | our hierarchy, lowest-weight tie-break | ours | 77 of 25,000 | 0.308% (0.247% to 0.385%) | 61 / 16 / 61 (p = 2e-07) | 118 |
| X memory, 25,000 shots | our hierarchy, lowest weight among the converged candidates | ours | 67 of 25,000 | 0.268% (0.211% to 0.340%) | 53 / 14 / 69 (p = 7e-10) | 118 |
| X memory, 25,000 shots | TCBP-JOBS (NVIDIA, GPU; answers only, no time) | NVIDIA; run by us | 53 of 25,000 | 0.212% (0.162% to 0.277%) | 45 / 8 / 77 (p = 3e-15) |  |
| X memory, 25,000 shots | our hierarchy, neural tie-break (network at training step 30,000) | ours | 66 of 25,000 | 0.264% (0.208% to 0.336%) | the laptop's table gives the count only | 118 |
| X memory, 25,000 shots | our hierarchy, neural tie-break (network at training step 40,000) | ours | 65 of 25,000 | 0.260% (0.204% to 0.331%) | the laptop's table gives the count only | 118 |
| X memory, 25,000 shots | our hierarchy, neural tie-break (network at training step 50,000) | ours | 63 of 25,000 | 0.252% (0.197% to 0.322%) | the laptop's table gives the count only | 118 |
| X memory, 25,000 shots | our hierarchy, neural tie-break (network at training step 60,000) | ours | 62 of 25,000 | 0.248% (0.194% to 0.318%) | the laptop's table gives the count only | 118 |
| X memory, 25,000 shots | our hierarchy, neural tie-break (network at training step 70,000) | ours | 59 of 25,000 | 0.236% (0.183% to 0.304%) | the laptop's table gives the count only | 118 |
| X memory, 25,000 shots | our hierarchy, neural tie-break (network at training step 80,000) | ours | 59 of 25,000 | 0.236% (0.183% to 0.304%) | the laptop's table gives the count only | 118 |
| X memory, 25,000 shots | our hierarchy, neural tie-break (network at training step 90,000) | ours | 62 of 25,000 | 0.248% (0.194% to 0.318%) | the laptop's table gives the count only | 118 |
| X memory, 25,000 shots | our hierarchy, neural tie-break (network at training step 100,000) | ours | 59 of 25,000 | 0.236% (0.183% to 0.304%) | the laptop's table gives the count only | 118 |
| X memory, 25,000 shots | our hierarchy, neural tie-break (network at training step 110,000) | ours | 60 of 25,000 | 0.240% (0.187% to 0.309%) | the laptop's table gives the count only | 118 |
| X memory, 25,000 shots | our hierarchy, neural tie-break (network at training step 120,000) | ours | 60 of 25,000 | 0.240% (0.187% to 0.309%) | 45 / 15 / 77 (p = 3e-11) | 118 |
| Z memory, 25,000 shots | relay-BP (kit baseline) | the kit | 123 of 25,000 | 0.492% (0.413% to 0.587%) | the reference | 171 |
| Z memory, 25,000 shots | PriorTilt, seed A | IonQ | 106 of 25,000 | 0.424% (0.351% to 0.513%) | 63 / 43 / 60 (p = 0.11) | 55.6 |
| Z memory, 25,000 shots | PriorTilt, seed B | IonQ | 107 of 25,000 | 0.428% (0.354% to 0.517%) | 63 / 44 / 60 (p = 0.14) | 57.1 |
| Z memory, 25,000 shots | our hierarchy, lowest-weight tie-break | ours | 85 of 25,000 | 0.340% (0.275% to 0.420%) | 73 / 12 / 50 (p = 1e-06) | 131 |
| Z memory, 25,000 shots | our hierarchy, lowest weight among the converged candidates | ours | 73 of 25,000 | 0.292% (0.232% to 0.367%) | 64 / 9 / 59 (p = 4e-10) | 131 |
| Z memory, 25,000 shots | TCBP-JOBS (NVIDIA, GPU; answers only, no time) | NVIDIA; run by us | 60 of 25,000 | 0.240% (0.187% to 0.309%) | 46 / 14 / 77 (p = 1e-11) |  |

The neural tie-break against the lowest-weight tie-break on the same 25,000 X shots (network at training step 120,000, recomputed here from its per-shot logits): both fail 54, only the network rule fails 6, only the weight rule fails 23 (exact p = 0.002). Against the better classical tie-break, lowest weight among the converged candidates: both fail 52, only the network rule fails 8, only the classical rule fails 15 (exact p = 0.21).

The same decodes as pairs (shot k of X with shot k of Z), the kit's score; times per pair:

| Decoder | Built by | Failed pairs | Rate per cycle ± error | 95% interval | Against relay-BP: both fail / only this row / only relay-BP (exact p) | Mean time per pair, ms |
|---|---|---|---|---|---|---|
| relay-BP (kit baseline) | the kit | 245 of 25,000 | 8.20e-04 ± 5.2e-05 | 7.2e-04 to 9.3e-04 | the reference | 338 |
| PriorTilt, seed A | IonQ | 207 of 25,000 | 6.93e-04 ± 4.8e-05 | 6.0e-04 to 7.9e-04 | 118 / 89 / 127 (p = 0.012) | 108 |
| PriorTilt, seed B | IonQ | 188 of 25,000 | 6.29e-04 ± 4.6e-05 | 5.5e-04 to 7.3e-04 | 115 / 73 / 130 (p = 8e-05) | 109 |
| our hierarchy, lowest-weight tie-break | ours | 162 of 25,000 | 5.42e-04 ± 4.3e-05 | 4.6e-04 to 6.3e-04 | 134 / 28 / 111 (p = 7e-13) | 249 |
| our hierarchy, lowest weight among the converged candidates | ours | 140 of 25,000 | 4.68e-04 ± 4.0e-05 | 4.0e-04 to 5.5e-04 | 117 / 23 / 128 (p = 7e-19) | 249 |
| TCBP-JOBS (NVIDIA, GPU) | NVIDIA; run by us | 113 of 25,000 | 3.77e-04 ± 3.6e-05 | 3.1e-04 to 4.5e-04 | 91 / 22 / 154 (p = 1e-25) | no time: batch mode on a shared GPU |

Our hierarchy (lowest weight among the converged candidates) against TCBP-JOBS on the same 25,000 pairs: both fail 83, only ours fails 57, only TCBP-JOBS fails 30 (exact p = 0.005). **TCBP-JOBS fails fewer pairs than our hierarchy here.**

Where the failures are: the shots on which the two PriorTilt runs predict the same 12 bits ("agree"), and the others:

| Set | Shots | The two runs agree | They disagree | Share in "disagree" |
|---|---|---|---|---|
| all shots (X and Z memory) | 50,000 | 49,786 | 214 | 0.4% |
| shots PriorTilt, seed A, gets wrong | 208 | 63 | 145 | 69.7% |
| shots still wrong after the lowest-weight tie-break | 162 | 63 | 99 | 61.1% |

Sources: `report/data/exploration/priortilt/selector_*.npz` (per shot: relay-BP and PriorTilt under two seeds), and for the rows with the network `results_network_select_step*.txt` (the laptop's own table).
<!-- END table:exploration_p0.004 -->

<!-- BEGIN table:where_failures -->
Where the failures are. "Agree" means the two runs of PriorTilt (two random seeds) predict the same 12 bits; the hierarchy then takes that answer, and a wrong one cannot be repaired by any tie-break. X and Z memory shots are counted together.

| Shots | Set | Shots in the set | The two runs agree | They disagree | Share in "disagree" |
|---|---|---|---|---|---|
| exploration shots, p = 0.004 | all shots | 50,000 | 49,786 | 214 | 0.4% |
| exploration shots, p = 0.004 | shots wrong under one fast decode (PriorTilt, seed A) | 208 | 63 | 145 | 69.7% |
| exploration shots, p = 0.004 | shots wrong under the hierarchy, lowest-weight tie-break | 162 | 63 | 99 | 61.1% |
| test shots, p = 0.003, the timed decoder | all shots | 20,000 | 19,998 | 2 | 0.0% |
| test shots, p = 0.003, the timed decoder | shots wrong under our decoder, neural tie-break | 3 | 3 | 0 | 0.0% |
| test shots, p = 0.004, the timed decoder | all shots | 20,000 | 19,910 | 90 | 0.4% |
| test shots, p = 0.004, the timed decoder | shots wrong under our decoder, neural tie-break | 44 | 17 | 27 | 61.4% |
| test shots, p = 0.005, the timed decoder | all shots | 20,000 | 19,391 | 609 | 3.0% |
| test shots, p = 0.005, the timed decoder | shots wrong under our decoder, neural tie-break | 483 | 184 | 299 | 61.9% |
<!-- END table:where_failures -->

<!-- BEGIN table:exploration_p0.005 -->
**Exploration shots, not test shots**: X memory at p = 0.005, the team's `data/p0.005/validation_X.npz` (5,000 shots), decoded on the laptop beside a GPU training (compare the times with each other only; a time is per shot of one basis). The two PriorTilt runs disagree on 141 shots (2.8%).

| Decoder | Built by | Failures | Share of shots (95% interval) | Against relay-BP, same shots: both fail / only this row / only relay-BP (exact p) | Mean time per shot, ms |
|---|---|---|---|---|---|
| relay-BP (kit baseline) | the kit | 194 of 5,000 | 3.88% (3.38% to 4.45%) | the reference | 885 |
| PriorTilt, seed A | IonQ | 137 of 5,000 | 2.74% (2.32% to 3.23%) | 92 / 45 / 102 (p = 3e-06) | 193 |
| PriorTilt, seed B | IonQ | 152 of 5,000 | 3.04% (2.60% to 3.55%) | 107 / 45 / 87 (p = 3e-04) | 190 |
| our hierarchy, lowest-weight tie-break | ours | 129 of 5,000 | 2.58% (2.18% to 3.06%) | 110 / 19 / 84 (p = 6e-11) | 537 |
| our hierarchy, lowest weight among the converged candidates | ours | 108 of 5,000 | 2.16% (1.79% to 2.60%) | 91 / 17 / 103 (p = 3e-16) | 537 |

Source: `report/data/exploration/priortilt_p0.005/selector_validation_X.npz`.
<!-- END table:exploration_p0.005 -->

<!-- BEGIN table:exploration_ensemble -->
**Exploration shots, not test shots**: the team's validation files (5,000 shots per basis), each member's saved answers combined offline by `hierarchy.ensemble`. Members: R relay-BP, P1 and P2 PriorTilt under two seeds, T TCBP-JOBS (GPU), J joint relay-BP on the shots where P1, P2 and T are not unanimous. A rule that names "the network" uses the learned tie-break; the others need no learning of ours. An ORACLE is given the truth: a ceiling, not a decoder. A pair is shot k of X with shot k of Z.

| p | Row | Failures X, Z | Failed pairs | Rate per cycle ± error | Against relay-BP alone: both fail / only this / only relay-BP (exact p) | Against TCBP-JOBS alone: both fail / only this / only TCBP-JOBS (exact p) |
|---|---|---|---|---|---|---|
| 0.003 | R alone | 2, 2 | 4 of 5,000 | 6.67e-05 ± 3.3e-05 |  | 0 / 4 / 1 (p = 0.38) |
| 0.003 | P1 alone | 2, 2 | 4 of 5,000 | 6.67e-05 ± 3.3e-05 | 3 / 1 / 1 (p = 1.00) | 0 / 4 / 1 (p = 0.38) |
| 0.003 | P2 alone | 3, 2 | 5 of 5,000 | 8.34e-05 ± 3.7e-05 | 3 / 2 / 1 (p = 1.00) | 0 / 5 / 1 (p = 0.22) |
| 0.003 | T alone | 0, 1 | 1 of 5,000 | 1.67e-05 ± 1.7e-05 | 0 / 1 / 4 (p = 0.38) |  |
| 0.003 | ac: P1 where P1 = P2; else the lowest weight among the converged of R, P1, P2 | 2, 2 | 4 of 5,000 | 6.67e-05 ± 3.3e-05 | 4 / 0 / 0 (p = 1.00) | 0 / 4 / 1 (p = 0.38) |
| 0.003 | aw: P1 where P1 = P2; else the lowest weight of R, P1, P2 | 2, 2 | 4 of 5,000 | 6.67e-05 ± 3.3e-05 | 4 / 0 / 0 (p = 1.00) | 0 / 4 / 1 (p = 0.38) |
| 0.003 | bw: P1 where P1 = P2 = T; else the lowest weight of P1, P2, T, R | 0, 0 | 0 of 5,000 | 0.00e+00 ± 0.0e+00 | 0 / 0 / 4 (p = 0.12) | 0 / 0 / 1 (p = 1.00) |
| 0.003 | f: P1 where P1 = P2 = T; else J where it converged, else T **(the rule chosen on the training shots)** | 1, 0 | 1 of 5,000 | 1.67e-05 ± 1.7e-05 | 0 / 1 / 4 (p = 0.38) | 0 / 1 / 1 (p = 1.00) |
| 0.003 | gw: P1 where P1 = P2 = T; else most of J, T, P1, P2, R; a tie: that order | 1, 0 | 1 of 5,000 | 1.67e-05 ± 1.7e-05 | 1 / 0 / 3 (p = 0.25) | 0 / 1 / 1 (p = 1.00) |
| 0.003 | ORACLE (needs the truth): a perfect choice among P1, P2, T | 0, 0 | 0 of 5,000 | 0.00e+00 ± 0.0e+00 | 0 / 0 / 4 (p = 0.12) | 0 / 0 / 1 (p = 1.00) |
| 0.003 | ORACLE (needs the truth): a perfect choice among P1, P2, T, R | 0, 0 | 0 of 5,000 | 0.00e+00 ± 0.0e+00 | 0 / 0 / 4 (p = 0.12) | 0 / 0 / 1 (p = 1.00) |
| 0.003 | ORACLE (needs the truth): a perfect choice among P1, P2, T, R, J | 0, 0 | 0 of 5,000 | 0.00e+00 ± 0.0e+00 | 0 / 0 / 4 (p = 0.12) | 0 / 0 / 1 (p = 1.00) |
| 0.003 | ORACLE (needs the truth): a perfect choice among R, P1, P2 | 1, 1 | 2 of 5,000 | 3.33e-05 ± 2.4e-05 | 2 / 0 / 2 (p = 0.50) | 0 / 2 / 1 (p = 1.00) |
| 0.005 | R alone | 194, 189 | 373 of 5,000 | 6.44e-03 ± 3.3e-04 |  | 151 / 222 / 36 (p = 7e-34) |
| 0.005 | P1 alone | 137, 144 | 276 of 5,000 | 4.72e-03 ± 2.8e-04 | 188 / 88 / 185 (p = 4e-09) | 137 / 139 / 50 (p = 7e-11) |
| 0.005 | P2 alone | 152, 145 | 289 of 5,000 | 4.95e-03 ± 2.9e-04 | 198 / 91 / 175 (p = 3e-07) | 138 / 151 / 49 (p = 3e-13) |
| 0.005 | T alone | 91, 97 | 187 of 5,000 | 3.17e-03 ± 2.3e-04 | 151 / 36 / 222 (p = 7e-34) |  |
| 0.005 | ac: P1 where P1 = P2; else the lowest weight among the converged of R, P1, P2 | 108, 104 | 207 of 5,000 | 3.52e-03 ± 2.4e-04 | 177 / 30 / 196 (p = 5e-31) | 126 / 81 / 61 (p = 0.11) |
| 0.005 | aw: P1 where P1 = P2; else the lowest weight of R, P1, P2 | 129, 118 | 241 of 5,000 | 4.11e-03 ± 2.6e-04 | 206 / 35 / 167 (p = 8e-22) | 137 / 104 / 50 (p = 2e-05) |
| 0.005 | bw: P1 where P1 = P2 = T; else the lowest weight of P1, P2, T, R | 107, 113 | 218 of 5,000 | 3.71e-03 ± 2.5e-04 | 184 / 34 / 189 (p = 3e-27) | 182 / 36 / 5 (p = 8e-07) |
| 0.005 | f: P1 where P1 = P2 = T; else J where it converged, else T **(the rule chosen on the training shots)** | 91, 97 | 187 of 5,000 | 3.17e-03 ± 2.3e-04 | 151 / 36 / 222 (p = 7e-34) | 187 / 0 / 0 (p = 1.00) |
| 0.005 | gw: P1 where P1 = P2 = T; else most of J, T, P1, P2, R; a tie: that order | 90, 92 | 179 of 5,000 | 3.03e-03 ± 2.3e-04 | 162 / 17 / 211 (p = 9e-44) | 159 / 20 / 28 (p = 0.31) |
| 0.005 | ORACLE (needs the truth): a perfect choice among P1, P2, T | 53, 63 | 116 of 5,000 | 1.95e-03 ± 1.8e-04 | 105 / 11 / 268 (p = 4e-65) | 116 / 0 / 71 (p = 8e-22) |
| 0.005 | ORACLE (needs the truth): a perfect choice among P1, P2, T, R | 50, 55 | 105 of 5,000 | 1.77e-03 ± 1.7e-04 | 105 / 0 / 268 (p = 4e-81) | 105 / 0 / 82 (p = 4e-25) |
| 0.005 | ORACLE (needs the truth): a perfect choice among P1, P2, T, R, J | 50, 55 | 105 of 5,000 | 1.77e-03 ± 1.7e-04 | 105 / 0 / 268 (p = 4e-81) | 105 / 0 / 82 (p = 4e-25) |
| 0.005 | ORACLE (needs the truth): a perfect choice among R, P1, P2 | 75, 82 | 155 of 5,000 | 2.62e-03 ± 2.1e-04 | 155 / 0 / 218 (p = 5e-66) | 106 / 49 / 81 (p = 0.006) |

Source: `report/data/exploration/ensemble/ens_p<p>/validation_{X,Z}.npz` and `.json`.
<!-- END table:exploration_ensemble -->

<!-- BEGIN table:ablation_80k -->
**The ablation at scale, on exploration shots never used for a fit**: 80,000 fresh X-memory shots at p = 0.004 (Stim seeds 9401 to 9416), decoded in session mode as the plugin decodes, the final X network (step 120,000). One shot is one X-memory shot, not a pair. The two PriorTilt runs differ on 366 shots; on 302 of them at least one candidate is right, and the last column counts how often each tie-break picks a right one there.

| Decoder | Whose work | Failures | Share of shots (95% interval) | Right pick where one existed |
|---|---|---|---|---|
| PriorTilt, one run (seed A) | IonQ | 357 of 80,000 | 0.446% (0.402% to 0.495%) |  |
| PriorTilt, one run (seed B) | IonQ | 348 of 80,000 | 0.435% (0.392% to 0.483%) |  |
| our hierarchy, lowest-weight tie-break | ours, nothing learned | 298 of 80,000 | 0.372% (0.333% to 0.417%) | 191 of 302 |
| our hierarchy, lowest weight among the converged candidates | ours, nothing learned | 253 of 80,000 | 0.316% (0.280% to 0.358%) | 236 of 302 |
| our hierarchy, neural tie-break (final X network) | ours, learned | 237 of 80,000 | 0.296% (0.261% to 0.336%) | 252 of 302 |
| ORACLE: a perfect tie-break on the same candidates (a ceiling, not a decoder) | needs the truth | 187 of 80,000 | 0.234% (0.203% to 0.270%) |  |

Paired, on the same shots:

| Comparison | Both fail | Only the first fails | Only the second fails | Exact p |
|---|---|---|---|---|
| the neural tie-break against lowest weight | 211 | 26 | 87 | 7e-09 |
| the neural tie-break against lowest weight among the converged | 198 | 39 | 55 | 0.12 |
| lowest weight among the converged against lowest weight | 252 | 1 | 46 | 7e-13 |

Source: `report/data/exploration/ablation_80k/per_shot.npz` (recomputed here) and `summary.json` (checked against it), from branch `hierarchy` at commit `534d428`. Relay-BP alone was not decoded on these shots.
<!-- END table:ablation_80k -->

## F. Atom loss

<!-- BEGIN table:loss -->
X memory, p = 0.004, relay-BP at the kit's settings on shots simulated with Clifft under **our own simplified loss model** (after every CX each of its two atoms is lost with probability s; flags are perfect). It is not Infleqtion's hardware model. These are exploration numbers on shots made for this study; a shot here is one X-memory shot, not a pair. Each loss rate and machine has its own dataset, so rows of different datasets are not paired with each other.

| s per atom per CX | Dataset | Decoder | Failures | Share of shots (95% interval) | Paired on the same shots |
|---|---|---|---|---|---|
| 0.0002 | ICRN server, seed 5100 | loss-blind relay-BP (ignores the flags) | 81 of 3,000 | 2.70% (2.18% to 3.34%) |  |
| 0.0002 | ICRN server, seed 5100 | flag-aware relay-BP (the published method) | 160 of 20,000 | 0.80% (0.69% to 0.93%) | loss-blind only 63, this row only 5 (p = 8e-14); loss-blind ÷ this = 3.52 (2.53 to 5.43), on 3,000 shots |
| 0.0005 | ICRN server, seed 5200 | loss-blind relay-BP (ignores the flags) | 109 of 800 | 13.63% (11.42% to 16.18%) |  |
| 0.0005 | ICRN server, seed 5200 | flag-aware relay-BP (the published method) | 132 of 10,000 | 1.32% (1.11% to 1.56%) | loss-blind only 98, this row only 1 (p = 3e-28); loss-blind ÷ this = 9.08 (5.75 to 18.33), on 800 shots |
| 0.0005 | this PC, seed 2200 | loss-blind relay-BP (ignores the flags) | 57 of 400 | 14.25% (11.16% to 18.02%) |  |
| 0.0005 | this PC, seed 2200 | flag-aware relay-BP (the published method) | 29 of 2,000 | 1.45% (1.01% to 2.07%) | loss-blind only 53, this row only 1 (p = 6e-15); loss-blind ÷ this = 11.40 (5.90 to 53.00), on 400 shots |
| 0.001 | ICRN server, seed 5300 | loss-blind relay-BP (ignores the flags) | not measured | not measured | not measured |
| 0.001 | ICRN server, seed 5300 | flag-aware relay-BP (the published method) | 162 of 4,000 | 4.05% (3.48% to 4.71%) |  |
| 0.001 | ICRN server, seed 5300 | ORACLE: told the true loss round (a ceiling, not a decoder) | 96 of 4,000 | 2.40% (1.97% to 2.92%) | flag-aware only 90, this row only 24 (p = 4e-10); flag-aware ÷ this = 1.69 (1.44 to 2.01), on 4,000 shots |
| 0.001 | ICRN server, seed 7310 | flag-aware relay-BP (the published method) | 146 of 4,000 | 3.65% (3.11% to 4.28%) |  |
| 0.001 | ICRN server, seed 7310 | ours: flag-aware with the loss-time tracker | 122 of 4,000 | 3.05% (2.56% to 3.63%) | flag-aware only 66, this row only 42 (p = 0.026); flag-aware ÷ this = 1.20 (1.03 to 1.40), on 4,000 shots |
| 0.001 | ICRN server, seed 7310 | ORACLE: told the true loss round (a ceiling, not a decoder) | 82 of 4,000 | 2.05% (1.65% to 2.54%) | flag-aware only 84, this row only 20 (p = 2e-10); flag-aware ÷ this = 1.78 (1.49 to 2.15), on 4,000 shots |
| 0.001 | this PC, seed 2300 | loss-blind relay-BP (ignores the flags) | 173 of 300 | 57.67% (52.01% to 63.13%) |  |
| 0.001 | this PC, seed 2300 | flag-aware relay-BP (the published method) | 176 of 4,000 | 4.40% (3.81% to 5.08%) | loss-blind only 158, this row only 0 (p = 5e-48); loss-blind ÷ this = 11.53 (7.61 to 22.00), on 300 shots |
| 0.001 | this PC, seed 2300 | ORACLE: told the true loss round (a ceiling, not a decoder) | 107 of 4,000 | 2.67% (2.22% to 3.22%) | flag-aware only 94, this row only 25 (p = 1e-10); flag-aware ÷ this = 1.64 (1.42 to 1.94), on 4,000 shots |
| 0.001 | this PC, seed 7300 | flag-aware relay-BP (the published method) | 192 of 4,000 | 4.80% (4.18% to 5.51%) |  |
| 0.001 | this PC, seed 7300 | ours: flag-aware with the loss-time tracker | 147 of 4,000 | 3.67% (3.13% to 4.30%) | flag-aware only 75, this row only 30 (p = 1e-05); flag-aware ÷ this = 1.31 (1.17 to 1.48), on 4,000 shots |
| 0.001 | this PC, seed 7300 | ORACLE: told the true loss round (a ceiling, not a decoder) | 112 of 4,000 | 2.80% (2.33% to 3.36%) | flag-aware only 102, this row only 22 (p = 2e-13); flag-aware ÷ this = 1.71 (1.48 to 2.01), on 4,000 shots |
| 0.002 | this PC, seed 2400 | loss-blind relay-BP (ignores the flags) | 148 of 150 | 98.67% (95.27% to 99.63%) |  |
| 0.002 | this PC, seed 2400 | flag-aware relay-BP (the published method) | 173 of 750 | 23.07% (20.19% to 26.21%) | loss-blind only 115, this row only 0 (p = 5e-35); loss-blind ÷ this = 4.48 (3.42 to 6.21), on 150 shots |

An ORACLE row is given the true loss round on purpose: it is a ceiling for a tracker of the loss time, not a decoder result. Sources: `report/data/loss/pc/*.json`, `report/data/loss/server/*/tables.md`.
<!-- END table:loss -->

## G. The networks

<!-- BEGIN table:networks -->
The two networks, from their training logs (`report/data/networks/*.train.log`). "Shots wrong" is the share of shots with at least one of the 12 logical bits wrong, on error-model shots drawn for the training report: a training diagnostic, not a result on Stim shots.

| Network | Trained on | Size | Started from | Training step (last in the copied log) | Shots seen | Shots per second (median of the log) | GPU hours so far (shots seen ÷ that speed) | Shots wrong at p = 0.003 / 0.004 / 0.005 |
|---|---|---|---|---|---|---|---|---|
| X memory (`cascade_X_s1`) | a laptop GPU (RTX 4060, 8 GB) | width 128, depth 8, 893,697 parameters | random weights | 120,000 of 120,000 | 15,360,000 | 723 | 5.9 | 0.5% / 3.8% / 18.3% |
| Z memory (`cascade_Z_s1`) | the ICRN server (A100 slice, 20 GB) | width 128, depth 8, 893,697 parameters | the X network's weights, mirrored | 100,000 of 100,000 | 12,800,000 | 893 | 4.0 | 0.5% / 4.1% / 18.6% |

The X network alone, on the team's Stim shot files (exploration shots; `report/data/exploration/cascade_X_s1/SCORES.md`), share of shots wrong:

| Training step | `validation_X_p0.003` | `train_X_p0.003` | `validation_X_p0.004` | `train_X_p0.004` | `validation_X_p0.005` | `train_X_p0.005` |
|---|---|---|---|---|---|---|
| 20,000 |  |  | 42.12% of 5,000 |  |  |  |
| 30,000 |  |  | 26.60% of 5,000 | 27.00% of 20,000 |  |  |
| 40,000 | 6.16% of 5,000 | 6.10% of 20,000 | 21.86% of 5,000 | 21.66% of 20,000 | 47.54% of 5,000 | 48.62% of 20,000 |
| 50,000 | 3.66% of 5,000 | 3.96% of 20,000 | 16.04% of 5,000 | 15.43% of 20,000 | 38.88% of 5,000 | 39.13% of 20,000 |
| 60,000 | 2.70% of 5,000 | 2.77% of 20,000 | 12.32% of 5,000 | 12.02% of 20,000 | 32.92% of 5,000 | 33.03% of 20,000 |
| 70,000 | 2.24% of 5,000 | 2.46% of 20,000 | 10.52% of 5,000 | 10.35% of 20,000 | 30.18% of 5,000 | 30.18% of 20,000 |
| 80,000 | 1.20% of 5,000 | 1.26% of 20,000 | 7.32% of 5,000 | 6.69% of 20,000 | 23.18% of 5,000 | 23.51% of 20,000 |
| 90,000 | 0.82% of 5,000 | 0.91% of 20,000 | 6.04% of 5,000 | 5.39% of 20,000 | 20.58% of 5,000 | 20.76% of 20,000 |
| 100,000 | 0.72% of 5,000 | 0.81% of 20,000 | 5.32% of 5,000 | 4.76% of 20,000 | 19.56% of 5,000 | 19.54% of 20,000 |
| 110,000 | 0.76% of 5,000 | 0.66% of 20,000 | 5.02% of 5,000 | 4.30% of 20,000 | 18.02% of 5,000 | 18.32% of 20,000 |
| 120,000 | 0.56% of 5,000 | 0.68% of 20,000 | 4.74% of 5,000 | 3.95% of 20,000 | 17.16% of 5,000 | 17.73% of 20,000 |

The final networks alone on the **test shots** (the team's test files, `neural.evaluate_stim`; `report/data/test/net_p<p>/`), share of shots with any logical bit wrong. Alone they are far worse than relay-BP; they are only ever asked to choose among candidates.

| p | X network (step 120,000) | Z network (step 100,000) |
|---|---|---|
| 0.003 | 64 of 10,000 (0.6%) | 59 of 10,000 (0.6%) |
| 0.004 | 420 of 10,000 (4.2%) | 434 of 10,000 (4.3%) |
| 0.005 | 1,885 of 10,000 (18.9%) | 1,810 of 10,000 (18.1%) |
<!-- END table:networks -->

## H. What we measured before building, and what did not work

<!-- BEGIN table:background -->
Measurements that decided the design, and what we tried that did not work. **All on exploration shots or on small studies made for the purpose: none of this is a result on the test shots.** Each line names the copied file under `report/data/` it is read from.

| Question or attempt | What was measured | File |
|---|---|---|
| Is the right answer among 16 relay-BP candidates (other random draws), on the shots where relay-BP settled on a wrong answer? p = 0.004 | it is among them on 114 of 145 shots (79%); the lowest weight picks it on 71 (49%). The information is in the candidates and weight is a poor score: room for a learned one | `background/g2_server_p0.004_summary.json` |
| Joint decoding (X and Z checks together, relay-BP on all 1,728 detectors) on relay-BP's failures, p = 0.003 | fixes 43 of 44 failures and breaks 0 of 300 shots relay-BP had right; about 8 s per ordinary shot on our PC (8 processes), far too slow to run on every shot | `background/g2_summary.json` |
| Joint decoding (X and Z checks together, relay-BP on all 1,728 detectors) on relay-BP's failures, p = 0.004 | fixes 89 of 100 failures and breaks 1 of 200 shots relay-BP had right; about 31 s per ordinary shot on our PC (8 processes), far too slow to run on every shot | `background/g2_summary.json` |
| A cheaper relay-BP: stopping relay-BP after 3 solutions instead of 5; p = 0.004, the 5,000 validation pairs | 42 failed pairs against the kit's 51 (only the kit's fails 24, only this one 15, p = 0.20), 1.44 times fewer iterations | `background/g3_frontier_p0.004.csv` |
| A cheaper relay-BP: stopping relay-BP at its first solution; p = 0.004, the 5,000 validation pairs | 73 failed pairs against the kit's 51 (only the kit's fails 16, only this one 38, p = 0.004), 3.42 times fewer iterations | `background/g3_frontier_p0.004.csv` |
| A cheaper relay-BP: a fast relay-BP schedule (first leg 40, later legs 30, stop at the first solution); p = 0.004, the 5,000 validation pairs | 108 failed pairs against the kit's 51 (only the kit's fails 12, only this one 69, p = 7e-11), 2.97 times fewer iterations | `background/g3_frontier_p0.004.csv` |
| The same fast schedule (40 / 30 / 1) as a fallback, X memory, p = 0.004, 5,000 shots sampled for that comparison | 49 failures against the kit's 20 of 5000 (only the kit's fails 2, only the fast one 31, p = 1e-07) | `background/relay_schedule_paired_X_p0.004_5000_seed1.log` |
| A classic predecoder (commit to sure faults first, pass the rest to relay-BP), p = 0.004, 300 shots sampled for that study | failures 7/300 without it, 7/300 with BP's own confident picks, 5/300 with a trained model; hard shots 13.3% without, 12.3% and 14.0% with: no gain | `background/predecoder_real2_p0.004_300_300_300_20.log` |
| A plain convolutional network that predicts the 12 logical bits (no structure of the code in it), 0.7 M parameters | learns at very low noise only: at p = 0.0001 it has 8.85% of shots wrong after 500,224 training shots (answering "nothing flipped" gets 33.22%); that is 30 to 50 times below the challenge's noise levels | `background/train_X_p0.0001_seed1.log` |
| "Network first": the network answers where its confidence is at least c, else the hierarchy (X memory, p = 0.005, network at step 120,000; c fitted on train_X with the strict rule, scored on validation_X, 5,000 shots) | the network answers 18.58% of the shots; 107 failures against the hierarchy's 108 (only network-first fails 0, only the hierarchy 1, p = 1); 530.8 against 536.5 ms per shot on the laptop beside a GPU training: no gain in accuracy | `background/results_hierarchy_heldout_step120000.txt` |
| "Network first": the network answers where its confidence is at least c, else the hierarchy (X memory, p = 0.005, network at step 120,000; c fitted on train_X with the loose rule, scored on validation_X, 5,000 shots) | the network answers 60.20% of the shots; 106 failures against the hierarchy's 108 (only network-first fails 3, only the hierarchy 5, p = 0.727); 451.6 against 536.5 ms per shot on the laptop beside a GPU training: no gain in accuracy | `background/results_hierarchy_heldout_step120000.txt` |
| "Network first" at p = 0.004 (c fitted on train_X and validation_X), on the 80,000 fresh X shots of the ablation at scale, never used for a fit | the network answers 77% of the shots; 229 failures against the hierarchy's 237 with the neural tie-break (only network-first fails 17, only the hierarchy 25, p = 0.28): level in accuracy | `exploration/ablation_80k/summary.json` |
| "Network first" at p = 0.004 (c fitted on train_X only), on the 80,000 fresh X shots of the ablation at scale, never used for a fit | the network answers 80% of the shots; 242 failures against the hierarchy's 237 with the neural tie-break (only network-first fails 32, only the hierarchy 27, p = 0.60): level in accuracy | `exploration/ablation_80k/summary.json` |
| A learned loss-time tracker (a small recurrent network, a GRU, over the 13 detector rounds of a flagged atom) in place of the tracker's statistical model (X memory, p = 0.004, s = 0.001, validation shots, 40,164 lost data atoms) | right round 48.2% and within one round 77.4% of the time, against 50.9% and 79.3% for the model; mean log-probability of the true round -1.390 against -1.325: it does not beat the model, so it was not decoded | `loss/tracker/accuracy_val_learned.json` |
| The same kind of plain network at the challenge's noise, on the laptop GPU: run `x_4h` (run 1), 5.6 M parameters | Stopped at 0.93 h. It equalled the always-0 guess at every p, also at p = 0.0005 (`x_4h.ckpt_0.83h.eval_lowp.log`). About 2,500 shots/s, about 3,600 optimizer steps by 0.83 h | `background/plain_cnn_runs/README.md` |
| The same kind of plain network at the challenge's noise, on the laptop GPU: run `x_1h_w32` (run 2), 0.7 M parameters | 117.3 M shots; at p = 0.003, 90.55% of shots wrong and 67.76% of bits right | `background/plain_cnn_runs/README.md` |
| Under loss (X memory, p = 0.004, s = 0.001, 4,000 shots): exact weights for where a lost atom can have been | 189 failures against the flag-aware decoder's 176 (only flag-aware fails 46, only this one 59, p = 0.24): no gain | `loss/pc/L3_X_p0.004_s1e-3.json` |
| Under loss (X memory, p = 0.004, s = 0.001, 4,000 shots): stopping relay-BP at its first solution | 200 failures against the flag-aware decoder's 176 (only flag-aware fails 10, only this one 34, p = 4e-04): more failures | `loss/pc/L1_X_p0.004_s1e-3.json` |
| Under loss, a piece of the flag-aware rule left out: without the final-readout flip of a flagged data atom (X memory, p = 0.004, s = 0.001, 1,000 shots) | 68 of 1,000 failures; on the same shots only flag-aware fails 8, only this one 38 (p = 9e-06); flag-aware ÷ this = 0.56 (0.42 to 0.71) | `loss/server/L3_s1e-3_seed5300_jobD/tables.md` |
| Under loss, a piece of the flag-aware rule left out: without the columns for a flagged check atom of the other basis (X memory, p = 0.004, s = 0.001, 1,000 shots) | 39 of 1,000 failures; on the same shots only flag-aware fails 12, only this one 13 (p = 1.00); flag-aware ÷ this = 0.97 (0.75 to 1.27) | `loss/server/L3_s1e-3_seed5300_jobD/tables.md` |
| Under loss, a piece of the flag-aware rule left out: without trusting "no flag" (every unflagged atom also gets loss priors) (X memory, p = 0.004, s = 0.001, 1,000 shots) | 48 of 1,000 failures; on the same shots only flag-aware fails 11, only this one 21 (p = 0.11); flag-aware ÷ this = 0.79 (0.60 to 1.02) | `loss/server/L3_s1e-3_seed5300_jobD/tables.md` |
| Longer training of the X network, as a tie-breaker (X memory, p = 0.004, 25,000 exploration shots) | at training steps 30,000, 40,000, 50,000, 60,000, 70,000, 80,000, 90,000, 100,000, 110,000, 120,000 the hierarchy fails 66, 65, 63, 62, 59, 59, 62, 59, 60, 60 shots, while the network alone goes from 26.9% to 4.1% of shots wrong: the hard shots are hard for the network too | `exploration/priortilt/results_network_select_step*.txt` |
<!-- END table:background -->

## I. Checks made on every run

<!-- BEGIN table:checks -->
Checks made by `python -m report.make_tables` on this run:

- **Rows on the test shots:** of 33 registered rows, 30 are timed evaluator rows, 3 are combined offline from saved answers with no time in one process (ours, ensemble at p = 0.003; ours, ensemble at p = 0.004; ours, ensemble at p = 0.005), and 0 print "not measured" (none).
- **Same shots for every row:** each measured row names the sha256 of the test files it decoded, and the run stops unless they are the team's `data/p<p>/test_X.npz` and `test_Z.npz` (the values are in `report/rows.py`). A paired count is only made between such rows.
- **The rate uses pairs and 12 cycles:** for each evaluator row the per-shot file is re-counted (X failures, Z failures, pair = X or Z) and the rate per cycle is recomputed as 1 - (1 - P)^(1/12); the run stops if this differs from the evaluator's `summary.json`.
- **Times are per pair:** mean, median and p99 are recomputed from `seconds_total_pair` (the whole decode of one X shot plus one Z shot, every stage and rerun included) and must equal the evaluator's summary.
- **The paired count agrees with the team's `evaluation/compare_gain.py`:** `report/test_report.py` recomputes the two committed comparisons (`results/icrn/gain_p0.003_portable`, `results/mac/gain_p0.004`) from the per-shot files and gets the same counts.
- **Cross-check with the teammates' earlier runs.** p = 0.003: relay-BP's timed row and the teammates' earlier run (ICRN server, before the quiet window) give the same outcome on every one of the 10,000 pairs.
- **Cross-check with the teammates' earlier runs.** p = 0.004: relay-BP's timed row and the teammates' earlier run (a teammate's Mac (arm64)) give a different outcome on 65 of the 10,000 pairs (83 against 84 failed pairs).
- **Timed row against the offline decode of the same shots.** p = 0.003: relay-BP, timed row (one decoder through all 10,000 shots) 5 failed pairs, offline decode (a fresh decoder every 500 shots) 5; only the timed one fails 2, only the offline one 2, exact p = 1.00.
- **Timed row against the offline decode of the same shots.** p = 0.003: our decoder (neural tie-break), timed row (one decoder through all 10,000 shots) 3 failed pairs, offline decode (a fresh decoder every 500 shots) 4; only the timed one fails 1, only the offline one 2, exact p = 1.00.
- **Timed row against the offline decode of the same shots.** p = 0.003: the two PriorTilt runs disagree on 2 of 20,000 shots in our timed row and on 4 in the offline decode; 1 of those shots are in both. Both use the same seeds and the same test of agreement (the 12 predicted bits); PriorTilt draws random numbers from shot to shot, so the two modes give other draws and other disputed shots.
- **Timed row against the offline decode of the same shots.** p = 0.004: relay-BP, timed row (one decoder through all 10,000 shots) 83 failed pairs, offline decode (a fresh decoder every 500 shots) 90; only the timed one fails 32, only the offline one 39, exact p = 0.48.
- **Timed row against the offline decode of the same shots.** p = 0.004: our decoder (neural tie-break), timed row (one decoder through all 10,000 shots) 44 failed pairs, offline decode (a fresh decoder every 500 shots) 50; only the timed one fails 10, only the offline one 16, exact p = 0.33.
- **Timed row against the offline decode of the same shots.** p = 0.004: the two PriorTilt runs disagree on 90 of 20,000 shots in our timed row and on 75 in the offline decode; 51 of those shots are in both. Both use the same seeds and the same test of agreement (the 12 predicted bits); PriorTilt draws random numbers from shot to shot, so the two modes give other draws and other disputed shots.
- **Timed row against the offline decode of the same shots.** p = 0.005: relay-BP, timed row (one decoder through all 10,000 shots) 845 failed pairs, offline decode (a fresh decoder every 500 shots) 794; only the timed one fails 244, only the offline one 193, exact p = 0.017 (below 0.05; we read it as the run-to-run spread of a randomized decoder: relay-BP draws random strengths from a generator seeded once, so the two modes give each shot other draws, and the offline mode replays the same seed in every 500-shot chunk, which the paired test does not account for; a comparison that mixes the two modes is marked "mixed").
- **Timed row against the offline decode of the same shots.** p = 0.005: our decoder (neural tie-break), timed row (one decoder through all 10,000 shots) 479 failed pairs, offline decode (a fresh decoder every 500 shots) 490; only the timed one fails 106, only the offline one 117, exact p = 0.50.
- **Timed row against the offline decode of the same shots.** p = 0.005: the two PriorTilt runs disagree on 609 of 20,000 shots in our timed row and on 589 in the offline decode; 368 of those shots are in both. Both use the same seeds and the same test of agreement (the 12 predicted bits); PriorTilt draws random numbers from shot to shot, so the two modes give other draws and other disputed shots.
- **Rows run after the test results were seen** (their START in `test_rows.out` is after its TEST RUN END line; each is labelled "run after the test results were seen, at the team's request; an ablation, not our decoder" wherever it appears, and none is our decoder): ours, lowest weight at p = 0.003; ours, no network, 2 cores at p = 0.003; ours, lowest weight at p = 0.004; ours, no network, 2 cores at p = 0.004; ours, lowest weight at p = 0.005; ours, no network at p = 0.005; ours, no network, 2 cores at p = 0.005.
<!-- END table:checks -->

## J. Run after the test results were seen, at the team's request: ablations, not our decoder

These rows were run, or read from saved answers, after the test results of A and B had been seen. None of them is our
decoder, and none of them changes a frozen choice. `REPORT.md` section 4 says what they show.

### The hierarchy without the network, timed (server job ctl 120)

The rows themselves are in A (`test_p<p>.md`, `test_time.md`), each labelled; this is their comparison with our decoder.

<!-- BEGIN table:test_network_timed -->
**How much the network does, timed against timed.** Our decoder's timed row (the hierarchy with the neural tie-break) against the same hierarchy with a hand-written tie-break in place of the network, each a timed row of the team's evaluator on the same 10,000 test pairs, in one process ("1 core"; "2 cores": the two fast decodes side by side). Only the shots where the two fast decodes disagree can differ. A row marked "run after the test results were seen (an ablation)" is an ablation the team asked for after the test results were seen, not our decoder; the rows at p = 0.003 and 0.004 of the best hand-written rule on one core ran in the test run itself, among its pre-registered arms. Times are comparable to a few percent (`APPENDIX.md` A).

| p | Cores | Without the network: the tie-break | Failed pairs: with the network against without | Both fail | Only with the network fails / only without | Exact p | Mean time per pair, ms: with against without |
|---|---|---|---|---|---|---|---|
| 0.003 | 1 core | the best hand-written rule (lowest weight among the converged candidates) | 3 against 4 | 3 | 0 / 1 | 1.00 | 45.7 against 46.3 |
| 0.003 | 1 core | plain lowest weight [run after the test results were seen (an ablation)] | 3 against 4 | 3 | 0 / 1 | 1.00 | 45.7 against 46.9 |
| 0.003 | 2 cores | the best hand-written rule (lowest weight among the converged candidates) [run after the test results were seen (an ablation)] (each gives its one-core row's answers on every shot) | 3 against 4 | 3 | 0 / 1 | 1.00 | 28.9 against 29.0 |
| 0.004 | 1 core | the best hand-written rule (lowest weight among the converged candidates) | 44 against 52 | 36 | 8 / 16 | 0.15 | 216 against 218 |
| 0.004 | 1 core | plain lowest weight [run after the test results were seen (an ablation)] | 44 against 60 | 39 | 5 / 21 | 0.002 | 216 against 215 |
| 0.004 | 2 cores | the best hand-written rule (lowest weight among the converged candidates) [run after the test results were seen (an ablation)] (each gives its one-core row's answers on every shot) | 44 against 52 | 36 | 8 / 16 | 0.15 | 150 against 150 |
| 0.005 | 1 core | the best hand-written rule (lowest weight among the converged candidates) [run after the test results were seen (an ablation)] | 479 against 481 | 414 | 65 / 67 | 0.93 | 825 against 815 |
| 0.005 | 1 core | plain lowest weight [run after the test results were seen (an ablation)] | 479 against 553 | 434 | 45 / 119 | 7e-09 | 825 against 817 |
| 0.005 | 2 cores | the best hand-written rule (lowest weight among the converged candidates) [run after the test results were seen (an ablation)] (each gives its one-core row's answers on every shot) | 479 against 481 | 414 | 65 / 67 | 0.93 | 608 against 604 |
| 0.003, 0.004, 0.005 together | 1 core | the best hand-written rule (lowest weight among the converged candidates) | 526 against 537 |  | 73 / 84 | 0.42 |  |
| 0.003, 0.004, 0.005 together | 1 core | plain lowest weight | 526 against 617 |  | 50 / 141 | 3e-11 |  |
<!-- END table:test_network_timed -->

### TCBP-JOBS in place of relay-BP in our hierarchy (from the ensemble's saved answers)

<!-- BEGIN table:test_after_swap -->
**Run after the test results were seen, at the team's request; an ablation, not our decoder.** TCBP-JOBS in place of relay-BP in our hierarchy, as the team asked after the test results: no new rule is computed, the rows are rules the ensemble scored on the test shots among its pre-registered rules (`hierarchy.ensemble`; P1 and P2 PriorTilt under two seeds, T TCBP-JOBS, R relay-BP, all from the same saved answers). Rule d is the closest to that swap. Both comparisons stay in one decode mode: TCBP-JOBS alone (T) and our frozen decoder's offline decode (rule a: P1 where the two runs agree, else the network among R, P1, P2) come from the same files. Offline rows: no time.

| p | Rule (scored offline by the ensemble on the test shots) | Failures X, Z | Failed pairs | Rate per cycle ± error | Against TCBP-JOBS alone: failed pairs; both fail / only this rule / only TCBP-JOBS (exact p) | Against our frozen decoder's offline decode (rule a): failed pairs; both fail / only this rule / only rule a (exact p) |
|---|---|---|---|---|---|---|
| 0.003 | rule d: TCBP-JOBS in place of relay-BP (where at least two of the three, the two PriorTilt runs and TCBP-JOBS, agree, that answer; where all three differ, the network among them and relay-BP) | 1, 1 | 2 of 10,000 | 1.67e-05 ± 1.2e-05 | 2 against 3; 2 / 0 / 1 (p = 1.00) | 2 against 4; 2 / 0 / 2 (p = 0.50) |
| 0.003 | rule f, the ensemble's frozen rule, for reference (where the two PriorTilt runs and TCBP-JOBS all agree, that answer; elsewhere joint decoding where it converged, else TCBP-JOBS) | 1, 1 | 2 of 10,000 | 1.67e-05 ± 1.2e-05 | 2 against 3; 2 / 0 / 1 (p = 1.00) | 2 against 4; 2 / 0 / 2 (p = 0.50) |
| 0.004 | rule d: TCBP-JOBS in place of relay-BP (where at least two of the three, the two PriorTilt runs and TCBP-JOBS, agree, that answer; where all three differ, the network among them and relay-BP) | 19, 20 | 39 of 10,000 | 3.26e-04 ± 5.2e-05 | 39 against 41; 29 / 10 / 12 (p = 0.83) | 39 against 50; 35 / 4 / 15 (p = 0.019) |
| 0.004 | rule f, the ensemble's frozen rule, for reference (where the two PriorTilt runs and TCBP-JOBS all agree, that answer; elsewhere joint decoding where it converged, else TCBP-JOBS) | 8, 8 | 16 of 10,000 | 1.33e-04 ± 3.3e-05 | 16 against 41; 15 / 1 / 26 (p = 4e-07) | 16 against 50; 14 / 2 / 36 (p = 5e-09) |
| 0.005 | rule d: TCBP-JOBS in place of relay-BP (where at least two of the three, the two PriorTilt runs and TCBP-JOBS, agree, that answer; where all three differ, the network among them and relay-BP) | 204, 202 | 403 of 10,000 | 3.42e-03 ± 1.7e-04 | 403 against 360; 291 / 112 / 69 (p = 0.002) | 403 against 490; 378 / 25 / 112 (p = 2e-14) |
| 0.005 | rule f, the ensemble's frozen rule, for reference (where the two PriorTilt runs and TCBP-JOBS all agree, that answer; elsewhere joint decoding where it converged, else TCBP-JOBS) | 186, 177 | 360 of 10,000 | 3.05e-03 ± 1.6e-04 | 360 against 360; 360 / 0 / 0 (p = 1.00) | 360 against 490; 266 / 94 / 224 (p = 2e-13) |

**Where rule d and TCBP-JOBS part**, in shots (not pairs): per basis, shots only rule d fails / shots only TCBP-JOBS fails, split by what rule d did on the shot.

| p | On the shots where | X memory: only rule d / only TCBP-JOBS | Z memory: only rule d / only TCBP-JOBS |
|---|---|---|---|
| 0.003 | the two PriorTilt runs agree: rule d keeps their answer | 0 / 0 | 0 / 1 |
| 0.003 | all three differ: rule d takes the network's pick among P1, P2, T and R | 0 / 0 | 0 / 0 |
| 0.003 | the two PriorTilt runs disagree and TCBP-JOBS sides with one of them: rule d's answer is TCBP-JOBS's (checked on every shot) | 0 / 0 | 0 / 0 |
| 0.004 | the two PriorTilt runs agree: rule d keeps their answer | 4 / 6 | 4 / 4 |
| 0.004 | all three differ: rule d takes the network's pick among P1, P2, T and R | 1 / 0 | 1 / 2 |
| 0.004 | the two PriorTilt runs disagree and TCBP-JOBS sides with one of them: rule d's answer is TCBP-JOBS's (checked on every shot) | 0 / 0 | 0 / 0 |
| 0.005 | the two PriorTilt runs agree: rule d keeps their answer | 36 / 31 | 42 / 26 |
| 0.005 | all three differ: rule d takes the network's pick among P1, P2, T and R | 19 / 6 | 16 / 7 |
| 0.005 | the two PriorTilt runs disagree and TCBP-JOBS sides with one of them: rule d's answer is TCBP-JOBS's (checked on every shot) | 0 / 0 | 0 / 0 |

Source: `report/data/test/ensemble_p<p>/test_{X,Z}.npz` and `.json`.
<!-- END table:test_after_swap -->

## K. The learned predecoder on the test shots (`predecoder/`): secondary, one basis at a time

<!-- BEGIN table:operating_rule -->
**Which method at which noise level: an operating rule chosen after the test results were seen, from the measured rows; not a frozen decoder.** X memory only: failures of the 10,000 X test shots at each p, and the mean time per X shot. Our pipeline's time is the X half of its timed rows' own per-shot times on the test shots; the predecoder's is its clean one-thread sample on 1,000 validation X shots (exploration shots); each row says which. Unpaired, so a difference of a few shots is within the noise.

| p | Method | X test failures | Mean time per X shot, ms | Where the time comes from |
|---|---|---|---|---|
| 0.003 | our pipeline (the hierarchy with the network), one core | 1 of 10,000 | 23.7 | test shots, its timed row |
| 0.003 | our pipeline, its two fast decodes on two cores | 1 of 10,000 | 15.0 | test shots, its timed row |
| 0.003 | the predecoder (network and head, then relay-BP with the head's priors) | 2 of 10,000 | 52.9 | validation sample (exploration shots) |
| 0.003 | the predecoder with the fallback (TCBP-JOBS where its relay-BP did not converge; needs a GPU; the rule was designed after the first test result: post hoc at p = 0.005 and 0.004) | 1 of 10,000 | 52.9 | validation sample, plus TCBP-JOBS's GPU decode on the 0.1% of shots that fall back |
| 0.003 | TCBP-JOBS alone (NVIDIA, GPU), for reference | 1 of 10,000 | 2.0 | test shots; a GPU decode call alone |
| 0.003 | TCBP-JOBS with the head's priors (the TCBP version; designed after the test results were seen), a reference row, not one of the rule's options | 0 of 10,000 | not timed |  |
| 0.004 | our pipeline (the hierarchy with the network), one core | 18 of 10,000 | 104 | test shots, its timed row |
| 0.004 | our pipeline, its two fast decodes on two cores | 18 of 10,000 | 71.7 | test shots, its timed row |
| 0.004 | the predecoder (network and head, then relay-BP with the head's priors) | 15 of 10,000 | 94.0 | validation sample (exploration shots) |
| 0.004 | the predecoder with the fallback (TCBP-JOBS where its relay-BP did not converge; needs a GPU; the rule was designed after the first test result: post hoc at p = 0.005 and 0.004) | 10 of 10,000 | 94.0 | validation sample, plus TCBP-JOBS's GPU decode on the 0.3% of shots that fall back |
| 0.004 | TCBP-JOBS alone (NVIDIA, GPU), for reference | 20 of 10,000 | 3.9 | test shots; a GPU decode call alone |
| 0.004 | TCBP-JOBS with the head's priors (the TCBP version; designed after the test results were seen), a reference row, not one of the rule's options | 4 of 10,000 | not timed |  |
| 0.005 | our pipeline (the hierarchy with the network), one core | 245 of 10,000 | 420 | test shots, its timed row |
| 0.005 | our pipeline, its two fast decodes on two cores | 245 of 10,000 | 310 | test shots, its timed row |
| 0.005 | the predecoder (network and head, then relay-BP with the head's priors) | 151 of 10,000 | 212 | validation sample (exploration shots) |
| 0.005 | the predecoder with the fallback (TCBP-JOBS where its relay-BP did not converge; needs a GPU; the rule was designed after the first test result: post hoc at p = 0.005 and 0.004) | 107 of 10,000 | 212 | validation sample, plus TCBP-JOBS's GPU decode on the 0.8% of shots that fall back |
| 0.005 | TCBP-JOBS alone (NVIDIA, GPU), for reference | 186 of 10,000 | 8.7 | test shots; a GPU decode call alone |
| 0.005 | TCBP-JOBS with the head's priors (the TCBP version; designed after the test results were seen), a reference row, not one of the rule's options | 72 of 10,000 | 55.8 | an A100 slice, not a clean timing (network on validation shots, build and decode on the test shots) |
<!-- END table:operating_rule -->

<!-- BEGIN table:test_predecoder -->
**The learned predecoder on the test shots: secondary (chosen after the declared gate at p = 0.004 failed), one basis at a time (the pair score is its own section below), accuracy only; its times are not clean (a shared 12-worker pool on the team's PC), so iterations are shown instead.** Relay-BP at the kit's settings, given for each shot the priors that a second output (a "head") of our trained network computes from that shot's detectors; truth enters only the scoring. Per noise level and basis: the saved test file of that basis alone, so a row counts shots, not pairs, and cannot sit in the pair-scored tables. H against the others: both fail / only H fails / only the other fails (exact p), on the same shots. Counts from the run's own files (`predecoder.compare`, `predecoder.secondary`); TCBP-JOBS's and relay-BP's totals are checked against their rows in this report.

| p | Basis | Decoder | Failed shots | Share (95% interval) | H against it: both / only H / only it (exact p) | Mean BP iterations per shot; shots not converged |
|---|---|---|---|---|---|---|
| 0.005 | X | relay-BP with the head's priors, a decoder built per shot (H; ours) | 151 of 10,000 | 1.51% (1.29% to 1.77%) |  | 711; 90 not converged |
| 0.005 | X | relay-BP with the kit's priors, a decoder built per shot (the declared comparison; the kit) | 433 of 10,000 | 4.33% (3.95% to 4.75%) | 125 / 26 / 308 (p = 2e-62) | 2,436; 258 not converged |
| 0.005 | X | TCBP-JOBS (NVIDIA, GPU; its row of this report) | 186 of 10,000 | 1.86% (1.61% to 2.14%) | 90 / 61 / 96 (p = 0.006) |  |
| 0.005 | X | relay-BP, the evaluator's timed row (one decoder through all shots; its row of this report) | 432 of 10,000 | 4.32% (3.94% to 4.74%) | 124 / 27 / 308 (p = 1e-61) |  |
| 0.004 | X | relay-BP with the head's priors, a decoder built per shot (H; ours) [run after the p = 0.005 X test result (the registered extension)] | 15 of 10,000 | 0.15% (0.09% to 0.25%) |  | 167; 12 not converged |
| 0.004 | X | relay-BP with the kit's priors, a decoder built per shot (the declared comparison; the kit) | 39 of 10,000 | 0.39% (0.29% to 0.53%) | 12 / 3 / 27 (p = 8e-06) | 514; 13 not converged |
| 0.004 | X | TCBP-JOBS (NVIDIA, GPU; its row of this report) | 20 of 10,000 | 0.20% (0.13% to 0.31%) | 8 / 7 / 12 (p = 0.36) |  |
| 0.004 | X | relay-BP, the evaluator's timed row (one decoder through all shots; its row of this report) | 36 of 10,000 | 0.36% (0.26% to 0.50%) | 10 / 5 / 26 (p = 2e-04) |  |
| 0.003 | X | relay-BP with the head's priors, a decoder built per shot (H; ours) [run after the p = 0.005 X test result (the registered extension)] | 2 of 10,000 | 0.02% (0.01% to 0.07%) |  | 61; 1 not converged |
| 0.003 | X | relay-BP with the kit's priors, a decoder built per shot (the declared comparison; the kit) | 7 of 10,000 | 0.07% (0.03% to 0.14%) | 1 / 1 / 6 (p = 0.12) | 144; 2 not converged |
| 0.003 | X | TCBP-JOBS (NVIDIA, GPU; its row of this report) | 1 of 10,000 | 0.01% (0.00% to 0.06%) | 1 / 1 / 0 (p = 1.00) |  |
| 0.003 | X | relay-BP, the evaluator's timed row (one decoder through all shots; its row of this report) | 2 of 10,000 | 0.02% (0.01% to 0.07%) | 1 / 1 / 1 (p = 1.00) |  |
| 0.005 | Z | relay-BP with the head's priors, a decoder built per shot (H; ours) [run after the p = 0.005 X test result (the registered extension)] | 132 of 10,000 | 1.32% (1.11% to 1.56%) |  | 695; 73 not converged |
| 0.005 | Z | relay-BP with the kit's priors, a decoder built per shot (the declared comparison; the kit) | 398 of 10,000 | 3.98% (3.61% to 4.38%) | 108 / 24 / 290 (p = 4e-59) | 2,364; 230 not converged |
| 0.005 | Z | TCBP-JOBS (NVIDIA, GPU; its row of this report) | 177 of 10,000 | 1.77% (1.53% to 2.05%) | 76 / 56 / 101 (p = 4e-04) |  |
| 0.005 | Z | relay-BP, the evaluator's timed row (one decoder through all shots; its row of this report) | 427 of 10,000 | 4.27% (3.89% to 4.68%) | 117 / 15 / 310 (p = 8e-73) |  |

Source: `report/data/test/predecoder/test_run_p<p>_<basis>_{summary,secondary}.json` (copied from `predecoder/logs/`).

**Not held out: the fallback rule, designed after the first test result; frozen on the validation file before this application; the rule was prompted by the X test shots at p = 0.005, so its X number there is post hoc, not held out.** The rule: for every shot, the answer of relay-BP with the head's priors (H) where that relay-BP converged (its own success flag, no truth), else TCBP-JOBS's answer (T). No knob, no threshold (`predecoder/logs/fallback_rule_frozen.md`). It is not an equal of the rows above.

| p | Basis | What this number is | Shots that fall back to TCBP-JOBS | The rule fails (share, 95% interval) | H fails | TCBP-JOBS fails | Rule against TCBP-JOBS: both / only the rule / only TCBP-JOBS (exact p) | Rule against H: both / only the rule / only H (exact p) |
|---|---|---|---|---|---|---|---|---|
| 0.005 | X | post hoc: the rule was prompted by these same test shots, so this number is not held out | 90 of 10,000 | 107 (1.07%, 0.89% to 1.29%) | 151 | 186 | 91 / 16 / 95 (p = 8e-15) | 106 / 1 / 45 (p = 1e-12) |
| 0.005 | Z | its clean test: designed after the first test result, frozen (14:32 UTC) before the Z test file was opened (15:44 UTC); run after the p = 0.005 X test result (the registered extension) | 73 of 10,000 | 100 (1.00%, 0.82% to 1.21%) | 132 | 177 | 76 / 24 / 101 (p = 2e-12) | 100 / 0 / 32 (p = 5e-10) |
| 0.004 | X | post hoc: the rule was designed after the first test result, and this test file was opened (for H) before the rule was frozen; run after the p = 0.005 X test result (the registered extension) | 12 of 10,000 | 10 (0.10%, 0.05% to 0.18%) | 15 | 20 | 8 / 2 / 12 (p = 0.013) | 10 / 0 / 5 (p = 0.062) |
| 0.003 | X | the rule was designed after the first test result and frozen (14:32 UTC) before this test file was opened (14:56 UTC); run after the p = 0.005 X test result (the registered extension) | 1 of 10,000 | 1 (0.01%, 0.00% to 0.06%) | 2 | 1 | 1 / 0 / 0 (p = 1.00) | 1 / 0 / 1 (p = 1.00) |

Source: `report/data/test/predecoder/fallback_test_<basis>_p<p>.{json,log}` (`predecoder.fallback`).

**The pair score at p = 0.005: run after the p = 0.005 X test result (the registered extension); the fallback rule's half on X is post hoc (designed after the first test result), its half on Z clean; accuracy only.** The kit's score: a pair of an X shot and a Z shot fails when either fails. Not a held-out result, and not in the pair-scored tables of our decoder. TCBP-JOBS's and relay-BP's counts are checked against their rows in this report.

| Decoder | Failures X, Z | Failed pairs | Rate per cycle ± error | Against TCBP-JOBS: both / only this / only TCBP-JOBS (exact p) | Against relay-BP's timed row: both / only this / only relay-BP (exact p) |
|---|---|---|---|---|---|
| relay-BP with the head's priors in both bases, H: run after the p = 0.005 X test result (the registered extension) | 151, 132 | 281 of 10,000 | 2.37e-03 ± 1.4e-04 | 167 / 114 / 193 (p = 8e-06) | 242 / 39 / 603 (p = 6e-131) |
| the fallback rule in both bases (designed after the first test result: post hoc on its X half) | 107, 100 | 206 of 10,000 | 1.73e-03 ± 1.2e-04 | 167 / 39 / 193 (p = 1e-25) | 179 / 27 / 666 (p = 1e-160) |
| TCBP-JOBS (its row of this report) | 186, 177 | 360 of 10,000 | 3.05e-03 ± 1.6e-04 |  |  |
| relay-BP, the evaluator's timed row (its row of this report) | 432, 427 | 845 of 10,000 | 7.33e-03 ± 2.5e-04 |  |  |

Source: `report/data/test/predecoder/pair_test_p0.005.{json,log}` (`predecoder.pair`).
<!-- END table:test_predecoder -->

## L. TCBP-JOBS with the head's priors (`predecoder/`, formerly the branch `predecoder-tcbp`): designed after the test results were seen

<!-- BEGIN table:predecoder_tcbp -->
**TCBP-JOBS with the head's priors: designed after the test results were seen; formerly a separate branch (predecoder-tcbp).** TCBP-JOBS (NVIDIA's GPU decoder) given, for each shot, the priors our network's head computes from that shot's detectors, in place of the kit's fixed priors.

**Exploration shots, not test shots:** the 5,000 shots of `data/p0.005/validation_X.npz`. TH against each row: both fail / only TH / only that row (exact p), on the same shots.

| Decoder | Failed shots | Share (95% interval) | TH against it: both / only TH / only it (exact p) |
|---|---|---|---|
| TCBP-JOBS with the head's priors, per shot (TH) | 33 of 5,000 | 0.66% (0.47% to 0.93%) |  |
| TCBP-JOBS with the kit's priors, per shot, the same run (the like-for-like control) | 100 of 5,000 | 2.00% (1.65% to 2.43%) | 25 / 8 / 75 (p = 9e-15) |
| TCBP-JOBS with the kit's priors, the stored file (another fit) | 91 of 5,000 | 1.82% (1.48% to 2.23%) | 25 / 8 / 66 (p = 2e-12) |
| relay-BP with the head's priors (H, the predecoder) | 79 of 5,000 | 1.58% (1.27% to 1.96%) | 29 / 4 / 50 (p = 4e-11) |
| the frozen fallback rule (post hoc on test X at p = 0.005) | 50 of 5,000 | 1.00% (0.76% to 1.32%) | 28 / 5 / 22 (p = 0.002) |
| H where H converged, else TH (designed after the test results) | 44 of 5,000 | 0.88% (0.66% to 1.18%) | 29 / 4 / 15 (p = 0.019) |

**Test shots: designed after the test results were seen; formerly a separate branch (predecoder-tcbp).** One basis at a time (X until the Z run lands), accuracy only. Its knobs were fitted at p = 0.005 and applied unchanged at p = 0.004 and 0.003, with no validation there; on Z memory X's knobs were applied unchanged, with no Z fit (declared beforehand), after a validation gate on Z that passed. The declared pair is TH against TCBP-JOBS's stored test file; at p = 0.005 the same run's kit-priors arm is the like-for-like control.

| p | Basis | TH fails | TCBP-JOBS, its stored test file (its row in this report), fails | TH against it, the declared pair: both / only TH / only TCBP-JOBS (exact p) | TCBP-JOBS with the kit's priors in the same run (the like-for-like control) fails | TH against that control (exact p) |
|---|---|---|---|---|---|---|
| 0.005 | X | 72 of 10,000 | 186 | 50 / 22 / 136 (p = 3e-21) | 186 | 50 / 22 / 136 (p = 3e-21) |
| 0.004 | X | 4 of 10,000 | 20 | 2 / 2 / 18 (p = 4e-04) | not run (X at p = 0.005 only) |  |
| 0.003 | X | 0 of 10,000 | 1 | 0 / 0 / 1 (p = 1.00) | not run (X at p = 0.005 only) |  |
| 0.005 | Z | 61 of 10,000 | 177 | 41 / 20 / 136 (p = 2e-22) | not run (X at p = 0.005 only) |  |
| 0.005 | pairs | 132 of 10,000 pairs (1.11e-03 ± 9.6e-05 per cycle) | 360 (3.05e-03 ± 1.6e-04 per cycle) | 91 / 41 / 269 (p = 3e-42) |  |  |

**Its time per X shot: not a clean timing.** An A100 MIG slice, one process, one shot at a time, 1 CPU thread; server job 5 ran beside job 6, another TCBP-JOBS process on the same slice (`runner_gpu.out`, `runner_cpu.out`); the network's time is from validation shots. Per X shot, not per pair; ms: mean / median / 99th percentile. For scale: TCBP-JOBS alone takes 8.7 ms per X shot (a GPU decode call), relay-BP's timed row 760, and the relay-BP predecoder 212 (`predecoder_latency.md`).

| Part | Mean | Median | 99th percentile | Shots |
|---|---|---|---|---|
| the network and the head on the A100, 32-bit floats | 9.4 | 8.5 | 19.8 | the first 1,000 validation X shots at p = 0.005 (exploration shots; server job 4) |
| TCBP-JOBS built with the head's priors and run, per shot | 46.4 | 44.6 | 69.2 | the 10,000 test X shots at p = 0.005 (server job 5; build 42.6, decode 3.8 ms on average) |
| the two together | 55.8 |  |  |  |

Source: `report/data/exploration/predecoder_tcbp/` (branch `predecoder-tcbp` at `dad4fd50`; job 4 from `server-logs`) and `report/data/test/predecoder_tcbp/` (job 5 from `server-logs`).
<!-- END table:predecoder_tcbp -->
