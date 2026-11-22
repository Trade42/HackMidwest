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

**Test shots: designed after the test results were seen; formerly a separate branch (predecoder-tcbp).** Decoded one basis at a time, accuracy only; the rows "pairs" score both bases as the kit does (a pair fails when its X or its Z shot fails). Its knobs were fitted on X at p = 0.005 and applied unchanged to Z (no Z fit, declared beforehand) and at p = 0.004 and 0.003: **no validation at these p** (X in run 2, Z in run 4, `predecoder/logs/tcbp_zlowp_declared.md`). Z at p = 0.005 first passed a validation gate on Z. The declared pair is TH against TCBP-JOBS's stored test file; only X at p = 0.005 has a same-run kit-priors arm, the like-for-like control.

| p | Basis | TH fails | TCBP-JOBS, its stored test file (its row in this report), fails | TH against it, the declared pair: both / only TH / only TCBP-JOBS (exact p) | TCBP-JOBS with the kit's priors in the same run (the like-for-like control) fails | TH against that control (exact p) |
|---|---|---|---|---|---|---|
| 0.005 | X | 72 of 10,000 | 186 | 50 / 22 / 136 (p = 3e-21) | 186 | 50 / 22 / 136 (p = 3e-21) |
| 0.004 | X | 4 of 10,000 | 20 | 2 / 2 / 18 (p = 4e-04) | not run (X at p = 0.005 only) |  |
| 0.003 | X | 0 of 10,000 | 1 | 0 / 0 / 1 (p = 1.00) | not run (X at p = 0.005 only) |  |
| 0.005 | Z | 61 of 10,000 | 177 | 41 / 20 / 136 (p = 2e-22) | not run (X at p = 0.005 only) |  |
| 0.004 | Z | 6 of 10,000 | 21 | 4 / 2 / 17 (p = 7e-04) | not run (X at p = 0.005 only) |  |
| 0.003 | Z | 1 of 10,000 | 2 | 1 / 0 / 1 (p = 1.00) | not run (X at p = 0.005 only) |  |
| 0.005 | pairs | 132 of 10,000 pairs (1.11e-03 ± 9.6e-05 per cycle) | 360 (3.05e-03 ± 1.6e-04 per cycle) | 91 / 41 / 269 (p = 3e-42) |  |  |
| 0.004 | pairs | 10 of 10,000 pairs (8.34e-05 ± 2.6e-05 per cycle) | 41 (3.42e-04 ± 5.3e-05 per cycle) | 6 / 4 / 35 (p = 3e-07) |  |  |
| 0.003 | pairs | 1 of 10,000 pairs (8.33e-06 ± 8.3e-06 per cycle) | 3 (2.50e-05 ± 1.4e-05 per cycle) | 1 / 0 / 2 (p = 0.50: no claim) |  |  |

**Its time per X shot: not a clean timing.** An A100 MIG slice, one process, one shot at a time, 1 CPU thread; server job 5 ran beside job 6, another TCBP-JOBS process on the same slice (`runner_gpu.out`, `runner_cpu.out`); the network's time is from validation shots. Per X shot, not per pair; ms: mean / median / 99th percentile. For scale: TCBP-JOBS alone takes 8.7 ms per X shot (a GPU decode call), relay-BP's timed row 760, and the relay-BP predecoder 212 (`predecoder_latency.md`).

| Part | Mean | Median | 99th percentile | Shots |
|---|---|---|---|---|
| the network and the head on the A100, 32-bit floats | 9.4 | 8.5 | 19.8 | the first 1,000 validation X shots at p = 0.005 (exploration shots; server job 4) |
| TCBP-JOBS built with the head's priors and run, per shot | 46.4 | 44.6 | 69.2 | the 10,000 test X shots at p = 0.005 (server job 5; build 42.6, decode 3.8 ms on average) |
| the two together | 55.8 |  |  |  |

**Its time per X shot after the speed work (branch `tcbp-speed`): exploration shots, each part timed alone, not one end-to-end timer.** The first 1,000 validation X shots at p = 0.005, the A100 MIG slice, one process, 1 CPU thread; ms, mean per X shot. Both columns were measured the same way on the same shots, so they compare like for like; the registered time above is another measurement, on other shots. The test numbers were made with the registered path and were not rerun. The decoded CUDA graph's logits give the same 12 bits and faults as the validation run's head arm (`ctl/130`) on every shot, and so do job 1's dense and CSR arms.

| Part | The registered path: eager network, H dense | With the speed work: CUDA graph, H as a scipy CSR matrix | Same answers, and notes |
|---|---|---|---|
| the network and head, float32, one shot at a time (job 3) | 8.3 | 1.7 | logits equal eager's bit for bit on 1,000 of 1,000 rows |
| TCBP-JOBS built with the shot's priors (job 1, the arms taking turns) | 41.2 | 34.0 | the 12 bits, the faults and the whole corrections equal on 1,000 of 1,000 shots |
| its decode (job 1) | 2.7 | 2.7 |  |
| **the sum, network + build + decode** | **52.2** | **38.5** | separate timers, not one end-to-end timer; the sum of the unrounded means |
| freeing the decoder, before the next shot (not in the sum; job 1) | 4.6 | 4.8 | medians 4.6 and 4.6 ms; the largest CSR free took 228 ms |

A fast path for easy shots (`predecoder/fastpath/`, branch `tcbp-fastpath`; exploration after the test results, on the 5,000 validation X shots at p = 0.005; no test file opened): a check sends a shot to TCBP-JOBS with the kit's priors, built once, and the rest to the TCBP version, with the rule that it add no failure. On its own time model (the TCBP version alone 51.1 ms: one timer over the priors, the CSR build, the decode and the free, plus the eager network's 9.4 ms; not comparable with the speed work's sums), the syndrome weight (round 1) saves 1.8 ms (49.3), and TCBP-JOBS's own decode time (round 2) saves more (43.4 ms mean) but worsens the 99th percentile (103 against 63.1 ms), with a threshold in ms that does not carry from one run to another. The head's entropy (C9, round 3) sends 35.4% of the shots to the fast path with 0 added failures, 38.8 ms mean and the same 99th percentile (63.1 ms): a third look at the same validation shots; not confirmed on fresh shots (its one held-out check, declared in `predecoder/fastpath/CRITERION_4.md`, is pending). At p = 0.004 (its validation X shots; only 12 shots where the cheap answer alone fails, so little power): C9's frozen threshold adds 2 failures, so it is a rule for p = 0.005 only; the criterion picks the syndrome weight there (64.2% fast, 24.2 ms mean against 53.8 for the TCBP version alone, 0 added), not frozen (picked on one half of the shots, it adds 2 on the other).

The TCBP pipeline with the head retrained on the neutral-atom noise (`neutral_atom_noise/`, task `tcbp-retrain`; exploration after the main test results, declared before any number; its own simulated Stim shots at factor 0.8, not the test files; run 1, the first 1,000 of 4,000 shots): on the matched base it was trained with, the retrained head makes TCBP-JOBS worse, 27 failures against 8 for the matched priors alone (only the retrained head 20, only the matched priors 1, sign test p = 2e-05). Against the kit-trained head on the generic base (119) it fails fewer, but those two differ in the head and the base both, so that is not a gain from retraining; the head-alone comparison (the retrained head against the kit head on the same matched base) is in run 2, not landed yet.

Source: `report/data/exploration/predecoder_tcbp/` (branch `predecoder-tcbp` at `dad4fd50`; job 4 from `server-logs`; `speed/` from branch `tcbp-speed` and `fastpath/` from branch `tcbp-fastpath`, both merged into `predecoder/`; `retrain/` from branch `tcbp-retrain`, merged into `neutral_atom_noise/`) and `report/data/test/predecoder_tcbp/` (job 5 from `server-logs`; Z and the pair score at p = 0.004 and 0.003 from branch `tcbp-zlowp`, merged into `predecoder/`).
