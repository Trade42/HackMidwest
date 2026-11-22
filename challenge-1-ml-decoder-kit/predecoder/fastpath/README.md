# tcbp-fastpath: a fast path for easy shots

**Exploration, after the test results; a separate branch (`tcbp-fastpath`).** Validation shots only:
`data/p0.005/validation_X.npz`, 5,000 Stim circuit shots, seed 201, X memory. No test file was opened. Nothing was
decoded here: the tool reads saved per-shot files.

**The question.** The TCBP version (TH: network + head, then TCBP-JOBS built with the shot's own priors) costs about
51 ms per X shot on the A100. TCBP-JOBS with the kit's priors, built once (F), costs about 13 ms. Can a cheap check
send the easy shots to F, keep TH's accuracy, and lower the mean time?

**The answer, in short: no useful fast path from these two checks (a negative result).** With the gate fixed in
advance (the rule may add no failure TH does not have), the best rule found sends 3.9% of the shots to F and lowers
the mean from 51.1 ms to 49.3 ms (1.8 ms, 3.5%, inside the timers' noise), with its threshold on the edge of these
shots. Its failures are TH's own 33, shot for shot. C1 (agreement with PriorTilt) adds 19 failures. Every check that
sends more shots to F adds failures: F is wrong on 75 shots where TH is right, and those shots are not easy to spot
cheaply. This is not worth a pipeline change.

## What was fixed first

`CRITERION.md`, committed (`b0a410e6`, 18:25:59 UTC by the commit) before any rule's number: the two checks, the gate
(no added failure: "only rule" = 0, since TH's own run-to-run spread is 0 changed outcomes on the 1,000 shots decoded
again), the choice (lowest mean time among the rules that pass), and the time model.

**Deviations after the first number** (both in `fastpath.py`, commit `a759d231`; `CRITERION.md` is unchanged):
- **The time model for C2.** `CRITERION.md` says every shot runs the cheap answer and the slow path "adds" to it. The
  first run used that model for C2 too: C2 (w = 115) came out at **61.9 ms mean, 102.9 ms p99**, slower than TH alone
  (51.1 / 63.1), so under the committed model no rule both passes the gate and is faster than TH. That model is wrong
  for C2: the syndrome weight is known before any decode, so F need not run on the shots that go to TH. With F run
  only on the easy shots, C2 is **49.3 ms mean, 63.1 ms p99**. C1 keeps the committed model (its check needs F's
  answer on every shot). The first run's log was overwritten; its numbers are the ones above.
- **A fast path must also be faster than TH alone** (`fastpath.py`, "candidates"). Under the corrected time model it
  changes nothing.

## Inputs (all X, validation, p = 0.005)

| what | file | branch, commit |
|---|---|---|
| TH (head arm) and F (fresh arm and its batch check) of job 3 | `files/predecoder-tcbp/validation_X_p0.005_per_shot.npz` | `server-logs` `9a99005a` |
| PriorTilt P2 | `priortilt/results_p005main/selector_validation_X.npz` | `laptop-artifacts` `b4ef168a` |
| TH's time with H as CSR, one timer (priors + build + decode + free), shots 0-999 | `files/tcbp-speed/check_per_shot.npz` | `server-logs` `9a99005a` |
| the network + head's time, A100 float32, one shot at a time | `files/predecoder-tcbp/gpu_network/meta_fp32.json` (mean 9.4, p99 19.8 ms; no per-shot values) | `server-logs` |

The tool checks that the truths agree across files, that P2's data hash is the validation file's, that F built once
equals F built per shot on all 5,000 shots, and that the CSR answers equal the dense ones.

## Command

Run from `challenge-1-ml-decoder-kit/`, with the kit's environment (numpy only):

    python predecoder/fastpath/fastpath.py --data data/p0.005/validation_X.npz \
        --job3 <server-logs: files/predecoder-tcbp/validation_X_p0.005_per_shot.npz> \
        --priortilt <laptop-artifacts: priortilt/results_p005main/selector_validation_X.npz> \
        --speed <server-logs: files/tcbp-speed/check_per_shot.npz> --out predecoder/fastpath/result.json

Output: `result.json` (every row, the full scan of w, the input hashes) and `result.log`. Ran on luminaty in under a
second.

## Results (validation_X p = 0.005, 5,000 shots; paired against TH on the same shots)

Times: A100 component timers, one process (F: job 3's `seconds_decode_fresh`; TH: tcbp-speed job 1, CSR; network:
job 4's mean). P2's time is the laptop CPU's (labelled). The p99 is over shots 0-999, the shots with TH's per-shot
CSR time, with the network at its mean (its per-shot times were not stored).

| rule | shots on the fast path | failures, share [95%] | both / only rule / only TH | exact p | mean ms | p99 ms | gate |
|---|---|---|---|---|---|---|---|
| TH alone (the TCBP version) | 0% | 33, 0.0066 [0.0047, 0.0093] | - | - | 51.1 | 63.1 | reference |
| F alone (TCBP-JOBS, kit priors, built once) | 100% | 100, 0.0200 [0.0165, 0.0243] | 25 / 75 / 8 | 9.1e-15 | 12.8 | 45.3 | fails |
| C1: F and P2 agree on the 12 bits -> F, else TH | 97.0% | 48, 0.0096 [0.0073, 0.0127] | 29 / 19 / 4 | 0.0026 | 191.8 (P2 on the laptop CPU dominates) | 2,714 | fails |
| **C2: syndrome weight <= 115 -> F, else TH (chosen)** | **3.9%** | **33, 0.0066 [0.0047, 0.0093]** | **33 / 0 / 0** | 1 | **49.3** | **63.1** | **passes** |

C2's scan (`result.json`, `c2_scan`), the trade-off once the gate is dropped:

| w | on the fast path | failures | only rule / only TH | mean ms |
|---|---|---|---|---|
| 110 | 2.1% | 33 | 0 / 0 | 50.1 |
| 115 | 3.9% | 33 | 0 / 0 | 49.3 |
| 120 | 6.9% | 35 | 2 / 0 | 48.0 |
| 130 | 18.3% | 37 | 5 / 1 | 43.1 |
| 140 | 36.6% | 40 | 9 / 2 | 35.6 |
| 150 | 58.5% | 51 | 21 / 3 | 27.1 |
| 160 | 78.0% | 59 | 31 / 5 | 19.8 |

How to read it:
- The syndrome weight runs from 84 to 220 (median 147). F's extra failures start at weight 116, so only the
  lightest 3.9% of shots are safe for F. w = 115 is tuned to the edge of these shots; on new shots it would add a
  failure now and then (`FROZEN_RULE.md`).
- **Does "0 added failures at w = 115" survive another model fit of F?** F's answers vary from run to run (91 / 96 /
  100 failures with the kit's priors). A diagnostic with the two other kit-priors runs on the same shots, w not
  re-picked: job 1's fresh arm (96 failures) and the stored TCBP-JOBS file (91) each add **0** failures at w <= 115
  (and 0 at w <= 110); their lowest-weight extra failures are at weights 116 and 119. That is the only
  out-of-sample-like evidence here; new shots may still put one at or below 115.
- C1 fails for two reasons: F and P2 agree on a wrong answer on 19 shots where TH is right, and P2 is slow on its
  hard shots (189.5 ms mean on the laptop CPU), so the check costs more than TH.
- The bulk of TH's time is building TCBP-JOBS with the shot's priors (34 ms of 51). A cheaper path for most shots
  needs either a decoder whose priors can change after it is built, or a check that finds F's 75 extra failures;
  neither syndrome weight nor agreement with PriorTilt does.

## Not done, and what is left

- No test file opened; no new decode.
- Truth-free signals not tried (deadline): F's own decode time (`seconds_decode_fresh`, a stand-in for its iteration
  count) and its correction's weight (`n_faults_fresh`) against the syndrome weight, both already in job 3's file;
  and the distance between the head's priors and the kit's. That last one costs the 9.4 ms network, but skipping the
  34 ms build is where any real saving would come from. TCBP-JOBS's `converged` flag is False on most shots
  (`README_tcbp.md`), so it was not used.
- The p99 of the whole pipeline was not measured as one timer; the network's p99 (19.8 ms) is not inside it.
- Z memory and other p: not looked at.

## Round 2 at p = 0.005: five more checks (`CRITERION_2.md`, Part A)

Exploration, after the test results; a separate branch. The criterion was committed (`17f86351`, 18:34:29 UTC by the
commit) before any of these numbers. The same validation shots, the same TH and F (job 3), the same gate (0 added
failures), the same time model (A100 timers; TH's CSR time 41.7 ms; network 9.4 ms at its mean).

    PYTHONPATH=. python predecoder/fastpath/fastpath2.py --data data/p0.005/validation_X.npz \
        --arms <server-logs: files/predecoder-tcbp/validation_X_p0.005_per_shot.npz> \
        --logits <orchestration/scratch/predecoder/runs/val_p0.005_cpu/logits> --settings predecoder/logs/fit_tcbp_X_p0.005.json \
        --speed <server-logs: files/tcbp-speed/check_per_shot.npz> --time-model csr_total \
        --alt job1=<files/predecoder-tcbp/validation_X_p0.005_fresh_per_shot.npz>:fresh stored=<files/tcbp_p0.005/validation_X.tcbp.npz>:T \
        --out predecoder/fastpath/result2_p0.005.json

It checks that every logits row's hash equals job 3's head arm (5,000 of 5,000) and that F built once equals F
built per shot. Output: `result2_p0.005.json`, `result2_p0.005.log`.

| check, at its t* (largest t with 0 added failures) | fast path | failures (both / only rule / only TH) | mean ms | p99 ms |
|---|---|---|---|---|
| TH alone | 0% | 33 | 51.1 | 63.1 |
| C2 syndrome weight <= 115 (round 1's choice) | 3.9% | 33 (33 / 0 / 0) | 49.3 | 63.1 |
| C3 total prior move | no safe t (the lowest-move shot is an F-only failure) | 33 | 51.1 | 63.1 |
| C4 largest prior move <= 4.539 | 0.2% | 33 (33 / 0 / 0) | 51.0 | 63.1 |
| C5 columns moved by more than 1 | no safe t | 33 | 51.1 | 63.1 |
| **C6 F's decode time <= 5.51 ms (chosen by the criterion)** | **40.1%** | **33 (33 / 0 / 0)** | **43.4** | **102.9** |
| C7 F's correction weight <= 48 | 12.9% | 33 (33 / 0 / 0) | 57.3 | 102.9 |

**By the declared criterion, C6 is chosen: 43.4 ms mean, against 51.1 for TH alone (15% lower), with TH's 33
failures shot for shot.** But it is weak, and it should not go into the pipeline as it stands:
- **Its p99 is worse: 102.9 ms against 63.1.** F runs first on every shot, so a hard shot pays F's slow decode
  (F's p99 is 76.7 ms) and then the whole TCBP version.
  - Not declared, a diagnostic: if F's decode could be stopped at 5.51 ms, the rule would cost 35.2 ms mean and
    68.3 ms p99. Whether cudaq-qec 0.9.0 can stop a decode was not tested.
- **F's times were not measured as a pipeline would run it.** They come from job 3, where F was built fresh for each
  shot and the GPU was shared with the knob fit's last rows. Neither the 12.8 ms mean nor the 5.51 ms threshold was
  measured on a decoder built once and running alone.
- **The threshold is a time in ms, measured on one run.** In job 1, F's decodes on the same shots were faster
  (median 5.4 ms against job 3's 7.2; rank correlation 0.78). At the same 5.51 ms, job 1 would send 51% of the shots
  to F and add 1 failure. An iteration count would not depend on the machine or the load, but no file stores one.
- **The half split:** C6's t picked on the odd shots adds 1 failure on the even shots. C2 picked on the odd shots
  (t = 127) adds 3 on the even shots; C4 picked on the even shots adds 2; C7 picked on the odd shots adds 1.
  Every threshold sits on the edge of these 5,000 shots.
- **Another model fit of F** at each t*: job 1's fresh arm adds 1 failure under C6 (with job 1's own times) and 0
  under C2, C4 and C7. The stored file adds 0 under C2, C4 and C7; it has no per-shot times, so C6 is unmeasured.
- C3 and C5 fail as declared: most of the head's moves are downward, with the head surer than the kit that a column
  did not fire. C5's "moved by more than 1" covers 6,712 to 8,492 of the 8,784 columns. Counting only upward moves
  (the faults the head suspects) was not declared and was not tried.

Round 2 in one line: the only check with a real saving is F's own decode time. It saves 15% of the mean and costs a
worse p99, with a threshold in ms that does not carry from one run to another.

## Round 3 at p = 0.005: the head's own confidence (`CRITERION_3.md`)

Exploration, after the test results; a separate branch. Declared (`cbfbfbcd`, 18:40:33 UTC by the commit) before
any number about these features, not even their range. **This is the third look at the same 5,000 shots**, so a pass
here is weaker evidence than a pass in round 1. Command: round 2's, with `--round3 --out predecoder/fastpath/result3_p0.005.json`
(output `result3_p0.005.json`, `result3_p0.005.log`).

| check, at its t* | fast path | failures (both / only rule / only TH) | mean ms | p99 ms | 95% bound on added per fast shot |
|---|---|---|---|---|---|
| TH alone | 0% | 33 | 51.1 | 63.1 | (TH's own rate 0.0066) |
| C6 F's decode time <= 5.51 ms (round 2's choice) | 40.1% | 33 (33 / 0 / 0) | 43.4 | 102.9 | 0.0015 |
| C8 upward prior move <= 225.3 | 23.5% | 33 (33 / 0 / 0) | 42.8 | 63.1 | 0.0026 |
| **C9 the head's entropy <= 69.1 bits (chosen)** | **35.4%** | **33 (33 / 0 / 0)** | **38.8** | **63.1** | **0.0017** |

- **C9 is chosen by the criterion over every check of rounds 1 to 3: 38.8 ms mean, against 51.1 ms for TH alone
  (24% lower). Its p99 is TH's own, 63.1 ms, and its failures are TH's 33, shot for shot.** C9 is the sum over the
  8,784 columns of the binary entropy of the head's raw probability. A shot the network is sure about, either way,
  goes to F after the network; the rest go to the TCBP version. The network has already run, so a hard shot pays no
  more than TH alone.
- **The diagnostics:**
  - the half split: t picked on the even shots (70.6) adds 1 failure on the odd shots; picked on the odd shots
    (69.1), it adds 0 on the even ones;
  - the other model fits of F: job 1's fresh arm and the stored file both add 0 at t*.
  So on new shots expect C9 to add about 0 to 2 failures per 5,000. The threshold is on the edge, like every other.
- C8 (upward moves only) also passes, at 42.8 ms. C9 sends more shots to the fast path for the same cost per shot.
- The 95% bound for 0 added failures among the fast shots (3 / fast shots, for C9 3 / 1,771 = 0.0017) is below TH's
  own failure rate (0.0066), but not far below it.
