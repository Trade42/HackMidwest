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
