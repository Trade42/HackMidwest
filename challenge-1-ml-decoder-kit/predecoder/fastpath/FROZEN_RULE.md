# tcbp-fastpath: the frozen rule

Exploration, after the test results; a separate branch (`tcbp-fastpath`). Chosen on validation_X p = 0.005 only
(5,000 Stim shots, seed 201) by the criterion committed first (`CRITERION.md`, commit `b0a410e6`, 18:25:59 UTC). No test file
was opened.

**The rule (C2, w = 115):** count the shot's detection events among its 936 basis detectors.
- At most 115: answer with TCBP-JOBS with the kit's priors, one decoder built once (no network).
- Otherwise: run the TCBP version (network + head, TCBP-JOBS built with the shot's priors) and answer with it.

Frozen in commit `a759d231` (18:27:54 UTC).

**Deviation after the first number** (`README.md` has it in full): under `CRITERION.md`'s time model (the cheap
answer runs on every shot) C2 came out at 61.9 ms mean, 102.9 ms p99, slower than the TCBP version alone (51.1 /
63.1), so no rule both passed and was faster. Because the weight is known before any decode, C2 was re-timed with
the cheap answer only on its easy shots: 49.3 ms mean, 63.1 ms p99. This rule is chosen under the corrected model.

**A negative result:** the saving is 1.8 ms (3.5%), inside the timers' noise. Not worth a pipeline change. Numbers: `result.json`, `result.log`, `README.md`.

**Caveat, stated with the choice:** 115 sits exactly at the edge. The lowest-weight validation shot that the cheap
answer gets wrong and the TCBP version gets right has weight 116. On new shots this w is expected to add a failure
now and then; a margin (for example w = 110: 2.1% of shots on the fast path, 50.1 ms mean) would be safer. The
saving is small either way: 51.1 -> 49.3 ms mean.

Diagnostic, w not re-picked: with the two other kit-priors runs of TCBP-JOBS on the same shots (job 1's fresh arm,
96 failures; the stored file, 91) as the cheap answer, the rule also adds 0 failures at w <= 115 (their lowest-weight
extra failures: weights 116 and 119).
