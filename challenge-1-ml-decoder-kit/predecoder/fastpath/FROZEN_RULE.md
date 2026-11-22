# tcbp-fastpath: the frozen rule

Exploration, after the test results; a separate branch (`tcbp-fastpath`). Chosen on validation_X p = 0.005 only
(5,000 Stim shots, seed 201) by the criterion committed first (`CRITERION.md`, commit `b0a410e6`). No test file
was opened.

**The rule (C2, w = 115):** count the shot's detection events among its 936 basis detectors.
- At most 115: answer with TCBP-JOBS with the kit's priors, one decoder built once (no network).
- Otherwise: run the TCBP version (network + head, TCBP-JOBS built with the shot's priors) and answer with it.

Frozen at 2026-11-22 18:28 UTC (the commit time decides). Numbers: `result.json`, `result.log`, `README.md`.

**Caveat, stated with the choice:** 115 sits exactly at the edge. The lowest-weight validation shot that the cheap
answer gets wrong and the TCBP version gets right has weight 116. On new shots this w is expected to add a failure
now and then; a margin (for example w = 110: 2.1% of shots on the fast path, 50.1 ms mean) would be safer. The
saving is small either way: 51.1 -> 49.3 ms mean.
