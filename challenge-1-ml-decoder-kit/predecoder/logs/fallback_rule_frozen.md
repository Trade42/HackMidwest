# The fallback rule, frozen

Frozen 2026-11-22 14:32:15 UTC, before test_Z (data/p0.005/test_Z.npz) or any other test file of the
extension is opened, and before the rule is applied to any test file.

**The rule, for every shot and both bases: the answer of relay-BP with the head's priors (H) if that
relay-BP converged (its own success flag, recorded per shot by compare.py as converged_head); otherwise
TCBP-JOBS's answer (T).** No knob, no threshold. Tool: predecoder/fallback.py.

Origin: Sal's idea (14:21 UTC), prompted by the X test result at p = 0.005 (90 of H's 151 failures were
unconverged), so its X p = 0.005 test number is post hoc and is labelled "designed after the first test
result".

Validation (data/p0.005/validation_X.npz, 5,000 shots, seed 201; H from this PC's CPU in float32, the head
arm re-run at 1bf0e7b, logs/validation_X_p0.005_cpu_head_arm.log; T from origin/server-logs
files/tcbp_p0.005/validation_X.tcbp.npz, sha256 876c47c6...; logs/fallback_validation_X_p0.005.log):
H 79, T 91, the rule 50 failures; it falls back on 44 shots; the rule against T 38 / 12 / 53
(both, only the rule, only T), exact two-sided p = 2.8e-7; against H 50 / 0 / 29. Validation supports the
rule as worded; nothing in it was changed after this number.
