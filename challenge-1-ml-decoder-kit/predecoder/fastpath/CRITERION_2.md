# tcbp-fastpath, round 2: more checks, written before any of their numbers

Exploration, after the test results; a separate branch (`tcbp-fastpath`). Validation shots only. No test file is
opened. Written 2026-11-22 after dq-orch-6's messages (Sal, 18:30 UTC: "you can try more"); the commit time decides.
Round 1 (`CRITERION.md`, `README.md`) chose C2 (syndrome weight <= 115) at p = 0.005.

## Part A: p = 0.005, `data/p0.005/validation_X.npz` (5,000 shots, seed 201)

The same shape as round 1: a cheap check says "easy" -> answer F (TCBP-JOBS with the kit's priors, built once:
job 3's fresh arm); otherwise answer TH (the TCBP version: job 3's head arm). Each check is one number s per shot,
"easy" when s <= t, and every t that occurs is scanned.

Checks on the network's view (they need the network, 9.4 ms, on every shot). q is the shot's priors exactly as TH
got them (log-odds mix with alpha 0.4, clip at eps 1e-4, cap at 0.5: `predecoder/logs/fit_tcbp_X_p0.005.json`,
`priors.shot_priors`, `tcbp.capped`), pi the kit's priors, and m_j = |logit(q_j) - logit(pi_j)|, the move of column j:
- **C3, the total move:** s = sum over the 8,784 columns of m_j.
- **C4, the largest move:** s = max of m_j.
- **C5, the columns moved a lot:** s = the number of columns with m_j > 1 (a prior moved by more than a factor e in
  its odds).

Checks on F's own output (no network unless the shot is hard):
- **C6, F's decode time:** s = job 3's `seconds_decode_fresh` (a machine-dependent stand-in for its iteration count).
- **C7, F's correction weight:** s = job 3's `n_faults_fresh`.

Logits: luminaty's CPU float32 bytes behind job 3 (`orchestration/scratch/predecoder/runs/val_p0.005_cpu/logits/`,
sha256 of logits.npy `14eea54d...`); every row's hash is checked against job 3's `logits_hash_head`.

## The gate and the choice (unchanged from round 1)

- Gate: the rule adds no failure: "only rule" (the rule fails, TH alone does not) = 0 on all 5,000 shots.
- For each check, t* = the largest t that passes the gate.
- Choice: the lowest mean time per shot among: the checks at their t*, round 1's C2 (49.3 ms), and TH alone
  (51.1 ms). The p99 is reported beside it.

## The time model (A100 component timers, one process; separate timers)

- C3-C5: network on every shot (9.4 ms, its mean; per-shot values not stored), then F's decode on the easy shots
  (`seconds_decode_fresh`), or TH's priors + build + decode + free with H as CSR on the hard ones (tcbp-speed job 1,
  41.7 ms mean; per shot for shots 0-999, which give the p99).
- C6-C7: F's decode on every shot, then network + TH (9.4 + 41.7 ms) on the hard ones.

## Diagnostics, reported, not used for the choice

- **Half split:** t is picked on the even shots by the gate and counted on the odd shots, and the other way round:
  the added failures on the half it was not picked on.
- **Another model fit of F:** the added failures at t* when F's answers come from job 1's fresh arm (96 failures) or
  the stored TCBP-JOBS file (91). C6 and C7 use that run's own decode time and correction weight where it has them
  (job 1: both; the stored file: correction weight only).

## Part B: p = 0.004, `data/p0.004/validation_X.npz` (5,000 shots, seed 201), once the server job's files arrive

Server job: `orchestration/scratch/tcbp-fastpath/job1_validation_X_p0.004.sh.txt` (TH and F on these shots, the
p = 0.005 knobs unchanged, logits: dq-predecoder's CPU float32 bytes `783faa41...`). Checked first: the batch check
equals the fresh arm on every shot, and every head row's hash equals the logits file's.

- The same checks (C2 syndrome weight, C3-C7), the same gate (only rule = 0 on all 5,000), the same t*, the same
  choice (lowest mean among them and TH alone), the same diagnostics (half split only: there is no other F run at
  p = 0.004).
- The time model: network 9.4 ms (measured at p = 0.005; the network's time does not depend on p); TH = that job's
  per-shot `seconds_decode_head` + tcbp-speed's CSR priors + build + free means; F = that job's per-shot
  `seconds_decode_fresh`. The p99 is over all 5,000 shots, with the network and TH's build and free at their means.
