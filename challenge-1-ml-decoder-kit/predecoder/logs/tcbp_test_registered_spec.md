# The registered spec of the test runs: TCBP-JOBS with the head's priors

**Designed after the test results were seen; a separate branch** (`predecoder-tcbp`, merged nowhere unless Sal says so).
When this was designed, the team's earlier test runs on these same shots had already been seen. At p = 0.005, X:
H (relay-BP with the head's priors) 151, TCBP-JOBS 186, the fallback rule 107. The p = 0.004 and 0.003 X results of
dq-predecoder's extension had been run as well.

Written 2026-11-22 15:52 UTC, before any number of these runs exists. Sal's word (15:36 UTC, relayed by dq-orch-6):
"didnt i say we just use tcbp for everything that relay bp was used for".

**The validation result that allows it** (`logs/tcbp_validation_X_p0.005_score.log`):
- TH, TCBP-JOBS with the head's priors (log-odds alpha 0.4, eps 1e-4, capped at 0.5), fails **33** of 5,000.
- The stored TCBP-JOBS file fails 91. Paired: both 25, only TH 8, only TCBP-JOBS 66, exact two-sided p = 1.8e-12.
  **The kill line is passed.**
- Against the same run's kit-priors arm (100): 8 against 75, p = 9.1e-15.
- Against H (79): 4 against 50.
- Against the frozen fallback rule (50): 5 against 22, p = 0.0015.
- (b2) fails 44 (reported, not a candidate; §2b).

**Common to every run below:**
- **Code:** branch `predecoder-tcbp`, the commit that adds this file. The decode path of `predecoder/tcbp.py` is
  unchanged since `83563dbe`, which made the fit and the validation numbers. Later commits changed only the
  `score` and `logits-diff` commands, and `evaluate.py`'s new `--precision`, which no run here uses.
- **Knobs:** `predecoder/logs/fit_tcbp_X_p0.005.json`, sha256 `7697010678a369c77949a6a4233a3087824fa4d2753bcadc4869f2eaee91b0d8`
  (log-odds, alpha 0.4, eps 1e-4; priors capped at 0.5). Unchanged for every p and for Z: there is no other fit.
- **Arm:** (a) TH, one decoder per shot, `decode()`, TCBP-JOBS (cudaq-qec 0.9.0, preset `tcbp-jobs`).
  The model is fitted once per run; it is not repeatable across runs (validation: 91 / 96 / 100 for three fits with
  the kit's priors). Each run's model is the one its numbers come from.
- **Logits:** made by dq-predecoder's registered test runs on luminaty's CPU in float32, with head `38ab667e...` and
  trunk `d8d1c70f...`. They reach the server as these exact bytes, in parts with sha256 checks, reassembled and
  checked before use.
- **Server:** A100, 1 process, 1 CPU thread. A run refuses to start if its output folder exists. Each runs once.
- **Scoring here:** `python -m predecoder.tcbp score --final ...`, against the stored files named below. Every
  number is reported whatever it shows. Nothing is decoded a second time on a test file.

**Run 1 (primary), X at p = 0.005:**
- Shots: `data/p0.005/test_X.npz` (10,000, seed 301), sha256 `1b13521cddb2bf5e5d4c7f1e85a8ecaaa3c72a0b99754a3f76b4a5a4a912fc8b`.
- Logits: `scratch/predecoder/test-run/logits_test_p0.005/logits.npy`, sha256 `4d7401eb89ca2efcc46ea95987315798756d5f172d0591383ee8c6ab6384fd63`.
- Command: `decode --data data/p0.005/test_X.npz --logits <in> --settings predecoder/logs/fit_tcbp_X_p0.005.json
  --arms head fresh --batch-check --final`. The head arm runs first.
- **Declared pair:** TH against the stored TCBP-JOBS test file `files/clean/tcbp_p0.005/test_X.tcbp.npz`
  (`server-logs`, sha256 `ba3b9ab71d5737a5d39efcbe5fceaada8625e0d75404344f8ac4a4dae60ae467`, 186 failures), exact two-sided.
- Beside it:
  - against the same run's fresh arm;
  - against H (151), from `scratch/predecoder/test-run/compare_test_p0.005/per_shot.npz`, sha256
    `f8ef9f2fec2e695f2a35a07ba56df56b38bc2bab084eb03a8eae435384ac7b21`;
  - against the fallback rule (107);
  - (b2).

**Run 2 (extension), X at p = 0.004 and 0.003, head arm only:**
- p = 0.004: shots `data/p0.004/test_X.npz`, sha256 `53a7cee301600d36f3dad806a88f59ad6d84b681044b2548268c2ea9b5689e94`;
  logits `logits_test_p0.004/logits.npy`, sha256 `77b1b93c19a6492ad4f48452b71c646841c1e6ef8536d1489e41aba6200b27af`.
  Declared pair: against `files/clean/tcbp_p0.004/test_X.tcbp.npz`, sha256 `46d2c27b936b37d57d39a5c4ddea1036fce56ffec33ea435181c5fe299569165`.
- p = 0.003: shots `data/p0.003/test_X.npz`, sha256 `f7bc181399a0a8fd1324950c98a8419c0082d7c34940c03dfb34c9e2cefc1ff3`;
  logits `logits_test_p0.003/logits.npy`, sha256 `a3b8de744277e1b8cac1300717f54d8d963e472e6acd8ad2352097c7d5c68752`.
  Declared pair: against `files/clean/tcbp_p0.003/test_X.tcbp.npz`, sha256 `b22bc0ccc87543767219389a95033e5cfc20087f83f68543e06e9e84e0355107`.
- The knobs were fitted at p = 0.005 and are applied unchanged. No validation run at these p exists for TH.

**Run 3 (extension), Z at p = 0.005, one job with a gate inside it (dq-orch-6, 15:49 UTC):**
- The Z head's logits come from dq-predecoder (CPU float32). Their paths and sha256s are recorded in
  `logs/tcbp_declarations.md` §3 when they arrive, before the job.
- Knobs: X's frozen file, unchanged. There is no Z fit.
- Step 1: TH on `data/p0.005/validation_Z.npz`.
- Step 2 runs only if step 1 fails fewer shots than the stored `files/tcbp_p0.005/validation_Z.tcbp.npz`, paired,
  exact two-sided p < 0.05: then TH on `data/p0.005/test_Z.npz` (`--final`). Otherwise the job prints SKIPPED and
  does not open test_Z.
- Step 3, here: the PAIR score of TH (X from run 1, Z from step 2) against TCBP-JOBS's 360 failed pairs (stored test_X
  `ba3b9ab7...` + test_Z `23ae160e...`), paired over the 10,000 pairs, exact two-sided.
