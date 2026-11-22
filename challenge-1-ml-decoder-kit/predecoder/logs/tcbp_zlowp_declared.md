# Run 4 (an extension): TCBP-JOBS with the Z head's priors, Z at p = 0.004 and 0.003, and the kit's pair score there

**Designed after the test results were seen; no validation at these p (as for X); a separate branch** (`tcbp-zlowp`,
from `main` at `e6c2a5a0`, merged nowhere unless Sal says so). Brief: `orchestration/briefs/tcbp-zlowp.md` (dq-orch-6).
Written before any number of this run exists: no Z logits at these p have been made, and no TH decode of these
`test_Z` files has run. This file's commit time is the time that counts; no clock time is typed here.

**Why.** Run 3 (`logs/tcbp_test_registered_spec.md`) gave TH on Z only at p = 0.005, so the kit's pair score (shot k of X
with shot k of Z) exists only there. X at p = 0.004 and 0.003 was decoded by run 2. This run adds Z at those two p, so
the pair score exists at all three.

**What was known when this was written:**
- run 2, TH on X: p = 0.004, 4 of 10,000 against TCBP-JOBS's 20 (only TH 2, only T 18); p = 0.003, 0 against 1;
- run 3, TH on Z at p = 0.005: 61 against 177; the pair score there, 132 against 360;
- TCBP-JOBS alone, the stored test files (`report/REPORT.md`): 41 failed pairs at p = 0.004 and 3 at p = 0.003. Read
  again here from the files before this was written, as a check of the files: X 20 and Z 21 at p = 0.004, X 1 and Z 2
  at p = 0.003.

## What runs

**The network, here (luminaty CPU, float32).** The Z head on the Z trunk, as in run 3:
- head `083147d2ea941fbae1bc0c3096e06d3bb3a07ca46fd9360fd9304226e5825a61` (laptop-artifacts `a2c3d5df`, `predecoder_Z_s1/head.pt`, step 20,000);
- trunk `fad24e80ec4e2c5ea7b54717c6b309b244d90ee9fb50a60128da997840230700` (server-logs `ce700eaa`, `files/ckpt/cascade_Z_s1_final_step100000.pt`).

Once per file, from this branch's clean checkout, lanes A, B, C, D (locks in that order), `nice -n 15 taskset -c 0-15`:

    python -m predecoder.evaluate --head <head.pt> --trunk <trunk> --data data/p0.004/test_Z.npz \
        --out <scratch>/logits_test_Z_p0.004 --device cpu --threads 16 --time-shots 200 --final
    (the same with p0.003)

The logits' sha256s are added to this file, in a second commit, before the files leave this PC.

**Why 16 threads.** That is the thread count of run 3's test_Z logits. A pre-flight on the first 1,000 shots of
`data/p0.005/validation_Z.npz` (not held out) checked this venv and this checkout:
- at 4 threads (the registered validation_Z bytes' count) it gives those bytes on 1,000 of 1,000 rows;
- at 16 threads, 995 of 1,000 rows are equal, and the other 5 differ by at most 0.0156 (one float16 step);
- two runs at 16 threads are byte-identical.

So the thread count is part of the logits' identity, and a reviewer reproduces these bytes with `--threads 16`.

**The decode, on the server (A100, 1 process, 1 CPU thread; the orchestrator's job).** The head arm only, once per file,
with `--final`, from this branch's commit named in the job. The job refuses to start a p whose output folder exists.

    python -m predecoder.tcbp decode --data data/p0.004/test_Z.npz --logits <logits_test_Z_p0.004> \
        --settings predecoder/logs/fit_tcbp_X_p0.005.json --arms head --out <folder> --final
    (the same with p0.003)

- **Knobs:** X's frozen TCBP file `predecoder/logs/fit_tcbp_X_p0.005.json`, sha256
  `7697010678a369c77949a6a4233a3087824fa4d2753bcadc4869f2eaee91b0d8` (log-odds, alpha 0.4, eps 1e-4, priors capped at
  0.5), unchanged. There is no Z fit and no fit at these p.
- **Code:** `predecoder/tcbp.py`'s decode path is the one of runs 2 and 3. Since run 3's commit `76521b78` only its
  docstring changed.
- **Shots:**
  - p = 0.004: `data/p0.004/test_Z.npz`, 10,000 shots, seed 302, sha256 `b61c1325de8852385d19ee5ce4eb41ea939b668f83fb3e994ce0f590218b3d56`;
  - p = 0.003: `data/p0.003/test_Z.npz`, 10,000 shots, seed 302, sha256 `f5ac5b336f4155870fd8a6792ad61e21d5d76f04e05a9c09419879b6512e7f49`.
- **No validation and no gate.** Run 3 had a validation_Z gate at p = 0.005. Run 2 (X at these p) had none, and this
  run follows run 2.

## What is scored, here, whatever it shows

**1. The declared pair, per p:** TH on test_Z against the stored TCBP-JOBS test_Z file, paired shot by shot, exact
two-sided (`predecoder.decode.paired`), with `tcbp score --final --logits <this PC's folder>`. The decoded rows' logits
hashes must equal this PC's file on 10,000 of 10,000 shots.
- p = 0.004: `server-logs:files/clean/tcbp_p0.004/test_Z.tcbp.npz` (`5c594898`), sha256 `4f0d86d51a261c9f7f1c9fd8d6971078edcdb70bbd053d46fd531dad485c7664`;
- p = 0.003: `server-logs:files/clean/tcbp_p0.003/test_Z.tcbp.npz` (`0e2e9574`), sha256 `16b3ba9f7fc38ccdbed0115c29362768c285b85e77f7a017c4208dbb47870ac2`.

**2. The kit's pair score, per p:** `python -m predecoder.tcbp pair --final`. A pair fails when its X shot or its Z
shot fails. The rate per cycle is `1 - (1 - P)^(1/12)`, with its error from `decoders.per_cycle_rate`. TH's pairs are
compared with TCBP-JOBS's pairs, exact two-sided, over the 10,000 pairs. The X side is run 2's, unchanged:

| p | X shots | TH on X: run 2's per-shot file (server-logs) | TCBP-JOBS on X: stored |
|---|---|---|---|
| 0.004 | `data/p0.004/test_X.npz` `53a7cee3...` | `files/predecoder-tcbp/test_X_p0.004/per_shot.npz` (`9535d886`), sha256 `28eacf973e970f6dbbb58d5d0fb9535e754f2f46a4a1110c7cac7f53bb5f28db` | `files/clean/tcbp_p0.004/test_X.tcbp.npz`, `46d2c27b...` |
| 0.003 | `data/p0.003/test_X.npz` `f7bc1813...` | `files/predecoder-tcbp/test_X_p0.003/per_shot.npz` (`df876277`), sha256 `aa1284bd139ade8b1fd4ec5a578c2dbb489164a0a0bb80ee57873c4251178f0f` | `files/clean/tcbp_p0.003/test_X.tcbp.npz`, `b22bc0cc...` |

As a check of the inputs, the pair command must print X counts of 4 (p = 0.004) and 0 (p = 0.003) for TH, and 20 and
1 for TCBP-JOBS.

## Limits, stated in advance

- **No same-run kit-priors arm.** Run 4 decodes the head arm only, so every comparison is with a stored TCBP-JOBS file
  from another model fit. That carries TCBP-JOBS's run-to-run spread (91, 96 and 100 on the same validation shots).
- **Small counts.** TCBP-JOBS alone fails 21 and 2 of these Z shots. At p = 0.003 a significant difference is not
  expected, and a p of 0.05 or more is reported as no claim.
- **The knobs were fitted on X at p = 0.005.** They are applied here unchanged, to Z, at lower p, with no validation.
- Each test file is decoded once. If the job fails partway, that is reported, and no test file is decoded a second
  time without the orchestrator's word.
