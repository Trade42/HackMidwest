# predecoder-tcbp: handoff

Branch `predecoder-tcbp`, from `origin/predecoder` at `e96cc020`. It is Sal's separate branch; on his word at 15:47 UTC it
went to main by PR #9 at 16:28:53 UTC (dq-orch-6). Full write-up:
`challenge-1-ml-decoder-kit/predecoder/README_tcbp.md`. Declarations: `predecoder/logs/tcbp_declarations.md`. Registered
test spec: `predecoder/logs/tcbp_test_registered_spec.md`.

## What changed

- `predecoder/tcbp.py` (new). TCBP-JOBS with one shot's own priors from the head: one model per run, one decoder
  per shot. Subcommands:
  - `decode`: arms `head` and `fresh`, plus `--batch-check`;
  - `fit`: the knob fit on train subsets;
  - `score`: the paired tests against the stored TCBP-JOBS files, H and the fallbacks;
  - `pair`: the kit's X+Z pair score;
  - `logits-diff`;
  - `probe`.
  No torch, so it runs in the teammates' TCBP env.
- Priors are capped at 0.5, because TCBP-JOBS refuses more.
- `predecoder/test_tcbp.py` (new): 7 tests with the relay stand-in.
- `predecoder/README_tcbp.md` (new): the full write-up. `predecoder/README.md`: a pointer to it with the headline
  numbers.
- `predecoder/evaluate.py`: `--precision fp32`, the network on a GPU in float32 with no autocast and no TF32. The
  default is unchanged.
- `predecoder/logs/`:
  - the fit (`fit_tcbp_X_p0.005.json`, frozen; `fit_tcbp_X_p0.005_full.json`);
  - every server job's log (`tcbp_job*`), the scores, the declarations and the spec.

## Numbers (A100 server, 1 process; X memory unless said)

- Validation X p = 0.005 (5,000 shots, seed 201):
  - TH (TCBP-JOBS with the head's priors) 33, against TCBP-JOBS alone 91 (stored): 8 against 66, p = 1.8e-12;
  - against the same run's kit-priors arm, 100: 8 against 75;
  - against H (relay-BP with the head's priors) 79, and the fallback rule 50.
- Time per shot on the A100: network fp32 9.4 ms + build 41.8 ms + decode 4.4 ms, about 56 ms mean. The GPU logits
  change no outcome on 1,000 validation shots.
- Test (registered at `dad4fd50`; "designed after the test results were seen; a separate branch"):
  - X p = 0.005: TH 72 of 10,000 against TCBP-JOBS alone 186 (22 against 136, p = 2.9e-21). The same-run kit-priors
    arm fails the same 186 shots. Also H 151 and the fallback rule 107.
  - X p = 0.004: TH 4 against 20 (p = 0.0004).
  - X p = 0.003: 0 against 1 (no claim).
  - Z p = 0.005 (X's knobs): validation_Z 35 against 97 (gate GO); test_Z TH 61 against 177 (20 against 136,
    p = 2.2e-22).
  - The kit's pair score at p = 0.005: TH 132 failed pairs of 10,000 [X 72 Z 61] against TCBP-JOBS 360 (41 against
    269, p = 3.0e-42). Rate per cycle 1.11e-3 ± 0.10e-3 against 3.05e-3 ± 0.16e-3.
- Merged into main by PR #9 (16:28:53 UTC) after the X results; the Z rows came after.
- Commands for each number: README_tcbp.md, "Commands". Tests: `python -m pytest -p no:cacheprovider
  predecoder/test_tcbp.py`, 7 pass (luminaty, 1 core, 51 s).

## Untested, and caveats

- TCBP-JOBS's answers vary from run to run (cause not isolated; the model fit is the leading candidate). With the
  kit's priors: 91, 96 and 100 failures on the same validation shots, but on test the run equals the stored file shot
  for shot. Comparisons with stored files may mix runs; the like-for-like is TH against the same run's fresh arm.
- The kit's and the repo's root READMEs were not changed (not this task's paths). Sal asked for "the pipeline readme"
  to be updated; that is left to whoever owns those files.
- The knobs were fitted at p = 0.005 on X and applied unchanged to p = 0.004, p = 0.003 and Z.
- The head was not retrained for TCBP-JOBS.
- The latency is three separate timers; their sum's p99 was not measured as one timer.
- Typed times are estimates; the commit times decide. The spec's "Written 15:52 UTC" was in fact 15:42:59 UTC
  (`dad4fd50`), and two others were a few minutes late (README erratum).
- `evaluate.py --precision`: the CPU path was tested by 6 existing tests, and the fp32 GPU path ran in job 4.

## What is left

- Merging to integration and main (the orchestrator's).
- A Z knob fit.
- A head retrained for TCBP-JOBS.
- Passing H as a scipy sparse matrix, which may build the decoder faster.
