# tcbp-fastpath: C9 frozen, and its one held-out check, declared before the shots exist

Exploration, after the test results; a separate branch (`tcbp-fastpath`). This file is committed before the fresh
shots are generated. The commit time decides. Asked by dq-orch-6 (18:43 UTC), because C9's threshold was chosen
on the same 5,000 validation shots that were looked at three times.

## The frozen rule (C9)

For each X shot at p = 0.005:
1. Run the network and head (X head `38ab667e...` on trunk `d8d1c70f...`), giving one logit per column of H
   (8,784 columns).
2. Compute s = the sum over the columns of the binary entropy, in bits, of sigmoid(logit). The logits are cast to
   float16 first, as stored. This is `prior_moves(...)["C9"]` in `fastpath2.py` at commit `6277aee7`.
3. If s <= **69.14651567152464** bits, answer with F: TCBP-JOBS with the kit's priors, one decoder built once.
4. Otherwise answer with TH: TCBP-JOBS built with the shot's head priors (`predecoder/logs/fit_tcbp_X_p0.005.json`,
   alpha 0.4, eps 1e-4, capped at 0.5).

The threshold is t* from `result3_p0.005.json`, unchanged. Nothing about the rule is refitted on the new shots.

## The held-out shots

- `make_dataset.py --basis X --p 0.005 --shots 5000 --seed 2026112251`: the kit's Stim circuit sampler.
  - The seed has not been used today. The team's files use 101, 102, 201, 202, 301, 302 and 401, and a search of the
    repo and `orchestration/` finds no 2026112251.
  - The file is named `holdout_X_p0.005_s2026112251.npz` and is kept in `orchestration/scratch/tcbp-fastpath/holdout/`
    (a dataset: not committed). Its sha256 is recorded at creation.
- The logits: `predecoder.evaluate` on luminaty's CPU, float32 (`--device cpu --precision fp32`), lanes A-D.
- A server job, run by dq-orch-6, with the same tool and settings as job 3 (`predecoder.tcbp decode --arms head fresh
  --batch-check`, code at main `e6c2a5a0`):
  - TH is the head arm;
  - F is the batch check (built once, `decode_batch`), which must equal the fresh arm on every shot;
  - F's time per shot is the fresh arm's `seconds_decode_fresh` (decode only).
  - Every head row's logits hash must equal luminaty's file.

## What is reported, once

The rule is applied once to all 5,000 shots, by a scoring step with no free parameter:
- the failures of the rule, of TH and of F;
- paired against TH: both fail, only the rule ("added"), only TH, and the exact two-sided p;
- the fast share;
- the 95% upper bound on the added-failure rate among the fast shots;
- the mean and p99 time per shot.

The verdict:
- **holds:** 0 added failures;
- **holds with a cost:** added > 0 but not significant against only-TH (exact p >= 0.05). The count is stated as
  the cost;
- **fails:** added significantly above only-TH (p < 0.05).

The time model uses today's component times, as dq-orch-6 gave them on the A100, one process:
- network + head: 1.7 ms (as a CUDA graph) on every shot;
- F: the job's per-shot decode time;
- TH: TH's per-shot decode time from the job, plus priors 0.16 ms (tcbp-speed job 1), CSR build 34.0 ms and free
  4.6 ms at their means;
- the p99 is over all 5,000 shots.

The same numbers with round 2's network time (9.4 ms) are reported beside these, for comparison with TH alone.

## p = 0.004

That line (`CRITERION_2.md` Part B, `CRITERION_3.md` §2) goes on unchanged. Nothing from it changes the frozen rule.
