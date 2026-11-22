# The retrained head's knobs refitted on train shots, then one test: declared before any number

**Label: exploration, task tcbp-retrain (branch `tcbp-retrain`).** Written 2026-11-22 before 18:49 UTC (pushed in `2144eac3` at 18:49:20; this time corrected afterwards). The
orchestrator `dq-orch-6` asked for it for Sal ("Sal wants accuracy pushed"). At that time:
- no decode of the two files below existed;
- run 1 of `tcbp_retrain_declared.md` was known: r 27, m 8, h 119 of 1,000; the retrained head at alpha 0.4 is worse
  than matched priors;
- run 2 was running, and none of its numbers had been seen.

**Why.** The frozen knobs (log-odds, alpha 0.4, eps 1e-4) were fitted for the kit-trained head on the kit's noise.
The retrained head is far more confident: 66 of the 87 fired columns per shot are above 0.5, and about 9 columns per
shot that did not fire are above 0.5 too. So the knobs are refitted for it on train shots, and the chosen knobs are
tested once on fresh shots.

## The head, the base, the decoder

The same retrained head (`b42d71d5...`, laptop-artifacts `07930fcc`) and the same matched base (`181d3c74...`, the
vector it was trained with) as `tcbp_retrain_declared.md`.
- **Mixing:** log-odds mixing with that base. predecoder.tcbp's own fit cannot do this: it mixes with the kit's priors
  at the file's p, so `neutral_atom_noise/tcbp_retrain.py fit` does it here.
- **Decoder:** TCBP-JOBS (cudaq-qec preset `tcbp-jobs`), one decoder built per shot, one CPU thread, the server's A100
  slice. Every prior is capped at 0.5.
- **Logits:** `predecoder.evaluate --base-priors priors_matched.npy`, fp32 on the server's GPU, as in runs 1 and 2.
  Shipping them from luminaty would put about 100 MB of logits into git.

## The shots (Stim, `make_shots.py` on luminaty, factor 0.8, X memory, no loss; both seeds never used before)

- **Train:** `X_f0.8_seed5800.npz`, 1,500 shots, seed 5800, sha256 `cb298991b34b44b44ff6a54651ccc185c186a310131fd8d698947cbd34ad804b`
  (converted: `2adc8b256b8952d9...`). Only the fit reads it.
- **Test:** `X_f0.8_seed4900.npz`, 4,000 shots, seed 4900, sha256 `8f61f30b65740f6306be4a846de77d7addc13633de1ee424b690e19bad072dc9`
  (converted: `97807a384cc3da68...`). It is read once, by the test below, after the fit has chosen.
  - Neither seed 4800 nor seed 4100 is used.

## The fit (`tcbp_retrain fit`, on all 1,500 train shots)

- **Arms:** arm m (matched priors, no head) once, as the control. Then the retrained head at every setting of
  `FIT_GRID`: log-odds, **alpha 0.1, 0.2, 0.3, 0.4, 0.6, 0.8 and 1.0** (1.0: the head's probabilities as they are),
  each at **eps 1e-4 and 1e-6**, so 14 settings.
- **Choice rule, fixed now:** the setting with the **fewest train failures**. Ties go to fewer corrections that do
  not reproduce the syndrome, then fewer shots TCBP-JOBS flags as not converged, then the smaller alpha, then the
  larger eps.
  - The control m is not a candidate. The chosen setting goes to the test even if it fails more train shots than m.
- **A stopped fit chooses nothing,** and then the arm rf is not run.

## The test (`tcbp_retrain decode --refit <fit.json>`, the test file, in the same server job right after the fit)

- **Arms:** inside each block of 100 the order is m, r, rf:
  - **m:** matched;
  - **r:** the retrained head at the frozen knobs (alpha 0.4, eps 1e-4), as in runs 1 and 2;
  - **rf:** the retrained head at the knobs the fit chose. If they are alpha 0.4 and eps 1e-4, rf repeats r's priors:
    a reproducibility check.
- **Pairs:**
  - **primary: rf vs m;**
  - secondary: rf vs r and r vs m.
  - Exact two-sided sign test on the shots where exactly one arm fails; 95% Wilson intervals.
- **Stopping rule:** no new block after the job's start + 60 minutes. The scored set is the longest prefix of whole
  blocks that all three arms finished.
- **Claim rule:** the refitted head adds to the matched priors only if rf fails fewer test shots than m with p < 0.05.
  Otherwise we say it does not, with the numbers.
- **Expectation.** On train, small alphas should sit near m and alpha 0.4 above it, as in run 1. Whether some
  alpha beats m we do not know. If the best alpha is the smallest (0.1), rf will be close to m and the test will
  likely show no significant difference.
