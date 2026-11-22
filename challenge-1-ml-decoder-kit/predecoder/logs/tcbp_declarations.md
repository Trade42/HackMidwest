# predecoder-tcbp: declarations, written before the numbers they govern

## 1. The knob fit (declared 2026-11-22 14:48 UTC, before any decode of TCBP-JOBS with the head's priors)

- The orchestrator overruled the brief's p = 0.004 at 14:47 UTC: the fit runs on **train_X at p = 0.005**. Of its 20,000
  shots, TCBP-JOBS alone fails 390 (`files/tcbp_p0.005/train_X.tcbp.npz` on `server-logs`). At p = 0.004 it fails only 37.
- Subsets: `python -m predecoder.subset --data data/p0.005/train_X.npz --per-shot <train_X.tcbp.npz p0.005> --arm T
  --right 2000 --seed 5`. That keeps the 390 failed shots and a random 2,000 of the right ones, the right file weighted by
  (right shots in all) / 2,000.
- Logits: `predecoder.evaluate`, the head `38ab667e...` on the final X trunk `d8d1c70f...`, on luminaty's CPU in float32.
  The server is sent these exact bytes, checked by sha256.
- The decoder: TCBP-JOBS (cudaq-qec 0.9.0, preset `tcbp-jobs`, no overrides). One model is fitted per run
  (`init_tcbp_model`), and one decoder is built per shot with that shot's priors (`error_rate_vec`), then `decode()`.
- The grid, the same as the one that made `logs/fit_X_p0.004.json`: log-odds mixing, alpha 0.1, 0.2, ..., 0.7, eps 1e-4.
  The extension rows (alpha 0.05 below; 0.8, 0.9 and 1.0 above) are decoded in the same run. They join the candidates
  only on the side where the declared grid's choice sits at its edge. The control row is the kit's priors (alpha 0).
- The rule, in order:
  1. the lowest weighted failures: the failed shots left wrong plus the right shots broken, with the right ones
     counted at their weight;
  2. then fewer weighted shots whose correction does not reproduce the syndrome;
  3. then fewer weighted shots the decoder flags as not converged;
  4. then the smaller alpha.
  (relay-BP's tie-breaks, its hard shots and iterations, do not exist for TCBP-JOBS.)
- The control row (the kit's priors, built per shot, `decode()`) should fail all 390 failed shots and none of the
  2,000 right ones, because the stored file was built once and run with `decode_batch`. Any difference is reported
  before a validation number exists.
- The chosen knobs are committed in `predecoder/logs/fit_tcbp_X_p0.005.json` before any validation decode.

## 2. The validation arms (to be completed after the fit and before any validation number: (a), and (b1) or (b2))
