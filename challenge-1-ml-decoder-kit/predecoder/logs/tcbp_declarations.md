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

## 1b. A change forced by the decoder (15:10 UTC, during the fit; no validation number with the head's priors exists)

The fit's first run stopped at alpha 0.3, on the failed file: `ValueError: ERROR: osd_method=4 (JOBS) requires every
error rate to be at most 0.5 (column 2652 has prior LLR -0.215245)`. When the head is confident, the mixed priors go above
0.5, and TCBP-JOBS refuses them. From commit 3 of this branch on, every shot's priors are therefore capped at 0.5 after
mixing and clipping (`tcbp.PRIOR_CAP`). Rows below the cap are unchanged. The rows already decoded (the control, alpha
0.1, alpha 0.2) passed the decoder's own check on every shot, so no prior of theirs was above 0.5, and the cap leaves
them as they were. Their cached parts are reused. The train numbers seen before this change: the control left 341 of
the 390 failed wrong and broke 5 of the 2,000 right; alpha 0.1, 175 and 3; alpha 0.2, 134 and 3. Nothing else was
changed. The rule, the grid and the extension stay as declared in section 1. The runs log how many shots and columns
the cap changed.

## 2. The validation arms (declared 2026-11-22 15:06 UTC; no number of TCBP-JOBS with the head's priors exists yet; the fit is running)

What was known when this was written: job 1 (server, 14:54 to 14:59 UTC) decoded `data/p0.005/validation_X.npz` with
TCBP-JOBS and the kit's priors, one decoder built per shot, using `decode()`. It fails 96, against the stored file's 91
(decoder built once, `decode_batch`). The two disagree on 44 shots' bits (only per-shot 17, only stored 12, p = 0.46).
The decoder's own `converged` flag is False on 4,858 of the 5,000 shots, even when the answer is right. No priors
from the head were involved.

- **Arm (a), TH:** TCBP-JOBS with the head's priors (the knobs frozen by section 1), one decoder per shot, `decode()`.
- **Arm (b2):** H where H converged, else TH. H is relay-BP with the head's priors (the luminaty CPU float32 run of
  `logs/validation_X_p0.005_cpu_head_arm.log`: 79 failures, 44 unconverged; per-shot file
  `scratch/predecoder/runs/val_p0.005_cpu/compare/per_shot.npz`). TH's priors use the same logits bytes (sha256
  `14eea54d...`). This is the frozen fallback rule (`logs/fallback_rule_frozen.md`) with T replaced by TH.
  (b1), "TH where its flag says converged, else H", is not used: that flag is False on 97% of shots, so (b1) would be
  H almost everywhere.
- **The choice:** the arm with fewer failures on these 5,000 validation shots, with a tie going to (a). It is then
  frozen for the test run. Choosing the better of two arms on the same shots favours the chosen arm slightly, and
  that is stated beside its numbers.
- **The kill line (the brief's):** the chosen arm fails fewer shots than TCBP-JOBS alone, the stored file
  (`files/tcbp_p0.005/validation_X.tcbp.npz`, 91), paired, exact two-sided p < 0.05. Reported beside it, whatever they
  show:
  - the chosen arm against the kit's-priors arm decoded per shot in the same run, with the same model ("fresh", the
    like-for-like control);
  - against H (79);
  - against the frozen fallback 'H where converged, else T' (50).
- **Job 3** decodes fresh and head in one process on one model. The fresh arm is decoded again rather than reused
  from job 1, so that model fit is no confound in head against fresh. `--batch-check` also decodes the kit's priors
  with one decoder built once and `decode_batch` on the same model. Fresh against batch then measures batch against
  one shot at a time, and job 1's fresh against job 3's fresh measures one model fit against another.

## 2b. The arm for the test run: (a), on Sal's word (recorded 2026-11-22 15:40 UTC)

The orchestrator (dq-orch-6) relayed Sal, 15:36 UTC, verbatim: "but like why are we even using relay bp!!! didnt i say we
just use tcbp for everything that relay bp was used for." Arm (b2) contains relay-BP, so **(a) TH, TCBP-JOBS with the
head's priors, is the arm for the test run.** (b2) is computed and reported beside it, but it is not a candidate.
What was known when this was written:
- the head arm's count on validation_X p = 0.005, from job 3's log: 33 of 5,000;
- the same run's fresh arm: 100 of 5,000.
(b2)'s validation number had not been computed, and the per-shot file had not yet arrived. The kill line of §2
(TH against the stored TCBP-JOBS file, 91, paired, exact two-sided p < 0.05) is read on the per-shot file before the
test run is registered.

## 3. The test runs (15:52 UTC)

Registered in `logs/tcbp_test_registered_spec.md` before any number of them exists. Z (run 3) is one job with a gate
inside it (dq-orch-6, 15:49 UTC): X's frozen knobs, no Z fit; validation_Z first; test_Z only if validation_Z beats
the stored TCBP-JOBS validation_Z file, paired, p < 0.05; then the pair score against TCBP-JOBS's 360. The Z head's
logits (from dq-predecoder) are recorded here, with their sha256, when they arrive:
- validation_Z: `scratch/predecoder/z/logits_validation_p0.005/logits.npy`, sha256
  `db786d0a04f8f72157e7c812ab690708c0c3208529365eb51c915cbf46a53a49`. Z head `083147d2...` (laptop-artifacts `a2c3d5d`,
  `predecoder_Z_s1/head.pt`), Z trunk `fad24e80...`, luminaty CPU, 4 threads. Shots `data/p0.005/validation_Z.npz`
  `24f12efa...`. The gate's stored file is `files/tcbp_p0.005/validation_Z.tcbp.npz`, sha256 `b0ba027a704c734a8c77b55e557b698dfc851e27e273b07df623cbbf76b27619`.
- test_Z: shots `data/p0.005/test_Z.npz`, `b86ae320d13fa555d8358fe399156d28d4e52b01fabf8eafa1b30e60bdcd0cef`. The stored
  TCBP-JOBS file is `files/clean/tcbp_p0.005/test_Z.tcbp.npz`, `23ae160e02bf9f6085ff17d2b0d89a5b376c7f589bbfde6da89dd09a831de037`.
  The logits' sha256 is added when dq-predecoder sends them.
- The knobs: X's TCBP file (`76970106...`: log-odds alpha 0.4, eps 1e-4, cap 0.5), NOT dq-predecoder's relay-BP Z
  knobs (`fit_Z_p0.004.json`, alpha 0.2).
