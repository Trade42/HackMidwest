# TCBP pipeline on the neutral-atom-shaped noise: declared before any TCBP number

**Label: exploration, after the main test results; a separate branch (`infleqtion-noise`).** Written 2026-11-22
about 16:05 UTC by the implementer of task `tcbp-noise`, before the server job ran and before any TCBP-JOBS decode
of this noise existed. Every number below is reported whatever it shows.

## The shots

Question 1's X shot file at factor 0.8: `X_f0.8_seed4100.npz` (made by `make_shots.py` on luminaty, Stim 1.16.0,
no atom loss), 2,000 shots, seed 4100, sha256 `4e6e2867557ced29a25b1c0380b1aa2b6f2d9922bdb4fe1300e2de0a409365e8`.
It is the file of this branch's BP-OSD and relay-BP rows at factor 0.8 (`logs/shots.log`).

The predecoder's tools read only the kit's shot format (`evaluation.data.load_dataset`), so `tcbp_check.py convert`
writes the same shots in that format (detectors unpacked, the kit's 936 X detector ids, `basis` X, `seed` 4100 and
`p` as below). Only the file format changes; the converted file's sha256 is recorded in `SHA256SUMS` beside it.

## The arms (TCBP-JOBS, cudaq-qec preset `tcbp-jobs`, one decoder built per shot, one CPU thread, the server's A100)

All three decode the kit's H and L (936 x 8,784, X memory). Only the prior vector differs.

- **(g) generic:** the kit's uniform model at **p = 0.0326371, BP-OSD's fitted p** for this file (`fit.json`:
  multiplier 2.744, fitted on the train file `X_f0.8_seed5100.npz`, seed 5100, not on these shots). Why BP-OSD's and
  not relay-BP's (0.045692): TCBP-JOBS is belief propagation followed by an ordered-statistics search (OSD method
  4, JOBS), the same family as BP-OSD; relay-BP's p was fitted for relay-BP's ensemble of BP runs. **Neither p was
  fitted for TCBP-JOBS**, so (g) is a fair generic picture but not TCBP-JOBS's own best uniform p.
- **(m) matched:** `priors.matched_priors("X", noise)` with the file's own noise (from its `meta`), the error model
  of the very circuit that made the shots, written onto the kit's columns; a column the model never fires gets
  1e-9 (`FLOOR`), as in this branch's other matched rows. If TCBP-JOBS refuses a prior that small, the script
  raises the floor to 1e-6 for this arm and says so in its log and summary; nothing else changes.
- **(h) head:** the network and head of the TCBP pipeline (trunk `d8d1c70f...`, head `38ab667e...`, run on the A100
  in float32), at the **frozen TCBP knobs** of `predecoder/logs/fit_tcbp_X_p0.005.json` on `predecoder-tcbp`
  (sha256 `7697010678a3...`): log-odds mixing, alpha 0.4, eps 1e-4, every prior capped at 0.5. **No refit.**
  The network's own p input and the base of the mixing are both **p = 0.0326371**, the same p as (g) (the
  converted file's `p`): a deployed pipeline would be told the fitted p, and with alpha 0 the arm is exactly (g),
  so h vs g differs only by the head. This p is about 6.5 times above the network's training range (0.003 to
  0.005).

## The pairs

Exact two-sided sign tests on the shots where exactly one arm fails (binomial, one half): **h vs g**, **h vs m**,
**m vs g**. Beside them, on the same shots, this branch's existing rows on this file: BP-OSD matched and BP-OSD at
the fitted generic p (2,000 shots), relay-BP matched and relay-BP at its fitted generic p (500 shots), each paired
with the TCBP arms on the shots both decoded. Every failure count has its 95% Wilson interval. A failure is a shot
with at least one of the 12 logical bits wrong (X memory only; no X+Z pairs here).

## The stopping rule

Decode time at this noise is unmeasured. The script decodes blocks of 100 shots in order, the three arms one after
the other inside each block (m, g, h), and starts no new block after the deadline it is given (17:12 UTC). **The
scored set is the longest prefix of whole blocks that all three arms finished.** A shot outside it is not scored
for any arm. If all 20 blocks finish, the scored set is all 2,000 shots.

## The expectation, stated before any number

The network saw only p = 0.003 to 0.005 kit noise. The fitted uniform p here is about 0.033 (0.046 for relay-BP),
about ten times higher, and the noise has another shape: mostly phase errors after the gates, readout and reset
worse than the gates, hook faults almost absent. So **(h) is a transfer test that may not help**, and may hurt.
**(m) vs (g) is the solid comparison**: on this file BP-OSD failed 91 times with matched priors against 442 at the
fitted generic p, and relay-BP 9 against 65 of 500; we expect TCBP-JOBS to show the same direction.
