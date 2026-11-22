# Z memory: what is fixed before any Z number exists

Declared 2026-11-22 14:22:44 UTC by the predecoder implementer, on the orchestrator's GO (dq-orch-5, Sal's
steer of 14:19 to 14:21 UTC: the bar is TCBP-JOBS). No Z result of the head exists at this time.

- The Z head trains on the final Z trunk (server-logs ce700ea, files/ckpt/cascade_Z_s1_final_step100000.pt,
  sha256 fad24e80ec4e2c5ea7b54717c6b309b244d90ee9fb50a60128da997840230700; logicals light, all detectors)
  with the X head's recipe: seed 1, 20,000 steps, batch 64, p-train 0.003 0.004 0.005.
- The Z knobs are fitted exactly as X's were: on train_Z p = 0.004, the shots the fresh arm fails plus a
  sample of 2,000 right ones (subset seed 5), log-odds grid alpha 0.1 to 0.7, eps 1e-4, the same rule.
- **If the Z head.pt is not back by 15:45 UTC, Z uses X's frozen knobs (log-odds, alpha 0.2, eps 1e-4)
  and no Z fit is made.**
- Then one validation check (validation_Z p = 0.005, head against fresh), then one Z test run at p = 0.005
  (arms fresh and head), from a frozen commit named before it runs.
- The TCBP-JOBS fallback rule ("H where relay-BP with the head's priors converged, else T") has no knob;
  it is scored on validation first and applied once to the saved test files, labelled "designed after the
  first test result".
