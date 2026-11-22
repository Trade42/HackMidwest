# tcbp-fastpath: the criterion, written before any rule's number

Exploration, after the test results; a separate branch (`tcbp-fastpath`). Validation shots only
(`data/p0.005/validation_X.npz`, 5,000 Stim shots, seed 201). No test file is opened.
Written 2026-11-22 18:26 UTC (the commit time decides), from dq-orch-6's message of 18:24 UTC (Sal: "we still want
the performance better than anything else, the speed/latency and even the accuracy").

## The rule's shape

For each shot: run a cheap answer and a cheap check. If the check says "easy", answer with the cheap answer.
Otherwise run the TCBP version (network + head, then TCBP-JOBS built with the shot's priors) and answer with it.

- **TH, the TCBP version:** job 3's head arm (`server-logs: files/predecoder-tcbp/validation_X_p0.005_per_shot.npz`).
- **F, the cheap answer:** TCBP-JOBS with the kit's priors, built once: the same run's fresh arm (its batch check,
  one decoder built once, gives the same answer on all 5,000 shots).

## The two candidate checks (no truth)

- **C1 (agreement):** F and PriorTilt P2 (laptop-artifacts `priortilt/results_p005main/selector_validation_X.npz`)
  give the same 12 logical bits.
- **C2 (syndrome weight):** the shot's number of detection events (of its 936 basis detectors) is at most w.
  Every w is scanned; the largest w that passes the gate is C2's candidate.

## The gate (accuracy first)

A rule passes only if it adds no failure: the shots where the rule fails and TH alone does not ("only rule") are 0.
The allowed exception, a difference that is not significant (exact two-sided binomial p >= 0.05 on the discordant
shots) AND no larger than TH's own run-to-run spread, is 0 here: TH's measured spread is 0 changed outcomes on the
1,000 shots decoded again (job 4 on the A100's logits; tcbp-speed job 1: dense, dense again, CSR, CSC).
So the gate is "only rule = 0".

## The choice

Among the rules that pass, the lowest MEAN time per shot. Its p99 is reported beside it.

## The time model (A100 components, one process, one CPU thread; separate timers, stated)

- F: decode only, per shot, job 3's fresh arm `seconds_decode_fresh` (a decoder built once costs nothing per shot).
- C1 adds P2's own time (laptop CPU, `seconds_P2`), run beside F: max(F, P2) per shot. Two machines; labelled.
- C2 adds nothing measurable (a sum over 936 bits).
- The slow path adds the network + head (9.4 ms mean, 19.8 ms p99, job 4, per-shot values not stored) and TH's
  priors + build + decode + free with H as CSR (tcbp-speed job 1, one timer, 41.7 ms mean, p99 53.7).
- The TCBP version alone, the reference: 9.4 + 41.7 = 51.1 ms mean.
