# Pre-registered hypotheses

Written before any evaluation result existed (Sat 2026-11-21, 18:25 CST). The five
predictions are the ones in the team plan. This file fixes how each one is judged, so a
result cannot be reinterpreted after the fact. `python reproduce.py` evaluates them
exactly as written here and writes `results/<experiment>/hypotheses.md`.

## Shared rules

- Unit of analysis: the question. Every comparison is paired over the same questions.
- Intervals: 95% percentile bootstrap over questions, 10,000 resamples, seed 0.
- Verdicts: **supported** when the interval of the stated difference lies entirely on the
  predicted side of zero; **contradicted** when it lies entirely on the other side;
  **not supported** otherwise. An "indistinguishable" prediction is supported when the
  interval contains zero, and the interval width is reported next to it.
- `tau` and `theta` are chosen on the 50 calibration questions only
  (`scripts/calibrate.py`); the 150 evaluation questions are never used for tuning.
- "Single" means the answer from the lowest-energy selection. "Vote" means the majority
  answer over the 3 lowest-energy distinct selections, ties to the lowest energy.

## The hypotheses

| | Prediction | Quantity | Supported when |
|---|---|---|---|
| H1 | Selection helps | EM(lane 2a, single) minus EM(lane 1a) | interval above 0 |
| H2 | Ideal QAOA at p = 2 is close to exact | (a) share of questions where lane 3a-p2's top selection is an exact optimum; (b) EM(3a-p2, single) minus EM(2a, single) | (a) Wilson interval above 0.5 and (b) interval contains 0 |
| H3 | Error detection helps the solver | approximation ratio of lane 3c (kept shots) minus lane 3b, both on the noisy backend | interval above 0 |
| H4 | The benefit shrinks downstream | [AR(3c) minus AR(3b)] minus [EM(3c) minus EM(3b)], both on a 0 to 1 scale | interval above 0. Reported as "not applicable" if H3 is not supported, since there is then no solver gain to pass on |
| H5 | The vote helps | EM(lane 2a, vote) minus EM(lane 2a, single) | interval above 0 |

Approximation ratio is the mean-sample form, `(E_max - mean energy) / (E_max - E_min)`
over the kept shots before repair. It is used instead of "best sample" on purpose: at
10 paragraphs there are 1,024 selections, so 4,000 shots from a uniformly random
sampler already contain the optimum about 98% of the time, and a best-sample metric
cannot tell the tiers apart. For the same reason every sampling lane is also reported
at a small equal raw-shot budget, and a uniformly random sampler (`rand`) is run as a
control.

## Secondary quantities reported whichever way they come out

- H3 and H4 at each noise level in the sweep, not only the default one.
- Share of shots kept, and optimum probability times kept share (equal raw shots).
- The same comparisons with the textbook (direct) encoding in place of the bus encoding.
- Prompt tokens and latency per stage for every lane.
