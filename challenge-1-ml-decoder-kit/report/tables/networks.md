The two networks, from their training logs (`report/data/networks/*.train.log`). "Shots wrong" is the share of shots with at least one of the 12 logical bits wrong, on error-model shots drawn for the training report: a training diagnostic, not a result on Stim shots.

| Network | Trained on | Size | Started from | Training step (last in the copied log) | Shots seen | Shots per second (median of the log) | GPU hours so far (shots seen ÷ that speed) | Shots wrong at p = 0.003 / 0.004 / 0.005 |
|---|---|---|---|---|---|---|---|---|
| X memory (`cascade_X_s1`) | a laptop GPU (RTX 4060, 8 GB) | width 128, depth 8, 893,697 parameters | random weights | 120,000 of 120,000 | 15,360,000 | 723 | 5.9 | 0.5% / 3.8% / 18.3% |
| Z memory (`cascade_Z_s1`) | the ICRN server (A100 slice, 20 GB) | width 128, depth 8, 893,697 parameters | the X network's weights, mirrored | 100,000 of 100,000 | 12,800,000 | 893 | 4.0 | 0.5% / 4.1% / 18.6% |

The X network alone, on the team's Stim shot files (exploration shots; `report/data/exploration/cascade_X_s1/SCORES.md`), share of shots wrong:

| Training step | `validation_X_p0.003` | `train_X_p0.003` | `validation_X_p0.004` | `train_X_p0.004` | `validation_X_p0.005` | `train_X_p0.005` |
|---|---|---|---|---|---|---|
| 20,000 |  |  | 42.12% of 5,000 |  |  |  |
| 30,000 |  |  | 26.60% of 5,000 | 27.00% of 20,000 |  |  |
| 40,000 | 6.16% of 5,000 | 6.10% of 20,000 | 21.86% of 5,000 | 21.66% of 20,000 | 47.54% of 5,000 | 48.62% of 20,000 |
| 50,000 | 3.66% of 5,000 | 3.96% of 20,000 | 16.04% of 5,000 | 15.43% of 20,000 | 38.88% of 5,000 | 39.13% of 20,000 |
| 60,000 | 2.70% of 5,000 | 2.77% of 20,000 | 12.32% of 5,000 | 12.02% of 20,000 | 32.92% of 5,000 | 33.03% of 20,000 |
| 70,000 | 2.24% of 5,000 | 2.46% of 20,000 | 10.52% of 5,000 | 10.35% of 20,000 | 30.18% of 5,000 | 30.18% of 20,000 |
| 80,000 | 1.20% of 5,000 | 1.26% of 20,000 | 7.32% of 5,000 | 6.69% of 20,000 | 23.18% of 5,000 | 23.51% of 20,000 |
| 90,000 | 0.82% of 5,000 | 0.91% of 20,000 | 6.04% of 5,000 | 5.39% of 20,000 | 20.58% of 5,000 | 20.76% of 20,000 |
| 100,000 | 0.72% of 5,000 | 0.81% of 20,000 | 5.32% of 5,000 | 4.76% of 20,000 | 19.56% of 5,000 | 19.54% of 20,000 |
| 110,000 | 0.76% of 5,000 | 0.66% of 20,000 | 5.02% of 5,000 | 4.30% of 20,000 | 18.02% of 5,000 | 18.32% of 20,000 |
| 120,000 | 0.56% of 5,000 | 0.68% of 20,000 | 4.74% of 5,000 | 3.95% of 20,000 | 17.16% of 5,000 | 17.73% of 20,000 |

The final networks alone on the **test shots** (the team's test files, `neural.evaluate_stim`; `report/data/test/net_p<p>/`), share of shots with any logical bit wrong. Alone they are far worse than relay-BP; they are only ever asked to choose among candidates.

| p | X network (step 120,000) | Z network (step 100,000) |
|---|---|---|
| 0.003 | not measured | not measured |
| 0.004 | not measured | not measured |
| 0.005 | not measured | not measured |
