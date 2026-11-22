The two networks, from their training logs (`report/data/networks/*.train.log`). "Shots wrong" is the share of shots with at least one of the 12 logical bits wrong, on error-model shots drawn for the training report: a training diagnostic, not a result on Stim shots.

| Network | Trained on | Size | Started from | Training step (last in the copied log) | Shots seen | Shots per second (median of the log) | GPU hours so far (shots seen ÷ that speed) | Shots wrong at p = 0.003 / 0.004 / 0.005 |
|---|---|---|---|---|---|---|---|---|
| X memory (`cascade_X_s1`) | a laptop GPU (RTX 4060, 8 GB) | width 128, depth 8, 893,697 parameters | random weights | 100,000 of 120,000 | 12,800,000 | 723 | 4.9 | 0.6% / 4.5% / 20.1% |
| Z memory (`cascade_Z_s1`) | the ICRN server (A100 slice, 20 GB) | width 128, depth 8, 893,697 parameters | the X network's weights, mirrored | 64,000 of 100,000 | 8,192,000 | 876 | 2.6 | 0.8% / 5.1% / 21.9% |

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
