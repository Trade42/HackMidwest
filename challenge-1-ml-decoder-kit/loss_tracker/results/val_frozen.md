## Validation (seed 7200), shots 0 to 1,499; the frozen pair: tracker version order1_explain1, variant tracker
Every condition on the same 1500 shots.

### reference exact

| condition | failures / shots | fail fraction (95% Wilson) | per cycle | unconverged (of them right) | mean iterations | paired shots: exact fails, cond fails | exact-only / cond-only (sign test) | exact ÷ cond (95% bootstrap) |
|---|---|---|---|---|---|---|---|---|
| exact | 69 / 1500 | 0.0460 (0.0365–0.0578) | 3.92e-03 | 40 (0) | 2560 | reference |  |  |
| tracker | 54 / 1500 | 0.0360 (0.0277–0.0467) | 3.05e-03 | 31 (1) | 2004 | 1500: 69, 54 | 30 / 15 (p=0.036) | 1.28 (1.05–1.62) |
| oracle-round | 31 / 1500 | 0.0207 (0.0146–0.0292) | 1.74e-03 | 14 (0) | 1528 | 1500: 69, 31 | 43 / 5 (p=1.4e-08) | 2.23 (1.69–3.09) |
| tracker>exact | 49 / 1500 | 0.0327 (0.0248–0.0429) | 2.76e-03 | 21 (0) | 2371 | 1500: 69, 49 | 29 / 9 (p=0.0017) | 1.41 (1.17–1.77) |


### The share of the oracle's gain

| condition | reference | failures: reference, condition, ORACLE | share of the gain recovered (95% bootstrap) |
|---|---|---|---|
| tracker | exact | 69, 54, 31 | 0.39 (0.09–0.68) |
| tracker>exact | exact | 69, 49, 31 | 0.53 (0.27–0.80) |
