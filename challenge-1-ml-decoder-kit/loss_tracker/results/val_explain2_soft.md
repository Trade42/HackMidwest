## Validation (seed 7200), shots 0 to 1,499; tracker version order1_explain2, soft posterior (not the frozen pair)
Every condition on the same 1500 shots.

### reference exact

| condition | failures / shots | fail fraction (95% Wilson) | per cycle | unconverged (of them right) | mean iterations | paired shots: exact fails, cond fails | exact-only / cond-only (sign test) | exact ÷ cond (95% bootstrap) |
|---|---|---|---|---|---|---|---|---|
| exact | 69 / 1500 | 0.0460 (0.0365–0.0578) | 3.92e-03 | 40 (0) | 2560 | reference |  |  |
| tracker | 51 / 1500 | 0.0340 (0.0260–0.0444) | 2.88e-03 | 33 (1) | 2012 | 1500: 69, 51 | 29 / 11 (p=0.0064) | 1.35 (1.10–1.67) |
| oracle-round | 31 / 1500 | 0.0207 (0.0146–0.0292) | 1.74e-03 | 14 (0) | 1528 | 1500: 69, 31 | 43 / 5 (p=1.4e-08) | 2.23 (1.69–3.09) |


### The share of the oracle's gain

| condition | reference | failures: reference, condition, ORACLE | share of the gain recovered (95% bootstrap) |
|---|---|---|---|
| tracker | exact | 69, 51, 31 | 0.47 (0.18–0.71) |
