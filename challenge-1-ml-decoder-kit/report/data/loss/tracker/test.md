## Test (seed 7300): the frozen decoder (tracker version order1_explain1, variant tracker) against the references
Every condition on the same 4000 shots.

### reference exact

| condition | failures / shots | fail fraction (95% Wilson) | per cycle | unconverged (of them right) | mean iterations | paired shots: exact fails, cond fails | exact-only / cond-only (sign test) | exact ÷ cond (95% bootstrap) |
|---|---|---|---|---|---|---|---|---|
| starter | 192 / 4000 | 0.0480 (0.0418–0.0551) | 4.09e-03 | 129 (0) | 2415 | 4000: 196, 192 | 57 / 53 (p=0.78) | 1.02 (0.92–1.13) |
| exact | 196 / 4000 | 0.0490 (0.0427–0.0561) | 4.18e-03 | 145 (0) | 2599 | reference |  |  |
| tracker | 147 / 4000 | 0.0367 (0.0313–0.0430) | 3.12e-03 | 101 (2) | 2141 | 4000: 196, 147 | 81 / 32 (p=4.5e-06) | 1.33 (1.18–1.51) |
| oracle-round | 112 / 4000 | 0.0280 (0.0233–0.0336) | 2.36e-03 | 64 (0) | 1684 | 4000: 196, 112 | 103 / 19 (p=3.8e-15) | 1.75 (1.52–2.03) |
| tracker>exact | 128 / 4000 | 0.0320 (0.0270–0.0379) | 2.71e-03 | 77 (0) | 2589 | 4000: 196, 128 | 80 / 12 (p=1.7e-13) | 1.53 (1.36–1.73) |
| exact>starter | 158 / 4000 | 0.0395 (0.0339–0.0460) | 3.35e-03 | 97 (0) | 3247 | 4000: 196, 158 | 38 / 0 (p=7.3e-12) | 1.24 (1.17–1.34) |
| oracle-round>exact | 106 / 4000 | 0.0265 (0.0220–0.0320) | 2.24e-03 | 53 (0) | 1973 | 4000: 196, 106 | 103 / 13 (p=1.5e-18) | 1.85 (1.60–2.14) |


### reference starter

| condition | failures / shots | fail fraction (95% Wilson) | per cycle | unconverged (of them right) | mean iterations | paired shots: starter fails, cond fails | starter-only / cond-only (sign test) | starter ÷ cond (95% bootstrap) |
|---|---|---|---|---|---|---|---|---|
| starter | 192 / 4000 | 0.0480 (0.0418–0.0551) | 4.09e-03 | 129 (0) | 2415 | reference |  |  |
| exact | 196 / 4000 | 0.0490 (0.0427–0.0561) | 4.18e-03 | 145 (0) | 2599 | 4000: 192, 196 | 53 / 57 (p=0.78) | 0.98 (0.88–1.09) |
| tracker | 147 / 4000 | 0.0367 (0.0313–0.0430) | 3.12e-03 | 101 (2) | 2141 | 4000: 192, 147 | 75 / 30 (p=1.3e-05) | 1.31 (1.17–1.48) |
| oracle-round | 112 / 4000 | 0.0280 (0.0233–0.0336) | 2.36e-03 | 64 (0) | 1684 | 4000: 192, 112 | 102 / 22 (p=1.7e-13) | 1.71 (1.48–2.01) |
| tracker>exact | 128 / 4000 | 0.0320 (0.0270–0.0379) | 2.71e-03 | 77 (0) | 2589 | 4000: 192, 128 | 86 / 22 (p=4e-10) | 1.50 (1.32–1.72) |
| exact>starter | 158 / 4000 | 0.0395 (0.0339–0.0460) | 3.35e-03 | 97 (0) | 3247 | 4000: 192, 158 | 53 / 19 (p=7.6e-05) | 1.22 (1.11–1.34) |
| oracle-round>exact | 106 / 4000 | 0.0265 (0.0220–0.0320) | 2.24e-03 | 53 (0) | 1973 | 4000: 192, 106 | 107 / 21 (p=4.5e-15) | 1.81 (1.56–2.14) |


### The share of the oracle's gain

| condition | reference | failures: reference, condition, ORACLE | share of the gain recovered (95% bootstrap) |
|---|---|---|---|
| tracker | exact | 196, 147, 112 | 0.58 (0.38–0.78) |
| tracker>exact | exact | 196, 128, 112 | 0.81 (0.64–1.03) |
| exact>starter | exact | 196, 158, 112 | 0.45 (0.33–0.61) |
| tracker | starter | 192, 147, 112 | 0.56 (0.36–0.77) |
| tracker>exact | starter | 192, 128, 112 | 0.80 (0.61–1.03) |
| exact>starter | starter | 192, 158, 112 | 0.42 (0.25–0.60) |
