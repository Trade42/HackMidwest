## X memory, p = 0.004, s = 0.001, seed 7310
Every condition on the same 4000 shots.

### reference exact

| condition | failures / shots | fail fraction (95% Wilson) | per cycle | unconverged (of them right) | mean iterations | paired shots: exact fails, cond fails | exact-only / cond-only (sign test) | exact ÷ cond (95% bootstrap) |
|---|---|---|---|---|---|---|---|---|
| starter | 146 / 4000 | 0.0365 (0.0311–0.0428) | 3.09e-03 | 98 (2) | 2289 | 4000: 161, 146 | 56 / 41 (p=0.15) | 1.10 (0.97–1.25) |
| exact | 161 / 4000 | 0.0403 (0.0346–0.0468) | 3.42e-03 | 116 (2) | 2528 | reference |  |  |
| tracker | 122 / 4000 | 0.0305 (0.0256–0.0363) | 2.58e-03 | 84 (1) | 2057 | 4000: 161, 122 | 80 / 41 (p=0.0005) | 1.32 (1.13–1.54) |
| oracle-round | 82 / 4000 | 0.0205 (0.0165–0.0254) | 1.72e-03 | 53 (1) | 1595 | 4000: 161, 82 | 100 / 21 (p=1.6e-13) | 1.96 (1.64–2.40) |


### reference starter

| condition | failures / shots | fail fraction (95% Wilson) | per cycle | unconverged (of them right) | mean iterations | paired shots: starter fails, cond fails | starter-only / cond-only (sign test) | starter ÷ cond (95% bootstrap) |
|---|---|---|---|---|---|---|---|---|
| starter | 146 / 4000 | 0.0365 (0.0311–0.0428) | 3.09e-03 | 98 (2) | 2289 | reference |  |  |
| exact | 161 / 4000 | 0.0403 (0.0346–0.0468) | 3.42e-03 | 116 (2) | 2528 | 4000: 146, 161 | 41 / 56 (p=0.15) | 0.91 (0.80–1.03) |
| tracker | 122 / 4000 | 0.0305 (0.0256–0.0363) | 2.58e-03 | 84 (1) | 2057 | 4000: 146, 122 | 66 / 42 (p=0.026) | 1.20 (1.03–1.40) |
| oracle-round | 82 / 4000 | 0.0205 (0.0165–0.0254) | 1.72e-03 | 53 (1) | 1595 | 4000: 146, 82 | 84 / 20 (p=1.6e-10) | 1.78 (1.49–2.15) |


### The share of the oracle's gain

| condition | reference | failures: reference, condition, ORACLE | share of the gain recovered (95% bootstrap) |
|---|---|---|---|
| tracker | exact | 161, 122, 82 | 0.49 (0.25–0.70) |
| tracker | starter | 146, 122, 82 | 0.38 (0.07–0.63) |
