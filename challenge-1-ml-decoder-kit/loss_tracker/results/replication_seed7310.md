## Replication on the team's server (seed 7310, 4,000 shots): the frozen decoder against the references
Every condition on the same 4000 shots.

### reference exact

| condition | failures / shots | fail fraction (95% Wilson) | per cycle | unconverged (of them right) | mean iterations | paired shots: exact fails, cond fails | exact-only / cond-only (sign test) | exact ÷ cond (95% bootstrap) |
|---|---|---|---|---|---|---|---|---|
| starter | 146 / 4000 | 0.0365 (0.0311–0.0428) | 3.09e-03 | 98 (2) | 2289 | 4000: 161, 146 | 56 / 41 (p=0.15) | 1.10 (0.97–1.25) |
| exact | 161 / 4000 | 0.0403 (0.0346–0.0468) | 3.42e-03 | 116 (2) | 2528 | reference |  |  |
| tracker | 122 / 4000 | 0.0305 (0.0256–0.0363) | 2.58e-03 | 84 (1) | 2057 | 4000: 161, 122 | 80 / 41 (p=0.0005) | 1.32 (1.13–1.54) |
| oracle-round | 82 / 4000 | 0.0205 (0.0165–0.0254) | 1.72e-03 | 53 (1) | 1595 | 4000: 161, 82 | 100 / 21 (p=1.6e-13) | 1.96 (1.64–2.40) |
| tracker>exact | 100 / 4000 | 0.0250 (0.0206–0.0303) | 2.11e-03 | 55 (0) | 2430 | 4000: 161, 100 | 80 / 19 (p=4.4e-10) | 1.61 (1.39–1.89) |
| exact>starter | 124 / 4000 | 0.0310 (0.0261–0.0368) | 2.62e-03 | 71 (1) | 3042 | 4000: 161, 124 | 38 / 1 (p=1.5e-10) | 1.30 (1.20–1.43) |
| oracle-round>exact | 76 / 4000 | 0.0190 (0.0152–0.0237) | 1.60e-03 | 40 (0) | 1835 | 4000: 161, 76 | 99 / 14 (p=6.1e-17) | 2.12 (1.77–2.58) |


### reference starter

| condition | failures / shots | fail fraction (95% Wilson) | per cycle | unconverged (of them right) | mean iterations | paired shots: starter fails, cond fails | starter-only / cond-only (sign test) | starter ÷ cond (95% bootstrap) |
|---|---|---|---|---|---|---|---|---|
| starter | 146 / 4000 | 0.0365 (0.0311–0.0428) | 3.09e-03 | 98 (2) | 2289 | reference |  |  |
| exact | 161 / 4000 | 0.0403 (0.0346–0.0468) | 3.42e-03 | 116 (2) | 2528 | 4000: 146, 161 | 41 / 56 (p=0.15) | 0.91 (0.80–1.03) |
| tracker | 122 / 4000 | 0.0305 (0.0256–0.0363) | 2.58e-03 | 84 (1) | 2057 | 4000: 146, 122 | 66 / 42 (p=0.026) | 1.20 (1.03–1.40) |
| oracle-round | 82 / 4000 | 0.0205 (0.0165–0.0254) | 1.72e-03 | 53 (1) | 1595 | 4000: 146, 82 | 84 / 20 (p=1.6e-10) | 1.78 (1.49–2.15) |
| tracker>exact | 100 / 4000 | 0.0250 (0.0206–0.0303) | 2.11e-03 | 55 (0) | 2430 | 4000: 146, 100 | 71 / 25 (p=2.9e-06) | 1.46 (1.25–1.72) |
| exact>starter | 124 / 4000 | 0.0310 (0.0261–0.0368) | 2.62e-03 | 71 (1) | 3042 | 4000: 146, 124 | 40 / 18 (p=0.0054) | 1.18 (1.06–1.32) |
| oracle-round>exact | 76 / 4000 | 0.0190 (0.0152–0.0237) | 1.60e-03 | 40 (0) | 1835 | 4000: 146, 76 | 88 / 18 (p=3e-12) | 1.92 (1.60–2.35) |


### The share of the oracle's gain

| condition | reference | failures: reference, condition, ORACLE | share of the gain recovered (95% bootstrap) |
|---|---|---|---|
| tracker | exact | 161, 122, 82 | 0.49 (0.25–0.70) |
| tracker>exact | exact | 161, 100, 82 | 0.77 (0.58–0.98) |
| exact>starter | exact | 161, 124, 82 | 0.47 (0.33–0.63) |
| tracker | starter | 146, 122, 82 | 0.38 (0.07–0.63) |
| tracker>exact | starter | 146, 100, 82 | 0.72 (0.46–0.98) |
| exact>starter | starter | 146, 124, 82 | 0.34 (0.14–0.57) |
