### X memory, p = 0.004, s = 0.0002, seed 5100: reference blind

| condition | failures / shots | fail fraction (95% Wilson) | per cycle | unconverged (of them right) | mean iterations | paired shots: blind fails, cond fails | blind-only / cond-only (sign test) | blind ÷ cond (95% bootstrap) |
|---|---|---|---|---|---|---|---|---|
| blind | 81 / 3000 | 0.0270 (0.0218–0.0334) | 2.28e-03 | 51 (1) | 1641 | reference |  |  |
| starter | 160 / 20000 | 0.0080 (0.0069–0.0093) | 6.69e-04 | 73 (3) | 696 | 3000: 81, 23 | 63 / 5 (p=7.7e-14) | 3.52 (2.53–5.43) |
| starter@0.5/stop1 | 216 / 20000 | 0.0108 (0.0095–0.0123) | 9.04e-04 | 73 (3) | 218 | 3000: 81, 30 | 60 / 9 (p=2.2e-10) | 2.70 (2.02–3.95) |

### X memory, p = 0.004, s = 0.0002, seed 5100: reference starter

| condition | failures / shots | fail fraction (95% Wilson) | per cycle | unconverged (of them right) | mean iterations | paired shots: starter fails, cond fails | starter-only / cond-only (sign test) | starter ÷ cond (95% bootstrap) |
|---|---|---|---|---|---|---|---|---|
| starter | 160 / 20000 | 0.0080 (0.0069–0.0093) | 6.69e-04 | 73 (3) | 696 | reference |  |  |
| starter@0.5/stop1 | 216 / 20000 | 0.0108 (0.0095–0.0123) | 9.04e-04 | 73 (3) | 218 | 20000: 160, 216 | 23 / 79 (p=2.3e-08) | 0.74 (0.66–0.82) |
