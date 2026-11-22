### X memory, p = 0.004, s = 0.001, seed 5300: reference starter

| condition | failures / shots | fail fraction (95% Wilson) | per cycle | unconverged (of them right) | mean iterations | paired shots: starter fails, cond fails | starter-only / cond-only (sign test) | starter ÷ cond (95% bootstrap) |
|---|---|---|---|---|---|---|---|---|
| starter | 162 / 4000 | 0.0405 (0.0348–0.0471) | 3.44e-03 | 112 (1) | 2325 | reference |  |  |
| starter-nofinal | 68 / 1000 | 0.0680 (0.0540–0.0853) | 5.85e-03 | 47 (0) | 3424 | 1000: 38, 68 | 8 / 38 (p=9.2e-06) | 0.56 (0.42–0.71) |
| starter-nohook | 39 / 1000 | 0.0390 (0.0287–0.0529) | 3.31e-03 | 28 (0) | 2528 | 1000: 38, 39 | 12 / 13 (p=1) | 0.97 (0.75–1.27) |
| starter-ratebase | 48 / 1000 | 0.0480 (0.0364–0.0631) | 4.09e-03 | 34 (0) | 2623 | 1000: 38, 48 | 11 / 21 (p=0.11) | 0.79 (0.60–1.02) |
| exact | 181 / 4000 | 0.0452 (0.0392–0.0521) | 3.85e-03 | 131 (3) | 2554 | 4000: 162, 181 | 44 / 63 (p=0.081) | 0.90 (0.80–1.01) |

### X memory, p = 0.004, s = 0.001, seed 5300: reference exact@0.4

| condition | failures / shots | fail fraction (95% Wilson) | per cycle | unconverged (of them right) | mean iterations | paired shots: exact@0.4 fails, cond fails | exact@0.4-only / cond-only (sign test) | exact@0.4 ÷ cond (95% bootstrap) |
|---|---|---|---|---|---|---|---|---|
| exact@0.4 | 35 / 1000 | 0.0350 (0.0253–0.0483) | 2.96e-03 | 29 (1) | 2604 | reference |  |  |
| oracle-round@0.4 | 25 / 1000 | 0.0250 (0.0170–0.0366) | 2.11e-03 | 15 (0) | 1695 | 1000: 35, 25 | 14 / 4 (p=0.031) | 1.40 (1.08–1.95) |
| exact | 181 / 4000 | 0.0452 (0.0392–0.0521) | 3.85e-03 | 131 (3) | 2554 | 1000: 35, 40 | 8 / 13 (p=0.38) | 0.88 (0.68–1.10) |

### X memory, p = 0.004, s = 0.001, seed 5300: reference exact@0.25

| condition | failures / shots | fail fraction (95% Wilson) | per cycle | unconverged (of them right) | mean iterations | paired shots: exact@0.25 fails, cond fails | exact@0.25-only / cond-only (sign test) | exact@0.25 ÷ cond (95% bootstrap) |
|---|---|---|---|---|---|---|---|---|
| exact@0.25 | 58 / 1000 | 0.0580 (0.0451–0.0742) | 4.97e-03 | 42 (1) | 2879 | reference |  |  |
| oracle-round@0.25 | 30 / 1000 | 0.0300 (0.0211–0.0425) | 2.54e-03 | 24 (0) | 2093 | 1000: 58, 30 | 31 / 3 (p=7.7e-07) | 1.93 (1.51–2.67) |
| exact | 181 / 4000 | 0.0452 (0.0392–0.0521) | 3.85e-03 | 131 (3) | 2554 | 1000: 58, 40 | 24 / 6 (p=0.0014) | 1.45 (1.18–1.86) |

### X memory, p = 0.004, s = 0.001, seed 5300: reference oracle-round

| condition | failures / shots | fail fraction (95% Wilson) | per cycle | unconverged (of them right) | mean iterations | paired shots: oracle-round fails, cond fails | oracle-round-only / cond-only (sign test) | oracle-round ÷ cond (95% bootstrap) |
|---|---|---|---|---|---|---|---|---|
| oracle-round | 96 / 4000 | 0.0240 (0.0197–0.0292) | 2.02e-03 | 54 (0) | 1634 | reference |  |  |
| oracle-round@0.4 | 25 / 1000 | 0.0250 (0.0170–0.0366) | 2.11e-03 | 15 (0) | 1695 | 1000: 26, 25 | 9 / 8 (p=1) | 1.04 (0.75–1.45) |
| oracle-round@0.25 | 30 / 1000 | 0.0300 (0.0211–0.0425) | 2.54e-03 | 24 (0) | 2093 | 1000: 26, 30 | 9 / 13 (p=0.52) | 0.87 (0.60–1.20) |
