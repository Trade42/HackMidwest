### X memory, p = 0.004, s = 0.001, seed 5300: reference starter

| condition | failures / shots | fail fraction (95% Wilson) | per cycle | unconverged (of them right) | mean iterations | paired shots: starter fails, cond fails | starter-only / cond-only (sign test) | starter ÷ cond (95% bootstrap) |
|---|---|---|---|---|---|---|---|---|
| starter | 162 / 4000 | 0.0405 (0.0348–0.0471) | 3.44e-03 | 112 (1) | 2325 | reference |  |  |
| starter-oracle-round | 107 / 4000 | 0.0267 (0.0222–0.0322) | 2.26e-03 | 70 (0) | 1698 | 4000: 162, 107 | 86 / 31 (p=3.7e-07) | 1.51 (1.28–1.79) |
| oracle-round | 96 / 4000 | 0.0240 (0.0197–0.0292) | 2.02e-03 | 54 (0) | 1634 | 4000: 162, 96 | 90 / 24 (p=3.6e-10) | 1.69 (1.44–2.01) |
| oracle-slot | 87 / 4000 | 0.0217 (0.0177–0.0268) | 1.83e-03 | 70 (9) | 1545 | 4000: 162, 87 | 111 / 36 (p=4.3e-10) | 1.86 (1.54–2.30) |
| exact | 181 / 4000 | 0.0452 (0.0392–0.0521) | 3.85e-03 | 131 (3) | 2554 | 4000: 162, 181 | 44 / 63 (p=0.081) | 0.90 (0.80–1.01) |
| starter@0.5/stop1 | 178 / 4000 | 0.0445 (0.0385–0.0513) | 3.79e-03 | 112 (1) | 1020 | 4000: 162, 178 | 9 / 25 (p=0.009) | 0.91 (0.85–0.97) |

### X memory, p = 0.004, s = 0.001, seed 5300: reference exact

| condition | failures / shots | fail fraction (95% Wilson) | per cycle | unconverged (of them right) | mean iterations | paired shots: exact fails, cond fails | exact-only / cond-only (sign test) | exact ÷ cond (95% bootstrap) |
|---|---|---|---|---|---|---|---|---|
| exact | 181 / 4000 | 0.0452 (0.0392–0.0521) | 3.85e-03 | 131 (3) | 2554 | reference |  |  |
| oracle-round | 96 / 4000 | 0.0240 (0.0197–0.0292) | 2.02e-03 | 54 (0) | 1634 | 4000: 181, 96 | 106 / 21 (p=7.5e-15) | 1.89 (1.63–2.25) |
| oracle-slot | 87 / 4000 | 0.0217 (0.0177–0.0268) | 1.83e-03 | 70 (9) | 1545 | 4000: 181, 87 | 123 / 29 (p=5.6e-15) | 2.08 (1.73–2.58) |
