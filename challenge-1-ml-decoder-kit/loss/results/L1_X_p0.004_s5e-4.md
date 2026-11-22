### L1: X memory, p = 0.004, s = 5e-4: blind against flag-aware

| condition | failures / shots | fail fraction (95% Wilson) | per cycle | unconverged (of them right) | mean iterations | paired shots: blind fails, cond fails | blind-only / cond-only (sign test) | blind ÷ cond (95% bootstrap) |
|---|---|---|---|---|---|---|---|---|
| blind | 57 / 400 | 0.1425 (0.1116–0.1802) | 1.27e-02 | 45 (0) | 5762 | reference |  |  |
| starter | 29 / 2000 | 0.0145 (0.0101–0.0207) | 1.22e-03 | 15 (0) | 1117 | 400: 57, 5 | 53 / 1 (p=6.1e-15) | 11.40 (5.90–53.00) |
| starter@0.5/stop1 | 36 / 2000 | 0.0180 (0.0130–0.0248) | 1.51e-03 | 15 (0) | 406 | 400: 57, 5 | 53 / 1 (p=6.1e-15) | 11.40 (5.90–53.00) |
