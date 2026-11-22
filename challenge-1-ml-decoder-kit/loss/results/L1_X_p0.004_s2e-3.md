### L1: X memory, p = 0.004, s = 2e-3: blind against flag-aware

| condition | failures / shots | fail fraction (95% Wilson) | per cycle | unconverged (of them right) | mean iterations | paired shots: blind fails, cond fails | blind-only / cond-only (sign test) | blind ÷ cond (95% bootstrap) |
|---|---|---|---|---|---|---|---|---|
| blind | 148 / 150 | 0.9867 (0.9527–0.9963) | 3.02e-01 | 147 (0) | 18058 | reference |  |  |
| starter | 173 / 750 | 0.2307 (0.2019–0.2621) | 2.16e-02 | 151 (0) | 7663 | 150: 148, 33 | 115 / 0 (p=4.8e-35) | 4.48 (3.42–6.21) |
| starter@0.5/stop1 | 186 / 750 | 0.2480 (0.2184–0.2801) | 2.35e-02 | 151 (0) | 5048 | 150: 148, 35 | 113 / 0 (p=1.9e-34) | 4.23 (3.24–5.84) |
