### L1: X memory, p = 0.004, s = 1e-3: blind against flag-aware

| condition | failures / shots | fail fraction (95% Wilson) | per cycle | unconverged (of them right) | mean iterations | paired shots: blind fails, cond fails | blind-only / cond-only (sign test) | blind ÷ cond (95% bootstrap) |
|---|---|---|---|---|---|---|---|---|
| blind | 173 / 300 | 0.5767 (0.5201–0.6313) | 6.91e-02 | 161 (1) | 13369 | reference |  |  |
| starter | 176 / 4000 | 0.0440 (0.0381–0.0508) | 3.74e-03 | 108 (3) | 2339 | 300: 173, 15 | 158 / 0 (p=5.5e-48) | 11.53 (7.61–22.00) |
| starter@0.5/stop1 | 200 / 4000 | 0.0500 (0.0437–0.0572) | 4.27e-03 | 108 (3) | 1026 | 300: 173, 14 | 159 / 0 (p=2.7e-48) | 12.36 (8.00–24.57) |
