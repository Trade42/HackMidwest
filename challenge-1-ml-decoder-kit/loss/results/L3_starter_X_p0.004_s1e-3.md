### L3: the oracle against the starter's flag-aware decoder, s = 1e-3 (ORACLE rows use the true loss times)

| condition | failures / shots | fail fraction (95% Wilson) | per cycle | unconverged (of them right) | mean iterations | paired shots: starter fails, cond fails | starter-only / cond-only (sign test) | starter ÷ cond (95% bootstrap) |
|---|---|---|---|---|---|---|---|---|
| starter | 176 / 4000 | 0.0440 (0.0381–0.0508) | 3.74e-03 | 108 (3) | 2339 | reference |  |  |
| oracle-round | 107 / 4000 | 0.0267 (0.0222–0.0322) | 2.26e-03 | 63 (1) | 1655 | 4000: 176, 107 | 94 / 25 (p=1.3e-10) | 1.64 (1.42–1.94) |
| exact | 189 / 4000 | 0.0473 (0.0411–0.0543) | 4.03e-03 | 140 (4) | 2609 | 4000: 176, 189 | 46 / 59 (p=0.24) | 0.93 (0.83–1.04) |
| starter@0.5/stop1 | 200 / 4000 | 0.0500 (0.0437–0.0572) | 4.27e-03 | 108 (3) | 1026 | 4000: 176, 200 | 10 / 34 (p=0.00039) | 0.88 (0.82–0.95) |
