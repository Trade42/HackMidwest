### L3: the oracle on the loss time, same map as exact, s = 1e-3 (ORACLE rows use the true loss times)

| condition | failures / shots | fail fraction (95% Wilson) | per cycle | unconverged (of them right) | mean iterations | paired shots: exact fails, cond fails | exact-only / cond-only (sign test) | exact ÷ cond (95% bootstrap) |
|---|---|---|---|---|---|---|---|---|
| exact | 189 / 4000 | 0.0473 (0.0411–0.0543) | 4.03e-03 | 140 (4) | 2609 | reference |  |  |
| oracle-round | 107 / 4000 | 0.0267 (0.0222–0.0322) | 2.26e-03 | 63 (1) | 1655 | 4000: 189, 107 | 105 / 23 (p=1e-13) | 1.77 (1.52–2.07) |
