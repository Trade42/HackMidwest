### X memory, p = 0.004, s = 0.0005, seed 5200: reference blind

| condition | failures / shots | fail fraction (95% Wilson) | per cycle | unconverged (of them right) | mean iterations | paired shots: blind fails, cond fails | blind-only / cond-only (sign test) | blind ÷ cond (95% bootstrap) |
|---|---|---|---|---|---|---|---|---|
| blind | 109 / 800 | 0.1363 (0.1142–0.1618) | 1.21e-02 | 98 (1) | 5684 | reference |  |  |
| starter | 132 / 10000 | 0.0132 (0.0111–0.0156) | 1.11e-03 | 65 (3) | 1122 | 800: 109, 12 | 98 / 1 (p=3.2e-28) | 9.08 (5.75–18.33) |
| starter@0.5/stop1 | 177 / 10000 | 0.0177 (0.0153–0.0205) | 1.49e-03 | 65 (3) | 386 | 800: 109, 17 | 95 / 3 (p=9.9e-25) | 6.41 (4.30–11.50) |

### X memory, p = 0.004, s = 0.0005, seed 5200: reference starter

| condition | failures / shots | fail fraction (95% Wilson) | per cycle | unconverged (of them right) | mean iterations | paired shots: starter fails, cond fails | starter-only / cond-only (sign test) | starter ÷ cond (95% bootstrap) |
|---|---|---|---|---|---|---|---|---|
| starter | 132 / 10000 | 0.0132 (0.0111–0.0156) | 1.11e-03 | 65 (3) | 1122 | reference |  |  |
| starter@0.5/stop1 | 177 / 10000 | 0.0177 (0.0153–0.0205) | 1.49e-03 | 65 (3) | 386 | 10000: 132, 177 | 12 / 57 (p=3.7e-08) | 0.75 (0.67–0.83) |
