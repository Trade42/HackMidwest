## Test (seed 7300): with the SECONDARY row tracker_order1_coupled2 (the version before the frozen one; not the frozen decoder)
Every condition on the same 4000 shots.

### reference exact

| condition | failures / shots | fail fraction (95% Wilson) | per cycle | unconverged (of them right) | mean iterations | paired shots: exact fails, cond fails | exact-only / cond-only (sign test) | exact ÷ cond (95% bootstrap) |
|---|---|---|---|---|---|---|---|---|
| starter | 192 / 4000 | 0.0480 (0.0418–0.0551) | 4.09e-03 | 129 (0) | 2415 | 4000: 196, 192 | 57 / 53 (p=0.78) | 1.02 (0.92–1.13) |
| exact | 196 / 4000 | 0.0490 (0.0427–0.0561) | 4.18e-03 | 145 (0) | 2599 | reference |  |  |
| tracker | 147 / 4000 | 0.0367 (0.0313–0.0430) | 3.12e-03 | 101 (2) | 2141 | 4000: 196, 147 | 81 / 32 (p=4.5e-06) | 1.33 (1.18–1.51) |
| oracle-round | 112 / 4000 | 0.0280 (0.0233–0.0336) | 2.36e-03 | 64 (0) | 1684 | 4000: 196, 112 | 103 / 19 (p=3.8e-15) | 1.75 (1.52–2.03) |
| tracker_order1_coupled2 | 144 / 4000 | 0.0360 (0.0307–0.0422) | 3.05e-03 | 98 (0) | 2188 | 4000: 196, 144 | 81 / 29 (p=7.3e-07) | 1.36 (1.21–1.55) |


### reference starter

| condition | failures / shots | fail fraction (95% Wilson) | per cycle | unconverged (of them right) | mean iterations | paired shots: starter fails, cond fails | starter-only / cond-only (sign test) | starter ÷ cond (95% bootstrap) |
|---|---|---|---|---|---|---|---|---|
| starter | 192 / 4000 | 0.0480 (0.0418–0.0551) | 4.09e-03 | 129 (0) | 2415 | reference |  |  |
| exact | 196 / 4000 | 0.0490 (0.0427–0.0561) | 4.18e-03 | 145 (0) | 2599 | 4000: 192, 196 | 53 / 57 (p=0.78) | 0.98 (0.88–1.09) |
| tracker | 147 / 4000 | 0.0367 (0.0313–0.0430) | 3.12e-03 | 101 (2) | 2141 | 4000: 192, 147 | 75 / 30 (p=1.3e-05) | 1.31 (1.17–1.48) |
| oracle-round | 112 / 4000 | 0.0280 (0.0233–0.0336) | 2.36e-03 | 64 (0) | 1684 | 4000: 192, 112 | 102 / 22 (p=1.7e-13) | 1.71 (1.48–2.01) |
| tracker_order1_coupled2 | 144 / 4000 | 0.0360 (0.0307–0.0422) | 3.05e-03 | 98 (0) | 2188 | 4000: 192, 144 | 81 / 33 (p=8e-06) | 1.33 (1.18–1.52) |


### The share of the oracle's gain

| condition | reference | failures: reference, condition, ORACLE | share of the gain recovered (95% bootstrap) |
|---|---|---|---|
| tracker | exact | 196, 147, 112 | 0.58 (0.38–0.78) |
| tracker_order1_coupled2 | exact | 196, 144, 112 | 0.62 (0.42–0.83) |
| tracker | starter | 192, 147, 112 | 0.56 (0.36–0.77) |
| tracker_order1_coupled2 | starter | 192, 144, 112 | 0.60 (0.38–0.81) |

Paired, `tracker` against `tracker_order1_coupled2`: failures 147 and 144; shots only `tracker` fails 42, only `tracker_order1_coupled2` fails 39; sign test p = 0.82.
