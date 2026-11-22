## X memory, p = 0.004, s = 0.001, seed 9100
Every condition on the same 40 shots.

### reference exact

| condition | failures / shots | fail fraction (95% Wilson) | per cycle | unconverged (of them right) | mean iterations | paired shots: exact fails, cond fails | exact-only / cond-only (sign test) | exact ÷ cond (95% bootstrap) |
|---|---|---|---|---|---|---|---|---|
| starter | 0 / 40 | 0.0000 (0.0000–0.0876) | 0.00e+00 | 0 (0) | 2073 | 40: 2, 0 | 2 / 0 (p=0.5) | inf (inf–inf) |
| exact | 2 / 40 | 0.0500 (0.0138–0.1650) | 4.27e-03 | 1 (0) | 2184 | reference |  |  |
| tracker | 0 / 40 | 0.0000 (0.0000–0.0876) | 0.00e+00 | 0 (0) | 1685 | 40: 2, 0 | 2 / 0 (p=0.5) | inf (inf–inf) |
| oracle-round | 0 / 40 | 0.0000 (0.0000–0.0876) | 0.00e+00 | 0 (0) | 1598 | 40: 2, 0 | 2 / 0 (p=0.5) | inf (inf–inf) |


### reference starter

| condition | failures / shots | fail fraction (95% Wilson) | per cycle | unconverged (of them right) | mean iterations | paired shots: starter fails, cond fails | starter-only / cond-only (sign test) | starter ÷ cond (95% bootstrap) |
|---|---|---|---|---|---|---|---|---|
| starter | 0 / 40 | 0.0000 (0.0000–0.0876) | 0.00e+00 | 0 (0) | 2073 | reference |  |  |
| exact | 2 / 40 | 0.0500 (0.0138–0.1650) | 4.27e-03 | 1 (0) | 2184 | 40: 0, 2 | 0 / 2 (p=0.5) | 0.00 (0.00–inf) |
| tracker | 0 / 40 | 0.0000 (0.0000–0.0876) | 0.00e+00 | 0 (0) | 1685 | 40: 0, 0 | 0 / 0 (p=1) | inf (inf–inf) |
| oracle-round | 0 / 40 | 0.0000 (0.0000–0.0876) | 0.00e+00 | 0 (0) | 1598 | 40: 0, 0 | 0 / 0 (p=1) | inf (inf–inf) |


### The share of the oracle's gain

| condition | reference | failures: reference, condition, ORACLE | share of the gain recovered (95% bootstrap) |
|---|---|---|---|
| tracker | exact | 2, 0, 0 | 1.00 (1.00–1.00) |
| tracker | starter | 0, 0, 0 | nan (nan–nan) |
