1500 decoded shots of val_X_p0.004_s1e-3_seed7200.npz; tracker coupled, sweeps 1, explain True.

| shots where | shots | lost data atoms per shot | mean P(true round) | smallest P(true round) in the shot, median | atoms more than one round off, per shot |
|---|---|---|---|---|---|
| all shots | 1500 | 10.0 | 0.379 | 0.046 | 2.10 |
| no decoder fails | 1412 | 10.0 | 0.381 | 0.047 | 2.05 |
| exact fails, the tracker and the ORACLE do not (the tracker's gain) | 25 | 12.2 | 0.366 | 0.032 | 2.84 |
| exact and the tracker fail, the ORACLE does not (the gain left) | 18 | 11.2 | 0.329 | 0.022 | 3.61 |
| the tracker fails, exact and the ORACLE do not | 14 | 10.9 | 0.355 | 0.046 | 2.50 |
| all three fail | 21 | 10.8 | 0.338 | 0.023 | 3.14 |
| the ORACLE fails, the tracker does not | 9 | 10.0 | 0.368 | 0.026 | 2.11 |
