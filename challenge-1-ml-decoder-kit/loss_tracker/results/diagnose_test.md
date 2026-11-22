4000 decoded shots of test_X_p0.004_s1e-3_seed7300.npz; tracker coupled, sweeps 1, explain True.

| shots where | shots | lost data atoms per shot | mean P(true round) | smallest P(true round) in the shot, median | atoms more than one round off, per shot |
|---|---|---|---|---|---|
| all shots | 4000 | 10.0 | 0.385 | 0.048 | 2.01 |
| no decoder fails | 3760 | 9.9 | 0.386 | 0.049 | 1.97 |
| exact fails, the tracker and the ORACLE do not (the tracker's gain) | 65 | 12.6 | 0.380 | 0.041 | 2.85 |
| exact and the tracker fail, the ORACLE does not (the gain left) | 38 | 12.3 | 0.363 | 0.042 | 3.00 |
| the tracker fails, exact and the ORACLE do not | 25 | 10.8 | 0.352 | 0.043 | 2.68 |
| all three fail | 77 | 11.4 | 0.369 | 0.034 | 2.49 |
| the ORACLE fails, the tracker does not | 28 | 10.8 | 0.362 | 0.055 | 2.79 |
