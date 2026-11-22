300 shots of val_X_p0.004_s1e-3_seed7200.npz (9.9 flagged data atoms per shot), one process, numpy.

| tracker version | one shot at a time: mean ms | median ms | 99th percentile ms | batches of 100: ms per shot |
|---|---|---|---|---|
| order1_explain1 (the frozen tracker of step 3) | 8.13 | 7.89 | 13.91 | 4.44 |
| order1_explain2 | 9.70 | 9.47 | 14.27 | 5.51 |
| order1_coupled2 | 18.84 | 18.27 | 31.73 | 11.40 |
| order1_coupled0 | 6.57 | 6.45 | 9.29 | 3.39 |
| order0_coupled2 (independent rounds) | 4.24 | 4.05 | 6.97 | 1.44 |
