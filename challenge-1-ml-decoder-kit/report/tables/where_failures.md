Where the failures are. "Agree" means the two runs of PriorTilt (two random seeds) predict the same 12 bits; the hierarchy then takes that answer, and a wrong one cannot be repaired by any tie-break. X and Z memory shots are counted together.

| Shots | Set | Shots in the set | The two runs agree | They disagree | Share in "disagree" |
|---|---|---|---|---|---|
| exploration shots, p = 0.004 | all shots | 50,000 | 49,786 | 214 | 0.4% |
| exploration shots, p = 0.004 | shots wrong under one fast decode (PriorTilt, seed A) | 208 | 63 | 145 | 69.7% |
| exploration shots, p = 0.004 | shots wrong under the hierarchy, lowest-weight tie-break | 162 | 63 | 99 | 61.1% |
| test shots, p = 0.003, the timed decoder | all shots | 20,000 | 19,998 | 2 | 0.0% |
| test shots, p = 0.003, the timed decoder | shots wrong under our decoder, neural tie-break | 3 | 3 | 0 | 0.0% |
| test shots, p = 0.004, the timed decoder | all shots | 20,000 | 19,910 | 90 | 0.4% |
| test shots, p = 0.004, the timed decoder | shots wrong under our decoder, neural tie-break | 44 | 17 | 27 | 61.4% |
