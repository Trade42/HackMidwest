**Exploration shots, not test shots**: X memory at p = 0.005, the team's `data/p0.005/validation_X.npz` (5,000 shots), decoded on the laptop beside a GPU training (compare the times with each other only; a time is per shot of one basis). The two PriorTilt runs disagree on 141 shots (2.8%).

| Decoder | Built by | Failures | Share of shots (95% interval) | Against relay-BP, same shots: both fail / only this row / only relay-BP (exact p) | Mean time per shot, ms |
|---|---|---|---|---|---|
| relay-BP (kit baseline) | the kit | 194 of 5,000 | 3.88% (3.38% to 4.45%) | the reference | 885 |
| PriorTilt, seed A | IonQ | 137 of 5,000 | 2.74% (2.32% to 3.23%) | 92 / 45 / 102 (p = 3e-06) | 193 |
| PriorTilt, seed B | IonQ | 152 of 5,000 | 3.04% (2.60% to 3.55%) | 107 / 45 / 87 (p = 3e-04) | 190 |
| our hierarchy, lowest-weight tie-break | ours | 129 of 5,000 | 2.58% (2.18% to 3.06%) | 110 / 19 / 84 (p = 6e-11) | 537 |
| our hierarchy, lowest weight among the converged candidates | ours | 108 of 5,000 | 2.16% (1.79% to 2.60%) | 91 / 17 / 103 (p = 3e-16) | 537 |

Source: `report/data/exploration/priortilt_p0.005/selector_validation_X.npz`.
