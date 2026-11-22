### Rule 1: the tracker version (4,000 validation shots)

| version | mean log P(true round) | within 0.005 of the best | chosen |
|---|---|---|---|
| order1_explain4 | -1.3250 | yes |  |
| order1_explain1 | -1.3250 | yes | **yes** |
| order1_explain2 | -1.3251 | yes |  |
| order1_explain0 | -1.3620 |  |  |
| order1_coupled4 | -1.3682 |  |  |
| order1_coupled1 | -1.3683 |  |  |
| order1_coupled2 | -1.3684 |  |  |
| order1_coupled0 | -1.4079 |  |  |
| order0_coupled2 | -1.4481 |  |  |
| order1_drop | -1.4681 |  |  |
| order1_naive | -1.5361 |  |  |
| order0_naive | -1.6762 |  |  |

### Rule 2: the decoder variant (the first 1500 validation shots, tracker version `order1_coupled2`)

| variant | failures | unconverged | mean iterations | chosen |
|---|---|---|---|---|
| tracker | 45 | 27 | 2026 | **yes** |
| tracker-T2 | 48 | 25 | 2108 |  |
| tracker-slot | 48 | 30 | 2101 |  |
| tracker-hard0.6 | 51 | 26 | 2005 |  |
| tracker-T0.5 | 52 | 30 | 2100 |  |

### Record: the chosen version `order1_explain1` decoded on the same validation shots

| variant | failures with the chosen version | with the earlier version | only the chosen fails / only the earlier fails | sign test |
|---|---|---|---|---|
| tracker | 54 / 1500 | 45 / 1500 | 22 / 13 | p = 0.18 |
