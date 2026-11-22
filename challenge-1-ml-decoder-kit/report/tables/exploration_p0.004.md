**Exploration shots, not test shots**: the team's train and validation files at p = 0.004 (`data/p0.004/train_*.npz`, `validation_*.npz`; 25,000 shots per basis). The rules were chosen on these same shots, which flatters them. Decoded on a laptop (Ryzen 9 8945HS) beside a GPU training: compare the times with each other only. In this first table a time is per shot of one basis.

| Shots | Decoder | Built by | Failures | Share of shots (95% interval) | Against relay-BP, same shots: both fail / only this row / only relay-BP (exact p) | Mean time per shot, ms |
|---|---|---|---|---|---|---|
| X memory, 25,000 shots | relay-BP (kit baseline) | the kit | 122 of 25,000 | 0.488% (0.409% to 0.582%) | the reference | 167 |
| X memory, 25,000 shots | PriorTilt, seed A | IonQ | 102 of 25,000 | 0.408% (0.336% to 0.495%) | 55 / 47 / 67 (p = 0.075) | 52.0 |
| X memory, 25,000 shots | PriorTilt, seed B | IonQ | 82 of 25,000 | 0.328% (0.264% to 0.407%) | 52 / 30 / 70 (p = 8e-05) | 52.4 |
| X memory, 25,000 shots | our hierarchy, lowest-weight tie-break | ours | 77 of 25,000 | 0.308% (0.247% to 0.385%) | 61 / 16 / 61 (p = 2e-07) | 118 |
| X memory, 25,000 shots | TCBP-JOBS (NVIDIA, GPU; answers only, no time) | NVIDIA; run by us | 53 of 25,000 | 0.212% (0.162% to 0.277%) | 45 / 8 / 77 (p = 3e-15) |  |
| X memory, 25,000 shots | our hierarchy, neural tie-break (network at training step 30,000) | ours | 66 of 25,000 | 0.264% (0.208% to 0.336%) | the laptop's table gives the count only | 118 |
| X memory, 25,000 shots | our hierarchy, neural tie-break (network at training step 40,000) | ours | 65 of 25,000 | 0.260% (0.204% to 0.331%) | the laptop's table gives the count only | 118 |
| X memory, 25,000 shots | our hierarchy, neural tie-break (network at training step 50,000) | ours | 63 of 25,000 | 0.252% (0.197% to 0.322%) | the laptop's table gives the count only | 118 |
| X memory, 25,000 shots | our hierarchy, neural tie-break (network at training step 60,000) | ours | 62 of 25,000 | 0.248% (0.194% to 0.318%) | 47 / 15 / 75 (p = 9e-11) | 118 |
| Z memory, 25,000 shots | relay-BP (kit baseline) | the kit | 123 of 25,000 | 0.492% (0.413% to 0.587%) | the reference | 171 |
| Z memory, 25,000 shots | PriorTilt, seed A | IonQ | 106 of 25,000 | 0.424% (0.351% to 0.513%) | 63 / 43 / 60 (p = 0.11) | 55.6 |
| Z memory, 25,000 shots | PriorTilt, seed B | IonQ | 107 of 25,000 | 0.428% (0.354% to 0.517%) | 63 / 44 / 60 (p = 0.14) | 57.1 |
| Z memory, 25,000 shots | our hierarchy, lowest-weight tie-break | ours | 85 of 25,000 | 0.340% (0.275% to 0.420%) | 73 / 12 / 50 (p = 1e-06) | 131 |
| Z memory, 25,000 shots | TCBP-JOBS (NVIDIA, GPU; answers only, no time) | NVIDIA; run by us | 60 of 25,000 | 0.240% (0.187% to 0.309%) | 46 / 14 / 77 (p = 1e-11) |  |

The neural tie-break against the lowest-weight tie-break on the same 25,000 X shots (network at training step 60,000, recomputed here from its per-shot logits): both fail 56, only the network rule fails 6, only the weight rule fails 21 (exact p = 0.006).

The same decodes as pairs (shot k of X with shot k of Z), the kit's score; times per pair:

| Decoder | Built by | Failed pairs | Rate per cycle ± error | 95% interval | Against relay-BP: both fail / only this row / only relay-BP (exact p) | Mean time per pair, ms |
|---|---|---|---|---|---|---|
| relay-BP (kit baseline) | the kit | 245 of 25,000 | 8.20e-04 ± 5.2e-05 | 7.2e-04 to 9.3e-04 | the reference | 338 |
| PriorTilt, seed A | IonQ | 207 of 25,000 | 6.93e-04 ± 4.8e-05 | 6.0e-04 to 7.9e-04 | 118 / 89 / 127 (p = 0.012) | 108 |
| PriorTilt, seed B | IonQ | 188 of 25,000 | 6.29e-04 ± 4.6e-05 | 5.5e-04 to 7.3e-04 | 115 / 73 / 130 (p = 8e-05) | 109 |
| our hierarchy, lowest-weight tie-break | ours | 162 of 25,000 | 5.42e-04 ± 4.3e-05 | 4.6e-04 to 6.3e-04 | 134 / 28 / 111 (p = 7e-13) | 249 |
| TCBP-JOBS (NVIDIA, GPU) | NVIDIA; run by us | 113 of 25,000 | 3.77e-04 ± 3.6e-05 | 3.1e-04 to 4.5e-04 | 91 / 22 / 154 (p = 1e-25) | no time: batch mode on a shared GPU |

Our hierarchy (lowest-weight tie-break) against TCBP-JOBS on the same 25,000 pairs: both fail 85, only ours fails 77, only TCBP-JOBS fails 28 (exact p = 2e-06). **TCBP-JOBS fails fewer pairs than our hierarchy here.**

Where the failures are: the shots on which the two PriorTilt runs predict the same 12 bits ("agree"), and the others:

| Set | Shots | The two runs agree | They disagree | Share in "disagree" |
|---|---|---|---|---|
| all shots (X and Z memory) | 50,000 | 49,786 | 214 | 0.4% |
| shots PriorTilt, seed A, gets wrong | 208 | 63 | 145 | 69.7% |
| shots still wrong after the lowest-weight tie-break | 162 | 63 | 99 | 61.1% |

Sources: `report/data/exploration/priortilt/selector_*.npz` (per shot: relay-BP and PriorTilt under two seeds), and for the rows with the network `results_network_select_step*.txt` (the laptop's own table).
