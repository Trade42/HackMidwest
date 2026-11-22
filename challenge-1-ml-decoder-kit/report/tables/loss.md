X memory, p = 0.004, relay-BP at the kit's settings on shots simulated with Clifft under **our own simplified loss model** (after every CX each of its two atoms is lost with probability s; flags are perfect). It is not Infleqtion's hardware model. These are exploration numbers on shots made for this study; a shot here is one X-memory shot, not a pair. Each loss rate and machine has its own dataset, so rows of different datasets are not paired with each other.

| s per atom per CX | Dataset | Decoder | Failures | Share of shots (95% interval) | Paired on the same shots |
|---|---|---|---|---|---|
| 0.0002 | ICRN server, seed 5100 | loss-blind relay-BP (ignores the flags) | 81 of 3,000 | 2.70% (2.18% to 3.34%) |  |
| 0.0002 | ICRN server, seed 5100 | flag-aware relay-BP (the published method) | 160 of 20,000 | 0.80% (0.69% to 0.93%) | loss-blind only 63, this row only 5 (p = 8e-14); loss-blind ÷ this = 3.52 (2.53 to 5.43), on 3,000 shots |
| 0.0005 | ICRN server, seed 5200 | loss-blind relay-BP (ignores the flags) | 109 of 800 | 13.63% (11.42% to 16.18%) |  |
| 0.0005 | ICRN server, seed 5200 | flag-aware relay-BP (the published method) | 132 of 10,000 | 1.32% (1.11% to 1.56%) | loss-blind only 98, this row only 1 (p = 3e-28); loss-blind ÷ this = 9.08 (5.75 to 18.33), on 800 shots |
| 0.0005 | this PC, seed 2200 | loss-blind relay-BP (ignores the flags) | 57 of 400 | 14.25% (11.16% to 18.02%) |  |
| 0.0005 | this PC, seed 2200 | flag-aware relay-BP (the published method) | 29 of 2,000 | 1.45% (1.01% to 2.07%) | loss-blind only 53, this row only 1 (p = 6e-15); loss-blind ÷ this = 11.40 (5.90 to 53.00), on 400 shots |
| 0.001 | ICRN server, seed 5300 | loss-blind relay-BP (ignores the flags) | not measured | not measured | not measured |
| 0.001 | ICRN server, seed 5300 | flag-aware relay-BP (the published method) | 162 of 4,000 | 4.05% (3.48% to 4.71%) |  |
| 0.001 | ICRN server, seed 5300 | ORACLE: told the true loss round (a ceiling, not a decoder) | 96 of 4,000 | 2.40% (1.97% to 2.92%) | flag-aware only 90, this row only 24 (p = 4e-10); flag-aware ÷ this = 1.69 (1.44 to 2.01), on 4,000 shots |
| 0.001 | this PC, seed 2300 | loss-blind relay-BP (ignores the flags) | 173 of 300 | 57.67% (52.01% to 63.13%) |  |
| 0.001 | this PC, seed 2300 | flag-aware relay-BP (the published method) | 176 of 4,000 | 4.40% (3.81% to 5.08%) | loss-blind only 158, this row only 0 (p = 5e-48); loss-blind ÷ this = 11.53 (7.61 to 22.00), on 300 shots |
| 0.001 | this PC, seed 2300 | ORACLE: told the true loss round (a ceiling, not a decoder) | 107 of 4,000 | 2.67% (2.22% to 3.22%) | flag-aware only 94, this row only 25 (p = 1e-10); flag-aware ÷ this = 1.64 (1.42 to 1.94), on 4,000 shots |
| 0.002 | this PC, seed 2400 | loss-blind relay-BP (ignores the flags) | 148 of 150 | 98.67% (95.27% to 99.63%) |  |
| 0.002 | this PC, seed 2400 | flag-aware relay-BP (the published method) | 173 of 750 | 23.07% (20.19% to 26.21%) | loss-blind only 115, this row only 0 (p = 5e-35); loss-blind ÷ this = 4.48 (3.42 to 6.21), on 150 shots |

An ORACLE row is given the true loss round on purpose: it is a ceiling for a tracker of the loss time, not a decoder result. Sources: `report/data/loss/pc/*.json`, `report/data/loss/server/*/tables.md`.
