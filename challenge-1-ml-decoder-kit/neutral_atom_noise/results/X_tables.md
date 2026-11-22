| Basis | Factor | Decoder | Picture of the noise | Failures / shots | Failure fraction (95% Wilson) | Per cycle | Unconverged | Mean iterations | Mean ms per shot |
|---|---|---|---|---|---|---|---|---|---|
| X | 0.6 | relay-BP | matched | 2 / 2000 | 0.10% (0.027%–0.36%) | 8.34e-05 | 0 | 156 | 155 |
| X | 0.6 | BP-OSD | matched | 5 / 2000 | 0.25% (0.11%–0.58%) | 2.09e-04 | 558 | 38 | 319 |
| X | 0.7 | relay-BP | matched | 13 / 2000 | 0.65% (0.38%–1.11%) | 5.43e-04 | 4 | 398 | 255 |
| X | 0.7 | BP-OSD | matched | 31 / 2000 | 1.55% (1.09%–2.19%) | 1.30e-03 | 840 | 49 | 456 |
| X | 0.7 | BP-OSD | generic, same two-qubit error (p = 0.0104) | 26 / 375 | 6.93% (4.78%–9.96%) | 5.97e-03 | 310 | 73 | 1259 |
| X | 0.8 | relay-BP | matched | 39 / 2000 | 1.95% (1.43%–2.65%) | 1.64e-03 | 18 | 978 | 787 |
| X | 0.8 | BP-OSD | matched | 91 / 2000 | 4.55% (3.72%–5.55%) | 3.87e-03 | 1177 | 60 | 998 |
| X | 0.8 | BP-OSD | generic, same two-qubit error (p = 0.0119), also the best p on train shots | 613 / 2000 | 30.65% (28.67%–32.71%) | 3.00e-02 | 1883 | 78 | 1394 |
| X | 0.9 | relay-BP | matched | 31 / 500 | 6.20% (4.40%–8.67%) | 5.32e-03 | 16 | 2589 | 2464 |
| X | 0.9 | BP-OSD | matched | 238 / 2000 | 11.90% (10.55%–13.39%) | 1.05e-02 | 1482 | 69 | 830 |
| X | 0.9 | BP-OSD | generic, same two-qubit error (p = 0.0134) | 1113 / 2000 | 55.65% (53.46%–57.81%) | 6.55e-02 | 1965 | 79 | 1793 |
| X | 1 | relay-BP | matched | 42 / 250 | 16.80% (12.68%–21.93%) | 1.52e-02 | 30 | 5481 | 4967 |
| X | 1 | BP-OSD | matched | 599 / 2000 | 29.95% (27.98%–31.99%) | 2.92e-02 | 1750 | 75 | 963 |

Paired, on the same shots (the generic row against the matched row of its point; **bold** marks the better generic row of a point):

| Basis | Factor | Decoder | Generic picture | Shots | Failures: generic, matched | Only generic wrong | Only matched wrong | Exact sign test p | Mean iterations: generic ÷ matched | Unconverged: generic, matched |
|---|---|---|---|---|---|---|---|---|---|---|
| X | 0.7 | BP-OSD | **generic, same two-qubit error (p = 0.0104)** | 375 | 26, 9 | 21 | 4 | 0.00091 | 73 ÷ 47 = 1.6 | 310, 149 |
| X | 0.8 | BP-OSD | **generic, same two-qubit error (p = 0.0119), also the best p on train shots** | 2000 | 613, 91 | 540 | 18 | 7.1e-135 | 78 ÷ 60 = 1.3 | 1883, 1177 |
| X | 0.9 | BP-OSD | **generic, same two-qubit error (p = 0.0134)** | 2000 | 1113, 238 | 887 | 12 | 2.6e-244 | 79 ÷ 69 = 1.2 | 1965, 1482 |

Iterations are belief-propagation iterations per shot, the same on any machine. Relay-BP often has its answer inside its first leg of 80 iterations and never runs more than 80 + 300 x 60 = 18,080. BP-OSD stops BP at 80 iterations and then runs OSD on every unconverged shot, so for BP-OSD the unconverged count is the costly part.
