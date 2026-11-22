| Basis | Factor | Decoder | Picture of the noise | Failures / shots | Failure fraction (95% Wilson) | Per cycle | Unconverged | Mean iterations | Mean ms per shot |
|---|---|---|---|---|---|---|---|---|---|
| X | 0.6 | relay-BP | matched | 2 / 2000 | 0.10% (0.027%–0.36%) | 8.34e-05 | 0 | 156 | 155 |
| X | 0.6 | BP-OSD | matched | 5 / 2000 | 0.25% (0.11%–0.58%) | 2.09e-04 | 558 | 38 | 319 |
| X | 0.7 | relay-BP | matched | 13 / 2000 | 0.65% (0.38%–1.11%) | 5.43e-04 | 4 | 398 | 255 |
| X | 0.7 | relay-BP | generic, same two-qubit error (p = 0.0104) | 2 / 50 | 4.00% (1.10%–13.46%) | 3.40e-03 | 2 | 2801 | 2256 |
| X | 0.7 | BP-OSD | matched | 31 / 2000 | 1.55% (1.09%–2.19%) | 1.30e-03 | 840 | 49 | 456 |
| X | 0.7 | BP-OSD | generic, same two-qubit error (p = 0.0104) | 227 / 2000 | 11.35% (10.03%–12.81%) | 9.99e-03 | 1681 | 74 | 1193 |
| X | 0.8 | relay-BP | matched | 39 / 2000 | 1.95% (1.43%–2.65%) | 1.64e-03 | 18 | 978 | 787 |
| X | 0.8 | relay-BP | generic, same two-qubit error (p = 0.0119) | 112 / 500 | 22.40% (18.96%–26.26%) | 2.09e-02 | 102 | 8413 | 7593 |
| X | 0.8 | relay-BP | generic, best p on train shots (p = 0.0457) | 65 / 500 | 13.00% (10.33%–16.23%) | 1.15e-02 | 54 | 4626 | 3530 |
| X | 0.8 | BP-OSD | matched | 91 / 2000 | 4.55% (3.72%–5.55%) | 3.87e-03 | 1177 | 60 | 998 |
| X | 0.8 | BP-OSD | generic, same two-qubit error (p = 0.0119) | 613 / 2000 | 30.65% (28.67%–32.71%) | 3.00e-02 | 1883 | 78 | 1394 |
| X | 0.8 | BP-OSD | generic, best p on train shots (p = 0.0326) | 442 / 2000 | 22.10% (20.34%–23.97%) | 2.06e-02 | 1757 | 75 | 1100 |
| X | 0.9 | relay-BP | matched | 118 / 2000 | 5.90% (4.95%–7.02%) | 5.05e-03 | 66 | 2391 | 1991 |
| X | 0.9 | relay-BP | generic, same two-qubit error (p = 0.0134) | 132 / 250 | 52.80% (46.62%–58.90%) | 6.06e-02 | 126 | 13813 | 12208 |
| X | 0.9 | BP-OSD | matched | 238 / 2000 | 11.90% (10.55%–13.39%) | 1.05e-02 | 1482 | 69 | 830 |
| X | 0.9 | BP-OSD | generic, same two-qubit error (p = 0.0134) | 1113 / 2000 | 55.65% (53.46%–57.81%) | 6.55e-02 | 1965 | 79 | 1793 |
| X | 1 | relay-BP | matched | 129 / 750 | 17.20% (14.67%–20.07%) | 1.56e-02 | 88 | 5484 | 4373 |
| X | 1 | BP-OSD | matched | 599 / 2000 | 29.95% (27.98%–31.99%) | 2.92e-02 | 1750 | 75 | 963 |

Paired, on the same shots (the generic row against the matched row of its point; **bold** marks the better generic row of a point):

| Basis | Factor | Decoder | Generic picture | Shots | Failures: generic, matched | Only generic wrong | Only matched wrong | Exact sign test p | Mean iterations: generic ÷ matched | Unconverged: generic, matched |
|---|---|---|---|---|---|---|---|---|---|---|
| X | 0.7 | relay-BP | **generic, same two-qubit error (p = 0.0104)** | 50 | 2, 0 | 2 | 0 | 0.5 | 2801 ÷ 636 = 4.4 | 2, 0 |
| X | 0.7 | BP-OSD | **generic, same two-qubit error (p = 0.0104)** | 2000 | 227, 31 | 202 | 6 | 5.2e-52 | 74 ÷ 49 = 1.5 | 1681, 840 |
| X | 0.8 | relay-BP | generic, same two-qubit error (p = 0.0119) | 500 | 112, 9 | 104 | 1 | 5.2e-30 | 8413 ÷ 1113 = 7.6 | 102, 4 |
| X | 0.8 | relay-BP | **generic, best p on train shots (p = 0.0457)** | 500 | 65, 9 | 57 | 1 | 4.1e-16 | 4626 ÷ 1113 = 4.2 | 54, 4 |
| X | 0.8 | BP-OSD | generic, same two-qubit error (p = 0.0119) | 2000 | 613, 91 | 540 | 18 | 7.1e-135 | 78 ÷ 60 = 1.3 | 1883, 1177 |
| X | 0.8 | BP-OSD | **generic, best p on train shots (p = 0.0326)** | 2000 | 442, 91 | 372 | 21 | 3.6e-84 | 75 ÷ 60 = 1.3 | 1757, 1177 |
| X | 0.9 | relay-BP | **generic, same two-qubit error (p = 0.0134)** | 250 | 132, 14 | 118 | 0 | 6e-36 | 13813 ÷ 2630 = 5.3 | 126, 7 |
| X | 0.9 | BP-OSD | **generic, same two-qubit error (p = 0.0134)** | 2000 | 1113, 238 | 887 | 12 | 2.6e-244 | 79 ÷ 69 = 1.2 | 1965, 1482 |

Iterations are belief-propagation iterations per shot, the same on any machine. Relay-BP often has its answer inside its first leg of 80 iterations and never runs more than 80 + 300 x 60 = 18,080. BP-OSD stops BP at 80 iterations and then runs OSD on every unconverged shot, so for BP-OSD the unconverged count is the costly part.
