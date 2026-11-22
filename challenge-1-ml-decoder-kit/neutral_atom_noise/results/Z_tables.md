| Basis | Factor | Decoder | Picture of the noise | Failures / shots | Failure fraction (95% Wilson) | Per cycle | Unconverged | Mean iterations | Mean ms per shot |
|---|---|---|---|---|---|---|---|---|---|
| Z | 0.7 | relay-BP | matched | 3 / 2000 | 0.15% (0.051%–0.44%) | 1.25e-04 | 1 | 193 | 148 |
| Z | 0.7 | BP-OSD | matched | 14 / 2000 | 0.70% (0.42%–1.17%) | 5.85e-04 | 788 | 46 | 416 |
| Z | 0.8 | relay-BP | matched | 13 / 2000 | 0.65% (0.38%–1.11%) | 5.43e-04 | 2 | 422 | 297 |
| Z | 0.8 | relay-BP | generic, same two-qubit error (p = 0.0119) | 67 / 500 | 13.40% (10.69%–16.67%) | 1.19e-02 | 65 | 6699 | 5650 |
| Z | 0.8 | BP-OSD | matched | 32 / 2000 | 1.60% (1.14%–2.25%) | 1.34e-03 | 1009 | 54 | 576 |
| Z | 0.8 | BP-OSD | generic, same two-qubit error (p = 0.0119) | 440 / 2000 | 22.00% (20.24%–23.87%) | 2.05e-02 | 1838 | 77 | 1175 |

Paired, on the same shots (the generic row against the matched row of its point; **bold** marks the better generic row of a point):

| Basis | Factor | Decoder | Generic picture | Shots | Failures: generic, matched | Only generic wrong | Only matched wrong | Exact sign test p | Mean iterations: generic ÷ matched | Unconverged: generic, matched |
|---|---|---|---|---|---|---|---|---|---|---|
| Z | 0.8 | relay-BP | **generic, same two-qubit error (p = 0.0119)** | 500 | 67, 4 | 64 | 1 | 3.6e-18 | 6699 ÷ 447 = 15.0 | 65, 0 |
| Z | 0.8 | BP-OSD | **generic, same two-qubit error (p = 0.0119)** | 2000 | 440, 32 | 415 | 7 | 8.4e-113 | 77 ÷ 54 = 1.4 | 1838, 1009 |

Iterations are belief-propagation iterations per shot, the same on any machine. Relay-BP often has its answer inside its first leg of 80 iterations and never runs more than 80 + 300 x 60 = 18,080. BP-OSD stops BP at 80 iterations and then runs OSD on every unconverged shot, so for BP-OSD the unconverged count is the costly part.
