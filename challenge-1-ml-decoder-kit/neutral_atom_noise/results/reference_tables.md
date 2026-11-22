| Basis | Factor | Decoder | Picture of the noise | Failures / shots | Failure fraction (95% Wilson) | Per cycle | Unconverged | Mean iterations | Mean ms per shot |
|---|---|---|---|---|---|---|---|---|---|
| X | kit p = 0.004 | relay-BP | generic, uniform p = 0.0040 | 3 / 2000 | 0.15% (0.051%–0.44%) | 1.25e-04 | 1 | 524 | 414 |
| X | kit p = 0.004 | BP-OSD | generic, uniform p = 0.0040 | 42 / 2000 | 2.10% (1.56%–2.83%) | 1.77e-03 | 733 | 52 | 383 |

Paired, on the same shots (the generic row against the matched row of its point; **bold** marks the better generic row of a point):

| Basis | Factor | Decoder | Generic picture | Shots | Failures: generic, matched | Only generic wrong | Only matched wrong | Exact sign test p | Mean iterations: generic ÷ matched | Unconverged: generic, matched |
|---|---|---|---|---|---|---|---|---|---|---|

Iterations are belief-propagation iterations per shot, the same on any machine. Relay-BP often has its answer inside its first leg of 80 iterations and never runs more than 80 + 300 x 60 = 18,080. BP-OSD stops BP at 80 iterations and then runs OSD on every unconverged shot, so for BP-OSD the unconverged count is the costly part.
