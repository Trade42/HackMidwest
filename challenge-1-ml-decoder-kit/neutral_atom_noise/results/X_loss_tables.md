| Basis | Factor | Decoder | Picture of the noise | Failures / shots | Failure fraction (95% Wilson) | Per cycle | Unconverged | Mean iterations | Mean ms per shot |
|---|---|---|---|---|---|---|---|---|---|
| X + loss | 0.4 | relay-BP | blind + matched | 10 / 150 | 6.67% (3.66%–11.84%) | 5.73e-03 | 7 | 3065 | 2467 |
| X + loss | 0.4 | relay-BP | starter + matched | 0 / 2000 | 0.00% (0.00%–0.19%) | 0.00e+00 | 0 | 133 | 126 |
| X + loss | 0.4 | relay-BP | starter + generic, same two-qubit error (p = 0.0060) | 3 / 1000 | 0.30% (0.10%–0.88%) | 2.50e-04 | 1 | 709 | 623 |
| X + loss | 0.6 | relay-BP | starter + matched | 9 / 100 | 9.00% (4.81%–16.23%) | 7.83e-03 | 6 | 2489 | 1926 |

Paired, on the same shots (the generic row against the matched row of its point; **bold** marks the better generic row of a point):

| Basis | Factor | Decoder | Generic picture | Shots | Failures: generic, matched | Only generic wrong | Only matched wrong | Exact sign test p | Mean iterations: generic ÷ matched | Unconverged: generic, matched |
|---|---|---|---|---|---|---|---|---|---|---|
| X + loss | 0.4 | relay-BP | **starter + generic, same two-qubit error (p = 0.0060)** | 1000 | 3, 0 | 3 | 0 | 0.25 | 709 ÷ 119 = 6.0 | 1, 0 |

Iterations are belief-propagation iterations per shot, the same on any machine. Relay-BP often has its answer inside its first leg of 80 iterations and never runs more than 80 + 300 x 60 = 18,080. BP-OSD stops BP at 80 iterations and then runs OSD on every unconverged shot, so for BP-OSD the unconverged count is the costly part.

What the loss flags add, on the same shots (the row that ignores the heralds against the flag-aware row, same Pauli priors):

| Factor | Decoder | Pauli priors | Shots | Failures: no flags, with flags | Only no-flags wrong | Only with-flags wrong | Exact sign test p | Mean iterations: no flags, with flags |
|---|---|---|---|---|---|---|---|---|
| 0.4 | relay-BP | matched | 150 | 10, 0 | 10 | 0 | 0.002 | 3065, 110 |
