| Basis | Factor | Decoder | Picture of the noise | Failures / shots | Failure fraction (95% Wilson) | Per cycle | Unconverged | Mean iterations | Mean ms per shot |
|---|---|---|---|---|---|---|---|---|---|
| X + loss | 0.3 | relay-BP | starter + matched | 0 / 1000 | 0.00% (0.00%–0.38%) | 0.00e+00 | 0 | 63 | 71 |
| X + loss | 0.4 | relay-BP | blind + matched | 18 / 300 | 6.00% (3.83%–9.28%) | 5.14e-03 | 9 | 3003 | 2450 |
| X + loss | 0.4 | relay-BP | starter + matched | 0 / 2000 | 0.00% (0.00%–0.19%) | 0.00e+00 | 0 | 133 | 126 |
| X + loss | 0.4 | relay-BP | starter + generic, same two-qubit error (p = 0.0060) | 3 / 1000 | 0.30% (0.10%–0.88%) | 2.50e-04 | 1 | 709 | 623 |
| X + loss | 0.5 | relay-BP | starter + matched | 8 / 1000 | 0.80% (0.41%–1.57%) | 6.69e-04 | 3 | 493 | 416 |
| X + loss | 0.6 | relay-BP | blind + matched | 62 / 100 | 62.00% (52.21%–70.90%) | 7.75e-02 | 51 | 13547 | 12289 |
| X + loss | 0.6 | relay-BP | starter + matched | 60 / 1000 | 6.00% (4.69%–7.65%) | 5.14e-03 | 44 | 2222 | 1755 |
| X + loss | 0.6 | relay-BP | starter + generic, same two-qubit error (p = 0.0089) | 47 / 150 | 31.33% (24.45%–39.14%) | 3.08e-02 | 45 | 9611 | 8723 |
| X + loss | 0.8 | relay-BP | starter + matched | 170 / 300 | 56.67% (51.01%–62.15%) | 6.73e-02 | 152 | 13217 | 10264 |

Paired, on the same shots (the generic row against the matched row of its point; **bold** marks the better generic row of a point):

| Basis | Factor | Decoder | Generic picture | Shots | Failures: generic, matched | Only generic wrong | Only matched wrong | Exact sign test p | Mean iterations: generic ÷ matched | Unconverged: generic, matched |
|---|---|---|---|---|---|---|---|---|---|---|
| X + loss | 0.4 | relay-BP | **starter + generic, same two-qubit error (p = 0.0060)** | 1000 | 3, 0 | 3 | 0 | 0.25 | 709 ÷ 119 = 6.0 | 1, 0 |
| X + loss | 0.6 | relay-BP | **starter + generic, same two-qubit error (p = 0.0089)** | 150 | 47, 10 | 37 | 0 | 1.5e-11 | 9611 ÷ 2096 = 4.6 | 45, 6 |

Iterations are belief-propagation iterations per shot, the same on any machine. Relay-BP often has its answer inside its first leg of 80 iterations and never runs more than 80 + 300 x 60 = 18,080. BP-OSD stops BP at 80 iterations and then runs OSD on every unconverged shot, so for BP-OSD the unconverged count is the costly part.

What the loss flags add, on the same shots (the row that ignores the heralds against the flag-aware row, same Pauli priors):

| Factor | Decoder | Pauli priors | Shots | Failures: no flags, with flags | Only no-flags wrong | Only with-flags wrong | Exact sign test p | Mean iterations: no flags, with flags |
|---|---|---|---|---|---|---|---|---|
| 0.4 | relay-BP | matched | 300 | 18, 0 | 18 | 0 | 7.6e-06 | 3003, 125 |
| 0.6 | relay-BP | matched | 100 | 62, 9 | 53 | 0 | 2.2e-16 | 13547, 2489 |
