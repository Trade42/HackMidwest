Dataset train_X_p0.004_s1e-3_seed7100.npz (seed 7100), first 20000 shots; 200376 lost data atoms (10.02 per shot). Each data atom shares its 6 checks with 21 other data atoms.

| Of the lost data atoms | share |
|---|---|
| another lost atom on an X check | 0.661 |
| another lost atom on a Z check | 0.662 |
| another lost atom on any check | 0.780 |
| a flagged check atom next to it | 0.350 |
| alone in every sense | 0.143 |
| a check already randomised when it was lost | 0.482 |
| another lost atom on X and on Z checks | 0.543 |

Triple-rounds from the loss round on, by the number of the 3 detector bits that are dirty (flagged check atom, hook, or another lost atom):

| check type | 0 dirty (clean) | 1 | 2 | 3 |
|---|---|---|---|---|
| X | 0.458 | 0.372 | 0.122 | 0.048 |
| Z | 0.478 | 0.364 | 0.110 | 0.047 |

Clean triples of lost atoms, rounds 2..12 of 13 (detector rounds 1..11): firing rate of each detector, in the atom's CX order, and the joint pattern.

| side | check type (CX positions) | round − loss round | triples | rate of detector 1 / 2 / 3 | none fires | exactly one | exactly two | all three |
|---|---|---|---|---|---|---|---|---|
| left | X (1, 5, 6) | <= -2 | 242003 | 0.145 / 0.144 / 0.144 | 0.642 | 0.291 | 0.058 | 0.009 |
| left | X (1, 5, 6) | -1 | 46793 | 0.145 / 0.140 / 0.146 | 0.644 | 0.289 | 0.058 | 0.008 |
| left | X (1, 5, 6) | 0 (the loss round) | 49763 | 0.143 / 0.379 / 0.440 | 0.394 | 0.290 | 0.275 | 0.041 |
| left | X (1, 5, 6) | +1 | 50036 | 0.499 / 0.500 / 0.502 | 0.193 | 0.288 | 0.343 | 0.176 |
| left | X (1, 5, 6) | +2 | 43271 | 0.500 / 0.499 / 0.500 | 0.195 | 0.307 | 0.302 | 0.196 |
| left | X (1, 5, 6) | >= +3 | 163388 | 0.500 / 0.500 / 0.500 | 0.197 | 0.303 | 0.303 | 0.197 |
| left | Z (2, 3, 4) | <= -2 | 244944 | 0.143 / 0.143 / 0.144 | 0.653 | 0.280 | 0.052 | 0.015 |
| left | Z (2, 3, 4) | -1 | 47482 | 0.144 / 0.145 / 0.146 | 0.650 | 0.282 | 0.053 | 0.016 |
| left | Z (2, 3, 4) | 0 (the loss round) | 50468 | 0.205 / 0.263 / 0.321 | 0.500 | 0.286 | 0.138 | 0.076 |
| left | Z (2, 3, 4) | +1 | 50767 | 0.501 / 0.502 / 0.498 | 0.290 | 0.211 | 0.208 | 0.292 |
| left | Z (2, 3, 4) | +2 | 43844 | 0.496 / 0.498 / 0.500 | 0.338 | 0.164 | 0.162 | 0.335 |
| left | Z (2, 3, 4) | >= +3 | 165606 | 0.500 / 0.499 / 0.499 | 0.338 | 0.163 | 0.164 | 0.336 |
| right | X (3, 4, 5) | <= -2 | 240627 | 0.144 / 0.144 / 0.144 | 0.652 | 0.280 | 0.053 | 0.015 |
| right | X (3, 4, 5) | -1 | 46519 | 0.144 / 0.142 / 0.143 | 0.654 | 0.277 | 0.054 | 0.015 |
| right | X (3, 4, 5) | 0 (the loss round) | 49558 | 0.264 / 0.321 / 0.382 | 0.446 | 0.270 | 0.157 | 0.128 |
| right | X (3, 4, 5) | +1 | 49804 | 0.500 / 0.501 / 0.502 | 0.287 | 0.214 | 0.208 | 0.291 |
| right | X (3, 4, 5) | +2 | 43067 | 0.497 / 0.498 / 0.497 | 0.339 | 0.164 | 0.162 | 0.335 |
| right | X (3, 4, 5) | >= +3 | 164097 | 0.499 / 0.499 / 0.497 | 0.339 | 0.162 | 0.165 | 0.334 |
| right | Z (1, 2, 6) | <= -2 | 244177 | 0.145 / 0.143 / 0.145 | 0.644 | 0.289 | 0.058 | 0.009 |
| right | Z (1, 2, 6) | -1 | 47197 | 0.144 / 0.142 / 0.145 | 0.646 | 0.286 | 0.058 | 0.009 |
| right | Z (1, 2, 6) | 0 (the loss round) | 50205 | 0.141 / 0.201 / 0.445 | 0.392 | 0.449 | 0.140 | 0.019 |
| right | Z (1, 2, 6) | +1 | 50457 | 0.504 / 0.506 / 0.496 | 0.182 | 0.316 | 0.316 | 0.186 |
| right | Z (1, 2, 6) | +2 | 43619 | 0.497 / 0.497 / 0.501 | 0.197 | 0.307 | 0.300 | 0.196 |
| right | Z (1, 2, 6) | >= +3 | 165723 | 0.498 / 0.499 / 0.499 | 0.197 | 0.306 | 0.302 | 0.195 |

An atom lost three or more cycles ago, clean triples (the fitted tables; patterns are written detector 1, 2, 3):

| side | check type (CX positions) | pattern probabilities 000, 100, 010, 110, 001, 101, 011, 111 | the same for a present atom (round 6) |
|---|---|---|---|
| left | X (1, 5, 6) | 0.197 0.196 0.053 0.053 0.053 0.053 0.197 0.197 | 0.644 0.103 0.092 0.017 0.094 0.016 0.025 0.009 |
| left | Z (2, 3, 4) | 0.338 0.053 0.054 0.056 0.056 0.055 0.053 0.336 | 0.651 0.093 0.092 0.019 0.096 0.017 0.017 0.015 |
| right | X (3, 4, 5) | 0.339 0.053 0.055 0.055 0.053 0.056 0.054 0.334 | 0.651 0.093 0.094 0.018 0.094 0.018 0.017 0.015 |
| right | Z (1, 2, 6) | 0.197 0.054 0.055 0.196 0.197 0.053 0.054 0.195 | 0.644 0.094 0.092 0.025 0.103 0.016 0.016 0.009 |

The same atoms: P(detector m of round r = detector m' of round r − 1). 0.5 means unrelated.

| side | check type | m = 1: m' = 1, 2, 3 | m = 2: m' = 1, 2, 3 | m = 3: m' = 1, 2, 3 | present atom, largest departure from its no-relation value |
|---|---|---|---|---|---|
| left | X | 0.499, 0.775, 0.774 | 0.498, 0.499, 0.499 | 0.498, 0.499, 0.500 | 0.015 |
| left | Z | 0.499, 0.498, 0.499 | 0.500, 0.499, 0.500 | 0.499, 0.497, 0.498 | 0.015 |
| right | X | 0.501, 0.500, 0.501 | 0.499, 0.499, 0.499 | 0.500, 0.498, 0.499 | 0.015 |
| right | Z | 0.499, 0.499, 0.773 | 0.500, 0.498, 0.777 | 0.501, 0.500, 0.499 | 0.015 |

A data atom that was never lost, its triple otherwise clean: the pattern when a check atom of the OTHER type next to it was flagged in this cycle or the one before.

| side | check type of the triple | flagged other-type neighbour | triple-rounds | none fires | exactly one | exactly two | all three |
|---|---|---|---|---|---|---|---|
| left | X | no | 9624964 | 0.6660 | 0.2732 | 0.0526 | 0.0082 |
| left | X | yes | 322775 | 0.5324 | 0.3222 | 0.1272 | 0.0182 |
| left | Z | no | 8072661 | 0.6505 | 0.2816 | 0.0525 | 0.0154 |
| left | Z | yes | 296431 | 0.5647 | 0.2824 | 0.0984 | 0.0544 |
| right | X | no | 9624368 | 0.6729 | 0.2653 | 0.0473 | 0.0145 |
| right | X | yes | 322297 | 0.5751 | 0.2732 | 0.0956 | 0.0561 |
| right | Z | no | 8072882 | 0.6433 | 0.2898 | 0.0580 | 0.0089 |
| right | Z | yes | 296682 | 0.5236 | 0.3300 | 0.1275 | 0.0190 |

A check atom flagged in cycle c: the firing rate of its own detectors in rounds c and c + 1: X checks 0.499 and 0.502 (103429 and 103429 reads); Z checks 0.500 and 0.501 (94973 and 95025 reads).

A data atom that was never lost, with exactly one of its 3 bits on a check that another lost atom has already left: that bit, and the other two bits.

| side | check type | bit | triple-rounds | rate of that bit | other two bits 00/10/01/11 when it is 0 | when it is 1 | largest difference |
|---|---|---|---|---|---|---|---|
| left | X | 1 | 1740977 | 0.500 | 0.762 0.103 0.104 0.031 | 0.762 0.103 0.104 0.031 | 0.001 |
| left | X | 2 | 1697194 | 0.500 | 0.754 0.112 0.112 0.022 | 0.752 0.113 0.113 0.022 | 0.001 |
| left | X | 3 | 1684373 | 0.500 | 0.756 0.111 0.110 0.023 | 0.756 0.111 0.109 0.023 | 0.001 |
| left | Z | 1 | 1484652 | 0.500 | 0.743 0.112 0.113 0.032 | 0.743 0.112 0.112 0.032 | 0.001 |
| left | Z | 2 | 1473791 | 0.500 | 0.742 0.113 0.113 0.032 | 0.743 0.113 0.112 0.032 | 0.001 |
| left | Z | 3 | 1466110 | 0.500 | 0.746 0.111 0.110 0.034 | 0.747 0.110 0.109 0.034 | 0.001 |
| right | X | 1 | 1719979 | 0.500 | 0.761 0.105 0.104 0.030 | 0.761 0.105 0.105 0.029 | 0.001 |
| right | X | 2 | 1707182 | 0.500 | 0.762 0.104 0.104 0.030 | 0.763 0.104 0.104 0.030 | 0.000 |
| right | X | 3 | 1700541 | 0.500 | 0.762 0.104 0.104 0.031 | 0.762 0.103 0.104 0.031 | 0.000 |
| right | Z | 1 | 1494850 | 0.500 | 0.738 0.117 0.119 0.026 | 0.739 0.117 0.119 0.026 | 0.000 |
| right | Z | 2 | 1485698 | 0.500 | 0.734 0.121 0.120 0.025 | 0.735 0.120 0.120 0.025 | 0.001 |
| right | Z | 3 | 1445591 | 0.500 | 0.745 0.110 0.110 0.034 | 0.747 0.110 0.109 0.034 | 0.002 |
