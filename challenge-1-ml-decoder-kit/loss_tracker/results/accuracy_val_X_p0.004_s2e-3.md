Dataset val_X_p0.004_s2e-3_seed7240.npz (seed 7240), rounds counted 1..12.

| version | atoms | most probable round is the true one | within one round | mean P(true round) | mean log P(true round) | mean error (rounds) |
|---|---|---|---|---|---|---|
| tracker | 77414 | 0.462 | 0.754 | 0.334 | -1.446 | +0.24 |
| tables_of_1e-3 | 77414 | 0.461 | 0.753 | 0.336 | -1.446 | +0.23 |

Kill line (version `tracker`): losses of rounds 1 to 9 within one round of the truth: 0.740 of 59133 (rounds 2 to 10: 0.730 of 58534). The line is 0.5.

By true loss round (version `tracker`):

| true round | atoms | most probable round is the true one | within one round (95% Wilson) | mean P(true round) | mean P(within one) | mean error (rounds) |
|---|---|---|---|---|---|---|
| 1 | 6797 | 0.610 | 0.808 (0.798–0.817) | 0.520 | 0.744 | +0.90 |
| 2 | 6758 | 0.466 | 0.808 (0.798–0.817) | 0.362 | 0.761 | +0.71 |
| 3 | 6668 | 0.450 | 0.757 (0.746–0.767) | 0.330 | 0.676 | +0.59 |
| 4 | 6669 | 0.431 | 0.728 (0.717–0.739) | 0.310 | 0.634 | +0.53 |
| 5 | 6670 | 0.423 | 0.719 (0.708–0.729) | 0.298 | 0.616 | +0.48 |
| 6 | 6498 | 0.426 | 0.716 (0.705–0.727) | 0.290 | 0.605 | +0.34 |
| 7 | 6384 | 0.413 | 0.701 (0.690–0.712) | 0.286 | 0.598 | +0.25 |
| 8 | 6320 | 0.406 | 0.706 (0.694–0.717) | 0.280 | 0.598 | +0.16 |
| 9 | 6369 | 0.411 | 0.708 (0.696–0.719) | 0.281 | 0.608 | +0.05 |
| 10 | 6198 | 0.418 | 0.726 (0.714–0.737) | 0.291 | 0.638 | -0.15 |
| 11 | 6093 | 0.417 | 0.844 (0.835–0.853) | 0.315 | 0.731 | -0.41 |
| 12 | 5990 | 0.673 | 0.828 (0.818–0.837) | 0.436 | 0.668 | -0.79 |

By what the atom shares (from the heralds alone):

| group | share of the flagged atoms | atoms | true round | within one round | mean log P(true round) |
|---|---|---|---|---|---|
| shares no check with another flagged data atom | 0.050 | 3833 | 0.514 | 0.789 | -1.307 |
| shares an X check only | 0.066 | 5137 | 0.494 | 0.781 | -1.353 |
| shares a Z check only | 0.065 | 5065 | 0.495 | 0.791 | -1.356 |
| shares X and Z checks | 0.819 | 63379 | 0.453 | 0.746 | -1.469 |
| alone, and none of its 6 check atoms was ever flagged | 0.021 | 1596 | 0.529 | 0.800 | -1.258 |

Reliability of the round posterior, every (atom, round) pair (expected calibration error 0.0015):

| posterior between | pairs | mean posterior | share that is the true round |
|---|---|---|---|
| 0.00 and 0.02 | 501258 | 0.004 | 0.005 |
| 0.02 and 0.05 | 107696 | 0.033 | 0.034 |
| 0.05 and 0.10 | 90729 | 0.072 | 0.072 |
| 0.10 and 0.20 | 93782 | 0.144 | 0.142 |
| 0.20 and 0.30 | 54442 | 0.246 | 0.248 |
| 0.30 and 0.40 | 32504 | 0.346 | 0.344 |
| 0.40 and 0.50 | 20598 | 0.446 | 0.443 |
| 0.50 and 0.60 | 12701 | 0.546 | 0.544 |
| 0.60 and 0.70 | 7737 | 0.645 | 0.629 |
| 0.70 and 0.80 | 3672 | 0.743 | 0.721 |
| 0.80 and 0.90 | 2184 | 0.849 | 0.816 |
| 0.90 and 1.00 | 1665 | 0.952 | 0.914 |

Reliability of the most probable round (expected calibration error 0.0062):

| its posterior between | atoms | mean posterior | share that is the true round |
|---|---|---|---|
| 0.10 and 0.20 | 1049 | 0.180 | 0.189 |
| 0.20 and 0.30 | 10278 | 0.260 | 0.261 |
| 0.30 and 0.40 | 19126 | 0.352 | 0.350 |
| 0.40 and 0.50 | 19002 | 0.447 | 0.444 |
| 0.50 and 0.60 | 12701 | 0.546 | 0.544 |
| 0.60 and 0.70 | 7737 | 0.645 | 0.629 |
| 0.70 and 0.80 | 3672 | 0.743 | 0.721 |
| 0.80 and 0.90 | 2184 | 0.849 | 0.816 |
| 0.90 and 1.00 | 1665 | 0.952 | 0.914 |

Temperature on the evidence (1 = as fitted; the tracker alone, before any decoding):

| temperature | mean log P(true round) | calibration error of the most probable round |
|---|---|---|
| 0.7 | -1.5284 | 0.1110 |
| 0.85 | -1.4648 | 0.0537 |
| 1.0 | -1.4455 | 0.0062 |
| 1.25 | -1.4585 | 0.0558 |
| 1.5 | -1.4944 | 0.1009 |
| 2.0 | -1.5813 | 0.1619 |
| 3.0 | -1.7361 | 0.2281 |

For scale, the uniform guess: mean P(true round) 0.083, mean log P -2.485.
