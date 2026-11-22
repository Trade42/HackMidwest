Dataset val_Z_p0.004_s1e-3_seed7220.npz (seed 7220), rounds counted 1..12.

| version | atoms | most probable round is the true one | within one round | mean P(true round) | mean log P(true round) | mean error (rounds) |
|---|---|---|---|---|---|---|
| tracker | 39908 | 0.507 | 0.795 | 0.379 | -1.317 | +0.18 |

Kill line (version `tracker`): losses of rounds 1 to 9 within one round of the truth: 0.779 of 30247 (rounds 2 to 10: 0.773 of 29991). The line is 0.5.

By true loss round (version `tracker`):

| true round | atoms | most probable round is the true one | within one round (95% Wilson) | mean P(true round) | mean P(within one) | mean error (rounds) |
|---|---|---|---|---|---|---|
| 1 | 3511 | 0.638 | 0.830 (0.817–0.842) | 0.548 | 0.779 | +0.76 |
| 2 | 3380 | 0.507 | 0.837 (0.824–0.849) | 0.401 | 0.801 | +0.55 |
| 3 | 3380 | 0.483 | 0.793 (0.780–0.807) | 0.366 | 0.719 | +0.45 |
| 4 | 3326 | 0.481 | 0.763 (0.749–0.778) | 0.353 | 0.680 | +0.37 |
| 5 | 3298 | 0.480 | 0.772 (0.757–0.786) | 0.344 | 0.674 | +0.31 |
| 6 | 3458 | 0.453 | 0.743 (0.729–0.758) | 0.329 | 0.653 | +0.30 |
| 7 | 3302 | 0.470 | 0.752 (0.737–0.766) | 0.331 | 0.652 | +0.14 |
| 8 | 3343 | 0.448 | 0.752 (0.737–0.766) | 0.324 | 0.655 | +0.11 |
| 9 | 3249 | 0.469 | 0.770 (0.756–0.785) | 0.337 | 0.671 | +0.09 |
| 10 | 3255 | 0.453 | 0.776 (0.762–0.790) | 0.333 | 0.692 | -0.10 |
| 11 | 3197 | 0.472 | 0.884 (0.873–0.895) | 0.365 | 0.787 | -0.28 |
| 12 | 3209 | 0.737 | 0.871 (0.859–0.882) | 0.509 | 0.738 | -0.60 |

By what the atom shares (from the heralds alone):

| group | share of the flagged atoms | atoms | true round | within one round | mean log P(true round) |
|---|---|---|---|---|---|
| shares no check with another flagged data atom | 0.220 | 8777 | 0.525 | 0.808 | -1.268 |
| shares an X check only | 0.120 | 4787 | 0.517 | 0.800 | -1.303 |
| shares a Z check only | 0.122 | 4883 | 0.510 | 0.802 | -1.305 |
| shares X and Z checks | 0.538 | 21461 | 0.498 | 0.787 | -1.343 |
| alone, and none of its 6 check atoms was ever flagged | 0.144 | 5734 | 0.532 | 0.819 | -1.238 |

Reliability of the round posterior, every (atom, round) pair (expected calibration error 0.0020):

| posterior between | pairs | mean posterior | share that is the true round |
|---|---|---|---|
| 0.00 and 0.02 | 283583 | 0.003 | 0.004 |
| 0.02 and 0.05 | 47742 | 0.033 | 0.036 |
| 0.05 and 0.10 | 38327 | 0.072 | 0.073 |
| 0.10 and 0.20 | 39187 | 0.145 | 0.145 |
| 0.20 and 0.30 | 24236 | 0.246 | 0.246 |
| 0.30 and 0.40 | 15456 | 0.348 | 0.346 |
| 0.40 and 0.50 | 10944 | 0.447 | 0.435 |
| 0.50 and 0.60 | 7937 | 0.548 | 0.539 |
| 0.60 and 0.70 | 5539 | 0.645 | 0.631 |
| 0.70 and 0.80 | 2872 | 0.745 | 0.725 |
| 0.80 and 0.90 | 1825 | 0.850 | 0.799 |
| 0.90 and 1.00 | 1248 | 0.946 | 0.922 |

Reliability of the most probable round (expected calibration error 0.0119):

| its posterior between | atoms | mean posterior | share that is the true round |
|---|---|---|---|
| 0.10 and 0.20 | 130 | 0.180 | 0.185 |
| 0.20 and 0.30 | 2756 | 0.264 | 0.254 |
| 0.30 and 0.40 | 7642 | 0.355 | 0.351 |
| 0.40 and 0.50 | 9959 | 0.449 | 0.440 |
| 0.50 and 0.60 | 7937 | 0.548 | 0.539 |
| 0.60 and 0.70 | 5539 | 0.645 | 0.631 |
| 0.70 and 0.80 | 2872 | 0.745 | 0.725 |
| 0.80 and 0.90 | 1825 | 0.850 | 0.799 |
| 0.90 and 1.00 | 1248 | 0.946 | 0.922 |

Temperature on the evidence (1 = as fitted; the tracker alone, before any decoding):

| temperature | mean log P(true round) | calibration error of the most probable round |
|---|---|---|
| 0.7 | -1.4121 | 0.1208 |
| 0.85 | -1.3407 | 0.0617 |
| 1.0 | -1.3173 | 0.0119 |
| 1.25 | -1.3282 | 0.0536 |
| 1.5 | -1.3651 | 0.1041 |
| 2.0 | -1.4569 | 0.1704 |
| 3.0 | -1.6230 | 0.2437 |

For scale, the uniform guess: mean P(true round) 0.083, mean log P -2.485.
