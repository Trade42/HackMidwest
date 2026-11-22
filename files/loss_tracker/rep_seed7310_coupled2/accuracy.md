Dataset data.npz (seed 7310), rounds counted 1..12.

| version | atoms | most probable round is the true one | within one round | mean P(true round) | mean log P(true round) | mean error (rounds) |
|---|---|---|---|---|---|---|
| tracker | 40303 | 0.492 | 0.780 | 0.368 | -1.361 | +0.16 |

Kill line (version `tracker`): losses of rounds 1 to 9 within one round of the truth: 0.767 of 30611 (rounds 2 to 10: 0.757 of 30346). The line is 0.5.

By true loss round (version `tracker`):

| true round | atoms | most probable round is the true one | within one round (95% Wilson) | mean P(true round) | mean P(within one) | mean error (rounds) |
|---|---|---|---|---|---|---|
| 1 | 3474 | 0.631 | 0.833 (0.820–0.845) | 0.554 | 0.779 | +0.76 |
| 2 | 3454 | 0.508 | 0.842 (0.829–0.853) | 0.403 | 0.803 | +0.58 |
| 3 | 3480 | 0.479 | 0.777 (0.763–0.790) | 0.359 | 0.706 | +0.51 |
| 4 | 3481 | 0.458 | 0.763 (0.749–0.777) | 0.338 | 0.670 | +0.40 |
| 5 | 3419 | 0.459 | 0.757 (0.742–0.771) | 0.331 | 0.659 | +0.31 |
| 6 | 3384 | 0.452 | 0.742 (0.727–0.757) | 0.321 | 0.639 | +0.25 |
| 7 | 3365 | 0.441 | 0.732 (0.716–0.746) | 0.317 | 0.635 | +0.20 |
| 8 | 3217 | 0.433 | 0.732 (0.716–0.747) | 0.312 | 0.636 | +0.09 |
| 9 | 3337 | 0.434 | 0.722 (0.706–0.737) | 0.309 | 0.637 | -0.02 |
| 10 | 3209 | 0.443 | 0.747 (0.732–0.762) | 0.328 | 0.675 | -0.17 |
| 11 | 3190 | 0.464 | 0.856 (0.844–0.868) | 0.351 | 0.761 | -0.39 |
| 12 | 3293 | 0.696 | 0.855 (0.842–0.866) | 0.480 | 0.715 | -0.69 |

By what the atom shares (from the heralds alone):

| group | share of the flagged atoms | atoms | true round | within one round | mean log P(true round) |
|---|---|---|---|---|---|
| shares no check with another flagged data atom | 0.218 | 8767 | 0.531 | 0.813 | -1.242 |
| shares an X check only | 0.120 | 4841 | 0.495 | 0.787 | -1.353 |
| shares a Z check only | 0.123 | 4938 | 0.514 | 0.791 | -1.312 |
| shares X and Z checks | 0.540 | 21757 | 0.471 | 0.762 | -1.421 |
| alone, and none of its 6 check atoms was ever flagged | 0.143 | 5757 | 0.546 | 0.825 | -1.204 |

Reliability of the round posterior, every (atom, round) pair (expected calibration error 0.0028):

| posterior between | pairs | mean posterior | share that is the true round |
|---|---|---|---|
| 0.00 and 0.02 | 280423 | 0.003 | 0.004 |
| 0.02 and 0.05 | 50462 | 0.033 | 0.038 |
| 0.05 and 0.10 | 40883 | 0.072 | 0.074 |
| 0.10 and 0.20 | 41177 | 0.144 | 0.144 |
| 0.20 and 0.30 | 25281 | 0.246 | 0.243 |
| 0.30 and 0.40 | 15857 | 0.347 | 0.342 |
| 0.40 and 0.50 | 11148 | 0.447 | 0.440 |
| 0.50 and 0.60 | 7594 | 0.547 | 0.525 |
| 0.60 and 0.70 | 5171 | 0.646 | 0.635 |
| 0.70 and 0.80 | 2643 | 0.745 | 0.712 |
| 0.80 and 0.90 | 1711 | 0.850 | 0.800 |
| 0.90 and 1.00 | 1286 | 0.947 | 0.913 |

Reliability of the most probable round (expected calibration error 0.0166):

| its posterior between | atoms | mean posterior | share that is the true round |
|---|---|---|---|
| 0.10 and 0.20 | 233 | 0.179 | 0.163 |
| 0.20 and 0.30 | 3340 | 0.263 | 0.255 |
| 0.30 and 0.40 | 8228 | 0.354 | 0.340 |
| 0.40 and 0.50 | 10097 | 0.449 | 0.441 |
| 0.50 and 0.60 | 7594 | 0.547 | 0.525 |
| 0.60 and 0.70 | 5171 | 0.646 | 0.635 |
| 0.70 and 0.80 | 2643 | 0.745 | 0.712 |
| 0.80 and 0.90 | 1711 | 0.850 | 0.800 |
| 0.90 and 1.00 | 1286 | 0.947 | 0.913 |

Temperature on the evidence (1 = as fitted; the tracker alone, before any decoding):

| temperature | mean log P(true round) | calibration error of the most probable round |
|---|---|---|
| 0.7 | -1.4605 | 0.1236 |
| 0.85 | -1.3863 | 0.0661 |
| 1.0 | -1.3607 | 0.0166 |
| 1.25 | -1.3687 | 0.0488 |
| 1.5 | -1.4032 | 0.0972 |
| 2.0 | -1.4916 | 0.1640 |
| 3.0 | -1.6530 | 0.2359 |

For scale, the uniform guess: mean P(true round) 0.083, mean log P -2.485.
