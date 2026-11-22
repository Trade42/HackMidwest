Dataset test_X_p0.004_s1e-3_seed7300.npz (seed 7300), rounds counted 1..12.

| version | atoms | most probable round is the true one | within one round | mean P(true round) | mean log P(true round) | mean error (rounds) |
|---|---|---|---|---|---|---|
| order1_explain1 | 39909 | 0.514 | 0.798 | 0.383 | -1.305 | +0.16 |

Kill line (version `order1_explain1`): losses of rounds 1 to 9 within one round of the truth: 0.784 of 30237 (rounds 2 to 10: 0.778 of 30016). The line is 0.5.

By true loss round (version `order1_explain1`):

| true round | atoms | most probable round is the true one | within one round (95% Wilson) | mean P(true round) | mean P(within one) | mean error (rounds) |
|---|---|---|---|---|---|---|
| 1 | 3489 | 0.625 | 0.828 (0.815–0.840) | 0.550 | 0.774 | +0.78 |
| 2 | 3393 | 0.513 | 0.853 (0.840–0.864) | 0.403 | 0.807 | +0.51 |
| 3 | 3415 | 0.507 | 0.792 (0.778–0.805) | 0.379 | 0.724 | +0.39 |
| 4 | 3346 | 0.484 | 0.774 (0.759–0.788) | 0.349 | 0.683 | +0.34 |
| 5 | 3321 | 0.482 | 0.766 (0.751–0.780) | 0.347 | 0.674 | +0.26 |
| 6 | 3420 | 0.459 | 0.755 (0.740–0.769) | 0.336 | 0.661 | +0.23 |
| 7 | 3254 | 0.474 | 0.753 (0.738–0.768) | 0.337 | 0.657 | +0.19 |
| 8 | 3288 | 0.465 | 0.764 (0.749–0.778) | 0.331 | 0.660 | +0.10 |
| 9 | 3311 | 0.470 | 0.764 (0.749–0.778) | 0.332 | 0.668 | +0.02 |
| 10 | 3268 | 0.474 | 0.777 (0.763–0.791) | 0.341 | 0.689 | -0.11 |
| 11 | 3249 | 0.497 | 0.889 (0.877–0.899) | 0.373 | 0.788 | -0.25 |
| 12 | 3155 | 0.724 | 0.868 (0.855–0.879) | 0.509 | 0.740 | -0.62 |

By what the atom shares (from the heralds alone):

| group | share of the flagged atoms | atoms | true round | within one round | mean log P(true round) |
|---|---|---|---|---|---|
| shares no check with another flagged data atom | 0.219 | 8743 | 0.525 | 0.808 | -1.262 |
| shares an X check only | 0.120 | 4773 | 0.531 | 0.810 | -1.275 |
| shares a Z check only | 0.119 | 4758 | 0.518 | 0.808 | -1.310 |
| shares X and Z checks | 0.542 | 21635 | 0.505 | 0.790 | -1.328 |
| alone, and none of its 6 check atoms was ever flagged | 0.144 | 5734 | 0.536 | 0.819 | -1.227 |

Reliability of the round posterior, every (atom, round) pair (expected calibration error 0.0021):

| posterior between | pairs | mean posterior | share that is the true round |
|---|---|---|---|
| 0.00 and 0.02 | 284077 | 0.003 | 0.004 |
| 0.02 and 0.05 | 47500 | 0.033 | 0.035 |
| 0.05 and 0.10 | 38750 | 0.072 | 0.071 |
| 0.10 and 0.20 | 38614 | 0.144 | 0.142 |
| 0.20 and 0.30 | 24095 | 0.246 | 0.248 |
| 0.30 and 0.40 | 15254 | 0.347 | 0.350 |
| 0.40 and 0.50 | 11062 | 0.447 | 0.439 |
| 0.50 and 0.60 | 7894 | 0.548 | 0.543 |
| 0.60 and 0.70 | 5521 | 0.645 | 0.633 |
| 0.70 and 0.80 | 2978 | 0.744 | 0.720 |
| 0.80 and 0.90 | 1847 | 0.850 | 0.814 |
| 0.90 and 1.00 | 1316 | 0.947 | 0.919 |

Reliability of the most probable round (expected calibration error 0.0105):

| its posterior between | atoms | mean posterior | share that is the true round |
|---|---|---|---|
| 0.10 and 0.20 | 129 | 0.183 | 0.233 |
| 0.20 and 0.30 | 2659 | 0.264 | 0.263 |
| 0.30 and 0.40 | 7524 | 0.355 | 0.362 |
| 0.40 and 0.50 | 10041 | 0.449 | 0.442 |
| 0.50 and 0.60 | 7894 | 0.548 | 0.543 |
| 0.60 and 0.70 | 5521 | 0.645 | 0.633 |
| 0.70 and 0.80 | 2978 | 0.744 | 0.720 |
| 0.80 and 0.90 | 1847 | 0.850 | 0.814 |
| 0.90 and 1.00 | 1316 | 0.947 | 0.919 |

Temperature on the evidence (1 = as fitted; the tracker alone, before any decoding):

| temperature | mean log P(true round) | calibration error of the most probable round |
|---|---|---|
| 0.7 | -1.3949 | 0.1166 |
| 0.85 | -1.3265 | 0.0575 |
| 1.0 | -1.3053 | 0.0105 |
| 1.25 | -1.3186 | 0.0581 |
| 1.5 | -1.3572 | 0.1077 |
| 2.0 | -1.4511 | 0.1735 |
| 3.0 | -1.6192 | 0.2469 |

For scale, the uniform guess: mean P(true round) 0.083, mean log P -2.485.
