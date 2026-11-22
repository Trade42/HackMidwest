Dataset val_X_p0.004_s1e-3_seed7200.npz (seed 7200), rounds counted 1..12.

| version | atoms | most probable round is the true one | within one round | mean P(true round) | mean log P(true round) | mean error (rounds) |
|---|---|---|---|---|---|---|
| order0_naive | 40164 | 0.429 | 0.712 | 0.339 | -1.676 | -0.14 |
| order0_coupled2 | 40164 | 0.469 | 0.760 | 0.359 | -1.448 | +0.20 |
| order1_naive | 40164 | 0.454 | 0.739 | 0.349 | -1.536 | +0.01 |
| order1_drop | 40164 | 0.461 | 0.747 | 0.334 | -1.468 | +0.12 |
| order1_coupled0 | 40164 | 0.481 | 0.766 | 0.354 | -1.408 | +0.13 |
| order1_coupled1 | 40164 | 0.494 | 0.779 | 0.365 | -1.368 | +0.16 |
| order1_coupled2 | 40164 | 0.494 | 0.779 | 0.367 | -1.368 | +0.16 |
| order1_coupled4 | 40164 | 0.494 | 0.779 | 0.367 | -1.368 | +0.16 |
| order1_explain0 | 40164 | 0.496 | 0.781 | 0.368 | -1.362 | +0.14 |
| order1_explain1 | 40164 | 0.509 | 0.793 | 0.379 | -1.325 | +0.17 |
| order1_explain2 | 40164 | 0.509 | 0.793 | 0.381 | -1.325 | +0.16 |
| order1_explain4 | 40164 | 0.509 | 0.793 | 0.381 | -1.325 | +0.16 |

Kill line (version `order1_explain1`): losses of rounds 1 to 9 within one round of the truth: 0.779 of 30393 (rounds 2 to 10: 0.772 of 30173). The line is 0.5.

By true loss round (version `order1_explain1`):

| true round | atoms | most probable round is the true one | within one round (95% Wilson) | mean P(true round) | mean P(within one) | mean error (rounds) |
|---|---|---|---|---|---|---|
| 1 | 3472 | 0.629 | 0.825 (0.812–0.837) | 0.544 | 0.771 | +0.77 |
| 2 | 3462 | 0.512 | 0.849 (0.836–0.860) | 0.406 | 0.809 | +0.53 |
| 3 | 3382 | 0.504 | 0.795 (0.781–0.808) | 0.374 | 0.721 | +0.44 |
| 4 | 3329 | 0.476 | 0.769 (0.754–0.783) | 0.352 | 0.682 | +0.39 |
| 5 | 3418 | 0.477 | 0.769 (0.754–0.782) | 0.343 | 0.671 | +0.25 |
| 6 | 3376 | 0.455 | 0.753 (0.738–0.767) | 0.333 | 0.659 | +0.19 |
| 7 | 3255 | 0.453 | 0.735 (0.719–0.749) | 0.324 | 0.641 | +0.20 |
| 8 | 3430 | 0.471 | 0.757 (0.742–0.771) | 0.335 | 0.661 | +0.15 |
| 9 | 3269 | 0.461 | 0.756 (0.741–0.770) | 0.327 | 0.662 | +0.06 |
| 10 | 3252 | 0.471 | 0.765 (0.751–0.780) | 0.339 | 0.684 | -0.12 |
| 11 | 3231 | 0.462 | 0.870 (0.858–0.881) | 0.360 | 0.774 | -0.32 |
| 12 | 3288 | 0.729 | 0.871 (0.859–0.882) | 0.505 | 0.736 | -0.60 |

By what the atom shares (from the heralds alone):

| group | share of the flagged atoms | atoms | true round | within one round | mean log P(true round) |
|---|---|---|---|---|---|
| shares no check with another flagged data atom | 0.220 | 8834 | 0.534 | 0.807 | -1.271 |
| shares an X check only | 0.120 | 4802 | 0.527 | 0.802 | -1.285 |
| shares a Z check only | 0.116 | 4662 | 0.516 | 0.801 | -1.298 |
| shares X and Z checks | 0.544 | 21866 | 0.493 | 0.783 | -1.362 |
| alone, and none of its 6 check atoms was ever flagged | 0.142 | 5701 | 0.542 | 0.811 | -1.243 |

Reliability of the round posterior, every (atom, round) pair (expected calibration error 0.0027):

| posterior between | pairs | mean posterior | share that is the true round |
|---|---|---|---|
| 0.00 and 0.02 | 285345 | 0.003 | 0.004 |
| 0.02 and 0.05 | 48157 | 0.033 | 0.037 |
| 0.05 and 0.10 | 38789 | 0.072 | 0.075 |
| 0.10 and 0.20 | 39355 | 0.144 | 0.142 |
| 0.20 and 0.30 | 24258 | 0.246 | 0.244 |
| 0.30 and 0.40 | 15432 | 0.348 | 0.342 |
| 0.40 and 0.50 | 11024 | 0.448 | 0.442 |
| 0.50 and 0.60 | 7994 | 0.547 | 0.542 |
| 0.60 and 0.70 | 5461 | 0.646 | 0.621 |
| 0.70 and 0.80 | 3018 | 0.744 | 0.722 |
| 0.80 and 0.90 | 1809 | 0.849 | 0.807 |
| 0.90 and 1.00 | 1326 | 0.946 | 0.916 |

Reliability of the most probable round (expected calibration error 0.0120):

| its posterior between | atoms | mean posterior | share that is the true round |
|---|---|---|---|
| 0.10 and 0.20 | 145 | 0.184 | 0.166 |
| 0.20 and 0.30 | 2754 | 0.265 | 0.267 |
| 0.30 and 0.40 | 7702 | 0.355 | 0.349 |
| 0.40 and 0.50 | 9955 | 0.449 | 0.444 |
| 0.50 and 0.60 | 7994 | 0.547 | 0.542 |
| 0.60 and 0.70 | 5461 | 0.646 | 0.621 |
| 0.70 and 0.80 | 3018 | 0.744 | 0.722 |
| 0.80 and 0.90 | 1809 | 0.849 | 0.807 |
| 0.90 and 1.00 | 1326 | 0.946 | 0.916 |

Temperature on the evidence (1 = as fitted; the tracker alone, before any decoding):

| temperature | mean log P(true round) | calibration error of the most probable round |
|---|---|---|
| 0.7 | -1.4233 | 0.1208 |
| 0.85 | -1.3498 | 0.0606 |
| 1.0 | -1.3250 | 0.0120 |
| 1.25 | -1.3345 | 0.0555 |
| 1.5 | -1.3705 | 0.1042 |
| 2.0 | -1.4614 | 0.1725 |
| 3.0 | -1.6263 | 0.2461 |

For scale, the uniform guess: mean P(true round) 0.083, mean log P -2.485.
