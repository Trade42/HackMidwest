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

Kill line (version `order1_coupled2`): losses of rounds 1 to 9 within one round of the truth: 0.767 of 30393 (rounds 2 to 10: 0.758 of 30173). The line is 0.5.

By true loss round (version `order1_coupled2`):

| true round | atoms | most probable round is the true one | within one round (95% Wilson) | mean P(true round) | mean P(within one) | mean error (rounds) |
|---|---|---|---|---|---|---|
| 1 | 3472 | 0.631 | 0.825 (0.812–0.837) | 0.545 | 0.771 | +0.78 |
| 2 | 3462 | 0.515 | 0.846 (0.834–0.858) | 0.405 | 0.806 | +0.53 |
| 3 | 3382 | 0.493 | 0.788 (0.774–0.802) | 0.368 | 0.714 | +0.46 |
| 4 | 3329 | 0.469 | 0.759 (0.744–0.773) | 0.344 | 0.671 | +0.41 |
| 5 | 3418 | 0.462 | 0.757 (0.742–0.771) | 0.332 | 0.657 | +0.28 |
| 6 | 3376 | 0.437 | 0.737 (0.722–0.752) | 0.318 | 0.640 | +0.23 |
| 7 | 3255 | 0.433 | 0.716 (0.701–0.732) | 0.308 | 0.622 | +0.20 |
| 8 | 3430 | 0.452 | 0.734 (0.719–0.748) | 0.317 | 0.640 | +0.15 |
| 9 | 3269 | 0.432 | 0.737 (0.722–0.752) | 0.307 | 0.640 | +0.05 |
| 10 | 3252 | 0.453 | 0.744 (0.729–0.759) | 0.323 | 0.667 | -0.15 |
| 11 | 3231 | 0.448 | 0.856 (0.843–0.867) | 0.344 | 0.758 | -0.39 |
| 12 | 3288 | 0.700 | 0.846 (0.833–0.858) | 0.478 | 0.709 | -0.70 |

By what the atom shares (from the heralds alone):

| group | share of the flagged atoms | atoms | true round | within one round | mean log P(true round) |
|---|---|---|---|---|---|
| shares no check with another flagged data atom | 0.220 | 8834 | 0.534 | 0.807 | -1.271 |
| shares an X check only | 0.120 | 4802 | 0.517 | 0.794 | -1.309 |
| shares a Z check only | 0.116 | 4662 | 0.510 | 0.794 | -1.323 |
| shares X and Z checks | 0.544 | 21866 | 0.470 | 0.761 | -1.430 |
| alone, and none of its 6 check atoms was ever flagged | 0.142 | 5701 | 0.542 | 0.811 | -1.243 |

Reliability of the round posterior, every (atom, round) pair (expected calibration error 0.0031):

| posterior between | pairs | mean posterior | share that is the true round |
|---|---|---|---|
| 0.00 and 0.02 | 278874 | 0.003 | 0.005 |
| 0.02 and 0.05 | 50079 | 0.033 | 0.037 |
| 0.05 and 0.10 | 41164 | 0.072 | 0.075 |
| 0.10 and 0.20 | 41649 | 0.144 | 0.142 |
| 0.20 and 0.30 | 25154 | 0.246 | 0.246 |
| 0.30 and 0.40 | 15715 | 0.347 | 0.338 |
| 0.40 and 0.50 | 11038 | 0.447 | 0.438 |
| 0.50 and 0.60 | 7498 | 0.547 | 0.539 |
| 0.60 and 0.70 | 5194 | 0.646 | 0.615 |
| 0.70 and 0.80 | 2654 | 0.744 | 0.725 |
| 0.80 and 0.90 | 1664 | 0.849 | 0.800 |
| 0.90 and 1.00 | 1285 | 0.947 | 0.914 |

Reliability of the most probable round (expected calibration error 0.0141):

| its posterior between | atoms | mean posterior | share that is the true round |
|---|---|---|---|
| 0.05 and 0.10 | 1 | 0.098 | 1.000 |
| 0.10 and 0.20 | 269 | 0.180 | 0.152 |
| 0.20 and 0.30 | 3392 | 0.263 | 0.265 |
| 0.30 and 0.40 | 8212 | 0.354 | 0.347 |
| 0.40 and 0.50 | 9995 | 0.449 | 0.440 |
| 0.50 and 0.60 | 7498 | 0.547 | 0.539 |
| 0.60 and 0.70 | 5194 | 0.646 | 0.615 |
| 0.70 and 0.80 | 2654 | 0.744 | 0.725 |
| 0.80 and 0.90 | 1664 | 0.849 | 0.800 |
| 0.90 and 1.00 | 1285 | 0.947 | 0.914 |

Temperature on the evidence (1 = as fitted; the tracker alone, before any decoding):

| temperature | mean log P(true round) | calibration error of the most probable round |
|---|---|---|
| 0.7 | -1.4689 | 0.1217 |
| 0.85 | -1.3944 | 0.0622 |
| 1.0 | -1.3684 | 0.0141 |
| 1.25 | -1.3756 | 0.0514 |
| 1.5 | -1.4094 | 0.0988 |
| 2.0 | -1.4968 | 0.1653 |
| 3.0 | -1.6569 | 0.2366 |

For scale, the uniform guess: mean P(true round) 0.083, mean log P -2.485.
