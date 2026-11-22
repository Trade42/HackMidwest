# The network in fixed point, on the team's validation shots

Made by `python fpga_network/widths.py --checkpoint <checkpoint> --out <folder> --table` from the runs listed there. Checkpoint step 120000 (sha256 `d8d1c70f55223330...`). Shots: `challenge-1-ml-decoder-kit/data/p0.004/validation_X.npz` (5,000 Stim circuit shots, X memory, p = 0.004, seed 201; sha256 `b00ee09c35ed088a...`). Ranges calibrated on the first 200 shots of `train_X.npz`. Machine: luminaty (Ryzen 7 7730U), CPU. Every number here is **simulated** in software (torch, float64 holding whole numbers exactly), not on an FPGA.

A shot fails when any of its 12 logical bits is wrong (evaluate_stim's number). "Answers changed" counts the shots whose 12 predicted bits differ from the float network's on the same shot. The tie-break columns use the laptop's three candidates per shot (relay-BP, PriorTilt twice) and the hierarchy's network rule, on the shots where the two PriorTilt answers differ.

| run | weights | values | stream | ranges | shots | network fails alone | answers changed vs float | fails: both / float only / fixed only | largest, median logit change | largest float |logit| of a flipped bit | tie-break shots | choices changed | network-rule failures, float -> fixed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| float (evaluate_stim) | float32 | float32 | float32 | | 5,000 | 237 (4.74%, [4.18%, 5.36%]) | | | | | 24 | | 14 |
| w16 | 16 | 16 | 16 | largest | 5,000 | 239 (4.78%, [4.22%, 5.41%]) | 14 | 236 / 1 / 3 | 3.161, 0.0165 | 1.539 | 24 | 0 | 14 -> 14 |
|  | saturated values: 102 (b6.o 42, b1.c 29, r.pool 21, b5.c 5) | | | | | | | | | | | | |
| w12 | 12 | 12 | 16 | largest | 1,000 | 61 (6.10%, [4.78%, 7.76%]) | 17 | 56 / 2 / 5 | 4.504, 0.0569 | 2.864 | 4 | 0 | 4 -> 4 |
|  | saturated values: 26 (b6.o 11, r.pool 7, b1.c 4, r.sum 2) | | | | | | | | | | | | |
| w10 | 10 | 10 | 16 | largest | 1,000 | 421 (42.10%, [39.08%, 45.18%]) | 414 | 53 / 5 / 368 | 8.802, 3.4825 | 7.641 | 4 | 1 | 4 -> 4 |
|  | saturated values: 28 (b6.o 11, r.pool 10, b1.c 4, r.sum 2) | | | | | | | | | | | | |
| w12p | 12 | 12 | 16 | 99.99th pct | 5,000 | 239 (4.78%, [4.22%, 5.41%]) | 58 | 226 / 11 / 13 | 9.556, 0.0743 | 7.073 | 24 | 0 | 14 -> 14 |
|  | saturated values: 79,259 (b1.in 35195, b4.d 9732, b1.o 8495, b3.d 6596) | | | | | | | | | | | | |
| w10p | 10 | 10 | 16 | 99.99th pct | 5,000 | 238 (4.76%, [4.20%, 5.39%]) | 95 | 220 / 17 / 18 | 9.674, 0.1921 | 7.073 | 24 | 0 | 14 -> 14 |
|  | saturated values: 80,348 (b1.in 35503, b4.d 9936, b1.o 8565, b3.d 6794) | | | | | | | | | | | | |
| w8p | 8 | 8 | 16 | 99.99th pct | 5,000 | 239 (4.78%, [4.22%, 5.41%]) | 208 | 193 / 44 / 46 | 13.492, 1.3013 | 12.584 | 24 | 2 | 14 -> 14 |
|  | saturated values: 66,407 (b1.in 31520, b4.d 7899, b1.o 6240, b3.d 5342) | | | | | | | | | | | | |
| w6p | 6 | 6 | 16 | 99.99th pct | 500 | 412 (82.40%, [78.82%, 85.49%]) | 408 | 29 / 1 / 383 | 16.206, 2.4738 | 10.414 | 1 | 0 | 1 -> 1 |
|  | saturated values: 10,863 (b1.in 3699, b4.d 2006, b1.o 1403, b3.d 1045) | | | | | | | | | | | | |
| w4p | 4 | 4 | 16 | 99.99th pct | 500 | 500 (100.00%, [99.24%, 100.00%]) | 500 | 30 / 0 / 470 | 46.346, 18.1851 | 12.682 | 1 | 0 | 1 -> 1 |
|  | saturated values: 64,034 (r.sum 51486, r.pool 10928, r.in 1562, b0.o 39) | | | | | | | | | | | | |
| w12s12p | 12 | 12 | 12 | 99.99th pct | 1,000 | 58 (5.80%, [4.51%, 7.42%]) | 15 | 54 / 4 / 4 | 5.460, 0.1579 | 2.243 | 4 | 0 | 4 -> 4 |
|  | saturated values: 25,139 (b1.in 7042, s5 4145, s1 2223, b4.d 1915) | | | | | | | | | | | | |
| w8s8p | 8 | 8 | 8 | 99.99th pct | 1,000 | 201 (20.10%, [17.73%, 22.70%]) | 204 | 50 / 8 / 151 | 12.774, 6.3462 | 12.158 | 4 | 3 | 4 -> 4 |
|  | saturated values: 20,936 (b1.in 6337, s5 3606, s6 1582, b4.d 1544) | | | | | | | | | | | | |
| w8 | 8 | 8 | 16 | largest | 500 | 407 (81.40%, [77.75%, 84.57%]) | 406 | 25 / 5 / 382 | 17.859, 2.7663 | 11.199 | 1 | 0 | 1 -> 1 |
|  | saturated values: 5 (b1.c 2, b6.o 2, b5.c 1) | | | | | | | | | | | | |
| w8s8 | 8 | 8 | 8 | largest | 500 | 291 (58.20%, [53.83%, 62.44%]) | 293 | 26 / 4 / 265 | 14.999, 6.6748 | 12.530 | 1 | 0 | 1 -> 1 |
|  | saturated values: 16 (b6.o 6, b5.c 4, r.sum 3, b1.c 2) | | | | | | | | | | | | |
| w6 | 6 | 6 | 16 | largest | 500 | 499 (99.80%, [98.88%, 99.96%]) | 499 | 30 / 0 / 469 | 40.055, 8.7644 | 13.052 | 1 | 1 | 1 -> 1 |
|  | saturated values: 24,801 (r.sum 17142, r.pool 7650, b1.c 5, b2.c 2) | | | | | | | | | | | | |
| w4 | 4 | 4 | 16 | largest | 500 | 500 (100.00%, [99.24%, 100.00%]) | 500 | 30 / 0 / 470 | 54.095, 7.5977 | 13.052 | 1 | 1 | 1 -> 1 |
|  | saturated values: 58 (r.sum 58) | | | | | | | | | | | | |

The float row against the laptop's GPU run of the same checkpoint on the same file (`laptop_step120000_validation_X_p0.004.npz`): largest |logit difference| 1.40e-03, 0 of 60,000 bits of different sign, failures 237 there and 237 here.

Tie-break on 75 more shots, of `train_X.npz` where P1 != P2 (after the 200 calibration shots): float: network-rule failures 22; w16: choices changed 0, failures 22; w12: choices changed 1, failures 22; w10: choices changed 6, failures 23; w12p: choices changed 0, failures 22; w10p: choices changed 1, failures 22; w8p: choices changed 5, failures 23; w6p: choices changed 18, failures 24; w4p: choices changed 46, failures 48; w12s12p: choices changed 0, failures 22; w8s8p: choices changed 9, failures 25; w8: choices changed 6, failures 22.

## Reading the table

**What matters is the tie-break.** In our decoder the network only chooses among the two or three answers of the
fast decoders, on the shots where they disagree (`hierarchy/README.md`). The last three columns count those
choices on the 24 such shots of `validation_X.npz`, and the line under the table on 75 more from `train_X.npz`:
99 in all. With none of 99 changed, the true share of changed choices is below 3.7% (95% Wilson interval);
more tie-break shots would be needed to say more.

- **16 bits** changes 14 of 5,000 answers (2.8 in a thousand). Twelve are bits whose float logit is within 0.08 of
  zero (rounding at the decision boundary). The other two (shots 3714 and 3851) saturate nothing: the float network
  itself swings those logits by 3 to 6 when nudged by the size of the 16-bit rounding (`logs/w16_changes_final.log`).
  **No tie-break choice changes** (0 of 99).
- **12 bits** (percentile ranges) fails as often alone (239 of 5,000, as at 16 bits; the float network 237) and
  changes 58 answers, but **no tie-break choice** (0 of 99).
- **10 bits** (percentile ranges): 238 failures alone, 95 answers changed, **1 of 99 tie-break choices** changed and
  no tie-break failure.
- **8 bits** (percentile ranges): 239 failures alone, 208 answers changed, **7 of 99 tie-break choices** changed, and
  on the train shots one more tie-break failure (23 against 22).
- **6 and 4 bits** break the network (82% and 100% of shots wrong).
- **Ranges set by the largest value seen** (no value of the calibration shots saturates) break the network at 10 bits
  and below: a few rare large values fix a coarse scale for all the others. The 99.99th percentile lets about 16
  values per shot saturate (80,000 over 5,000 shots, out of several million) and keeps the rest fine.
- **The residual stream** (the running sum the blocks add to) **needs more than 8 bits**. With percentile ranges,
  an 8-bit stream (`w8s8p`) breaks the network: 20% of 1,000 shots wrong and 204 answers changed, against 51 with the
  stream at 16 bits on the same shots. A 12-bit stream (`w12s12p`) does as well as a 16-bit one: 15 answers changed
  against 17 on the same 1,000 shots, and no tie-break choice of 79. (With ranges from the largest value, every 8-bit
  run is broken, so `w8s8` against `w8` says nothing about the stream.) The small head at the end stays at 16 bits in
  every run (`logs/diagnostics_train_shots.log`).

**The smallest width that changes nothing that matters: 12 bits**, for the weights and every value, with percentile
ranges and only the small head at 16 bits. With the stream at 16 (`w12p`, 5,000 shots) the network alone fails as
often as at 16 bits and none of 99 tie-break choices changes; with the stream at 12 as well (`w12s12p`, 1,000 shots)
none of 79 changes. 10 bits changes one choice of 99 and no outcome; 8 bits seven. All of this is **simulated** in software on 5,000 validation shots
(1,000 for the rows with ranges from the largest value and for the two narrow-stream rows, 500 for the narrowest),
not on an FPGA.
