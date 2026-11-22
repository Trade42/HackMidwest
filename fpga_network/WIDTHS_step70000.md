# The network in fixed point at step 70,000 (a first table: 1,000 or 500 shots per width; see WIDTHS.md for the final checkpoint)

Made by `python fpga_network/widths.py --checkpoint <checkpoint> --out <folder> --table` from the runs listed there. Checkpoint step 70000 (sha256 `692a796f94401dbd...`). Shots: `challenge-1-ml-decoder-kit/data/p0.004/validation_X.npz` (5,000 Stim circuit shots, X memory, p = 0.004, seed 201; sha256 `b00ee09c35ed088a...`). Ranges calibrated on the first 200 shots of `train_X.npz`. Machine: luminaty (Ryzen 7 7730U), CPU. Every number here is **simulated** in software (torch, float64 holding whole numbers exactly), not on an FPGA.

A shot fails when any of its 12 logical bits is wrong (evaluate_stim's number). "Answers changed" counts the shots whose 12 predicted bits differ from the float network's on the same shot. The tie-break columns use the laptop's three candidates per shot (relay-BP, PriorTilt twice) and the hierarchy's network rule, on the shots where the two PriorTilt answers differ.

| run | weights | values | stream | ranges | shots | network fails alone | answers changed vs float | fails: both / float only / fixed only | largest, median logit change | largest float |logit| of a flipped bit | tie-break shots | choices changed | network-rule failures, float -> fixed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| float (evaluate_stim) | float32 | float32 | float32 | | 5,000 | 526 (10.52%, [9.70%, 11.40%]) | | | | | 24 | | 14 |
| w16 | 16 | 16 | 16 | largest | 1,000 | 109 (10.90%, [9.12%, 12.98%]) | 2 | 109 / 1 / 0 | 1.019, 0.0200 | 0.011 | 4 | 0 | 4 -> 4 |
|  | saturated values: 18 (s7 5, b1.in 5, b6.o 5, b5.c 1) | | | | | | | | | | | | |
| w12 | 12 | 12 | 16 | largest | 1,000 | 110 (11.00%, [9.21%, 13.09%]) | 16 | 107 / 3 / 3 | 7.722, 0.0695 | 5.758 | 4 | 0 | 4 -> 4 |
|  | saturated values: 18 (b6.o 6, s7 5, b1.in 5, b5.c 1) | | | | | | | | | | | | |
| w10 | 10 | 10 | 16 | largest | 1,000 | 304 (30.40%, [27.63%, 33.32%]) | 277 | 100 / 10 / 204 | 7.718, 2.2859 | 6.361 | 4 | 1 | 4 -> 4 |
|  | saturated values: 23 (b6.o 7, b1.in 7, s7 5, h 2) | | | | | | | | | | | | |
| w12p | 12 | 12 | 16 | p99.99 | not run | | | | | | | | |
| w10p | 10 | 10 | 16 | p99.99 | not run | | | | | | | | |
| w8p | 8 | 8 | 16 | 99.99th pct | 1,000 | 127 (12.70%, [10.78%, 14.91%]) | 79 | 97 / 13 / 30 | 8.324, 0.9217 | 7.718 | 4 | 1 | 4 -> 4 |
|  | saturated values: 16,955 (b1.in 7430, b2.o 3203, b1.o 1844, b4.d 1158) | | | | | | | | | | | | |
| w6p | 6 | 6 | 16 | p99.99 | not run | | | | | | | | |
| w4p | 4 | 4 | 16 | p99.99 | not run | | | | | | | | |
| w8 | 8 | 8 | 16 | largest | 500 | 162 (32.40%, [28.45%, 36.62%]) | 150 | 52 / 7 / 110 | 9.152, 2.0578 | 7.095 | 1 | 0 | 1 -> 1 |
|  | saturated values: 13 (b6.o 7, s7 5, b5.c 1) | | | | | | | | | | | | |
| w8s8 | 8 | 8 | 8 | max | not run | | | | | | | | |
| w6 | 6 | 6 | 16 | largest | 500 | 499 (99.80%, [98.88%, 99.96%]) | 499 | 59 / 0 / 440 | 29.788, 5.7677 | 12.673 | 1 | 1 | 1 -> 1 |
|  | saturated values: 15,935 (r.pool 14173, r.sum 1611, h 141, b1.in 9) | | | | | | | | | | | | |
| w4 | 4 | 4 | 16 | largest | 500 | 500 (100.00%, [99.24%, 100.00%]) | 500 | 59 / 0 / 441 | 36.500, 6.5485 | 12.673 | 1 | 1 | 1 -> 1 |
|  | saturated values: 1 (r.in 1) | | | | | | | | | | | | |

The float row against the laptop's GPU run of the same checkpoint on the same file (`laptop_step70000_validation_X_p0.004.npz`): largest |logit difference| 1.14e-03, 0 of 60,000 bits of different sign, failures 526 there and 526 here.

Tie-break on 75 more shots, of `train_X.npz` where P1 != P2 (after the 200 calibration shots): float: network-rule failures 21; w16: choices changed 0, failures 21; w12: choices changed 0, failures 21; w10: choices changed 6, failures 21; w8p: choices changed 1, failures 21.

The runs were made in lane B on 2026-11-22, 4 processes on cores 8-11, with fpga_network/widths.py --run <name> --shots <N>; the float run took 333 s for 5,000 shots.
