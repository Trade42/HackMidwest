# The neural network on an FPGA

The team's decoder uses a neural network (`challenge-1-ml-decoder-kit/neural/`, width 128, depth 8, 893,697 weights)
as its **tie-breaker**: on the few shots where two fast decodes disagree (about 0.4% at p = 0.004), the network says
which answer is most likely (`hierarchy/README.md`). This folder asks whether that network could run on an FPGA, the
Urbana board's Spartan-7 XC7S50 or a larger chip of the same family, and builds the hardware for it in simulation.

**Every number here carries one of three words.** *Simulated*: computed by a program (Verilator for the hardware,
torch or numpy for the arithmetic), on luminaty (Ryzen 7 7730U, Linux). *Synthesised*: Yosys 0.71+77 for the Xilinx
7 series, **not placed and routed**. *Estimated*: a formula, given beside the number. **Nothing here has run on an
FPGA.** A time for the network is the tie-breaker's time on the shots that need it, never a whole decoder's time.

## The three sentences

1. The network does not fit inside the Urbana's chip: its weights need 10.7 Mb at 12 bits (7.15 Mb at 8) against the
   chip's 2.76 Mb of block RAM, so on that board they and the working values would live in its 128 MB DDR3 memory;
   of the 7-series chips compared, only the largest Artix-7 (XC7A200T) would hold it all, and only at 8 bits.
2. Its numbers can be made small without changing what it does for us: with 12-bit weights and values (the running
   sum and the small head kept at 16 bits) it gets as many of 5,000 validation shots wrong as at 16 bits (239; the
   float network 237) and changes none of the 99 tie-break choices it makes in our decoder; 8 bits changes 7 of them
   (simulated in software, on the final checkpoint).
3. We built a 32-unit engine that walks the network layer by layer; in simulation it reproduces our integer reference
   bit for bit on whole shots, needs 22.7 million clocks per shot (0.23 s at an assumed 100 MHz, for the tie-breaker's
   rare shots only), and synthesises to 8,463 LUTs, 2,809 flip-flops and 73 DSP slices of the XC7S50, without its
   memories.

## What was done, step by step

### 1. What it needs (`SIZING.md`, from `sizing.py`)

Counted from the final checkpoint (step 120,000; the counts are those of the architecture): 893,697 weights; **493.5 million multiply-adds per shot** (PyTorch's counter
finds 533.8 million, because the model computes and discards some products at the first and last rounds); the largest
set of values kept at once is 354,816 numbers. Against the chips (AMD's DS180): the weights need 10.72 Mb at 12 bits and
7.15 Mb at 8, one shot's values 5.22 or 4.76 Mb (the residual stream at 16 bits), and the XC7S50 has 2.76 Mb of block
RAM. At 12 bits no chip compared holds both; at 8 bits only the XC7A200T (13.46 Mb). **Estimated** lower bound on time: multiply-adds / (DSP slices x 100 MHz) = 41 ms
per shot on the XC7S50's 120 DSP slices, 6.7 ms on an XC7A200T's 740. Reading the weights from the Urbana's DDR3 once
per shot: about 0.42 ms at the memory's peak rate (**estimated**).

### 2. Smaller numbers (`WIDTHS.md`, from `widths.py`)

On the final checkpoint (step 120,000), the team's `data/p0.004/validation_X.npz` (5,000 Stim shots, seed 201) and
99 tie-break shots (24 of that file and 75 of `train_X.npz`), **simulated** in software (`WIDTHS.md` has the table
and its reading; `WIDTHS_step70000.md` a first table on step 70,000):

| weights and values (stream and head 16) | network wrong alone, of 5,000 | answers changed | tie-break choices changed, of 99 |
|---|---:|---:|---:|
| float (as `neural.evaluate_stim` runs it; equals the laptop's GPU run: 237) | 237 | | |
| 16 bits | 239 | 14 | 0 |
| **12 bits** (percentile ranges) | 239 | 58 | **0** |
| 10 bits (percentile ranges) | 238 | 95 | 1 (no tie-break failure more) |
| 8 bits (percentile ranges) | 239 | 208 | 7 (one tie-break failure more, on train shots) |
| 6 and 4 bits | 412 and 500, of 500 | | |

- **The smallest width that changes nothing that matters is 12 bits.** With none of 99 choices changed, the true
  share is below 3.7% (95% Wilson interval).
- At 16 bits, 12 of the 14 changed answers are bits within 0.08 of zero; the other two are shots where the float
  network itself swings by 3 to 6 under a nudge of the size of the rounding (`logs/w16_changes_final.log`): the
  network's sensitivity, not an error. This stays inside the brief's limit of a few shots in a thousand.
- Ranges must let the rarest values saturate: set by the largest value seen, 10 bits already breaks the network
  (42% of shots wrong); set by the 99.99th percentile, it works.

How the fixed-point network is made (`fixed_point.py`): every batch normalisation that follows a linear step is folded
into it (the folded network gives the model's logits to within 4.5e-5); the two that act on the residual stream stay
as a scale and a shift per channel. Weights get one power of two per output channel; values one per tensor, from 200
train shots (never the scored shots): either the largest value seen, or (the runs marked "percentile") the 99.99th
percentile, so the rarest values saturate; the 16-bit tensors always take the largest. Both kinds of range are in
`WIDTHS.md`, side by side. **One rounding rule everywhere**: add half, round down, saturate. SiLU is a table.

Two choices were made on smoke tests of 10 to 20 **validation** shots (disclosed, because they were then scored on
the same file), and then checked on 10 train shots that chose nothing (`logs/diagnostics_train_shots.log`):
- the small head at the end (33,281 weights, 0.1% of the work) stays at 16 bits: at 8 bits it alone moved the logits
  by 0.77 (median; on the train shots 1.12) where any other layer at 8 bits moved them by under 0.04;
- the residual stream stays at 16 bits: at 8 bits it alone changed 9 of 10 answers (on the train shots 10 of 10); its
  range grows from 3.8 to 45.6 over the blocks, so every block rounds it again, more coarsely.

### 3. A reference in whole numbers (`reference.py`)

The whole network in numpy int64: products, sums, shifts, saturation and table look-ups, no floating point. It equals
the torch fixed-point version on every shot tested: 10 shots at 16 and at 8 bits with a smoke calibration, 20 at 16
and at 8 bits with the real calibration of step 70,000, and 20 at 12 bits on the final checkpoint
(`logs/reference_smoke.log`, `logs/reference_step70000.log`, `logs/reference_final.log`). The widest possible sum is 40 bits with the sign at 16
bits (`reference.accumulator_bits`), so the torch version, which carries whole numbers in float64 (53 bits), is exact.
This is the specification the hardware must meet.

### 4. Hardware (`rtl/`)

`rtl/nn_engine.v` is an engine with **32 multiply-add units** ("lanes"). On every clock it reads one value and gives it
to all 32 lanes, each with its own weight; after the K inputs of a step, it writes the 32 results, one per clock
through a three-stage pipeline (round, saturate, SiLU table or residual sum, write). A table of layer descriptions tells it what to do; it knows five kinds of step:
a linear step, the two collecting steps of the code's graph (with a table of each node's six neighbours, and the
missing time offsets at the first and last rounds), the per-channel normalisation, and the readout's sums over time.
The memories (values, weights, biases, shifts, tables, neighbours) are outside the engine, so synthesis counts the
engine itself.

**Simulated** with Verilator 5.055 against `reference.py`, value by value:

| test | values compared | result | clocks | log |
|---|---:|---|---:|---|
| one linear step (block 0, `down`), shot 0 | 59,904 | equal | 314,499 | `logs/engine_down_w16_shot0.log` |
| the whole block 0, shot 0 | 654,336 | equal | 2,619,666 | `logs/engine_block0_w16_shot0.log` |
| block 7, shot 1 | 654,336 | equal | 2,619,666 | `logs/engine_block7_w16_shot1.log` |
| **the whole network, 46 steps, shots 0, 1, 2, 3** (16 bits) | 539,136 and the 12 logits, each | **equal, each** | **22,702,433** | `logs/engine_net_w16_shot{0,1,2,3}.log` |
| the whole network at 8 bits (`w8p`), shot 0 | 539,136 and the 12 logits | equal | 22,702,433 | `logs/engine_net_w8p_shot0.log` |
| the whole network at 8 bits, step 70,000's real calibration, shot 4 | 539,136 and the 12 logits | equal | 22,702,433 | `logs/engine_net_w8p_shot4_realcal.log` |
| **the final checkpoint**, its real calibration, 12 bits and 16 bits, shot 3714 (the shot most sensitive to rounding) | 539,136 and the 12 logits, each | **equal, each** | 22,702,433 | `logs/engine_net_w12p_final_shot3714.log`, `logs/engine_net_w16_final_shot3714.log` |

The "values compared" of a whole shot are the residual stream after the last block and every tensor of the readout
(the normalised values, the sums over time, the collecting step, the pooling, the head), and the 12 logits as 48-bit
whole numbers. The first rows use ranges from a 12-shot smoke calibration; the engine's equality with the reference
does not depend on them, and the last rows use the real calibrations.

**Synthesised** (Yosys, `synth_xilinx -family xc7`, not placed and routed; `logs/yosys_engine.log`), the engine without
its memories: **8,463 LUTs** (26% of the XC7S50's 32,600), **2,809 flip-flops** (4% of 65,200), **73 DSP48E1** (61% of
120): 32 for the lanes and 41 for address arithmetic that counters could replace. Its timing is unknown (not placed and
routed).

**Where the clocks go** (from the simulated count of a block, 2,619,666): the multiply-adds themselves are 1,861,632
clocks (71%); writing the 32 results of a position takes 35 clocks, 453,600 in all (17%); the normalisation, one value
per clock, 239,616 (9%). An earlier version that wrote at 3 clocks per value needed 3,889,455 clocks per block and
33,388,774 per shot (its logs are in the git history). **Estimated**: overlapping the writes with the next sums would
save most of the 17%; using all 120 DSP slices as lanes would move the time toward the 41 ms bound of step 1.

## What was not done

- No FPGA ran any of this; there is no bitstream and no timing result (no placement and routing).
- The memories were not built: the engine reads them as ports. On the Urbana the weights would have to stream from DDR3
  (`SIZING.md`); no memory controller exists here.
- The engine's tests ran mostly on the step-70,000 checkpoint, and on the final one for shot 3714 at 12 and 16 bits.
- The SiLU tables are exact tables; at 16 bits one has 65,536 entries (1 Mb). A real design would approximate SiLU.

## Commands

From the top of the repository, with the kit's Python 3.12 environment, torch (CPU build), and Verilator and Yosys on
the PATH (YosysHQ's OSS CAD Suite):

```sh
python fpga_network/sizing.py --checkpoint <checkpoint.pt> > fpga_network/SIZING.md
python fpga_network/widths.py --checkpoint <checkpoint.pt> --out <folder> --run float      # then w16, w12, w10, w8p, ...
python fpga_network/widths.py --checkpoint <checkpoint.pt> --out <folder> --table \
    --tie-break <selector_validation_X.npz> --tie-break-train <selector_train_X.npz> --anchor <laptop eval .npz>
python fpga_network/reference.py --checkpoint <checkpoint.pt> --calibration <folder>/calibration.json --run w16 --shots 20
python fpga_network/rtl/run_engine.py --checkpoint <checkpoint.pt> --calibration <folder>/calibration.json \
    --run w16 --test net --shot 0 --work <scratch folder>
cd fpga_network/rtl && yosys -p "read_verilog nn_engine.v; chparam -set WA 15 -set CA 11 -set TA 22 -set GA 12 nn_engine; \
    synth_xilinx -family xc7 -top nn_engine -noiopad; stat"
```

The checkpoints are on the branch `laptop-artifacts`: the final one at commit `2f752e9`,
`cascade_X_s1/checkpoint_final_step120000.pt` (sha256 `d8d1c70f552233300f1c10997b69292092042eb8517f5b6526b607c1f6dc4933`),
and `cascade_X_s1/checkpoint_step70000.pt` (sha256 `692a796f94401dbdd7d522967fb407e14f75a2d6537613263edfd0f4896b12f8`).
The laptop's candidate files for the tie-break are `priortilt/results_8b/selector_{validation,train}_X.npz` there
(sha256 `60db8f10...` and `d23883ca...`, as in `hierarchy/README.md`), and its own float runs of the same checkpoints on
`validation_X.npz` are `cascade_X_s1/eval/step{70000,120000}_validation_X_p0.004.npz`. The calibrations used are
`logs/calibration_step70000.json` and `logs/calibration_final.json`; the lane logs `logs/lane_B_hold1_step70000.log` and
`logs/lane_BD_final.log`. No `test_*` or `go_*` file was opened.
