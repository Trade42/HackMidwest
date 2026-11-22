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

1. The network does not fit inside the Urbana's chip: its weights need 7.15 Mb at 8 bits against the chip's 2.76 Mb of
   block RAM, so on that board they would stream from its 128 MB DDR3 memory; only the largest Artix-7 (XC7A200T)
   of the chips compared would hold weights and working values on the chip.
2. In fixed point the network's answers survive: at 16 bits it changes PENDING of 5,000 validation answers, and at
   PENDING bits, with the running sum and the small head kept at 16 bits, PENDING (simulated in software).
3. We built a 32-unit engine that walks the network layer by layer; in simulation it reproduces our integer reference
   bit for bit on whole shots, needs 33.4 million clocks per shot (0.33 s at an assumed 100 MHz, about a third of a
   second for the tie-breaker's rare shots), and synthesises to 6,533 LUTs, 2,668 flip-flops and 67 DSP slices of the
   XC7S50, without its memories.

## What was done, step by step

### 1. What it needs (`SIZING.md`, from `sizing.py`)

Counted from the step-70,000 checkpoint: 893,697 weights; **493.5 million multiply-adds per shot** (PyTorch's counter
finds 533.8 million, because the model computes and discards some products at the first and last rounds); the largest
set of values kept at once is 354,816 numbers. Against the chips (AMD's DS180): at 8 bits the weights need 7.15 Mb and
the XC7S50 has 2.76 Mb of block RAM. **Estimated** lower bound on time: multiply-adds / (DSP slices x 100 MHz) = 41 ms
per shot on the XC7S50's 120 DSP slices, 6.7 ms on an XC7A200T's 740. Reading the weights from the Urbana's DDR3 once
per shot: about 0.42 ms at the memory's peak rate (**estimated**).

### 2. Smaller numbers (`WIDTHS.md`, from `widths.py`)

PENDING: the table on the team's `data/p0.004/validation_X.npz` (5,000 Stim shots, seed 201), with the float network
beside it.

How the fixed-point network is made (`fixed_point.py`): every batch normalisation that follows a linear step is folded
into it (the folded network gives the model's logits to within 4.5e-5); the two that act on the residual stream stay
as a scale and a shift per channel. Weights get one power of two per output channel; values one per tensor, from the
largest value on 200 train shots (never the scored shots). **One rounding rule everywhere**: add half, round down,
saturate. SiLU is a table.

Two choices were made on smoke tests of 10 to 20 **validation** shots (disclosed, because they were then scored on
the same file), and then checked on 10 train shots that chose nothing (`logs/diagnostics_train_shots.log`):
- the small head at the end (33,281 weights, 0.1% of the work) stays at 16 bits: at 8 bits it alone moved the logits
  by 0.77 (median; on the train shots 1.12) where any other layer at 8 bits moved them by under 0.04;
- the residual stream stays at 16 bits: at 8 bits it alone changed 9 of 10 answers (on the train shots 10 of 10); its
  range grows from 3.8 to 45.6 over the blocks, so every block rounds it again, more coarsely.

### 3. A reference in whole numbers (`reference.py`)

The whole network in numpy int64: products, sums, shifts, saturation and table look-ups, no floating point. It equals
the torch fixed-point version on every shot tested (logs below). The widest possible sum is 40 bits with the sign at 16
bits (`reference.accumulator_bits`), so the torch version, which carries whole numbers in float64 (53 bits), is exact.
This is the specification the hardware must meet.

### 4. Hardware (`rtl/`)

`rtl/nn_engine.v` is an engine with **32 multiply-add units** ("lanes"). On every clock it reads one value and gives it
to all 32 lanes, each with its own weight; after the K inputs of a step, it writes the 32 results one by one (round,
saturate, SiLU table or residual sum). A table of layer descriptions tells it what to do; it knows five kinds of step:
a linear step, the two collecting steps of the code's graph (with a table of each node's six neighbours, and the
missing time offsets at the first and last rounds), the per-channel normalisation, and the readout's sums over time.
The memories (values, weights, biases, shifts, tables, neighbours) are outside the engine, so synthesis counts the
engine itself.

**Simulated** with Verilator 5.055 against `reference.py`, value by value:

| test | values compared | result | clocks | log |
|---|---:|---|---:|---|
| one linear step (block 0, `down`), shot 0 | 59,904 | equal | 428,691 | `logs/engine_down_w16_shot0.log` |
| the whole block 0, shot 0 | 654,336 | equal | 3,889,455 | `logs/engine_block_w16_shot0.log` |
| block 7, shot 1 | 654,336 | equal | 3,889,455 | `logs/engine_block7_w16_shot1.log` |
| **the whole network, 46 steps, shot 0** | 539,136, and the 12 logits | **equal** | **33,388,774** | `logs/engine_net_w16_shot0.log` |
| the whole network at 8 bits (`w8p`), shot 0 | 539,136 and 12 logits | equal | 33,388,774 | `logs/engine_net_w8p_shot0.log` |
| the whole network, shots 1, 2 and 3 | 539,136 and 12 logits, each | equal, each | 33,388,774 each | `logs/engine_net_w16_shots1-3.log` |

**Synthesised** (Yosys, `synth_xilinx -family xc7`, not placed and routed; `logs/yosys_engine.log`), the engine without
its memories: **6,533 LUTs** (20% of the XC7S50's 32,600), **2,668 flip-flops** (4% of 65,200), **67 DSP48E1** (56% of
120): 32 for the lanes and 35 for address arithmetic that counters could replace. Its timing is unknown (not placed and
routed).

**Where the clocks go** (from the simulated count of a block, 3,889,455): the multiply-adds themselves are 1,861,632
clocks (48%); writing the 32 results of each position one by one at 3 clocks each is 32%; the normalisation, one value
at 3 clocks, is 18%. **Estimated**: overlapping the writes with the next sums and pipelining the normalisation would
bring a block near 2.2 million clocks (about 20 million per shot with the readout, 0.2 s at 100 MHz); 120 lanes would
move it toward the 41 ms bound of step 1.

## What was not done

- No FPGA ran any of this; there is no bitstream and no timing result (no placement and routing).
- The memories were not built: the engine reads them as ports. On the Urbana the weights would have to stream from DDR3
  (`SIZING.md`); no memory controller exists here.
- Only one checkpoint was used for the hardware tests (step 70,000); the width table is PENDING for the final one.
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

The checkpoints are on the branch `laptop-artifacts` (`cascade_X_s1/checkpoint_step70000.pt`, sha256
`692a796f94401dbdd7d522967fb407e14f75a2d6537613263edfd0f4896b12f8`); the laptop's candidate files for the tie-break
are `priortilt/results_8b/selector_{validation,train}_X.npz` there. No `test_*` or `go_*` file was opened.
