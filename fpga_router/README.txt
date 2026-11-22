Router on the Urbana FPGA (XC7S50-CSGA324): the first design, kept as a record

THE TRAINED BOARD DEMO IS ../fpga_board/ (a teammate's design: 17 features with the convergence flag, the teammates'
frozen model results/router_p0.004/model_X.json, Vivado scripts). Build and load that one. This folder is the
earlier design, checked here with open tools, and a second, independent trained router on 16 whole-number features.
The check of ../fpga_board/ is CHECK_OF_fpga_board.md; how to build it and load it on the board, with Vivado on
the same machine or on another one, is BOARD_NOTES.md.

What the router is. Relay-BP first finds one solution. If that took 1 to 80 iterations the shot is easy and the
solution is kept. Otherwise a logistic model says "escalate" (run full relay-BP) or "accept". The router chooses
between a fast and a slow decoder; it is not a decoder, and its latency is never a decoder's latency.

Files
  router_core.v       weighted sum of 16 features, bias, compare: 4 clocks
  router_frontend.v   123-byte packet -> 16 features (13 popcounts of the syndrome, one per round, and 3 numbers)
  uart_rx.v uart_tx.v 8N1 serial, 115,200 baud at 100 MHz
  router_top.v        top level: clk_100mhz, uart_rxd, uart_txd, led[15:0]; the easy-shot rule (EASY_ITER)
  tb_router.v         testbench: every packet in over the UART pin, every reply compared byte for byte; prints
                      PASS or FAIL and the latency in clocks
  sim.sh              ./sim.sh <folder>: Verilator build and run on one folder of weights and vectors
  router_fpga.py      host side: features, fixed point, export, send (from shots), sendmem (from vectors.mem)
  train_router.py     the trained router of trained_X_p0.004/
  standin/            made-up weights and 300 shots (the design's first hardware test; not a trained router).
                      These features come from router_fpga.py "shots" (its own relay-BP schedule and an unclipped
                      posterior), not from the teammates' feature files: a hardware regression test only.
  trained_X_p0.004/   the trained router for X memory at p = 0.004 and all 5,000 validation shots as vectors
  urbana_router.xdc   pins (checked in CHECK_OF_fpga_board.md, against RealDigital's master file)
  logs/               every number below

Tools: OSS CAD Suite 2026-11-21 for linux-x64 (Verilator 5.055, Yosys 0.71+77), unpacked in user space; g++ 16.2.
Python 3.12 with the kit's requirements.txt (numpy 2.5.5, scipy 1.18.1). Machine: luminaty (Ryzen 7 7730U, Linux).

Simulated (Verilator 5.055), from fpga_router/:
  ./sim.sh standin                     PASS, 300 shots                          logs/baseline_as_committed.log (before the
                                                                                RTL change), logs/sim_standin.log (after)
  ./sim.sh trained_X_p0.004            PASS, 5,000 shots, bit for bit            logs/sim_trained_X_p0.004.log
  ./sim.sh trained_X_p0.004 standin    FAIL (as it must: wrong weights)         logs/sim_cross_check.log
  Latency, clocks, the same on every shot: from the last packet byte handed over by uart_rx to the verdict, 5;
  to the start bit of the reply on the txd pin, 8. At an assumed 100 MHz clock (not placed and routed): 50 ns and
  80 ns. The serial link costs far more: 123 bytes x 10 bits / 115,200 baud = 10.7 ms in, 6 bytes = 0.52 ms out.
  These cover the router alone, not relay-BP, which runs on the host.

Synthesised (Yosys synth_xilinx -family xc7, not placed and routed; the XC7S50 has 32,600 LUTs, 65,200 flip-flops,
120 DSP slices and 75 blocks of 36 Kb, per AMD's DS180):
  stand-in weights   1,105 LUTs, 611 flip-flops, 16 DSP48E1, no block RAM        logs/baseline_as_committed.log
  trained weights    1,145 LUTs (+1 SRL16E), 662 flip-flops, 15 DSP48E1, none   logs/yosys_trained_X_p0.004.log
  (LUTs = LUT2..LUT6 cells; flip-flops = FDRE + FDSE. One trained weight is a power of two, so Yosys used a shift.)

The trained router (python train_router.py ../challenge-1-ml-decoder-kit 0.004 X trained_X_p0.004;
logs/train_X_p0.004.log, trained_X_p0.004/report.json). X memory, p = 0.004; train = data/p0.004/train_X.npz
(20,000 shots, Stim seed 101, 2,760 hard), validation = validation_X.npz (5,000, seed 201, 691 hard, 36 of them
wrong). No test_* or go_* file is opened. The label (the first solution's 12 logical bits are wrong) is only
fitted and scored; the features come from detector bits and relay-BP's first solution only. Same recipe as the
teammates' (evaluation/router_train.py: standardise, L2 = 1, L-BFGS on the hard shots, the threshold that accepts
no wrong hard validation shot), on the 16 numbers the board receives: their features without "converged"
(the 14 unconverged validation shots all have the largest iteration count and are all escalated), and the mean
|posterior| sent as round(16 log2(1 + m)), 18 different values on these shots.
  model, each at its own threshold           AUC   hard validation shots escalated / accepted, wrong accepted
  A  theirs, 17 features, floats            0.797   683 / 8, 0
  B  16 features, floats                     0.821   638 / 53, 0
  C  the board's whole numbers, floats       0.822   640 / 51, 0
  D  the board's model, 16-bit weights       0.822   639 / 52, 0
  The thresholds are fitted on these same validation shots, so "0 wrong accepted" is by construction; on the
  2,760 hard train shots D accepts 1 wrong shot of 120, A none. D and A decide alike on 647 of 691 hard
  validation shots (4,956 of 5,000 with the easy ones); on the other 44 only A escalates.

Not done here: placement and routing, timing at 100 MHz, the board itself. Nothing in this folder has run on a board.
