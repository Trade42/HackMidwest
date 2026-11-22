# What the network needs from an FPGA

Made by `python fpga_network/sizing.py --checkpoint <checkpoint>` from `x_final_step120000.pt` (step 120000, width 128, depth 8, bottleneck 4, reads 1728 detector bits, X memory). Nothing here was run on an FPGA: the network numbers are counted from the model, the chip numbers come from AMD's data sheet, and every time is an **estimate** with its formula.

The network is our decoder's tie-breaker (hierarchy/): it runs only on the shots where the two fast decodes disagree, about 0.4% of shots at p = 0.004. Its time is never a whole decoder's time.

## The network, counted

- **Weights:** 893,697 (count of the model's parameters).
- **Multiply-adds per shot:** 493,507,584 needed; PyTorch's counter finds 533,818,368 in one forward pass (it also computes products that model.py then discards, at the first and last round of a collecting step).
- **Activations:** the residual stream holds 239,616 numbers per shot (13 rounds x 144 checks x 128); the largest set kept at once is about 354,816 numbers.

| part | weights | multiply-adds per shot |
|---|---:|---:|
| embedding (a table of 6 vectors: no multiplications) | 768 | 0 |
| block 0: normalisation in (per channel scale and shift, then SiLU) | 256 | 239,616 |
| block 0: down 128 -> 32 | 4,128 | 7,667,712 |
| block 0: checks -> data qubits (6 neighbours x 2 times) | 24,768 | 21,233,664 |
| block 0: data qubits -> checks (6 neighbours x 2 times) | 24,768 | 21,233,664 |
| block 0: up 32 -> 128 | 4,288 | 7,667,712 |
| readout: normalisation, SiLU, average over time | 256 | 239,616 |
| readout: checks -> data qubits at width 128 | 393,728 | 28,311,552 |
| readout: average over each logical's qubits, small network per logical | 33,281 | 617,472 |
| **blocks 1 to 7: the same as block 0, each** | | |
| **all 8 blocks** | **465,664** | **464,338,944** |
| **whole network** | **893,697** | **493,507,584** |

The normalisations become a scale and a shift per channel in inference; their multiplications are counted, SiLU is not (it becomes a table).

## Against the chips

Memory at b bits per number: weights x b, and the activations x b. One block RAM of 36 Kb holds 36,864 bits.

| bits per number | weights, Mb | activations kept at once, Mb | both, Mb |
|---:|---:|---:|---:|
| 16 | 14.30 | 5.68 | 19.98 |
| 8 | 7.15 | 2.84 | 9.99 |
| 6 | 5.36 | 2.13 | 7.49 |
| 4 | 3.57 | 1.42 | 4.99 |
| 12, with the residual stream at 16 (WIDTHS.md: the stream needs 16 bits) | 10.72 | 5.22 | 15.94 |
| 8, with the residual stream at 16 (WIDTHS.md: the stream needs 16 bits) | 7.15 | 4.76 | 11.91 |

SiLU tables (fixed_point.py: one entry per possible input): 33 tables in the blocks and the readout, plus one in the head, which widths.py keeps at 16 bits.

| bits per value | body tables, Mb | the head's table (16 bits), Mb |
|---:|---:|---:|
| 16 | 34.603 | 1.049 |
| 12 | 1.622 | 1.049 |
| 10 | 0.338 | 1.049 |
| 8 | 0.068 | 1.049 |
| 6 | 0.013 | 1.049 |
| 4 | 0.002 | 1.049 |

At 16 bits a table is large (65,536 entries); a hardware design would then approximate SiLU in pieces. That was not done here: the tables are exact.

A chip "holds all" when its block RAM takes the weights, one shot's values and the SiLU tables.

| chip | LUTs | DSP slices | block RAM, Mb | holds at 12 bits (stream 16) | holds at 8 bits (stream 16) | time per shot at 100 MHz, every DSP busy |
|---|---:|---:|---:|---|---|---:|
| XC7S50 (Urbana's chip) | 32,600 | 120 | 2.76 | neither | neither | 41.1 ms |
| XC7S75 | 48,000 | 140 | 3.32 | neither | neither | 35.3 ms |
| XC7S100 (largest Spartan-7) | 64,000 | 160 | 4.42 | neither | neither | 30.8 ms |
| XC7A100T (Artix-7) | 63,400 | 240 | 4.98 | neither | neither | 20.6 ms |
| XC7A200T (largest Artix-7) | 134,600 | 740 | 13.46 | the weights | all | 6.7 ms |

The time column is an **estimate**, a lower bound: multiply-adds per shot / (DSP slices x 100 MHz), as if every DSP slice did one useful multiply-add on every clock; reading weights from outside the chip would lengthen it.

The Urbana board has, outside the FPGA: 128 MB of DDR3 memory: "a 1Gbit DDR3 organized as a 64M x 16, single-rank memory that can operate up to 1066MHz (ISSI part number IS43TR16640CL-125JBL)" (the schematic's block diagram says 256 MB; the part number and the manual say 1 Gbit); 16 MB of QSPI flash (Micron MT25QL128), which holds a configuration; a microSD socket.

## In plain words

- **The weights do not fit on the Urbana's chip.** At 8 bits they are 7.15 Mb; the XC7S50 has 2.76 Mb of block RAM in all. Even at 4 bits (3.57 Mb) they take more than all of it.
- **On the Urbana board they would live in the DDR3 memory** (0.89 MB at 8 bits, against 128 MB), and stream into the chip layer by layer. One shot reads every weight once: about 0.42 ms at the memory's peak rate (an **estimate**: 893,697 bytes / 2.13 GB/s; a real memory controller reaches less), small beside the arithmetic below. A batch of shots shares one reading.
- **One shot's values do not fit either**, once the residual stream keeps its 16 bits (WIDTHS.md): 5.22 Mb at 12 bits and 4.76 Mb at 8 bits, against the XC7S50's 2.76 Mb (the stream alone is 3.83 Mb). On the Urbana they too would go to the DDR3 memory, part by part.
- **At 12 bits (stream 16), of the chips in the table, none holds the weights, one shot's values and the SiLU tables together** (18.61 Mb).
- **At 8 bits (stream 16), of the chips in the table, only the XC7A200T (largest Artix-7) holds the weights, one shot's values and the SiLU tables together** (13.02 Mb, of 13.46 Mb: just).
  No Spartan-7 holds the weights alone at 8 bits: the largest, the XC7S100, has 4.42 Mb.
- **Time:** about 41 ms per shot on the XC7S50 at best (estimate, above). For a tie-breaker that runs on about 0.4% of shots this is the cost of those shots only.
