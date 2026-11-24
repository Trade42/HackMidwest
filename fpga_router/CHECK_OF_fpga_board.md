# Historical audit of the trained Urbana router

This audit records the state at commit `134de6e` on November 22, 2026. The LED-pin issue described below was fixed in the current `fpga_board/vivado/urbana_router.xdc`. For current build and test instructions, see [the board README](../fpga_board/README.md).

Checked by the fpga worker on 2026-11-22, at commit `134de6e` (GJIMB, "Add trained Urbana FPGA router demo"),
on luminaty (Ryzen 7 7730U, Linux). Nothing in `fpga_board/` was changed: every run below reads its files and
writes only to a scratch folder or to `fpga_router/logs/`.

Tools: OSS CAD Suite 2026-11-21 for linux-x64 (Icarus Verilog 14.0 devel, Verilator 5.055, Yosys 0.71+77), unpacked
in user space; Python 3.12.15, numpy 2.5.5, scipy 1.18.1 in a fresh venv built from the kit's `requirements.txt`
(not the tracked `.venv312`).

**In short:** the design does what its README says, bit for bit, on all 5,000 validation shots of X memory at
p = 0.004, including through the UART. At the audited commit, two things needed attention before the board: **two LED pins in the constraints
file were wrong, and one of them (A18) is a ground ball of the chip, so Vivado was expected to refuse to write the
bitstream**
(finding 1, a two-line fix); and the exporter's rounded threshold lets a board image for Z at p = 0.003 accept a
wrong candidate that the software escalates (finding 2, a three-line fix). The committed X p = 0.004 image is not
affected by the second.

## What was run, and what agrees

| What | Command (from `fpga_board/`) | Result here | Log |
|---|---|---|---|
| core testbench, Icarus | the README's `iverilog ... tb_core` then `vvp` | `PASS core: 5000 saved validation shots, score and verdict` | `logs/check_fpga_board_testbenches.log` |
| UART testbench, Icarus | the README's `iverilog ... tb_uart` then `vvp` | `PASS UART: 32 original saved shots, byte-exact replies` | same |
| both, Verilator | `verilator --binary --timing -Igenerated ...` | both PASS (only width warnings, none of which changes a value) | same |
| UART testbench on **all 5,000** shots | its own exporter with `--uart-vectors 5000` into scratch, its own `tb_uart.sv` unchanged, Verilator | `PASS UART: 5000 original saved shots, byte-exact replies` | `logs/check_fpga_board_uart5000_and_yosys.log` |
| `verify_reference.py` | the README's command with the venv above | `5000 saved shots: software escalates 683, fixed hardware escalates 683, mismatches 0` | `logs/check_fpga_board_python.log` |
| re-export of `generated/` | the README's `export` command, `--out` in scratch | all 7 files identical byte for byte | same |
| Yosys | the README's line | 983 LUTs (LUT2 to LUT6), 688 flip-flops (FDRE + FDSE), 17 DSP48E1, 96 CARRY4, no block RAM: **the README's numbers exactly** | `logs/check_fpga_board_uart5000_and_yosys.log` |
| latency in clocks | `check_fpga_board/tb_latency.sv` (new, in this folder) on its 32 UART vectors, Verilator | last byte handed over by `uart_rx` to the core's `out_valid`: **5 clocks**; to the start bit of the reply on `txd`: **8 clocks**; the same on every packet | `logs/check_fpga_board_latency.log` |
| the pins | `vivado/urbana_router.xdc` line by line against RealDigital's master file and the chip's package pin list | 17 of 19 agree; `led[8]` and `led[9]` do not: see finding 1 | `logs/check_fpga_board_pins.log` |
| the other five frozen models | its exporter into scratch for X and Z at p = 0.003, 0.004, 0.005, then `verify_reference.py` on each | see finding 2 | `logs/check_fpga_board_python.log`, `logs/check_fpga_board_threshold_edge.log` |

So, for the committed image: the RTL computes exactly the fixed-point model's 56-bit score and verdict on every
validation shot, the packet format is parsed correctly on every shot (the 32-shot UART test alone left the parser
tested on 32 shots), and the fixed-point verdicts equal the kit's own float rule (`evaluation/router.py`) on all
5,000 shots. The latency at an assumed 100 MHz clock (not placed and routed) is 50 ns to the verdict and 80 ns to
the reply's first bit; the README's 40 ns is the core alone. The serial link costs 124 x 10 / 115,200 = 10.8 ms in
and 8 x 10 / 115,200 = 0.69 ms out per shot. All of these are the router's, not a decoder's.

## Findings

Severity: **high** = the bitstream cannot be built, or the board can give a wrong answer; **medium** = a check can mislead; **low** = documentation or
robustness.

### 1. High: `led[8]` and `led[9]` are on the wrong pins, and A18 is not an I/O pin

`vivado/urbana_router.xdc:19-20`:

```
set_property -dict {PACKAGE_PIN B18 IOSTANDARD LVCMOS33} [get_ports {led[8]}]
set_property -dict {PACKAGE_PIN A18 IOSTANDARD LVCMOS33} [get_ports {led[9]}]
```

RealDigital's master constraints file for the Urbana (`URBANA BOARD CONSTRAINTS V2I1 1/3/2023`, linked as "Urbana
constraints file" from https://www.realdigital.org/hardware/urbana, file
https://www.realdigital.org/downloads/1f07b1146ec59e165634a5b6c782a17d.txt, sha256 `cffacac8...063f48`), lines 38-39:

```
set_property -dict {PACKAGE_PIN C17 IOSTANDARD LVCMOS33} [get_ports {LED[8]}]
set_property -dict {PACKAGE_PIN B18 IOSTANDARD LVCMOS33} [get_ports {LED[9]}]
```

So B18 is LED 9, not LED 8, and **A18 is not an I/O pin at all**: it is absent from the chip's package pin list
(prjxray-db `spartan7/xc7s50csga324-1/package_pins.csv`, 212 pins, no A18), and RealDigital's schematic (sheet of
IC11G, "POWER", XC7S50-CSGA324) lists A18 among the GND balls. So Vivado cannot place an output there, and the
port is expected to stay unplaced, which `write_bitstream` refuses (its check for I/O without a pin). This was not
run here: there is no Vivado on luminaty; the pin facts are what was checked. The README's
sentence "LED8/LED9 are B18/A18; these correct two wrong assignments in the older integration XDC" is the wrong
way round: the older file (`fpga_router/urbana_router.xdc`) had C17 and B18, which match the master file.

**Fix:** in `vivado/urbana_router.xdc`, `led[8]` on C17 and `led[9]` on B18; and drop that README sentence.

All the other 17 pins agree with the master file (`logs/check_fpga_board_pins.log`): clock N15; the 14 other LEDs;
and the UART: the FPGA receives on B16 and transmits on A16. The master file names these two the other way round
(`UART_TXD` on B16, `UART_RXD` on A16) because it names them from the USB chip's side: RealDigital's reference
manual (figure 16) draws "TXD (B16)" from the FTDI FT2232 into the FPGA; the schematic joins FT2232 pin 38
(channel B TXD, an output per FTDI's FT2232H data sheet) to B16; and MIT 6.205's Urbana file says "labeled from
the perspective of the FPGA! note the inversion from RealDigital official documentation" with `uart_rxd` on B16.
So `uart_rxd` on B16 is right: do not "fix" it to match the master file's names. The master file's own clock line
has no `create_clock` (it is commented out); fpga_board's file adds one, at 10 ns, which is right.

### 2. High (for images other than the committed one): the rounded threshold can let a wrong candidate through

`tools/board_demo.py:88`: `tq = round(math.log(threshold / (1 - threshold)) * (1 << frac))`.

Each frozen model's threshold is chosen as the largest value that still escalates every wrong hard validation
shot, so it sits a few floating-point steps below one particular wrong shot (for X at p = 0.004 the risk of shot
1691 is 4.3e-18 above the threshold). The board compares a rounded score with a rounded threshold, so whether that
shot is escalated is decided by rounding. Run on all six frozen models (`check_fpga_board/threshold_edge.py`):

| model | board's THRESH minus the boundary shot's fixed score | the board |
|---|---|---|
| X p = 0.003 | -2,905 | escalates it |
| **Z p = 0.003** | **+3,793** | **accepts shot 4358, a wrong candidate that the software escalates** (wrong hard shots accepted: board 1, software 0) |
| X p = 0.004 (committed) | -155 (155 units of 2^-20 to spare) | escalates it |
| Z p = 0.004 | -4,211 | escalates it |
| X p = 0.005 | -7,392 | escalates it |
| Z p = 0.005 | -1,840 | escalates it |

The exporter reports this case (`float_fixed_verdict_flips: 1` for Z p = 0.003) and the README warns of it, but
nothing stops the image from being built.

**Smallest fix:** set the integer threshold from the frozen rule's own verdicts on the validation shots, and refuse
to export when no integer separates them. It uses the software verdicts, not the labels:

```python
# after the loop has the fixed scores and the software verdicts (risk > threshold, not easy) of every shot:
tq = min(score for score, s in zip(scores, software_verdicts) if s) - 1
assert all(score <= tq for score, s, e in zip(scores, software_verdicts, easy) if not s and not e)
```

Tested here on all six models: the board's verdict then equals the kit's rule on every one of the 5,000 shots of
each, with a gap of 7,434 to 96,185 units between the highest accepted and the lowest escalated score. The
committed X p = 0.004 image is not affected (its verdicts already agree on every shot); its THRESH would move by
154 units.

### 3. Medium: `float_fixed_verdict_flips` is measured against a re-derived float score, not the kit's rule

`tools/board_demo.py:147-152` recomputes the float score from the folded weights (`bias + sum(raw * row)`), not as
the kit does (`w0 + ((x - mean) / scale) @ w`). At a threshold a few floating-point steps from a shot, the two
orders of arithmetic disagree. For Z at p = 0.005 the exporter reports 1 flip (shot 4964) where the hardware
agrees with the kit's rule on every shot (`verify_reference.py`: mismatches 0). It is a false alarm there, and the
same effect could hide a real flip elsewhere. **Fix:** with finding 2's fix the parity holds by construction; keep
`verify_reference.py` (the kit's own arithmetic) as the acceptance check, and say in the README that the
exporter's counter is approximate at the threshold.

### 4. Low: the packet parser is tested on 32 shots only

`tb/tb_uart.sv` runs `uart_vectors.mem`, 32 shots by default (`--uart-vectors 32`); `tb/tb_core.sv` feeds the core
directly. So the 124-byte parsing (popcounts, the two-byte iteration count up to 18,080, the converged byte, the
Q8.8 posterior) was checked on 32 shots, of which few are hard. Run here on all 5,000 it passes (above), so this is
coverage, not a bug. **Fix:** export with `--uart-vectors 5000` and run `tb_uart` under Verilator (62 s here; Icarus
needs about 50 s for 32 shots, so hours for 5,000).

### 5. Low: what a committed board log needs

`fpga_board/.gitignore` ignores `build/` and `*.log`. The Vivado reports (`build/*.rpt`) and a saved `send`
printout named `*.log` will not show up in `git status`. To commit the board run's evidence: save the `send`
output as a `.txt`, and copy the three reports out of `build/` (or `git add -f`).

### 6. Low: `build.tcl` checks setup slack only

`vivado/build.tcl:19-23` takes `get_timing_paths -max_paths 1`, the worst setup path. Hold slack is not checked
(Vivado's router normally fixes hold, so this is a belt-and-braces point). **Fix:** also read
`[get_timing_paths -hold -max_paths 1]` and fail on a negative value. The asynchronous pins (`uart_rxd`, `uart_txd`,
`led`) have no input or output delay, which is right for this design: they are unconstrained, not failing.

### 7. Low: the README's `verify_reference.py` line names the tracked `.venv312`

`README.md`, "Verify locally": `../challenge-1-ml-decoder-kit/.venv312/bin/python` is a teammate's macOS venv that
was committed by mistake; it does not run on Debian or Windows. Any Python with numpy and scipy works (the README
says so in its next paragraph).

## Read and found right

- **Signed widths in the sum** (`rtl/router_core.sv`). A product is `$signed({1'b0, x})` (17 bits) times a signed
  24-bit weight, held in 41 bits: the largest, 65,535 x 2^23, needs 40. Four products sum into 43 bits; the bias and
  five group sums into 56. Every operand is signed, so every widening is a sign extension (these are Verilator's
  `WIDTHEXPAND` notes). The largest score on the 5,000 shots fits easily; the exporter asserts the 56-bit range.
- **The easy-shot rule** (`router_core.sv:61`): converged and 1 to 80 iterations, the kit's `first_leg_converged`;
  the exporter checks it against the saved flag on every shot (`board_demo.py:141`). Its pipeline register lines up
  with the score: tested by `tb_core` on every shot.
- **UART bit order**: least significant bit first in `uart_rx`, `uart_tx`, the testbench's sender and monitor, and
  the exporter's packing (`board_demo.py:107-108`); the popcounts do not depend on bit order anyway. Replies are
  little endian, as `send` decodes them. 8N1, `CLKS_PER_BIT = 868` = 100 MHz / 115,200 (0.006% off).
- **The 20 ms timeout** (`router_top.sv:20-25`): 2,000,000 clocks of silence reset the parser, so a lost byte costs
  one shot and the next packet is parsed right; tested by `tb_uart` with a broken packet in the middle. A byte at
  115,200 baud lasts 87 us, far inside 20 ms.
- **Resets**: a 15-clock power-on reset; `uart_rx`, `uart_tx` and the counters start from declared values, which the
  FPGA loads with the bitstream. There is no reset button: reloading the bitstream restarts it.
- **A malformed packet**: too short, it is completed by the next packet's bytes (one wrong reply, which `send` counts
  as a mismatch) until 20 ms of silence; `send` waits up to 2 s for a reply, so the parser always resyncs before the
  next packet. Too long, the extra bytes start a packet that the timeout drops. A corrupted byte gives a reply for
  the corrupted packet, again counted by `send`. The board never hangs.
- **`build.tcl`**: reads every `rtl/*.sv` and the constraints, part `xc7s50csga324-1`, synthesis, placement,
  routing, the three reports, refuses to write a bitstream if the worst setup slack at 10 ns is negative. Every top
  port has a pin and an I/O standard, so `write_bitstream`'s pin checks pass.
- **`program.tcl`**: opens the local hardware server, requires exactly one XC7S50 on the JTAG chain, loads
  `build/router_top.bit` into the FPGA's configuration memory (lost at power-off, as the README says).

