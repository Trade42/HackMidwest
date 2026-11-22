# Gross-code router on the Urbana FPGA

This directory is a standalone board demo for the **trained first-solution router** in the decoder kit. The included host script replays real saved shots: 936 basis-syndrome bits plus statistics from the first Relay-BP candidate. The FPGA counts the 13 syndrome cycles, computes a 17-feature fixed-point logistic score, applies the software router's first-leg bypass, and returns `FAST` (accept first candidate) or `SLOW` (rerun full Relay-BP). The decoders run in software in the evaluated pipeline; the replay script reads their saved first-candidate features. This board does not decode a syndrome or demonstrate end-to-end acceleration.

The checked-in image is the frozen **X-memory, p=0.004** model. It was exported from `results/router_p0.004/model_X.json` and the corresponding 5,000 original validation shots. Re-export to use another basis or error rate; each frozen model has its own threshold. `generated/manifest.json` records hashes of the model and original shot file. The previous `fpga_router/` on `origin/integration` used stand-in 16-feature weights and a different policy; its vectors and resource count do not apply here.

## Files and wire format

- `rtl/router_frontend.sv`: accepts a 124-byte packet. Bytes 0–116 are 13 cycles of 72 detector bits, little bit first within each byte. Bytes 117–123 are `iterations` as `u16 LE`, `candidate_converged` as `u8`, clipped mean absolute posterior as `u16 Q8.8 LE`, and correction weight as `u16 LE`.
- `rtl/router_core.sv`: 17 signed constant products, grouped additions, bias, threshold, first-leg bypass. `out_valid` is asserted on the fourth rising edge after `in_valid` is sampled. The score is a signed 56-bit integer with `FRAC` fractional bits. The reply is seven score bytes, little endian two's complement, then verdict byte 0 or 1.
- `rtl/router_top.sv`: 100 MHz clock, 115200 8N1 UART, 20 ms incomplete-packet timeout, LEDs. `led[0]` is the last verdict (`1 = SLOW`); `led[15:2]` count received shots modulo 16384; `led[1]` is off.
- `tools/board_demo.py`: exports coefficients and original saved-shot packets, checks float versus fixed decisions, and compares board replies byte for byte. Export uses the standard library; `send` needs `pyserial`.
- `generated/`: the ready-to-build X-memory image and reproducible vectors. `core_vectors.mem` holds all 5,000 validation shots; `uart_vectors.mem` holds 32 for the slower serial simulation.
- `vivado/`: Urbana constraints, batch build, and JTAG programming scripts.

The router only needs the **13 cycle counts**, so the FPGA's transmitted decision depends on the actual saved syndrome through those counts. Python packs the original 936 detector bits, not reconstructed counts. The first candidate and posterior statistics originate from the kit's saved first-solution feature file.

## Verify locally

From `fpga_board/` (run the simulators from this directory so they find `generated/`):

```bash
iverilog -g2012 -s tb_core -Igenerated -o /tmp/router_core.vvp \
  tb/tb_core.sv rtl/router_core.sv
vvp /tmp/router_core.vvp

iverilog -g2012 -s tb_uart -Igenerated -o /tmp/router_uart.vvp \
  tb/tb_uart.sv rtl/router_top.sv rtl/router_frontend.sv \
  rtl/router_core.sv rtl/uart_rx.sv rtl/uart_tx.sv
vvp /tmp/router_uart.vvp
```

Expected: `PASS core: 5000...` and `PASS UART: 32...`. The UART test also abandons a partial packet and checks that the timeout resynchronizes it. The exporter found **zero float-versus-fixed verdict changes on these 5,000 validation shots**. This is an agreement check on saved data, not a general proof for unseen shots.

For a separate comparison against the kit's exact NumPy/SciPy decision rule, use its Python environment:

```bash
../challenge-1-ml-decoder-kit/.venv312/bin/python tools/verify_reference.py \
  --model ../challenge-1-ml-decoder-kit/results/router_p0.004/model_X.json \
  --features ../challenge-1-ml-decoder-kit/results/router_p0.004/validation_X.npz \
  --generated generated
```

The expected result is `5000 saved shots: software escalates 683, fixed hardware escalates 683, mismatches 0`. On Debian, any environment with NumPy and SciPy installed works; the `.venv312` path is specific to this checkout.

To refresh or change models, from `fpga_board/`:

```bash
python3 tools/board_demo.py export \
  --model ../challenge-1-ml-decoder-kit/results/router_p0.004/model_X.json \
  --features ../challenge-1-ml-decoder-kit/results/router_p0.004/validation_X.npz \
  --data ../challenge-1-ml-decoder-kit/data/p0.004/validation_X.npz \
  --out generated --uart-vectors 32
```

Change `0.004` and `X` in all three paths together for another frozen model. The exporter checks the shot hash and model metadata before writing. Check `float_fixed_verdict_flips` and rerun `verify_reference.py` for each new image before claiming decision parity; some other frozen profiles have threshold-edge differences after quantization.

An open-source Spartan-7 synthesis mapping of this complete top level reported **983 LUTs, 688 flip-flops, 17 DSP48E1s, and 96 CARRY4s** with Yosys 0.71+post. Reproduce the mapping from `fpga_board/` with:

```bash
yosys -Q -T -p 'read_verilog -sv -Igenerated rtl/router_core.sv rtl/router_frontend.sv rtl/router_top.sv rtl/uart_rx.sv rtl/uart_tx.sv; synth_xilinx -family xc7 -top router_top; stat'
```

This is an unrouted mapping, not a Vivado utilization or timing result. The core has four registered stages: `in_valid` is sampled at stage 1 and `out_valid` rises at stage 4. Four 100 MHz clock periods give a nominal 40 ns pipeline budget. UART receive and frontend handoff add cycles; a 124-byte UART packet alone takes about 10.8 ms at 115200 baud, plus the eight-byte reply. Decoder time is additional.

## Transfer with GitHub

In this checkout, stage only this directory because other experiment files may already be modified:

```bash
git switch -c fpga
git add fpga_board
git commit -m "Add trained Urbana FPGA router demo"
git push -u origin fpga
```

If `fpga` already exists, switch to it instead of creating it. On Debian, clone the repository or fetch this branch, then `git switch fpga`. The frozen model, original shots, and this directory all need to be present. No Vivado install is needed on the Mac.

## Build and program on the Urbana board

1. Set power jumper **J16 to USB** and leave programming jumper **J14 unloaded** for direct JTAG programming. Connect Urbana's **programming/UART micro-USB port** to the Debian machine. Confirm the board is a Real Digital Urbana with **XC7S50-CSGA324-1**, and compare `vivado/urbana_router.xdc` with the master XDC for the exact board revision. The clock is N15 at 100 MHz. The UART FPGA input/output pins are B16/A16. LED8/LED9 are B18/A18; these correct two wrong assignments in the older integration XDC.
2. Source the installed Vivado environment, for example `source /opt/Xilinx/Vivado/<version>/settings64.sh`. In `fpga_board/`, run `vivado -mode batch -source vivado/build.tcl -nojournal -nolog`. The script synthesizes, places, routes, reports utilization/DRC/timing in `build/`, checks nonnegative worst path slack against the 10 ns clock constraint, and writes `build/router_top.bit` only after the check.
3. Inspect `build/timing_route.rpt`, `build/utilization_route.rpt`, and `build/drc.rpt`. Record LUT, FF, DSP and WNS as **Vivado routed estimates**. The Yosys counts above are a separate estimate.
4. With the board connected and visible to Vivado Hardware Manager, run `vivado -mode batch -source vivado/program.tcl -nojournal -nolog`. It requires exactly one XC7S50 JTAG device. The bitstream configures the SRAM FPGA; repeat programming after power cycling unless you separately set up flash.
5. Identify the USB serial interface with `ls -l /dev/serial/by-id/`. Use the stable `/dev/serial/by-id/...` path in place of `/dev/ttyUSB0`. Create a Python environment and install the only send-side dependency: `python3 -m venv .venv`, `. .venv/bin/activate`, `python -m pip install pyserial`. If serial access is denied, add your account to Debian's `dialout` group and log in again.
6. From `fpga_board/`, run `python tools/board_demo.py send --port /dev/serial/by-id/<your-uart> --dir generated --limit 100 --verbose`. Expect `100 board shots, 0 byte-exact mismatches`. The command also prints `SLOW` count and measured per-shot UART round-trip. For the complete saved set use `--limit 5000` (roughly a minute of wire time). The last verdict lights LED0 for `SLOW`; other LEDs show the shot count.

If the board gives no reply, check the serial device, USB cable, baud rate, and RX/TX pin directions in the board's master XDC. A failed timing or DRC gate must be resolved before treating the bitstream as a 100 MHz design. The board test has not been run in this checkout.

## Hackathon demonstration

1. Show the [challenge's hierarchical routing stretch goal](https://github.com/Infleqtion/MidQWest-2026/tree/main): the trained router decides when to accept the first Relay-BP answer and when to pay for full Relay-BP. Explain that the physical error sweep and logical-rate/latency comparison come from the software evaluation, while the FPGA row is the hardware cost of the router.
2. On screen, show `generated/manifest.json` and the local `PASS` tests. Then program the board and run 100 saved shots over UART with `--verbose`. Point to LED0 changing for `SLOW` and the count LEDs advancing. The terminal verifies the returned **score and decision** against the fixed-point model, not merely the LEDs.
3. Show the Vivado routed utilization and WNS reports from this board build. State 40 ns as the **core pipeline** only if Vivado closes 100 MHz timing. Separately show measured UART round-trip and measured decoder latency if available; the USB demo's wall time is dominated by serial transport.
4. For the noise-model presentation, use the software comparison for the Infleqtion-style noise branch. This FPGA image uses the saved no-loss X-memory router; it does not ingest atom-loss heralds or switch among noise-matched models.

Board references: [Real Digital Urbana manual](https://www.realdigital.org/doc/496fed57c6b275735fe24c85de5718c2) and [Real Digital Urbana schematic](https://www.realdigital.org/downloads/064514e67fbab05a00650ce69af02486.pdf). The challenge's [official repository](https://github.com/Infleqtion/MidQWest-2026/tree/main) calls hardware cost a stretch goal; this implementation estimates the cost of the router, not of the full decoder.
