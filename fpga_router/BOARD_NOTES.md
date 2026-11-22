# Putting `fpga_board/` on the Urbana board: two ways

For a reader who has the board (and, for building, Vivado) and has never seen this repository. The design is the
teammate's `fpga_board/`; its README has the details. These notes add what a check on luminaty found
(`CHECK_OF_fpga_board.md`) and the facts about the board, each with its source. **Nothing here has been run on a
board yet.** Until a `send` printout from the board is committed, no one may say "it runs on the board".

Everything comes from one branch: `git fetch origin fpga && git switch fpga` (or `git checkout fpga`).

## Before building: fix two pins

`fpga_board/vivado/urbana_router.xdc` puts `led[8]` on B18 and `led[9]` on A18. RealDigital's own constraints file
for the Urbana (version V2I1 of 2023-01-03, https://www.realdigital.org/downloads/1f07b1146ec59e165634a5b6c782a17d.txt,
lines 38-39) has LED 8 on **C17** and LED 9 on **B18**, and A18 is a ground pin of the chip, so Vivado cannot write a
bitstream with the file as it is. Change those two lines to:

```
set_property -dict {PACKAGE_PIN C17 IOSTANDARD LVCMOS33} [get_ports {led[8]}]
set_property -dict {PACKAGE_PIN B18 IOSTANDARD LVCMOS33} [get_ports {led[9]}]
```

(That file belongs to the teammate who wrote `fpga_board/`; the fix is his to commit.) Every other pin, including
the serial pins (the FPGA receives on B16 and sends on A16), matches RealDigital's file
(`logs/check_fpga_board_pins.log`).

## The board, in brief (RealDigital's Urbana reference manual and schematic)

- Chip: Xilinx/AMD Spartan-7 XC7S50 in the CSGA324 package. RealDigital's own example projects target the part
  `xc7s50csga324-1`, and so does `fpga_board/vivado/build.tcl`.
- One micro-USB port carries power, programming (JTAG) and the serial link ("a UART/COM port recognized by Windows
  and Linux"), through an FTDI FT2232H chip: JTAG on its channel A, the serial link on its channel B.
- Jumpers: J16 on USB (power from the USB port). J14 **removed** for programming over USB: the FPGA then waits to be
  programmed (with J14 fitted it loads itself from the on-board flash at power-on; programming over USB still works).
- 100 MHz oscillator on pin N15. 16 LEDs, lit when their pin is high.
- A loaded bitstream lives in the FPGA's configuration memory: it is lost when the board loses power. Load it again
  after every power cycle.

## Way 1: build and load on one machine (Vivado and the board together)

1. **Vivado.** Any recent edition covers the XC7S50 at no cost (AMD's release notes UG973 list the XC7S50 under the
   free Standard edition). Vivado 2025.2 and earlier need no licence file. From 2026.1 on, the free "BASIC" tier
   covers "all 7 Series" devices but needs a free licence file, renewed each year (AMD's Vivado licensing page). To
   save disk space, select only the 7 Series devices in the installer ("Select the required devices or device
   families"). On Windows, also let the installer add the cable drivers (needed for programming over USB).
2. **Build** (from `fpga_board/`, after the pin fix): `vivado -mode batch -source vivado/build.tcl -nojournal -nolog`.
   It writes `build/router_top.bit` only if timing at 100 MHz closes, and it writes `build/utilization_route.rpt`,
   `build/timing_route.rpt` and `build/drc.rpt`. On Windows, run it from the "Vivado Tcl Shell" (or
   `C:\Xilinx\Vivado\<version>\bin\vivado.bat`); on Linux after `source /opt/Xilinx/Vivado/<version>/settings64.sh`.
3. **Load:** plug in the board, then `vivado -mode batch -source vivado/program.tcl -nojournal -nolog`. It refuses
   unless it sees exactly one XC7S50.
4. **Run** the self-check below.

## Way 2: build on one machine, commit the bitstream, load on another

Useful when Vivado is on one computer (for example the teammate's Debian laptop) and the board on another (for
example Sal's Windows laptop).

1. On the Vivado machine: steps 1 and 2 of Way 1. Then commit the bitstream and the three reports. `build/` is in
   `fpga_board/.gitignore`, so copy them out first, for example to `fpga_board/bitstream/`:
   `router_top.bit`, `utilization_route.rpt`, `timing_route.rpt`, `drc.rpt`, and push to `fpga`.
2. On the board's machine: `git fetch origin fpga`, then load the `.bit` with one of:
   - **Vivado Lab Edition** (free, much smaller than Vivado, only the Hardware Manager), or Vivado itself:
     `vivado_lab -mode batch -source fpga_board/vivado/program.tcl` needs the file at `fpga_board/build/router_top.bit`;
     or open the Hardware Manager, "Open target", "Program device", choose the file.
   - **openFPGALoader** (open source, no Vivado at all). MIT's course 6.205, which uses the Urbana, loads its
     bitstreams with `openFPGALoader -b arty_s7_50 <file>.bit` and says "This command works for any bitstream built
     for the Urbana boards" (https://fpga.mit.edu/6205/F24/documentation/openFPGA; `arty_s7_50` is a board entry of
     the same chip, xc7s50csga324, with an FT2232 cable on its channel A). On Windows, openFPGALoader comes with
     MSYS2 (`pacman -S mingw-w64-ucrt-x86_64-openFPGALoader`) or in YosysHQ's OSS CAD Suite for Windows. It needs the
     WinUSB driver on the FT2232's JTAG interface (MIT's page uses the Zadig tool). **Not checked here:** MIT's page
     switches both interfaces of the "Digilent USB Device"; switching interface 1 too would probably remove the COM
     port, so switch only interface 0 if you need the serial link afterwards. None of this was run on luminaty.

## The self-check on the board

Python 3 with `pyserial` (`python -m pip install pyserial`; `verify_reference.py` also needs numpy and scipy).

1. Find the serial port. Windows: Device Manager, "Ports (COM & LPT)", "USB Serial Port (COMn)"; if two appear, try the
   higher-numbered one first (the serial link is the FT2232's second channel; not checked on a board). Linux: `ls -l /dev/serial/by-id/`.
2. From `fpga_board/`, first 100 shots, then all 5,000, keeping the printout as a `.txt` (`*.log` is git-ignored):

   ```
   python tools/board_demo.py send --port COM5 --dir generated --limit 100 --verbose
   python tools/board_demo.py send --port COM5 --dir generated --limit 5000 > board_send_5000.txt
   ```

   Every reply (the 56-bit score and the verdict) is compared byte for byte with the fixed-point model's, which
   `generated/expected.bin` holds and `verify_reference.py` checks against the kit's own rule.
3. **What a working board prints** (from `generated/expected.bin`):
   - `100 board shots, 0 byte-exact mismatches, 17 SLOW replies, ... ms/shot over UART`;
   - `5000 board shots, 0 byte-exact mismatches, 683 SLOW replies, ...`;
   - the time per shot should be about 11.5 ms: 124 + 8 bytes of 10 bits at 115,200 baud is 11.5 ms. That is the
     serial link, not the logic, which needs 8 clocks (80 ns at 100 MHz) from the last byte in to the first bit out.
4. **What the LEDs show:** LED 0 is the last verdict (1 = SLOW, escalate), LED 1 is off, LEDs 2 to 15 count the
   shots received since the bitstream was loaded, in binary (LED 2 the lowest bit). Freshly loaded, after the
   100-shot run: LEDs 4, 7 and 8 lit (100 = 64 + 32 + 4) and LED 0 lit (shot 99 escalates). After the further
   5,000 shots the count is 5,100: LEDs 4, 5, 7, 8, 9, 10, 11 and 14 lit (5,100 = 4,096 + 512 + 256 + 128 + 64 + 32
   + 8 + 4), and LED 0 off (shot 4,999 is accepted).
5. **Commit the evidence:** `board_send_5000.txt`, the three Vivado reports, and the Vivado and OS versions. Then
   the words "runs on the board" are earned, for the router (the decision between a fast and a slow decoder), not
   for a decoder.

## If it does not answer

- No reply at all: the wrong COM port (try the other one), or the serial pins swapped. The FPGA must receive on
  B16 and send on A16; RealDigital's file names them `UART_TXD` (B16) and `UART_RXD` (A16), from the USB chip's side.
- `send` stops with mismatches on the first shot only: a half-packet from an earlier run; the board drops it after
  20 ms of silence, so simply run `send` again.
- Vivado's build stops on timing: the design is not a 100 MHz design until that is fixed; do not load it.
- Nothing lights after loading: J14 or the USB power jumper J16; reload after any power cycle.

## The open-source build route (no Vivado): not available here

The only way to place and route for this chip without Vivado is openXC7 (nextpnr-xilinx and the prjxray database,
which does contain `xc7s50csga324-1`). Its own releases carry no binaries; its installer needs root (Nix daemon, or
apt with sudo); the one prebuilt Linux tarball comes from a personal GitHub repository, and running it was refused
by this session's permission check. Building nextpnr-xilinx from source did not fit the time box. So no bitstream
was built on luminaty, and no timing result exists other than Vivado's, when someone runs it.
