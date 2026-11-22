"""What the team's neural network (challenge-1-ml-decoder-kit/neural/) needs from an FPGA: weights, arithmetic, memory.

  python fpga_network/sizing.py --checkpoint <x_step70000.pt> > fpga_network/SIZING.md

Run from the top of the repository. Every number about the network is read from the checkpoint and from the model's
own code: the weights by counting the model's parameters, the operations by running one shot through the model under
PyTorch's operation counter, and the stored values from the shapes of the tensors one shot creates. The chip and board
numbers are typed in below, each with its source.

Words. A "weight" is one learned number. A "multiply-add" (MAC) is one multiplication added to a running sum, the
unit of work of a DSP slice. An "activation" is one number the network computes on the way and must keep until a
later step has used it.
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

KIT = Path(__file__).resolve().parents[1] / "challenge-1-ml-decoder-kit"

# AMD 7 Series FPGAs Data Sheet: Overview (DS180 v2.6.1, tables 2 and 4) and the 7 Series Product Selection Guide
# (XMP101 v1.8). LUTs = slices x 4 (DS180 note 1: "Each 7 series FPGA slice contains four LUTs and eight flip-flops");
# block RAM in blocks of 36 Kb.
CHIPS = {
    # name: (logic cells, slices, DSP48E1 slices, block RAM 36 Kb blocks)
    "XC7S50 (Urbana's chip)": (52_160, 8_150, 120, 75),
    "XC7S75": (76_800, 12_000, 140, 90),
    "XC7S100 (largest Spartan-7)": (102_400, 16_000, 160, 120),
    "XC7A100T (Artix-7)": (101_440, 15_850, 240, 135),
    "XC7A200T (largest Artix-7)": (215_360, 33_650, 740, 365),
}
# RealDigital Urbana reference manual (https://www.realdigital.org/doc/496fed57c6b275735fe24c85de5718c2) and schematic.
BOARD_MEMORY = ("128 MB of DDR3 memory: \"a 1Gbit DDR3 organized as a 64M x 16, single-rank memory that can operate up "
                "to 1066MHz (ISSI part number IS43TR16640CL-125JBL)\" (the schematic's block diagram says 256 MB; the part "
                "number and the manual say 1 Gbit); 16 MB of QSPI flash (Micron MT25QL128), which holds a configuration; a "
                "microSD socket")
DDR3_PEAK_BYTES_PER_S = 2 * 1066e6     # 16 bits = 2 bytes per transfer at the manual's 1066 (million transfers per second)
CLOCK_MHZ = 100          # the Urbana's oscillator; an assumed clock for the engine, not a timing result


def count_macs(model, shots):
    """Multiply-adds of one forward pass, as PyTorch runs it (its counter counts 2 operations per multiply-add)."""
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(shots)
    return counter.get_total_flops() // 2 // len(shots), counter.get_flop_counts()


def layer_table(model):
    """[(group, weights, multiply-adds needed per shot)] with the multiply-adds from the shapes of the code's graph.

    "Needed" leaves out the products that model.py computes and then throws away: a collecting step computes both
    time offsets for every round and keeps one of them at the first and the last round.
    """
    width, narrow = model.config["width"], model.blocks[0].down.out_features
    rounds, checks = model.rounds, model.checks                    # 13 rounds of check nodes, 144 checks
    cycles, data = rounds - 1, checks                              # 12 cycles of data nodes, 144 data qubits
    slots = model.blocks[0].to_data.linears[0].in_features // narrow   # 6 neighbours
    rows = []
    def weights(module):
        return sum(p.numel() for p in module.parameters())
    rows.append(("embedding (a table of 6 vectors: no multiplications)", weights(model.embedding), 0))
    for k, b in enumerate(model.blocks):
        rows.append((f"block {k}: normalisation in (per channel scale and shift, then SiLU)", weights(b.norm_in), rounds * checks * width))
        rows.append((f"block {k}: down {width} -> {narrow}", weights(b.down), rounds * checks * width * narrow))
        rows.append((f"block {k}: checks -> data qubits (6 neighbours x 2 times)", weights(b.to_data) + weights(b.norm_checks),
                     cycles * data * slots * narrow * narrow * 2))
        rows.append((f"block {k}: data qubits -> checks (6 neighbours x 2 times)", weights(b.to_checks) + weights(b.norm_data),
                     (cycles * checks * slots * narrow * narrow * 2)))
        rows.append((f"block {k}: up {narrow} -> {width}", weights(b.up) + weights(b.norm_out), rounds * checks * narrow * width))
    rows.append(("readout: normalisation, SiLU, average over time", weights(model.norm), rounds * checks * width))
    rows.append((f"readout: checks -> data qubits at width {width}", weights(model.to_data), data * slots * width * width * 2))
    pool = model.pool.shape[0] * data * width
    head = model.head[0].in_features * model.head[0].out_features + model.head[2].in_features
    rows.append(("readout: average over each logical's qubits, small network per logical", weights(model.head),
                 pool + model.pool.shape[0] * head))
    return rows


def activation_sizes(model):
    """Numbers per shot that must be kept at once, from the tensor shapes of one block."""
    width, narrow = model.config["width"], model.blocks[0].down.out_features
    rounds, checks = model.rounds, model.checks
    stream = rounds * checks * width               # the residual stream: kept for the whole block
    at_checks = rounds * checks * narrow           # a narrow tensor on the checks
    at_data = (rounds - 1) * checks * narrow       # a narrow tensor on the data qubits
    # The widest moment of a block: the stream, plus the input and the output of one collecting step.
    peak = stream + at_checks + at_data
    return {"residual stream": stream, "narrow, on the checks": at_checks, "narrow, on the data qubits": at_data,
            "largest set kept at once (stream + one step's input and output)": peak}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(KIT))
    from neural.model import count_parameters
    from neural.train import load_decoder
    torch.set_num_threads(1)
    model, saved = load_decoder(args.checkpoint)
    c = saved["config"]
    torch.manual_seed(0)
    shots = (torch.rand(2, model.num_inputs) < 0.03).to(torch.uint8)
    torch_macs, by_op = count_macs(model, shots)
    rows = layer_table(model)
    needed = sum(r[2] for r in rows)
    total_w = count_parameters(model)
    assert sum(r[1] for r in rows) == total_w, "the table misses some weights"
    act = activation_sizes(model)
    out = print

    out("# What the network needs from an FPGA\n")
    out(f"Made by `python fpga_network/sizing.py --checkpoint <checkpoint>` from `{Path(args.checkpoint).name}` (step "
        f"{saved['step']}, width {c['width']}, depth {c['depth']}, bottleneck {c['bottleneck']}, reads {model.num_inputs} "
        f"detector bits, {c['basis']} memory). Nothing here was run on an FPGA: the network numbers are counted from the "
        f"model, the chip numbers come from AMD's data sheet, and every time is an **estimate** with its formula.\n")
    out("The network is our decoder's tie-breaker (hierarchy/): it runs only on the shots where the two fast decodes "
        "disagree, about 0.4% of shots at p = 0.004. Its time is never a whole decoder's time.\n")
    out("## The network, counted\n")
    out(f"- **Weights:** {total_w:,} (count of the model's parameters).")
    out(f"- **Multiply-adds per shot:** {needed:,} needed; PyTorch's counter finds {torch_macs:,} in one forward pass "
        f"(it also computes products that model.py then discards, at the first and last round of a collecting step).")
    out(f"- **Activations:** the residual stream holds {act['residual stream']:,} numbers per shot "
        f"({model.rounds} rounds x {model.checks} checks x {c['width']}); the largest set kept at once is about "
        f"{act['largest set kept at once (stream + one step\'s input and output)']:,} numbers.\n")
    out("| part | weights | multiply-adds per shot |")
    out("|---|---:|---:|")
    blocks_w = sum(r[1] for r in rows if r[0].startswith("block"))
    blocks_m = sum(r[2] for r in rows if r[0].startswith("block"))
    for name, w, m in rows:
        if name.startswith("block") and not name.startswith("block 0"):
            continue
        out(f"| {name} | {w:,} | {m:,} |")
    out(f"| **blocks 1 to {c['depth'] - 1}: the same as block 0, each** | | |")
    out(f"| **all {c['depth']} blocks** | **{blocks_w:,}** | **{blocks_m:,}** |")
    out(f"| **whole network** | **{total_w:,}** | **{needed:,}** |\n")
    out("The normalisations become a scale and a shift per channel in inference; their multiplications are counted, "
        "SiLU is not (it becomes a table).\n")

    out("## Against the chips\n")
    out("Memory at b bits per number: weights x b, and the activations x b. One block RAM of 36 Kb holds 36,864 bits.\n")
    out("| bits per number | weights, Mb | activations kept at once, Mb | both, Mb |")
    out("|---:|---:|---:|---:|")
    peak = act["largest set kept at once (stream + one step's input and output)"]
    for b in (16, 8, 6, 4):
        out(f"| {b} | {total_w * b / 1e6:.2f} | {peak * b / 1e6:.2f} | {(total_w + peak) * b / 1e6:.2f} |")
    mixed = act["residual stream"] * 16 + (peak - act["residual stream"]) * 8
    out(f"| 8, with the residual stream at 16 (see WIDTHS.md: the stream needs 16 bits) | {total_w * 8 / 1e6:.2f} | "
        f"{mixed / 1e6:.2f} | {(total_w * 8 + mixed) / 1e6:.2f} |")
    out("")
    n_tables = 4 * c["depth"] + 1           # every SiLU of the blocks and of the readout; the head has one more
    out(f"SiLU tables (fixed_point.py: one entry per possible input): {n_tables} tables in the blocks and the readout, "
        f"plus one in the head, which widths.py keeps at 16 bits.\n")
    out("| bits per value | body tables, Mb | the head's table (16 bits), Mb |")
    out("|---:|---:|---:|")
    for b in (16, 12, 10, 8, 6, 4):
        out(f"| {b} | {n_tables * 2 ** b * b / 1e6:.3f} | {2 ** 16 * 16 / 1e6:.3f} |")
    out("")
    out("At 16 bits a table is large (65,536 entries); a hardware design would then approximate SiLU in pieces. That was "
        "not done here: the tables are exact.\n")
    out("| chip | LUTs | DSP slices | block RAM, Mb | holds at 8 bits | holds at 4 bits | time per shot at 100 MHz, every DSP busy |")
    out("|---|---:|---:|---:|---|---|---:|")
    for name, (cells, slices, dsp, bram) in CHIPS.items():
        mb = bram * 36_864 / 1e6
        def holds(b):
            w, a = total_w * b / 1e6, peak * b / 1e6
            return "all" if w + a <= mb else ("the weights" if w <= mb else ("the activations" if a <= mb else "neither"))
        ms = needed / (dsp * CLOCK_MHZ * 1e6) * 1e3
        out(f"| {name} | {slices * 4:,} | {dsp} | {mb:.2f} | {holds(8)} | {holds(4)} | {ms:.1f} ms |")
    out("")
    out(f"The time column is an **estimate**, a lower bound: multiply-adds per shot / (DSP slices x {CLOCK_MHZ} MHz), "
        f"as if every DSP slice did one useful multiply-add on every clock; reading weights from outside the chip would "
        f"lengthen it.\n")
    out(f"The Urbana board has, outside the FPGA: {BOARD_MEMORY}.\n")
    w8 = total_w * 8 / 1e6
    bram50 = CHIPS["XC7S50 (Urbana's chip)"][3] * 36_864 / 1e6
    out("## In plain words\n")
    out(f"- **The weights do not fit on the Urbana's chip.** At 8 bits they are {w8:.2f} Mb; the XC7S50 has {bram50:.2f} Mb of "
        f"block RAM in all. Even at 4 bits ({total_w * 4 / 1e6:.2f} Mb) they take more than all of it.")
    read_ms = total_w / DDR3_PEAK_BYTES_PER_S * 1e3
    out(f"- **On the Urbana board they would live in the DDR3 memory** ({total_w / 1e6:.2f} MB at 8 bits, against 128 MB), "
        f"and stream into the chip layer by layer. One shot reads every weight once: about {read_ms:.2f} ms at the memory's "
        f"peak rate (an **estimate**: {total_w:,} bytes / {DDR3_PEAK_BYTES_PER_S / 1e9:.2f} GB/s; a real memory controller "
        f"reaches less), small beside the arithmetic below. A batch of shots shares one reading.")
    stream = act["residual stream"]
    fit = [b for b in (16, 8, 6, 4) if peak * b / 1e6 <= bram50]
    out(f"- **The activations of one shot** need {peak * 8 / 1e6:.2f} Mb at 8 bits, {peak * 8 / 1e6 / bram50:.0%} of the XC7S50's "
        f"block RAM: " + (f"they fit only at {max(fit)} bits or fewer, with no room left for weights." if fit else "they do not fit.")
        + f" The residual stream alone is {stream * 8 / 1e6:.2f} Mb at 8 bits.")
    big = [name for name, (_, _, _, bram) in CHIPS.items() if (total_w + peak) * 8 <= bram * 36_864]
    out(f"- **Of the chips in the table, " + (f"only the {', '.join(big)} holds" if big else "none holds")
        + f" the weights and one shot's activations at 8 bits.** No Spartan-7 does: the largest, the XC7S100, has "
        f"{CHIPS['XC7S100 (largest Spartan-7)'][3] * 36_864 / 1e6:.2f} Mb.")
    out(f"- **Time:** about {needed / (120 * CLOCK_MHZ * 1e6) * 1e3:.0f} ms per shot on the XC7S50 at best (estimate, above). "
        f"For a tie-breaker that runs on about 0.4% of shots this is the cost of those shots only.")


if __name__ == "__main__":
    main()
