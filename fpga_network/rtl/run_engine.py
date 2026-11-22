"""Run the hardware engine (nn_engine.v) in simulation on one shot and compare every tensor it writes with reference.py.

  python fpga_network/rtl/run_engine.py --checkpoint CKPT --calibration DIR/calibration.json --run w16 \\
      --test down --shot 0 --work <scratch folder>
  ... --test block                        (block 0: normalisation + SiLU, down, both collecting steps, up + residual)

It builds the run's fixed-point network (widths.make_net), exports it as whole numbers (reference.export), computes
the shot with reference.forward_int (keeping every named tensor), writes the engine's memories and its table of
layer descriptions as hex files, builds the testbench with Verilator (once per work folder), runs it, and compares
the engine's output region with the reference's tensors, value by value. It prints PASS or FAIL and the clocks.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "challenge-1-ml-decoder-kit"))
import fixed_point as fp      # noqa: E402
import reference as ref       # noqa: E402
import widths                 # noqa: E402

N, VW, WW, AW = 32, 16, 16, 48
LINEAR, HOP_CD, HOP_DC, AFFINE = 0, 1, 2, 3
# the activation memory's regions (in values) for block k: its input stream, and what each step writes
BASE = {"s": 0, "in_s": 0x40000, "c_s": 0x80000, "d_s": 0x90000, "o_s": 0xA0000, "s_next": 0xC0000}


def word(values, bits):
    """Lane 0 in the lowest bits, as hex."""
    v = 0
    for lane, x in enumerate(values):
        v |= (int(x) & ((1 << bits) - 1)) << (lane * bits)
    return f"{v:0{(len(values) * bits + 3) // 4}x}"


def desc(**f):
    fields = [0] * 20
    names = ["kind", "flags", "P", "K", "G", "C_out", "in_base", "out_base", "res_base", "w_base", "b_base", "s_base",
             "t_base", "sat_bits", "res", "shape", "g_base", "chan_base"]
    for key, v in f.items():
        fields[names.index(key)] = int(v) & 0xFFFFFFFF
    return word(fields, 32)


def build(q, dump, k, test):
    """The memories and layer table of block k (or of its "down" step alone)."""
    b, fa, w = q["blocks"][k], q["fa"], q["width"]
    p = f"b{k}."
    s_in, s_out = f"s{k}", f"s{k + 1}"
    R, C = q["rounds"], q["checks"]
    mem = {"W": [], "B": [], "S": [], "C": [], "T": [], "G": [], "D": []}

    def table(name):
        tab, lo = q["tables"][name]
        base = len(mem["T"])
        mem["T"].extend(int(v) & 0xFFFF for v in tab)
        return base
    # weights, as wide words
    w_down = len(mem["W"])
    mem["W"] += [word(b["down_W"][:, kk], WW) for kk in range(b["down_W"].shape[1])]
    def hop_words(W):
        c = W.shape[1] // 2
        out = []
        for blk in range(2):
            for h in range(2):
                for j in range(W.shape[2]):
                    out.append(word(W[blk, h * c:(h + 1) * c, j], WW))
        return out
    w_cd = len(mem["W"]); mem["W"] += hop_words(b["cd_W"])
    w_dc = len(mem["W"]); mem["W"] += hop_words(b["dc_W"])
    w_up = len(mem["W"])
    for g in range(b["up_W"].shape[0] // N):
        mem["W"] += [word(b["up_W"][g * N:(g + 1) * N, kk], WW) for kk in range(b["up_W"].shape[1])]
    # biases and shifts
    b_down = len(mem["B"]); mem["B"].append(word(b["down_b"], AW))
    b_cd = len(mem["B"]); mem["B"] += [word(b["cd_b"][blk], AW) for blk in range(2)]
    b_dc = len(mem["B"]); mem["B"] += [word(b[key][blk], AW) for key in ("dc_b_first", "dc_b_mid", "dc_b_last") for blk in range(2)]
    b_up = len(mem["B"]); mem["B"] += [word(b["up_b"][g * N:(g + 1) * N], AW) for g in range(b["up_W"].shape[0] // N)]
    s_down = len(mem["S"]); mem["S"].append(word(b["down_frac"] - fa[p + "c"], 8))
    s_cd = len(mem["S"]); mem["S"].append(word(b["cd_frac"] - fa[p + "d"], 8))
    s_dc = len(mem["S"]); mem["S"].append(word(b["dc_frac"] - fa[p + "o"], 8))
    s_up = len(mem["S"]); mem["S"] += [word(b["up_frac"][g * N:(g + 1) * N] - fa[s_out], 8) for g in range(b["up_W"].shape[0] // N)]
    # the normalisation's per-channel a, c and shift: {shift, c, a}
    shifts = b["in_scale_frac"] + fa[s_in] - fa[p + "in"]
    for o in range(len(b["in_scale"])):
        v = (int(b["in_scale"][o]) & 0xFFFF) | ((int(b["in_shift"][o]) & ((1 << AW) - 1)) << WW) | ((int(shifts[o]) & 0xFF) << (WW + AW))
        mem["C"].append(f"{v:0{(WW + AW + 8) // 4}x}")
    mem["G"] = [int(v) for v in np.concatenate([q["graph"]["to_data"].ravel(), q["graph"]["to_checks"].ravel()])]
    g_cd, g_dc = 0, q["graph"]["to_data"].size
    t_in, t_c, t_d, t_o = table(p + "in"), table(p + "c"), table(p + "d"), table(p + "o")

    P = R * C
    layers, a_in, check = [], {}, []
    if test == "down":
        a_in[BASE["in_s"]] = dump[p + "in_s"]
        check = [(p + "c_s", BASE["c_s"])]
    else:
        a_in[BASE["s"]] = dump[s_in] if k > 0 else dump["s0"]
        check = [(p + "in_s", BASE["in_s"]), (p + "c_s", BASE["c_s"]), (p + "d_s", BASE["d_s"]), (p + "o_s", BASE["o_s"]),
                 (s_out, BASE["s_next"])]
        layers.append(desc(kind=AFFINE, flags=1, P=P, K=b["in_scale"].size, in_base=BASE["s"], out_base=BASE["in_s"],
                           t_base=t_in, sat_bits=w[p + "in"], chan_base=0))
    layers.append(desc(kind=LINEAR, flags=1, P=P, K=b["down_W"].shape[1], G=1, C_out=N, in_base=BASE["in_s"],
                       out_base=BASE["c_s"], w_base=w_down, b_base=b_down, s_base=s_down, t_base=t_c, sat_bits=w[p + "c"]))
    if test != "down":
        c_in = b["down_W"].shape[0]
        layers.append(desc(kind=HOP_CD, flags=1, P=(R - 1) * C, K=12 * c_in, G=1, C_out=N, in_base=BASE["c_s"],
                           out_base=BASE["d_s"], w_base=w_cd, b_base=b_cd, s_base=s_cd, t_base=t_d, sat_bits=w[p + "d"],
                           shape=(R - 1) | (C << 8) | (c_in << 16) | (R << 24), g_base=g_cd))
        layers.append(desc(kind=HOP_DC, flags=1, P=R * C, K=12 * c_in, G=1, C_out=N, in_base=BASE["d_s"],
                           out_base=BASE["o_s"], w_base=w_dc, b_base=b_dc, s_base=s_dc, t_base=t_o, sat_bits=w[p + "o"],
                           shape=R | (C << 8) | (c_in << 16) | ((R - 1) << 24), g_base=g_dc))
        G_up = b["up_W"].shape[0] // N
        res = (fa[s_in] - fa[s_out]) & 0xFF | (w[s_out] << 8)
        layers.append(desc(kind=LINEAR, flags=2, P=P, K=b["up_W"].shape[1], G=G_up, C_out=b["up_W"].shape[0],
                           in_base=BASE["o_s"], out_base=BASE["s_next"], res_base=BASE["s"], w_base=w_up, b_base=b_up,
                           s_base=s_up, t_base=0, sat_bits=w[s_out], res=res))
    mem["D"] = layers
    return mem, a_in, check


def write(work, mem, a_in, check):
    work.mkdir(parents=True, exist_ok=True)
    (work / "desc.hex").write_text("\n".join(mem["D"]) + "\n")
    lines = []
    for base, tensor in a_in.items():
        lines.append(f"@{base:x}")
        lines += [f"{int(v) & 0xFFFF:04x}" for v in tensor.ravel()]
    (work / "a_in.hex").write_text("\n".join(lines) + "\n")
    for name, key in (("w", "W"), ("bias", "B"), ("shift", "S"), ("chan", "C")):
        (work / f"{name}.hex").write_text("\n".join(mem[key]) + "\n")
    (work / "table.hex").write_text("\n".join(f"{v:04x}" for v in mem["T"]) + "\n")
    (work / "graph.hex").write_text("\n".join(f"{v:02x}" for v in mem["G"]) + "\n")
    lo = min(base for _, base in check)
    hi = max(base + 0x40000 for _, base in check)
    hi = min(hi, (1 << 20) - 1)
    (work / "tb_params.vh").write_text(f"localparam integer N_LAYERS = {len(mem['D'])};\n"
                                       f"localparam integer OUT_LO = {lo};\nlocalparam integer OUT_HI = {hi};\n")
    return lo


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--run", default="w16")
    parser.add_argument("--test", choices=["down", "block"], default="down")
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--shot", type=int, default=0)
    parser.add_argument("--work", required=True)
    args = parser.parse_args()
    import json
    from neural.train import load_decoder
    torch.set_num_threads(1)
    model, saved = load_decoder(args.checkpoint)
    net = widths.make_net(fp.fold(model), json.loads(Path(args.calibration).read_text()), args.run)
    det, _ = widths.load_shots(widths.DATA, args.shot + 1)
    det = det[args.shot:args.shot + 1]
    torch_logits = fp.run(net, torch.from_numpy(det), batch=1)
    q = ref.export(net)
    dump = {}
    logits, frac = ref.forward_int(q, det, dump)
    assert np.array_equal((torch_logits * 2.0 ** frac).astype(np.int64), logits)
    mem, a_in, check = build(q, dump, args.block, args.test)
    work = Path(args.work)
    lo = write(work, mem, a_in, check)
    obj = work / "obj"
    if not (obj / "Vtb_engine").exists():
        subprocess.run(["verilator", "--binary", "--timing", "-Wno-fatal", "-Wno-WIDTH", "--top-module", "tb_engine",
                        f"-I{work}", "--Mdir", str(obj), str(HERE / "tb_engine.v"), str(HERE / "nn_engine.v")],
                       check=True, stdout=subprocess.DEVNULL)
    else:
        # the parameters file changes between tests: rebuild when it does
        subprocess.run(["verilator", "--binary", "--timing", "-Wno-fatal", "-Wno-WIDTH", "--top-module", "tb_engine",
                        f"-I{work}", "--Mdir", str(obj), str(HERE / "tb_engine.v"), str(HERE / "nn_engine.v")],
                       check=True, stdout=subprocess.DEVNULL)
    out = subprocess.run([str(obj / "Vtb_engine")], cwd=work, capture_output=True, text=True, check=True).stdout
    clocks = int(out.split("DONE:")[1].split("in")[1].split()[0])
    got = np.array([int(x, 16) for x in (work / "a_out.hex").read_text().split() if not x.startswith("@") and not x.startswith("/")],
                   dtype=np.int64)
    got = np.where(got >= 1 << 15, got - (1 << 16), got)
    ok = True
    print(f"engine test '{args.test}', block {args.block}, run {args.run} ({widths.RUNS[args.run][:3]} bits), checkpoint step "
          f"{saved['step']}, shot {args.shot} of {widths.DATA.name}; simulated with Verilator:")
    for name, base in check:
        want = dump[name].ravel()
        have = got[base - lo:base - lo + len(want)]
        diff = int((have != want).sum())
        ok &= diff == 0
        print(f"  {name:8s} {len(want):7d} values: {'equal to the reference' if diff == 0 else f'{diff} DIFFER'}"
              + ("" if diff == 0 else f" (first at {int(np.flatnonzero(have != want)[0])}: engine {int(have[np.flatnonzero(have != want)[0]])}, reference {int(want[np.flatnonzero(have != want)[0]])})"))
    print(f"{'PASS' if ok else 'FAIL'}: {len(mem['D'])} layers in {clocks:,} clocks of the engine ({N} multiply-add lanes)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
