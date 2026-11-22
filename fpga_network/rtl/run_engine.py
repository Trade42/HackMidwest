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
             "t_base", "sat_bits", "res", "shape", "g_base", "chan_base", "stride"]
    for key, v in f.items():
        fields[names.index(key)] = int(v) & 0xFFFFFFFF
    return word(fields, 32)


LINEAR_T = 4   # flags: 1 SiLU, 2 residual, 4 strided input, 8 transposed output, 16 raw output
TSUM = 4
S_EVEN, S_ODD = 0x00000, 0xC0000            # the residual stream alternates between two regions
R_BASE = {"r.in_s": 0x40000, "r.sum": 0x80000, "r.data": 0x90000, "r.pool": 0xA0000, "h_s": 0xA2000, "logits": 0xA4000}


class Memories:
    """The engine's memories as lists of hex words, and the layer table."""

    def __init__(self, q):
        self.q = q
        self.W, self.B, self.S, self.C, self.T, self.D = [], [], [], [], [], []
        self.G = [int(v) for v in np.concatenate([q["graph"]["to_data"].ravel(), q["graph"]["to_checks"].ravel(),
                                                   q["graph"]["readout"].ravel()])]
        self.g = {"to_data": 0, "to_checks": q["graph"]["to_data"].size,
                  "readout": q["graph"]["to_data"].size + q["graph"]["to_checks"].size}

    def table(self, name):
        tab, lo = self.q["tables"][name]
        base = len(self.T)
        self.T.extend(int(v) & 0xFFFF for v in tab)
        return base

    def linear_weights(self, W):
        """W [c_out, K] -> groups of N lanes, K words each."""
        base = len(self.W)
        c_out = W.shape[0]
        Wp = np.zeros((((c_out + N - 1) // N) * N, W.shape[1]), np.int64)
        Wp[:c_out] = W
        for g in range(len(Wp) // N):
            self.W += [word(Wp[g * N:(g + 1) * N, kk], WW) for kk in range(W.shape[1])]
        return base

    def hop_weights(self, W):
        """W [2 blocks, 2 c, 6 c_in] -> for block, group of N lanes: K = 2 * 6 c_in words (h, slot, i)."""
        base = len(self.W)
        c = W.shape[1] // 2
        for blk in range(2):
            for g in range(c // N):
                for h in range(2):
                    for j in range(W.shape[2]):
                        self.W.append(word(W[blk, h * c + g * N:h * c + (g + 1) * N, j], WW))
        return base

    def words(self, key, rows, bits):
        base = len(getattr(self, key))
        getattr(self, key).extend(word(r, bits) for r in rows)
        return base

    def lanes(self, v):
        """[c] -> rows of N lanes (the last padded with zeros)."""
        v = np.asarray(v, np.int64)
        pad = np.zeros(((len(v) + N - 1) // N) * N, np.int64)
        pad[:len(v)] = v
        return pad.reshape(-1, N)

    def affine(self, scale, shift, frac):
        base = len(self.C)
        for o in range(len(scale)):
            v = (int(scale[o]) & 0xFFFF) | ((int(shift[o]) & ((1 << AW) - 1)) << WW) | ((int(frac[o]) & 0xFF) << (WW + AW))
            self.C.append(f"{v:0{(WW + AW + 8) // 4}x}")
        return base


def block_layers(m, k):
    """Block k as five layer descriptions: normalisation + SiLU, down, checks -> data, data -> checks, up + residual."""
    q = m.q
    b, fa, w = q["blocks"][k], q["fa"], q["width"]
    p = f"b{k}."
    s_in, s_out = f"s{k}", f"s{k + 1}"
    base_in, base_out = (S_EVEN, S_ODD) if k % 2 == 0 else (S_ODD, S_EVEN)
    R, C = q["rounds"], q["checks"]
    c_in = b["down_W"].shape[0]
    P = R * C
    L = []
    L.append(desc(kind=AFFINE, flags=1, P=P, K=b["in_scale"].size, in_base=base_in, out_base=BASE["in_s"],
                  t_base=m.table(p + "in"), sat_bits=w[p + "in"],
                  chan_base=m.affine(b["in_scale"], b["in_shift"], b["in_scale_frac"] + fa[s_in] - fa[p + "in"])))
    L.append(desc(kind=LINEAR, flags=1, P=P, K=b["down_W"].shape[1], G=1, C_out=N, in_base=BASE["in_s"],
                  out_base=BASE["c_s"], w_base=m.linear_weights(b["down_W"]), b_base=m.words("B", m.lanes(b["down_b"]), AW),
                  s_base=m.words("S", m.lanes(b["down_frac"] - fa[p + "c"]), 8), t_base=m.table(p + "c"), sat_bits=w[p + "c"]))
    L.append(desc(kind=HOP_CD, flags=1, P=(R - 1) * C, K=12 * c_in, G=1, C_out=N, in_base=BASE["c_s"], out_base=BASE["d_s"],
                  w_base=m.hop_weights(b["cd_W"]), b_base=m.words("B", [b["cd_b"][blk] for blk in range(2)], AW),
                  s_base=m.words("S", m.lanes(b["cd_frac"] - fa[p + "d"]), 8), t_base=m.table(p + "d"), sat_bits=w[p + "d"],
                  shape=(R - 1) | (C << 8) | (c_in << 16) | (R << 24), g_base=m.g["to_data"]))
    L.append(desc(kind=HOP_DC, flags=1, P=R * C, K=12 * c_in, G=1, C_out=N, in_base=BASE["d_s"], out_base=BASE["o_s"],
                  w_base=m.hop_weights(b["dc_W"]),
                  b_base=m.words("B", [b[key][blk] for key in ("dc_b_first", "dc_b_mid", "dc_b_last") for blk in range(2)], AW),
                  s_base=m.words("S", m.lanes(b["dc_frac"] - fa[p + "o"]), 8), t_base=m.table(p + "o"), sat_bits=w[p + "o"],
                  shape=R | (C << 8) | (c_in << 16) | ((R - 1) << 24), g_base=m.g["to_checks"]))
    L.append(desc(kind=LINEAR, flags=2, P=P, K=b["up_W"].shape[1], G=b["up_W"].shape[0] // N, C_out=b["up_W"].shape[0],
                  in_base=BASE["o_s"], out_base=base_out, res_base=base_in, w_base=m.linear_weights(b["up_W"]),
                  b_base=m.words("B", m.lanes(b["up_b"]), AW), s_base=m.words("S", m.lanes(b["up_frac"] - fa[s_out]), 8),
                  sat_bits=w[s_out], res=((fa[s_in] - fa[s_out]) & 0xFF) | (w[s_out] << 8)))
    return L, base_in, base_out


def readout_layers(m, base_in):
    """The readout: normalisation + SiLU, the time sums, checks -> data qubits, the pooling, the head."""
    q = m.q
    r, fa, w = q["readout"], q["fa"], q["width"]
    R, C = q["rounds"], q["checks"]
    width = r["in_scale"].size
    depth = len(q["blocks"])
    L = []
    L.append(desc(kind=AFFINE, flags=1, P=R * C, K=width, in_base=base_in, out_base=R_BASE["r.in_s"],
                  t_base=m.table("r.in"), sat_bits=w["r.in"],
                  chan_base=m.affine(r["in_scale"], r["in_shift"], r["in_scale_frac"] + fa[f"s{depth}"] - fa["r.in"])))
    L.append(desc(kind=TSUM, P=2, K=width, in_base=R_BASE["r.in_s"], out_base=R_BASE["r.sum"], sat_bits=w["r.sum"],
                  res=(fa["r.in_s"] - fa["r.sum"]) & 0xFF, shape=(R - 1) | (C << 8)))
    G = width // N
    L.append(desc(kind=HOP_CD, flags=0, P=C, K=12 * width, G=G, C_out=width, in_base=R_BASE["r.sum"], out_base=R_BASE["r.data"],
                  w_base=m.hop_weights(r["r.W"]),
                  b_base=m.words("B", [row for blk in range(2) for row in m.lanes(r["r.b"][blk])], AW),
                  s_base=m.words("S", m.lanes(r["r.W.frac"] - fa["r.data"]), 8), sat_bits=w["r.data"],
                  shape=1 | (C << 8) | (width << 16) | (2 << 24), g_base=m.g["readout"]))
    n_obs = r["r.pool"].shape[0]
    L.append(desc(kind=LINEAR, flags=4 | 8, P=width, K=C, G=1, C_out=N, in_base=R_BASE["r.data"], out_base=R_BASE["r.pool"],
                  w_base=m.linear_weights(r["r.pool"]), b_base=m.words("B", m.lanes(np.zeros(n_obs, np.int64)), AW),
                  s_base=m.words("S", m.lanes(r["r.pool.frac"] - fa["r.pool"]), 8), sat_bits=w["r.pool"], stride=width))
    L.append(desc(kind=LINEAR, flags=1, P=n_obs, K=width, G=r["h0_W"].shape[0] // N, C_out=r["h0_W"].shape[0],
                  in_base=R_BASE["r.pool"], out_base=R_BASE["h_s"], w_base=m.linear_weights(r["h0_W"]),
                  b_base=m.words("B", m.lanes(r["h0_b"]), AW), s_base=m.words("S", m.lanes(r["h0_W.frac"] - fa["h"]), 8),
                  t_base=m.table("h"), sat_bits=w["h"]))
    L.append(desc(kind=LINEAR, flags=16, P=n_obs, K=r["h2_W"].shape[1], G=1, C_out=N, in_base=R_BASE["h_s"],
                  out_base=R_BASE["logits"], w_base=m.linear_weights(r["h2_W"]), b_base=m.words("B", m.lanes(r["h2_b"]), AW),
                  s_base=m.words("S", m.lanes([0]), 8), sat_bits=16))
    return L


def build(q, dump, k, test):
    """(memories, initial activation regions, [(tensor name, base address)] to compare) of a test."""
    m = Memories(q)
    if test == "down":
        L, base_in, _ = block_layers(m, k)
        m.D = [L[1]]
        return m, {BASE["in_s"]: dump[f"b{k}.in_s"]}, [(f"b{k}.c_s", BASE["c_s"])]
    if test == "block":
        L, base_in, base_out = block_layers(m, k)
        m.D = L
        p = f"b{k}."
        return m, {base_in: dump[f"s{k}"]}, [(p + "in_s", BASE["in_s"]), (p + "c_s", BASE["c_s"]), (p + "d_s", BASE["d_s"]),
                                            (p + "o_s", BASE["o_s"]), (f"s{k + 1}", base_out)]
    for kk in range(len(q["blocks"])):
        L, base_in, base_out = block_layers(m, kk)
        m.D += L
    m.D += readout_layers(m, base_out)
    depth = len(q["blocks"])
    return m, {S_EVEN: dump["s0"]}, [(f"s{depth}", base_out), ("r.in_s", R_BASE["r.in_s"]), ("r.sum", R_BASE["r.sum"]),
                                     ("r.data", R_BASE["r.data"]), ("r.pool", R_BASE["r.pool"]), ("h_s", R_BASE["h_s"]),
                                     ("logits", R_BASE["logits"])]


def write(work, m, a_in, check):
    work.mkdir(parents=True, exist_ok=True)
    (work / "desc.hex").write_text("\n".join(m.D) + "\n")
    lines = []
    for base, tensor in a_in.items():
        lines.append(f"@{base:x}")
        lines += [f"{int(v) & 0xFFFF:04x}" for v in tensor.ravel()]
    (work / "a_in.hex").write_text("\n".join(lines) + "\n")
    for name, key in (("w", "W"), ("bias", "B"), ("shift", "S"), ("chan", "C")):
        (work / f"{name}.hex").write_text("\n".join(getattr(m, key)) + "\n")
    (work / "table.hex").write_text("\n".join(f"{v:04x}" for v in m.T) + "\n")
    (work / "graph.hex").write_text("\n".join(f"{v:02x}" for v in m.G) + "\n")
    assert len(m.W) <= 1 << 15 and len(m.B) <= 1 << 8 and len(m.S) <= 1 << 8 and len(m.C) <= 1 << 11 and len(m.T) <= 1 << 22
    assert len(m.D) <= 64
    lo = 0
    hi = (1 << 20) - 1
    (work / "tb_params.vh").write_text(f"localparam integer N_LAYERS = {len(m.D)};\n"
                                       f"localparam integer OUT_LO = {lo};\nlocalparam integer OUT_HI = {hi};\n")
    return lo


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--run", default="w16")
    parser.add_argument("--test", choices=["down", "block", "net"], default="down")
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
    if "logits" in dump:
        dump["logits_raw"] = dump["logits"]
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
        if name == "logits":                # lane 0 of each of the 12 positions, as three 16-bit words of a 48-bit number
            words = got[base - lo:base - lo + 12 * N * 3].reshape(12, N, 3)[:, 0] & 0xFFFF
            have = words[:, 0] | (words[:, 1] << 16) | (words[:, 2] << 32)
            have = np.where(have >= 1 << 47, have - (1 << 48), have)
        elif name == "r.pool":              # transposed: observable-major, 32 lanes of which 12 are observables
            n_obs = dump[name].shape[1]
            have = got[base - lo:base - lo + n_obs * dump[name].shape[2]]
        else:
            have = got[base - lo:base - lo + len(want)]
        diff = int((have != want).sum())
        ok &= diff == 0
        print(f"  {name:8s} {len(want):7d} values: {'equal to the reference' if diff == 0 else f'{diff} DIFFER'}"
              + ("" if diff == 0 else f" (first at {int(np.flatnonzero(have != want)[0])}: engine {int(have[np.flatnonzero(have != want)[0]])}, reference {int(want[np.flatnonzero(have != want)[0]])})"))
    print(f"{'PASS' if ok else 'FAIL'}: {len(mem.D)} layers in {clocks:,} clocks of the engine ({N} multiply-add lanes)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
