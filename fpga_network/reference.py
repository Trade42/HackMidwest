"""The network in whole numbers only: what an FPGA engine must reproduce, bit for bit.

  python fpga_network/reference.py --checkpoint CKPT --calibration DIR/calibration.json --run w16 --shots 20

It builds the fixed-point network of a run of widths.py (fixed_point.FixedNet), exports it as whole numbers
(export: weights, biases, SiLU tables, and the scale of every tensor as a number of fractional bits), and computes
the logits with numpy int64 arithmetic alone: products, sums, shifts, saturation and table look-ups. No floating-point
number enters forward_int. It then checks, shot by shot, that every logit equals the torch fixed-point version's
(the logit is a whole number at a known scale, so "equal" means equal).

The one rounding rule, as integers: to go from F fractional bits to f, with s = F - f,
    s > 0:  (v + 2^(s-1)) >> s        (add half, arithmetic shift right = round down)
    s <= 0: v << -s                   (exact)
then saturate to the tensor's width. The scales differ per output channel for the sums of a layer.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "challenge-1-ml-decoder-kit"))
sys.path.insert(0, str(HERE))
import fixed_point as fp      # noqa: E402

I64 = np.int64


# ---------------------------------------------------------------- export: the fixed-point network as whole numbers

def _int(real, frac):
    """A tensor of fp's rounded reals at `frac` fractional bits (per channel if frac is a vector on axis -1 or 0)."""
    v = real.numpy() * np.exp2(np.asarray(frac, np.float64))
    k = np.rint(v)
    assert np.array_equal(k, v), "a rounded value is not a whole number at its scale"
    return k.astype(I64)


def export(net):
    """The integer network of a FixedNet on which forward() has run once (so every weight has been rounded)."""
    f, q = net.f, {"fa": dict(net.fa), "width": dict(net.width), "blocks": []}
    def wcol(name, real, axis):
        """Whole-number weights and their fractional bits, one per output channel on `axis`."""
        frac = net.wfrac[name].numpy().astype(I64)
        if axis == "hop":
            c = real.shape[1] // 2
            shape = (1, 2 * c, 1)
            return _int(real, np.concatenate([frac, frac]).reshape(shape)), frac
        shape = [1] * real.ndim
        shape[axis] = -1
        return _int(real, frac.reshape(shape)), frac
    q["embedding"] = _int(net.wq["embedding"], net.fa["s0"])
    for k in range(len(f["blocks"])):
        p, b = f"b{k}.", {}
        b["in_scale"], b["in_scale_frac"] = wcol(p + "in.scale", net.wq[p + "in.scale"], 0)
        b["in_shift"] = _int(net.wq[p + "in.shift"], b["in_scale_frac"] + net.fa[f"s{k}"])
        for key, inp, axis in (("down", "in_s", 0), ("cd", "c_s", "hop"), ("dc", "d_s", "hop"), ("up", "o_s", 0)):
            W, wf = wcol(p + key + "_W", net.wq[p + key + "_W"], axis)
            b[key + "_W"], b[key + "_frac"] = W, wf + net.fa[p + inp]          # the sums' fractional bits per channel
        b["down_b"] = _int(net.wq[p + "down_b"], b["down_frac"])
        b["cd_b"] = _int(net.wq[p + "cd_b"], b["cd_frac"][None])
        for key in ("dc_b_first", "dc_b_mid", "dc_b_last"):
            b[key] = _int(net.wq[p + key], b["dc_frac"][None])
        b["up_b"] = _int(net.wq[p + "up_b"], b["up_frac"])
        q["blocks"].append(b)
    r = {}
    r["in_scale"], r["in_scale_frac"] = wcol("r.in.scale", net.wq["r.in.scale"], 0)
    r["in_shift"] = _int(net.wq["r.in.shift"], r["in_scale_frac"] + net.fa[f"s{len(f['blocks'])}"])
    for key, inp, axis in (("r.W", "r.sum", "hop"), ("r.pool", "r.data", 0), ("h0_W", "r.pool", 0), ("h2_W", "h_s", 0)):
        W, wf = wcol(key, net.wq[key], axis)
        r[key], r[key + ".frac"] = W, wf + net.fa[inp]
    r["r.b"] = _int(net.wq["r.b"], r["r.W.frac"][None])
    r["h0_b"] = _int(net.wq["h0_b"], r["h0_W.frac"])
    r["h2_b"] = _int(net.wq["h2_b"], r["h2_W.frac"])
    q["readout"] = r
    q["tables"] = {name: (tab.numpy().astype(I64), int(lo)) for name, (tab, lo) in net.tables.items()}
    t = f["tables"]
    q["graph"] = {key: t[key].numpy().astype(I64) for key in ("to_data", "to_checks", "readout", "input_place", "input_code", "empty_code")}
    q["rounds"], q["checks"] = f["rounds"], f["checks"]
    return q


# ---------------------------------------------------------------- the integer forward pass

def requant(v, frac_in, frac_out, bits):
    """Whole numbers at frac_in fractional bits (per channel on the last axis) -> frac_out, then saturate to `bits`."""
    s = np.asarray(frac_in, I64) - I64(frac_out)
    right = np.maximum(s, 0)
    half = np.where(s > 0, np.left_shift(I64(1), np.maximum(s - 1, 0)), 0)
    out = np.where(s > 0, np.right_shift(v + half, right), np.left_shift(v, np.maximum(-s, 0)))
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return np.clip(out, lo, hi)


def silu(q, name, v):
    table, lo = q["tables"][name]
    return table[v - lo]


def messages(x, neighbours, W):
    """Integer collecting step without biases: (same, after) [..., 144, c]."""
    out = []
    for block in range(2):
        side = x[..., neighbours[block], :].reshape(*x.shape[:-2], neighbours.shape[1] // 6, -1)
        out.append(side @ W[block].T)
    y = np.concatenate(out, axis=-2)
    c = W.shape[1] // 2
    return y[..., :c], y[..., c:]


def per_target(b):
    return np.repeat(b, 72, axis=0)


def accumulator_bits(q):
    """The widest sum of products any layer can make, in bits with the sign, from the weights and the widths alone:
    for every output channel, sum |w| x (largest whole number its input can hold) + |bias|. It bounds every shot, and
    shows that float64 (53 bits) holds every sum of fixed_point.py exactly."""
    w, fa = q["width"], q["fa"]
    def top(name):
        return (1 << (w[name] - 1))
    worst = {}
    for k, b in enumerate(q["blocks"]):
        p = f"b{k}."
        worst[p + "in"] = int((np.abs(b["in_scale"]) * top(f"s{k}") + np.abs(b["in_shift"])).max())
        worst[p + "down"] = int((np.abs(b["down_W"]).sum(1) * top(p + "in_s") + np.abs(b["down_b"])).max())
        worst[p + "cd"] = int((np.abs(b["cd_W"]).sum(2).reshape(2, 2, -1).sum(1).max(0) * top(p + "c_s") + np.abs(b["cd_b"]).max(0)).max())
        worst[p + "dc"] = int((np.abs(b["dc_W"]).sum(2).reshape(2, 2, -1).sum(1).max(0) * top(p + "d_s")
                               + np.abs(np.stack([b["dc_b_first"], b["dc_b_mid"], b["dc_b_last"]])).max((0, 1))).max())
        worst[p + "up"] = int((np.abs(b["up_W"]).sum(1) * top(p + "o_s") + np.abs(b["up_b"])).max())
    r = q["readout"]
    worst["r.W"] = int((np.abs(r["r.W"]).sum(2).reshape(2, 2, -1).sum(1).max(0) * top("r.sum") + np.abs(r["r.b"]).max(0)).max())
    worst["r.pool"] = int((np.abs(r["r.pool"]).sum(1) * top("r.data")).max())
    worst["h0"] = int((np.abs(r["h0_W"]).sum(1) * top("r.pool") + np.abs(r["h0_b"])).max())
    worst["h2"] = int((np.abs(r["h2_W"]).sum(1) * top("h_s") + np.abs(r["h2_b"])).max())
    bits = {name: int(v).bit_length() + 1 for name, v in worst.items()}
    assert max(bits.values()) <= 53, f"a sum could exceed float64's exact range: {bits}"
    return bits


def forward_int(q, shots, dump=None):
    """Detector bits [n, 1728] (0/1) -> logits as whole numbers [n, 12], and their fractional bits.
    dump: a dict that receives every named integer tensor (for the hardware's testbench)."""
    fa, w, g = q["fa"], q["width"], q["graph"]
    keep = (lambda name, v: dump.__setitem__(name, v.copy()) or v) if dump is not None else (lambda name, v: v)
    n, R, C = len(shots), q["rounds"], q["checks"]
    codes = np.tile(g["empty_code"], (n, 1))
    codes[:, g["input_place"]] = g["input_code"] + shots.astype(I64)
    x = keep("s0", q["embedding"][codes].reshape(n, R, C, -1))                        # stream s0
    for k, b in enumerate(q["blocks"]):
        p = f"b{k}."
        h = x * b["in_scale"] + b["in_shift"]                                            # frac: in_scale_frac + fa[s_k]
        h = keep(p + "in_s", silu(q, p + "in", keep(p + "in", requant(h, b["in_scale_frac"] + fa[f"s{k}"], fa[p + "in"], w[p + "in"]))))
        h = h @ b["down_W"].T + b["down_b"]
        h = keep(p + "c_s", silu(q, p + "c", keep(p + "c", requant(h, b["down_frac"], fa[p + "c"], w[p + "c"]))))
        same, after = messages(h, g["to_data"], b["cd_W"])
        h = same[:, :-1] + after[:, 1:] + per_target(b["cd_b"])
        h = keep(p + "d_s", silu(q, p + "d", keep(p + "d", requant(h, b["cd_frac"], fa[p + "d"], w[p + "d"]))))
        same, after = messages(h, g["to_checks"], b["dc_W"])
        h = np.concatenate([same[:, :1] + per_target(b["dc_b_first"]),
                            same[:, 1:] + after[:, :-1] + per_target(b["dc_b_mid"]),
                            after[:, -1:] + per_target(b["dc_b_last"])], axis=1)
        h = keep(p + "o_s", silu(q, p + "o", keep(p + "o", requant(h, b["dc_frac"], fa[p + "o"], w[p + "o"]))))
        up = h @ b["up_W"].T + b["up_b"]
        s_next = f"s{k + 1}"
        x = keep(s_next, requant(requant(x, fa[f"s{k}"], fa[s_next], w[s_next]) + requant(up, b["up_frac"], fa[s_next], w[s_next]),
                                 fa[s_next], fa[s_next], w[s_next]))
    r, depth = q["readout"], len(q["blocks"])
    h = x * r["in_scale"] + r["in_shift"]
    h = keep("r.in_s", silu(q, "r.in", requant(h, r["in_scale_frac"] + fa[f"s{depth}"], fa["r.in"], w["r.in"])))
    sums = np.stack([h[:, :-1].sum(axis=1), h[:, 1:].sum(axis=1)], axis=1)
    sums = keep("r.sum", requant(sums, fa["r.in_s"], fa["r.sum"], w["r.sum"]))
    same, after = messages(sums, g["readout"], r["r.W"])
    data = keep("r.data", requant(same[:, 0] + after[:, 1] + per_target(r["r.b"]), r["r.W.frac"], fa["r.data"], w["r.data"]))
    pooled = keep("r.pool", requant(np.einsum("oq,nqc->noc", r["r.pool"], data), r["r.pool.frac"][:, None], fa["r.pool"], w["r.pool"]))
    h = pooled @ r["h0_W"].T + r["h0_b"]
    h = keep("h_s", silu(q, "h", requant(h, r["h0_W.frac"], fa["h"], w["h"])))
    logits = keep("logits", (h @ r["h2_W"].T + r["h2_b"])[..., 0])
    return logits, int(r["h2_W.frac"][0])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True, help="calibration.json of widths.py")
    parser.add_argument("--run", default="w16")
    parser.add_argument("--shots", type=int, default=20)
    parser.add_argument("--save", help="write the integer network to this .npz (for the hardware's tables)")
    args = parser.parse_args()
    import widths
    from neural.train import load_decoder
    torch.set_num_threads(1)
    model, saved = load_decoder(args.checkpoint)
    folded = fp.fold(model)
    ranges = json.loads(Path(args.calibration).read_text())
    net = widths.make_net(folded, ranges, args.run)
    det, _ = widths.load_shots(widths.DATA, args.shots)
    torch_logits = fp.run(net, torch.from_numpy(det), batch=10)        # also rounds every weight once
    q = export(net)
    acc = accumulator_bits(q)
    print(f"the widest possible sum of products, with sign: {max(acc.values())} bits ({max(acc, key=acc.get)}); "
          f"float64 holds 53 exactly, so the torch version computes whole numbers exactly on any shot")
    int_logits, frac = forward_int(q, det)
    as_whole = torch_logits * 2.0 ** frac
    assert np.array_equal(as_whole, np.rint(as_whole)), "the torch logits are not whole numbers at the head's scale"
    same = np.all(as_whole.astype(I64) == int_logits, axis=1)
    print(f"run {args.run} ({widths.RUNS[args.run][:3]} bits for weights, values, stream), checkpoint step {saved['step']}, "
          f"the first {len(det)} shots of {widths.DATA.name}: integer reference equal to the torch fixed-point logits on "
          f"{int(same.sum())} of {len(det)} shots (all 12 logits, as whole numbers at 2^-{frac})")
    if not same.all():
        bad = np.flatnonzero(~same)
        print("differing shots:", bad[:10].tolist(), "largest difference in units:", int(np.abs(as_whole.astype(I64) - int_logits).max()))
        raise SystemExit(1)
    if args.save:
        flat = {}
        def put(prefix, d):
            for key, v in d.items():
                if isinstance(v, dict):
                    put(f"{prefix}{key}.", v)
                elif isinstance(v, tuple):
                    flat[f"{prefix}{key}.table"], flat[f"{prefix}{key}.lo"] = v[0], np.int64(v[1])
                elif isinstance(v, list):
                    for i, item in enumerate(v):
                        put(f"{prefix}{key}{i}.", item)
                else:
                    flat[prefix + key] = np.asarray(v)
        put("", q)
        np.savez_compressed(args.save, **flat)
        print(f"wrote {args.save}")


if __name__ == "__main__":
    main()
