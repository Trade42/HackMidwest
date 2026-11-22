"""The team's network (challenge-1-ml-decoder-kit/neural/model.py) rewritten as hardware would compute it.

Three steps, each checkable:

1. fold(model): the trained network in inference form, as plain tensors. Every batch normalisation that follows a
   linear step is folded into that step's weights and biases. The two that act on the residual stream (a block's
   input, and the readout's) cannot be folded: they stay as a scale and a shift per channel. The block's factor
   1/sqrt(2 depth) goes into its last linear step, and the readout's average over 12 rounds into the readout's
   linear step. forward(folded, shots) in "float" mode must give the model's own logits (to rounding).

2. FixedNet(folded, bits, ranges): the same network in fixed point. Every number that goes into a multiplier, every
   weight and every intermediate value, is a whole number of `bits` bits (two's complement) times a power of two:
     - a weight matrix has one power of two per output channel, the largest that keeps its channel in range;
     - an intermediate value has one power of two per tensor, from the largest value seen on calibration shots
       (train shots, never the shots that are scored);
     - a bias is a whole number at the scale of the sums it is added to (wide, as a hardware accumulator is);
     - after a sum of products, the result goes to the next tensor's scale by ONE RULE everywhere:
       add half of the last place kept, round down (an arithmetic shift right), then saturate to `bits` bits;
     - SiLU is a table with one entry per possible input (2^bits of them).
   Sums are exact: numbers are carried as float64, which holds every product and every sum here exactly
   (products of two 16-bit numbers summed over at most 768 terms stay below 2^43, far inside float64's 2^53).
   So this torch version computes exactly what an integer machine computes; reference.py checks it shot by shot.

3. The scores: widths.py runs this on the team's validation shots against the float network.
"""
import math

import numpy as np
import torch
from torch.nn import functional as F

F64 = torch.float64


# ---------------------------------------------------------------- 1. folding

def _bn(norm):
    """Eval-mode batch normalisation as y = s * x + t, per channel."""
    s = norm.weight.detach().double() / torch.sqrt(norm.running_var.double() + norm.eps)
    return s, norm.bias.detach().double() - s * norm.running_mean.double()


def _hop(hop, scale_out=None):
    """A collecting step as (W [2 target blocks, 2 c_out, slots c_in], same-bias, after-bias [2, c_out]), optionally
    with an output scale s [c_out] multiplied into both time offsets."""
    W = torch.stack([lin.weight.detach().double() for lin in hop.linears])
    b = torch.stack([lin.bias.detach().double() for lin in hop.linears])
    c = W.shape[1] // 2
    if scale_out is not None:
        W = W * torch.cat([scale_out, scale_out])[None, :, None]
        b = b * torch.cat([scale_out, scale_out])[None, :]
    return W, b[:, :c], b[:, c:]


def fold(model):
    """The inference network as a dict of float64 tensors (see the top of this file)."""
    model = model.eval()
    if model.config["residual"] != "branch":
        raise ValueError("only residual='branch' is folded")
    f = {"embedding": model.embedding.weight.detach().double(), "blocks": []}
    for block in model.blocks:
        a_in, c_in = _bn(block.norm_in)
        s_c, t_c = _bn(block.norm_checks)
        s_d, t_d = _bn(block.norm_data)
        s_o, t_o = _bn(block.norm_out)
        W_down = block.down.weight.detach().double() * s_c[:, None]
        b_down = block.down.bias.detach().double() * s_c + t_c
        W_cd, bs_cd, ba_cd = _hop(block.to_data, s_d)
        W_dc, bs_dc, ba_dc = _hop(block.to_checks, s_o)
        f["blocks"].append({
            "in_scale": a_in, "in_shift": c_in,
            "down_W": W_down, "down_b": b_down,
            # checks -> data: every data cycle has both time offsets, so one bias per target block
            "cd_W": W_cd, "cd_b": bs_cd + ba_cd + t_d[None],
            # data -> checks: round 0 has only "same", the last round only "after"
            "dc_W": W_dc, "dc_b_first": bs_dc + t_o[None], "dc_b_mid": bs_dc + ba_dc + t_o[None], "dc_b_last": ba_dc + t_o[None],
            "up_W": block.up.weight.detach().double() * block.branch_scale, "up_b": block.up.bias.detach().double() * block.branch_scale,
        })
    a_r, c_r = _bn(model.norm)
    W_r, bs_r, ba_r = _hop(model.to_data)
    cycles = model.rounds - 1
    f["readout"] = {"in_scale": a_r, "in_shift": c_r,
                    "W": W_r / cycles, "b": bs_r + ba_r,          # the input is a sum over 12 rounds, not their mean
                    "pool": model.pool.detach().double(),
                    "h0_W": model.head[0].weight.detach().double(), "h0_b": model.head[0].bias.detach().double(),
                    "h2_W": model.head[2].weight.detach().double(), "h2_b": model.head[2].bias.detach().double()}
    f["tables"] = {"to_data": model.blocks[0].to_data.neighbours, "to_checks": model.blocks[0].to_checks.neighbours,
                   "readout": model.to_data.neighbours,
                   "input_place": model.input_place, "input_code": model.input_code, "empty_code": model.empty_code}
    for block in model.blocks:      # every block uses the same neighbour tables (they come from the code's graph)
        assert torch.equal(block.to_data.neighbours, f["tables"]["to_data"])
        assert torch.equal(block.to_checks.neighbours, f["tables"]["to_checks"])
    f["rounds"], f["checks"], f["width"] = model.rounds, model.checks, model.config["width"]
    return f


# ---------------------------------------------------------------- the forward pass, shared by every mode

class Float:
    """No rounding: the folded float network. Also records the largest |value| of every named tensor (calibration)."""

    def __init__(self, folded):
        self.f, self.ranges = folded, {}

    def w(self, name, tensor):
        return tensor

    def act(self, name, x):
        m = float(x.abs().max())
        self.ranges[name] = max(self.ranges.get(name, 0.0), m)
        return x

    def silu(self, name, x):
        return self.act(name + "_s", F.silu(self.act(name, x)))


def messages(x, neighbours, W):
    """A collecting step without its biases: (same, after), each [..., 144 targets, c_out]."""
    out = []
    for block in range(2):
        side = x.index_select(-2, neighbours[block]).reshape(*x.shape[:-2], neighbours.shape[1] // 6, -1)
        out.append(side @ W[block].T)
    y = torch.cat(out, dim=-2)
    c = W.shape[1] // 2
    return y[..., :c], y[..., c:]


def per_target(b):
    """[2 target blocks, c] -> [144 targets, c]: every target of a block shares the block's bias."""
    return b.repeat_interleave(72, dim=0)


def forward(q, shots):
    """Detector bits [shots, 1728] -> logits [shots, 12], computed through q (Float or FixedNet)."""
    f, t = q.f, q.f["tables"]
    n, R, C = len(shots), f["rounds"], f["checks"]
    codes = t["empty_code"].repeat(n, 1)
    codes[:, t["input_place"]] = t["input_code"] + shots.long()
    x = q.w("embedding", f["embedding"])[codes].reshape(n, R, C, -1)
    x = q.act("s0", x)                       # the residual stream entering block 0; "s<k>" enters block k
    for k, b in enumerate(f["blocks"]):
        h = q.silu(f"b{k}.in", q.affine(f"b{k}.in", x, b, f"s{k}") if hasattr(q, "affine") else x * b["in_scale"] + b["in_shift"])
        h = q.silu(f"b{k}.c", h @ q.w(f"b{k}.down_W", b["down_W"]).T + q.w(f"b{k}.down_b", b["down_b"]))
        same, after = messages(h, t["to_data"], q.w(f"b{k}.cd_W", b["cd_W"]))
        h = q.silu(f"b{k}.d", same[:, :-1] + after[:, 1:] + per_target(q.w(f"b{k}.cd_b", b["cd_b"])))
        same, after = messages(h, t["to_checks"], q.w(f"b{k}.dc_W", b["dc_W"]))
        first = same[:, :1] + per_target(q.w(f"b{k}.dc_b_first", b["dc_b_first"]))
        mid = same[:, 1:] + after[:, :-1] + per_target(q.w(f"b{k}.dc_b_mid", b["dc_b_mid"]))
        last = after[:, -1:] + per_target(q.w(f"b{k}.dc_b_last", b["dc_b_last"]))
        h = q.silu(f"b{k}.o", torch.cat([first, mid, last], dim=1))
        up = h @ q.w(f"b{k}.up_W", b["up_W"]).T + q.w(f"b{k}.up_b", b["up_b"])
        x = q.add_stream(x, up, f"s{k + 1}") if hasattr(q, "add_stream") else q.act(f"s{k + 1}", x + up)
    r = f["readout"]
    depth = len(f["blocks"])
    h = q.silu("r.in", q.affine("r.in", x, r, f"s{depth}") if hasattr(q, "affine") else x * r["in_scale"] + r["in_shift"])
    sums = q.act("r.sum", torch.stack([h[:, :-1].sum(dim=1), h[:, 1:].sum(dim=1)], dim=1))
    same, after = messages(sums, t["readout"], q.w("r.W", r["W"]))
    data = q.act("r.data", same[:, 0] + after[:, 1] + per_target(q.w("r.b", r["b"])))
    pooled = q.act("r.pool", q.w("r.pool", r["pool"]) @ data)
    h = q.silu("h", pooled @ q.w("h0_W", r["h0_W"]).T + q.w("h0_b", r["h0_b"]))
    return (h @ q.w("h2_W", r["h2_W"]).T + q.w("h2_b", r["h2_b"])).squeeze(-1)


# ---------------------------------------------------------------- 2. fixed point

def round_half_up(x):
    """The one rounding rule: add half, round down. On whole-number multiples of a power of two it is exact."""
    return torch.floor(x + 0.5)


def frac_bits(largest, bits):
    """The largest number of fractional bits that keeps `largest` inside a signed `bits`-bit whole number."""
    top = 2 ** (bits - 1) - 1
    return int(math.floor(math.log2(top / largest))) if largest > 0 else bits - 1


class FixedNet:
    """The network in fixed point: weights of `bits` bits, intermediate values of `act_bits` bits (default: the same).
    See the top of this file. ranges: largest |value| per tensor name, from calibration shots.

    Names. A tensor of intermediate values is named after the step that makes it ("b3.c": block 3's sum after its
    first linear step, the input of a SiLU; "b3.c_s": that SiLU's output). Its scale is 2^-fa[name]: the value v is
    stored as the whole number v * 2^fa[name]. A weight tensor's per-channel fractional bits are in wfrac[name].
    """

    # weight tensors: the axis of the output channel (one power of two per channel)
    WEIGHTS = {"down_W": 0, "up_W": 0, "cd_W": "hop", "dc_W": "hop", "r.W": "hop", "r.pool": 0, "h0_W": 0, "h2_W": 0}
    # bias tensors: (the weight tensor of the same step, the tensor that step reads)
    BIASES = {"down_b": ("down_W", "in_s"), "cd_b": ("cd_W", "c_s"), "dc_b_first": ("dc_W", "d_s"),
              "dc_b_mid": ("dc_W", "d_s"), "dc_b_last": ("dc_W", "d_s"), "up_b": ("up_W", "o_s"),
              "r.b": ("r.W", "r.sum"), "h0_b": ("h0_W", "r.pool"), "h2_b": ("h2_W", "h_s")}

    # The head: the small network at the very end (33,281 weights, 0.1% of the multiply-adds). Its weights and its
    # own intermediate values keep HEAD_BITS whatever the width of the rest: at 8 bits it alone moved the logits by
    # 0.77 (median) where any other layer at 8 bits moved them by under 0.04 (a diagnostic on 10 validation shots).
    HEAD_BITS = 16
    HEAD_WEIGHTS, HEAD_VALUES = ("h0_W", "h2_W"), ("h", "h_s")

    def __init__(self, folded, bits, ranges, act_bits=None, head_bits=None):
        act_bits = act_bits or bits
        self.head_bits = head_bits or self.HEAD_BITS
        self.f, self.bits, self.act_bits, self.ranges = folded, bits, act_bits, dict(ranges)
        self.width = {name: self.head_bits if name in self.HEAD_VALUES else act_bits for name in self.ranges}
        self.fa = {name: frac_bits(m, self.width[name]) for name, m in self.ranges.items()}
        self.wq, self.wfrac, self.tables, self.saturated = {}, {}, {}, {}

    @staticmethod
    def _split(name):
        """'b3.down_W' -> ('b3.', 'down_W'); 'r.W' -> ('', 'r.W')."""
        if name[0] == "b" and name[1].isdigit():
            prefix, key = name.split(".", 1)
            return prefix + ".", key
        return "", name

    def whole(self, name, x):
        """The whole numbers of a tensor at its scale: add half, round down, saturate. Saturations are counted."""
        b = self.width[name]
        lo, hi = -(2 ** (b - 1)), 2 ** (b - 1) - 1
        k = x.mul(2.0 ** self.fa[name]).add_(0.5).floor_()
        if float(k.max()) > hi or float(k.min()) < lo:
            self.saturated[name] = self.saturated.get(name, 0) + int(((k < lo) | (k > hi)).sum())
            k.clamp_(lo, hi)
        return k

    def quant(self, name, x):
        """Round to the tensor's scale (add half, round down) and saturate."""
        return self.whole(name, x).div_(2.0 ** self.fa[name])

    def act(self, name, x):
        return self.quant(name, x)

    def _round_weight(self, name, tensor, axis):
        """One power of two per output channel. For a collecting step (axis "hop", W [2 blocks, 2 c, inputs]) the
        channel o of "same" and of "after" and of both target blocks share one power of two: they are summed."""
        bits = self.head_bits if name in self.HEAD_WEIGHTS else self.bits
        wlo, whi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        if axis == "hop":
            c = tensor.shape[1] // 2
            largest = torch.maximum(tensor[:, :c].abs().amax(dim=(0, 2)), tensor[:, c:].abs().amax(dim=(0, 2)))
            frac = torch.tensor([frac_bits(float(m), bits) for m in largest], dtype=F64)
            s = (2.0 ** torch.cat([frac, frac])).reshape(1, -1, 1)
        else:
            others = tuple(d for d in range(tensor.ndim) if d != axis)
            largest = tensor.abs().amax(dim=others) if others else tensor.abs()
            frac = torch.tensor([frac_bits(float(m), bits) for m in largest], dtype=F64)
            s = (2.0 ** frac).reshape([-1 if d == axis else 1 for d in range(tensor.ndim)])
        self.wfrac[name] = frac
        return round_half_up(tensor * s).clamp(wlo, whi) / s

    def w(self, name, tensor):
        """A weight, bias or table, rounded once and kept."""
        if name not in self.wq:
            prefix, key = self._split(name)
            if key == "embedding":
                self.wq[name] = self.quant("s0", tensor)                 # a table of values of the stream s0
            elif key in self.WEIGHTS:
                self.wq[name] = self._round_weight(name, tensor, self.WEIGHTS[key])
            elif key in self.BIASES:
                wname, inp = self.BIASES[key]
                frac = self.wfrac[prefix + wname] + self.fa[prefix + inp] if wname[:2] not in ("r.", "h0", "h2") \
                    else self.wfrac[wname] + self.fa[inp]
                self.wq[name] = round_half_up(tensor * 2.0 ** frac) / 2.0 ** frac   # at the scale of the sums
            else:
                raise KeyError(name)
        return self.wq[name]

    def affine(self, name, x, layer, stream):
        """The per-channel scale and shift of a normalisation on the stream: the scale rounded per channel, the shift at
        the scale of the products, then rounded to the SiLU's input by silu()."""
        if name + ".scale" not in self.wq:
            self.wq[name + ".scale"] = self._round_weight(name + ".scale", layer["in_scale"], 0)
            frac = self.wfrac[name + ".scale"] + self.fa[stream]
            self.wq[name + ".shift"] = round_half_up(layer["in_shift"] * 2.0 ** frac) / 2.0 ** frac
        return x * self.wq[name + ".scale"] + self.wq[name + ".shift"]

    def table(self, name):
        """The SiLU table of a tensor name: whole-number input -> whole-number output, one entry per possible input."""
        if name not in self.tables:
            b, fi, fo = self.width[name], self.fa[name], self.fa[name + "_s"]
            lo, hi = -(2 ** (b - 1)), 2 ** (b - 1) - 1
            k = torch.arange(lo, hi + 1, dtype=F64)
            self.tables[name] = (round_half_up(F.silu(k / 2.0 ** fi) * 2.0 ** fo).clamp(lo, hi), lo)
        return self.tables[name]

    def silu(self, name, x):
        """Round the sum to the SiLU's input scale, then read the result from the table. Above 16 bits (a diagnostic
        only: no hardware would hold such a table) the same entries are computed instead of stored."""
        if self.width[name] > 16:
            return self.quant(name + "_s", F.silu(self.quant(name, x)))
        k = self.whole(name, x)
        table, lo = self.table(name)
        return table[k.long().sub_(lo)].div_(2.0 ** self.fa[name + "_s"])

    def add_stream(self, x, up, stream):
        """The residual sum, at the scale of the next block's stream: the old stream and the block's output are each
        rounded to it, added, and the sum saturates."""
        return self.quant(stream, self.quant(stream, x) + self.quant(stream, up))


def calibrate(folded, shots, batch=50):
    """Largest |value| of every named tensor on these shots (they must be train shots)."""
    q = Float(folded)
    with torch.no_grad():
        for start in range(0, len(shots), batch):
            forward(q, shots[start:start + batch])
    return q.ranges


def run(q, shots, batch=50):
    """Logits [shots, 12] as float64 numpy, through q."""
    out = []
    with torch.no_grad():
        for start in range(0, len(shots), batch):
            out.append(forward(q, shots[start:start + batch]))
    return torch.cat(out).numpy()
