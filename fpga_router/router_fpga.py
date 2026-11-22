"""Host side of the router-on-FPGA demo: features, fixed-point model, test vectors, serial link.

  python router_fpga.py shots  KIT_DIR P N_SHOTS OUT.npz      real shots from the kit -> syndromes, BP statistics, labels
  python router_fpga.py export MODEL.npz SHOTS.npz [OUT_DIR]  weights -> router_params.vh, shots -> vectors.mem
  python router_fpga.py send   PORT MODEL.npz SHOTS.npz       run the shots through the board and compare bit for bit
  python router_fpga.py sendmem PORT VECTORS.mem [N]          the same from a vectors.mem file (first N shots): numpy and pyserial
  python router_fpga.py standin SHOTS.npz MODEL.npz          made-up weights, to test the hardware before router v1 exists

The trained router is made by train_router.py, which calls export_arrays() below.

MODEL.npz holds a logistic model that predicts "the first relay-BP solution is wrong":
  coef (16,), intercept (), mean (16,), scale (16,), threshold ()    threshold is a probability; above it the shot escalates
  optional: thresh_fixed ()   the threshold as the integer the board compares with; it then replaces `threshold`
            easy_iter ()      shots with 1 to easy_iter iterations are accepted without asking the model (0 or absent: no such rule)
From scikit-learn: np.savez("router_v1.npz", coef=clf.coef_[0], intercept=clf.intercept_[0],
                            mean=scaler.mean_, scale=scaler.scale_, threshold=0.2)
The model must be trained on features(...) below, as integers, or the board and the laptop will disagree.
"""
import sys
import numpy as np

ROUNDS, CHECKS = 13, 72
N_FEAT = ROUNDS + 3                  # 13 syndrome weights, iterations, solution weight, mean |posterior|
PACKET_BYTES = ROUNDS * CHECKS // 8 + 6
REPLY_BYTES = 6
XW, WW, AW = 16, 16, 40              # feature, weight and score widths in the Verilog


def features(syndrome, iterations, weight, mean_abs_post):
    """The 16 integer features, exactly as the board computes them. syndrome: (..., 936) bits in H's row order."""
    s = np.asarray(syndrome, np.int64).reshape(*np.shape(syndrome)[:-1], ROUNDS, CHECKS)
    m = np.clip(np.nan_to_num(np.asarray(mean_abs_post, float), nan=0.0, posinf=1e12), 0, 1e12)
    q = np.round(16 * np.log2(1 + m)).astype(np.int64)          # the mean spans 6 decades, so it travels as a log
    scalars = [np.clip(np.asarray(v, np.int64), 0, 65535) for v in (iterations, weight)] + [q]
    return np.concatenate([s.sum(-1), np.stack(scalars, -1)], -1)


def packets(syndrome, iterations, weight, mean_abs_post):
    """(shots, 123) bytes: 117 syndrome bytes (round r in bytes 9r..9r+8), then three little-endian uint16."""
    x = features(syndrome, iterations, weight, mean_abs_post)
    bits = np.packbits(np.asarray(syndrome, np.uint8), axis=-1, bitorder="little")
    tail = np.stack([x[..., ROUNDS + k] >> sh & 255 for k in range(3) for sh in (0, 8)], -1).astype(np.uint8)
    return np.concatenate([bits, tail], -1)


def quantize(model):
    """Fold the scaler into the weights and round to fixed point. Returns W (16 ints), BIAS, THRESH, FRAC, (w, b), EASY_ITER."""
    w = np.asarray(model["coef"], float) / np.asarray(model["scale"], float)
    b = float(model["intercept"]) - float(np.sum(w * np.asarray(model["mean"], float)))
    frac = int(min(24, np.floor(np.log2((2 ** (WW - 1) - 1) / max(np.abs(w).max(), 1e-12)))))
    W = np.round(w * 2 ** frac).astype(np.int64)
    BIAS = int(round(b * 2 ** frac))
    if "thresh_fixed" in model:                      # chosen on the board's own integer scores: no rounding in between
        THRESH = int(model["thresh_fixed"])
    else:
        t = float(model["threshold"])
        THRESH = int(round(np.log(t / (1 - t)) * 2 ** frac))
    easy = int(model["easy_iter"]) if "easy_iter" in model else 0
    assert np.abs(W).max() < 2 ** (WW - 1) and abs(BIAS) < 2 ** (AW - 2) and abs(THRESH) < 2 ** (AW - 2) and 0 <= easy < 2 ** XW
    return W, BIAS, THRESH, frac, (w, b), easy


def score_fixed(x, W, BIAS):
    return np.asarray(x, np.int64) @ W + BIAS


def verdict(x, score, THRESH, easy_iter=0):
    """1 = escalate, as the board decides it: the score is above the threshold and the shot is not an easy one.

    A shot is easy when its iteration count (feature 13) is 1 to easy_iter: relay-BP's first leg found the solution.
    With easy_iter = 0 no shot is easy.
    """
    it = np.asarray(x, np.int64)[..., ROUNDS]
    hard = (it == 0) | (it > easy_iter)
    return (np.asarray(score) > THRESH) & hard


def replies(score, escalate):
    """(shots, 6) bytes: the 40-bit score, little endian, then 1 if the shot escalates."""
    u = np.asarray(score, np.int64) & (2 ** AW - 1)
    return np.stack([u >> 8 * k & 255 for k in range(5)] + [np.asarray(escalate).astype(np.int64)], -1).astype(np.uint8)


def hexs(v, bits):
    return f"{int(v) & (2 ** bits - 1):0{bits // 4}x}"


def cmd_shots(kit, p, n, out):
    sys.path.insert(0, kit)
    from circuit import GROSS_CIRCUITS, dem_matrices, detector_ids, load_circuit
    from decoders import relay_decoder
    circ = load_circuit(GROSS_CIRCUITS["X"], p)
    H, L, pri = dem_matrices(circ, "X")
    ids = np.asarray(detector_ids(circ, "X"))
    det, obs = circ.compile_detector_sampler(seed=11).sample(n, separate_observables=True)
    S = det[:, ids].astype(np.uint8)
    leg = relay_decoder(H, pri, num_sets=0, seed=1)
    rest = relay_decoder(H, pri, num_sets=20, stop_nconv=1, seed=1)
    rows = []
    for k in range(n):
        r = leg.decode_detailed(S[k]); it = r.iterations
        if not r.success:
            r = rest.decode_detailed(S[k]); it += r.iterations
        e = np.asarray(r.decoding, np.uint8)
        wrong = (not r.success) or bool((((L @ e) % 2) != obs[k]).any())
        rows.append((it, int(e.sum()), float(np.mean(np.abs(np.nan_to_num(np.asarray(r.posterior_ratios, float), posinf=1e9, neginf=-1e9)))), wrong, it > 80))
    it, wt, post, wrong, hard = map(np.array, zip(*rows))
    np.savez_compressed(out, syndrome=S, iterations=it, weight=wt, mean_abs_post=post, wrong=wrong, hard=hard)
    print(f"{n} Stim shots at p={p}: {hard.mean():.1%} hard, {wrong.sum()} wrong first answers")
    print("mean |posterior| percentiles 1/50/99/max:", np.round(np.percentile(post, [1, 50, 99, 100]), 1).tolist())
    print("iterations max", int(it.max()), "| solution weight max", int(wt.max()), "| detectors per round max", int(S.reshape(n, ROUNDS, CHECKS).sum(-1).max()))


def cmd_export(model_file, shots_file, out_dir="."):
    export_arrays(np.load(model_file), np.load(shots_file), out_dir)


def export_arrays(model, d, out_dir="."):
    """model and d are mappings with the entries of MODEL.npz and SHOTS.npz. Returns (x, fixed scores, verdicts)."""
    W, BIAS, THRESH, frac, (w, b), easy = quantize(model)
    args = (d["syndrome"], d["iterations"], d["weight"], d["mean_abs_post"])
    x = features(*args)
    fixed, flt = score_fixed(x, W, BIAS), x @ w + b
    esc = verdict(x, fixed, THRESH, easy)
    if "thresh_fixed" in model:
        t_float = THRESH / 2 ** frac                # the unrounded weights against the same threshold
    else:
        t = float(model["threshold"])
        t_float = np.log(t / (1 - t))               # the unrounded weights against the unrounded threshold
    flips = int((esc != verdict(x, flt, t_float, easy)).sum())
    print(f"FRAC = {frac} bits. Largest logit error from rounding: {np.abs(fixed / 2 ** frac - flt).max():.4f}. Verdicts changed: {flips} of {len(x)}")
    with open(f"{out_dir}/router_params.vh", "w") as f:
        f.write("// Generated by router_fpga.py export. Do not edit by hand.\n")
        f.write(f"localparam integer FRAC = {frac};\n")
        f.write(f"localparam [{N_FEAT * WW - 1}:0] W_FLAT = {{" + ", ".join(f"{WW}'h{hexs(v, WW)}" for v in W[::-1]) + "};\n")
        f.write(f"localparam signed [{AW - 1}:0] BIAS = {AW}'sh{hexs(BIAS, AW)};\n")
        f.write(f"localparam signed [{AW - 1}:0] THRESH = {AW}'sh{hexs(THRESH, AW)};\n")
        f.write(f"localparam [{XW - 1}:0] EASY_ITER = {XW}'d{easy};\n")
    vec = np.concatenate([packets(*args), replies(fixed, esc)], -1)
    with open(f"{out_dir}/vectors.mem", "w") as f:
        f.writelines(" ".join(f"{v:02x}" for v in row) + "\n" for row in vec)
    with open(f"{out_dir}/tb_vectors.vh", "w") as f:
        f.write(f"localparam integer NVEC = {len(vec)};\n")
    print(f"wrote router_params.vh, vectors.mem ({len(vec)} shots, {vec.shape[1]} bytes each), tb_vectors.vh. Escalated: {esc.mean():.1%}")
    return x, fixed, esc


def cmd_standin(shots_file, out):
    d = np.load(shots_file)
    x = features(d["syndrome"], d["iterations"], d["weight"], d["mean_abs_post"]).astype(float)
    coef = np.r_[np.full(ROUNDS, 0.15), 1.6, 0.7, -0.9]          # NOT a trained router: plausible signs and sizes only
    np.savez(out, coef=coef, intercept=-2.5, mean=x.mean(0), scale=x.std(0) + 1e-9, threshold=0.2)
    print(f"wrote {out}: stand-in weights, not a trained router")


def cmd_send(port, model_file, shots_file):
    import time
    import serial                                   # pip install pyserial
    model, d = np.load(model_file), np.load(shots_file)
    W, BIAS, THRESH, _, _, easy = quantize(model)
    args = (d["syndrome"], d["iterations"], d["weight"], d["mean_abs_post"])
    x = features(*args)
    fixed = score_fixed(x, W, BIAS)
    pk, want = packets(*args), replies(fixed, verdict(x, fixed, THRESH, easy))
    bad, t0 = 0, time.perf_counter()
    with serial.Serial(port, 115200, timeout=1) as link:
        for k in range(len(pk)):
            link.write(pk[k].tobytes())
            got = link.read(REPLY_BYTES)
            bad += got != want[k].tobytes()
    print(f"{len(pk)} shots, {bad} mismatches, {(time.perf_counter() - t0) / len(pk) * 1e3:.1f} ms per shot over UART")


def read_mem(mem_file):
    """vectors.mem -> (packets, expected replies) as lists of bytes objects."""
    rows = [bytes(int(v, 16) for v in line.split()) for line in open(mem_file) if line.strip()]
    assert rows and all(len(r) == PACKET_BYTES + REPLY_BYTES for r in rows), "not a vectors.mem of this design"
    return [r[:PACKET_BYTES] for r in rows], [r[PACKET_BYTES:] for r in rows]


def features_from_packets(pk):
    """The 16 features from the packet bytes, as the board's front end computes them: popcounts and the tail."""
    a = np.frombuffer(b"".join(pk), np.uint8).reshape(len(pk), PACKET_BYTES)
    bits = np.unpackbits(a[:, :ROUNDS * CHECKS // 8], axis=1, bitorder="little").reshape(len(pk), ROUNDS, CHECKS)
    tail = a[:, ROUNDS * CHECKS // 8:].astype(np.int64)
    return np.concatenate([bits.sum(-1), tail[:, 0::2] + 256 * tail[:, 1::2]], -1)


def check_mem(mem_file, model_file):
    """Recompute every expected reply of a vectors.mem from its packets and MODEL.npz. Returns the shots that differ."""
    pk, want = read_mem(mem_file)
    W, BIAS, THRESH, _, _, easy = quantize(np.load(model_file))
    x = features_from_packets(pk)
    score = score_fixed(x, W, BIAS)
    mine = replies(score, verdict(x, score, THRESH, easy))
    return [k for k in range(len(pk)) if mine[k].tobytes() != want[k]]


def cmd_sendmem(port, mem_file, limit=None):
    """Send the packets of a vectors.mem file to the board and compare every reply with the Python fixed-point model.

    The expected replies are those of the file. When model.npz lies beside the file, every one of them is first
    recomputed here from its packet with the fixed-point model (check_mem), so the comparison is with the model and
    not only with a file. The board must hold the bitstream built from the router_params.vh of the same folder.
    """
    import os
    import time
    import serial                                   # pip install pyserial
    pk, want = read_mem(mem_file)
    model_file = os.path.join(os.path.dirname(os.path.abspath(mem_file)), "model.npz")
    if os.path.exists(model_file):
        differ = check_mem(mem_file, model_file)
        print(f"{model_file}: the fixed-point model gives the file's reply on {len(pk) - len(differ)} of {len(pk)} shots")
        assert not differ, f"the file and the model disagree on shots {differ[:10]}: rebuild the folder with train_router.py"
    else:
        print(f"no model.npz beside {mem_file}: the replies are compared with the file only")
    if limit:
        pk, want = pk[:int(limit)], want[:int(limit)]
    bad, short, escalated, t0 = 0, 0, 0, time.perf_counter()
    with serial.Serial(port, 115200, timeout=1) as link:
        time.sleep(0.05)                            # longer than the board's 20 ms resync: a half-received packet is dropped
        link.reset_input_buffer()
        for k in range(len(pk)):
            link.write(pk[k])
            got = link.read(REPLY_BYTES)
            short += len(got) != REPLY_BYTES
            bad += got != want[k]
            escalated += len(got) == REPLY_BYTES and got[5] == 1
            if got != want[k] and bad <= 5:
                print(f"shot {k}: got {got.hex(' ')}, want {want[k].hex(' ')}")
    seconds = time.perf_counter() - t0
    print(f"{len(pk)} shots, {bad} mismatches ({short} replies short or missing), {escalated} escalated, "
          f"{seconds / len(pk) * 1e3:.1f} ms per shot over UART (the link, not the logic)")
    print("PASS: the board's replies equal the file's" if bad == 0 else "FAIL")


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "shots":
        cmd_shots(a[0], float(a[1]), int(a[2]), a[3])
    elif cmd == "export":
        cmd_export(*a)
    elif cmd == "send":
        cmd_send(*a)
    elif cmd == "sendmem":
        cmd_sendmem(*a)
    elif cmd == "standin":
        cmd_standin(*a)
