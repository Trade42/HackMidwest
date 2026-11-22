"""Host side of the router-on-FPGA demo: features, fixed-point model, test vectors, serial link.

  python router_fpga.py shots  KIT_DIR P N_SHOTS OUT.npz      real shots from the kit -> syndromes, BP statistics, labels
  python router_fpga.py export MODEL.npz SHOTS.npz [OUT_DIR]  weights -> router_params.vh, shots -> vectors.mem
  python router_fpga.py send   PORT MODEL.npz SHOTS.npz       run the shots through the board and compare bit for bit
  python router_fpga.py standin SHOTS.npz MODEL.npz          made-up weights, to test the hardware before router v1 exists

MODEL.npz holds a logistic model that predicts "the first relay-BP solution is wrong":
  coef (16,), intercept (), mean (16,), scale (16,), threshold ()    threshold is a probability; above it the shot escalates
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
    """Fold the scaler into the weights and round to fixed point. Returns W (16 ints), BIAS, THRESH, FRAC."""
    w = np.asarray(model["coef"], float) / np.asarray(model["scale"], float)
    b = float(model["intercept"]) - float(np.sum(w * np.asarray(model["mean"], float)))
    t = float(model["threshold"])
    frac = int(min(24, np.floor(np.log2((2 ** (WW - 1) - 1) / max(np.abs(w).max(), 1e-12)))))
    W = np.round(w * 2 ** frac).astype(np.int64)
    BIAS, THRESH = int(round(b * 2 ** frac)), int(round(np.log(t / (1 - t)) * 2 ** frac))
    assert np.abs(W).max() < 2 ** (WW - 1) and abs(BIAS) < 2 ** (AW - 2) and abs(THRESH) < 2 ** (AW - 2)
    return W, BIAS, THRESH, frac, (w, b)


def score_fixed(x, W, BIAS):
    return np.asarray(x, np.int64) @ W + BIAS


def replies(score, THRESH):
    """(shots, 6) bytes: the 40-bit score, little endian, then 1 if the shot escalates."""
    u = np.asarray(score, np.int64) & (2 ** AW - 1)
    return np.stack([u >> 8 * k & 255 for k in range(5)] + [(np.asarray(score) > THRESH).astype(np.int64)], -1).astype(np.uint8)


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
    model, d = np.load(model_file), np.load(shots_file)
    W, BIAS, THRESH, frac, (w, b) = quantize(model)
    args = (d["syndrome"], d["iterations"], d["weight"], d["mean_abs_post"])
    x = features(*args)
    fixed, flt = score_fixed(x, W, BIAS), x @ w + b
    t = float(model["threshold"])
    flips = int(((fixed > THRESH) != (flt > np.log(t / (1 - t)))).sum())
    print(f"FRAC = {frac} bits. Largest logit error from rounding: {np.abs(fixed / 2 ** frac - flt).max():.4f}. Verdicts changed: {flips} of {len(x)}")
    with open(f"{out_dir}/router_params.vh", "w") as f:
        f.write("// Generated by router_fpga.py export. Do not edit by hand.\n")
        f.write(f"localparam integer FRAC = {frac};\n")
        f.write(f"localparam [{N_FEAT * WW - 1}:0] W_FLAT = {{" + ", ".join(f"{WW}'h{hexs(v, WW)}" for v in W[::-1]) + "};\n")
        f.write(f"localparam signed [{AW - 1}:0] BIAS = {AW}'sh{hexs(BIAS, AW)};\n")
        f.write(f"localparam signed [{AW - 1}:0] THRESH = {AW}'sh{hexs(THRESH, AW)};\n")
    vec = np.concatenate([packets(*args), replies(fixed, THRESH)], -1)
    with open(f"{out_dir}/vectors.mem", "w") as f:
        f.writelines(" ".join(f"{v:02x}" for v in row) + "\n" for row in vec)
    with open(f"{out_dir}/tb_vectors.vh", "w") as f:
        f.write(f"localparam integer NVEC = {len(vec)};\n")
    print(f"wrote router_params.vh, vectors.mem ({len(vec)} shots, {vec.shape[1]} bytes each), tb_vectors.vh. Escalated: {(fixed > THRESH).mean():.1%}")


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
    W, BIAS, THRESH, _, _ = quantize(model)
    args = (d["syndrome"], d["iterations"], d["weight"], d["mean_abs_post"])
    pk, want = packets(*args), replies(score_fixed(features(*args), W, BIAS), THRESH)
    bad, t0 = 0, time.perf_counter()
    with serial.Serial(port, 115200, timeout=1) as link:
        for k in range(len(pk)):
            link.write(pk[k].tobytes())
            got = link.read(REPLY_BYTES)
            bad += got != want[k].tobytes()
    print(f"{len(pk)} shots, {bad} mismatches, {(time.perf_counter() - t0) / len(pk) * 1e3:.1f} ms per shot over UART")


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "shots":
        cmd_shots(a[0], float(a[1]), int(a[2]), a[3])
    elif cmd == "export":
        cmd_export(*a)
    elif cmd == "send":
        cmd_send(*a)
    elif cmd == "standin":
        cmd_standin(*a)
