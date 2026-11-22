#!/usr/bin/env python3
"""Export frozen kit router + actual saved shots, or compare an Urbana board over UART.

Uses only the Python standard library for export. The send command needs pyserial.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
N = 17
FW = 16
WW = 24
AW = 56
PKT = 124
REPLY = 8
EXPECTED_NAMES = ["first_solution_iterations", "candidate_converged",
                  "mean_abs_posterior_ratio", "correction_weight"] + [
                      f"syndrome_weight_cycle_{i}" for i in range(13)]


def npy(z, name):
    blob = z.read(name + ".npy")
    if blob[:6] != b"\x93NUMPY":
        raise ValueError(f"bad NPY: {name}")
    major = blob[6]
    hlen, start = (struct.unpack_from("<H", blob, 8)[0], 10) if major == 1 else (
        struct.unpack_from("<I", blob, 8)[0], 12)
    header = ast.literal_eval(blob[start:start + hlen].decode("latin1"))
    if header["fortran_order"]:
        raise ValueError("Fortran-order NPY is unsupported")
    return header, memoryview(blob)[start + hlen:]


def load_rows(path):
    with zipfile.ZipFile(path) as z:
        h, body = npy(z, "features")
        if h["descr"] != "<f8" or len(h["shape"]) != 2 or h["shape"][1] != N:
            raise ValueError("feature array must be (shots,17) float64")
        rows = [row for row in struct.iter_unpack("<" + "d" * N, body)]
        eh, eb = npy(z, "first_leg_converged")
        if eh["descr"] != "|b1" or eh["shape"] != (len(rows),):
            raise ValueError("first-leg flags have the wrong shape")
        easy_flags = eb.tobytes()
        mh, mb = npy(z, "metadata")
        if not mh["descr"].startswith("<U") or mh["shape"] != ():
            raise ValueError("unsupported feature metadata")
        meta = json.loads(mb.tobytes().decode("utf-32le"))
    if meta["feature_names"] != EXPECTED_NAMES:
        raise ValueError("feature names or order changed")
    return rows, easy_flags, meta


def load_syndromes(path, n):
    with zipfile.ZipFile(path) as z:
        dh, db = npy(z, "detectors")
        ih, ib = npy(z, "basis_detectors")
        if dh["descr"] != "|u1" or dh["shape"] != (n, 1728):
            raise ValueError("expected (shots,1728) uint8 detectors")
        if ih["descr"] != "<i8" or ih["shape"] != (936,):
            raise ValueError("expected 936 basis detector indices")
        ids = [v[0] for v in struct.iter_unpack("<q", ib)]
        if any(i < 0 or i >= 1728 for i in ids):
            raise ValueError("basis detector index outside detector record")
        return db, ids


def quantize_model(model):
    if model["feature_names"] != EXPECTED_NAMES or len(model["weights"]) != N + 1:
        raise ValueError("frozen router feature schema changed")
    if not 0 < model["threshold"] < 1:
        raise ValueError("risk threshold outside (0,1)")
    raw = [model["weights"][i + 1] / model["scale"][i] for i in range(N)]
    bias = model["weights"][0] - sum(raw[i] * model["mean"][i] for i in range(N))
    effective = raw.copy()
    effective[2] /= 256  # posterior travels as unsigned Q8.8
    frac = min(20, math.floor(math.log2(((1 << (WW - 1)) - 1) / max(map(abs, effective)))))
    if frac < 0:
        raise ValueError("weights exceed 24-bit fixed-point range")
    weights = [round(w * (1 << frac)) for w in effective]
    bq = round(bias * (1 << frac))
    tq = round(math.log(model["threshold"] / (1 - model["threshold"])) * (1 << frac))
    if any(not -(1 << 23) <= w < (1 << 23) for w in weights) or not -(1 << 55) <= bq < (1 << 55):
        raise ValueError("fixed-point coefficient overflow")
    return weights, bq, tq, frac, raw, bias


def packet_and_features(row, detectors, ids, offset):
    if any(not math.isfinite(v) for v in row):
        raise ValueError("nonfinite feature")
    x = [round(row[0]), round(row[1]), round(row[2] * 256), round(row[3])]
    if not all(0 <= v < (1 << FW) for v in x) or x[1] not in (0, 1):
        raise ValueError("scalar feature outside packet range")
    syndrome = [int(detectors[offset + i]) for i in ids]
    if any(bit not in (0, 1) for bit in syndrome):
        raise ValueError("nonbinary syndrome")
    x += [sum(syndrome[72 * c:72 * (c + 1)]) for c in range(13)]
    if any(x[i] != round(row[i]) for i in list(range(4, N)) + [0, 1, 3]):
        raise ValueError("saved features do not match the original syndrome/statistics")
    payload = bytearray()
    for k in range(0, 936, 8):
        payload.append(sum(syndrome[k + j] << j for j in range(8)))
    payload.extend(struct.pack("<HBHH", x[0], x[1], x[2], x[3]))
    assert len(payload) == PKT
    return bytes(payload), x


def export(args):
    model_path, feat_path, data_path = map(Path, (args.model, args.features, args.data))
    model = json.loads(model_path.read_text())
    rows, easy_flags, feature_meta = load_rows(feat_path)
    identity = feature_meta["dataset"]
    data_digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if (identity["sha256"] != data_digest or identity["shots"] != len(rows)
            or feature_meta["basis"] != model["basis"] or feature_meta["p"] != model["p"]
            or model["validation_dataset"]["sha256"] != data_digest):
        raise ValueError("model, feature file, and original shot file do not belong together")
    detectors, ids = load_syndromes(data_path, len(rows))
    weights, bias_q, thresh_q, frac, raw, bias = quantize_model(model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    terms = ", ".join(f"24'h{w & ((1 << WW)-1):06x}" for w in reversed(weights))
    (out / "router_params.svh").write_text(
        f"// Frozen model {model['basis']} p={model['p']}; SHA256 {hashlib.sha256(model_path.read_bytes()).hexdigest()}\n"
        f"localparam integer FRAC = {frac};\n"
        f"localparam logic [407:0] W_FLAT = {{{terms}}};\n"
        f"localparam logic signed [55:0] BIAS = 56'sh{bias_q & ((1 << AW)-1):014x};\n"
        f"localparam logic signed [55:0] THRESH = 56'sh{thresh_q & ((1 << AW)-1):014x};\n")
    packet_file, reply_file = bytearray(), bytearray()
    core_lines, uart_lines = [], []
    flips = easy_count = escalated = 0
    for k, row in enumerate(rows):
        packet, x = packet_and_features(row, detectors, ids, k * 1728)
        easy = x[1] == 1 and 0 < x[0] <= 80
        if easy != bool(easy_flags[k]):
            raise ValueError(f"shot {k}: RTL first-leg bypass disagrees with saved software flag")
        score = bias_q + sum(a * b for a, b in zip(weights, x))
        if not -(1 << (AW-1)) <= score < (1 << (AW-1)):
            raise ValueError("56-bit score overflow")
        verdict = int(not easy and score > thresh_q)
        float_score = bias + sum(a * b for a, b in zip(raw, row))
        # The frozen software rule compares sigmoid risk, not a rounded logit
        # threshold. This matters for a validation point exactly on that edge.
        float_risk = 1 / (1 + math.exp(-float_score))
        float_verdict = int(not easy and float_risk > model["threshold"])
        flips += verdict != float_verdict
        easy_count += easy
        escalated += verdict
        reply = (score & ((1 << AW)-1)).to_bytes(7, "little") + bytes([verdict])
        packet_file.extend(packet)
        reply_file.extend(reply)
        packed_x = sum(v << (FW*i) for i, v in enumerate(x))
        core_lines.append(f"{packed_x | ((score & ((1 << AW)-1)) << (N*FW)) | (verdict << (N*FW+AW)):083x}\n")
        if k < args.uart_vectors:
            uart_lines.append(" ".join(f"{v:02x}" for v in packet + reply) + "\n")
    (out / "packets.bin").write_bytes(packet_file)
    (out / "expected.bin").write_bytes(reply_file)
    (out / "core_vectors.mem").write_text("".join(core_lines))
    (out / "uart_vectors.mem").write_text("".join(uart_lines))
    (out / "tb_meta.svh").write_text(f"localparam integer NCORE = {len(rows)};\nlocalparam integer NUART = {min(len(rows),args.uart_vectors)};\n")
    manifest = {"basis": model["basis"], "p": model["p"], "shots": len(rows),
                "easy": easy_count, "escalated_fixed": escalated, "float_fixed_verdict_flips": flips,
                "fractional_bits": frac, "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "original_shots_sha256": data_digest, "uart_vectors": min(len(rows), args.uart_vectors),
                "packet_bytes": PKT, "reply_bytes": REPLY}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def send(args):
    try:
        import serial
    except ImportError as exc:
        raise SystemExit("Install pyserial: python3 -m pip install pyserial") from exc
    directory = Path(args.dir)
    packet = (directory / "packets.bin").read_bytes()
    expected = (directory / "expected.bin").read_bytes()
    total = min(len(packet) // PKT, args.limit or len(packet) // PKT)
    mismatches = 0
    slow = 0
    started = time.perf_counter()
    with serial.Serial(args.port, 115200, timeout=2) as link:
        link.reset_input_buffer()
        for k in range(total):
            link.write(packet[k*PKT:(k+1)*PKT])
            got = link.read(REPLY)
            want = expected[k*REPLY:(k+1)*REPLY]
            if got != want:
                mismatches += 1
                if mismatches <= 8:
                    print(f"shot {k}: got {got.hex()} expected {want.hex()}")
            else:
                slow += bool(got[-1])
                if args.verbose and k < 20:
                    print(f"shot {k}: {'SLOW' if got[-1] else 'FAST'} score={int.from_bytes(got[:7], 'little', signed=True)}")
    elapsed = time.perf_counter() - started
    print(f"{total} board shots, {mismatches} byte-exact mismatches, "
          f"{slow} SLOW replies, {1000 * elapsed / total:.2f} ms/shot over UART")
    if mismatches:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    e = commands.add_parser("export")
    e.add_argument("--model", required=True)
    e.add_argument("--features", required=True)
    e.add_argument("--data", required=True)
    e.add_argument("--out", default="generated")
    e.add_argument("--uart-vectors", type=int, default=32)
    s = commands.add_parser("send")
    s.add_argument("--port", required=True)
    s.add_argument("--dir", default="generated")
    s.add_argument("--limit", type=int, default=100)
    s.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.command == "export":
        if args.uart_vectors < 1:
            parser.error("--uart-vectors must be positive")
        export(args)
    else:
        send(args)


if __name__ == "__main__":
    main()
