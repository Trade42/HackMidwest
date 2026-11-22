"""Decode a range of 500-shot chunks of one kit shot file with relay-BP or PriorTilt, on the kit's own problem.

  python -I decode_file.py --kit <kit folder> --pt <PriorTiltDecoder folder> --data <file.npz>
                           --decoder relay|priortilt --chunks 0:10 --out <result.npz>

The problem is the kit's: H, L, priors = dem_matrices(load_circuit(GROSS_CIRCUITS[basis], p), basis), with the
file's p and basis, and the syndrome is detectors[:, basis_detectors]. Each chunk of --chunk-size shots is decoded
in file order on a FRESH decoder (relay-BP's result depends on the order of the shots). A shot fails when any of
the 12 predicted observables, L @ e mod 2, differs from the file's observables; an unconverged shot is scored on
what the decoder returned. Only the decode call is timed, as in the kit's decoders.score().

relay-BP: decoders.relay_decoder(H, priors) with the kit's settings unchanged (decode_detailed, for iterations).
PriorTilt: the Rust core on the kit's H and priors, with the paper's three-phase schedule for BB144 at --pt-p
(default: the file's p), max_rounds 3000, initial_iters 60, round_len 6, f64 messages, seed = --seed-base + chunk.
"""
import argparse
import hashlib
import os
import sys
import time
from importlib import metadata

import numpy as np

FORBIDDEN_PREFIXES = ("test_", "go_")   # the team's held-out shots: never opened here


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True)
    ap.add_argument("--pt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--decoder", choices=["relay", "priortilt"], required=True)
    ap.add_argument("--chunks", required=True, help="a:b, chunk indices a <= k < b")
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pt-p", type=float, default=None)
    ap.add_argument("--seed-base", type=int, default=1)
    ap.add_argument("--precision", default="f64")
    return ap.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    args = parse_args()
    name = os.path.basename(args.data)
    if name.startswith(FORBIDDEN_PREFIXES):
        sys.exit(f"refusing to open held-out file {name}")
    sys.path.insert(0, args.kit)
    sys.path.insert(1, args.pt)
    from circuit import GROSS_CIRCUITS, dem_matrices, detector_ids, load_circuit

    data = np.load(args.data)
    p, basis = float(data["p"]), str(data["basis"])
    circuit = load_circuit(GROSS_CIRCUITS[basis], p)
    if not np.array_equal(data["basis_detectors"], detector_ids(circuit, basis)):
        sys.exit("basis_detectors in the file do not match the kit circuit")
    H, L, priors = dem_matrices(circuit, basis)
    L = L.tocsr()
    det = np.ascontiguousarray(data["detectors"][:, data["basis_detectors"]], dtype=np.uint8)
    obs = np.asarray(data["observables"], dtype=np.uint8)
    a, b = (int(v) for v in args.chunks.split(":"))
    n_chunks = -(-len(det) // args.chunk_size)
    b = min(b, n_chunks)

    if args.decoder == "relay":
        from decoders import RELAY_SETTINGS, relay_decoder
        desc = {"decoder": "relay-BP (kit relay_decoder)", "settings": repr(RELAY_SETTINGS),
                "relay_bp": metadata.version("relay-bp")}

        def fresh(chunk):
            return relay_decoder(H, priors)

        def run(dec, syndrome):
            r = dec.decode_detailed(syndrome)
            return np.asarray(r.decoding, np.uint8), int(r.iterations), bool(r.success)
    else:
        import priortilt
        from priortilt.params import tuned_config
        cfg = tuned_config("BB144", args.pt_p or p)
        desc = {"decoder": "PriorTilt three-phase (paper schedule)", "settings": cfg.describe(),
                "precision": args.precision, "seed_base": args.seed_base, "priortilt_rs": metadata.version("priortilt_rs")}

        def fresh(chunk):
            return priortilt._create_core(args.precision, H, priors, cfg, args.seed_base + chunk)

        def run(dec, syndrome):
            e = np.asarray(dec.decode(syndrome), np.uint8)
            return e, int(dec.iterations), bool(dec.converge)

    # Weight of a returned correction: the number of faults and the sum of log((1 - prior) / prior) over them.
    llr = np.log((1 - priors) / priors)
    idx, failed, seconds, iters, conv = [], [], [], [], []
    preds, n_faults, weights = [], [], []
    started = time.time()
    for chunk in range(a, b):
        dec = fresh(chunk)
        for k in range(chunk * args.chunk_size, min((chunk + 1) * args.chunk_size, len(det))):
            t0 = time.perf_counter()
            e, it, ok = run(dec, det[k])
            dt = time.perf_counter() - t0
            pred = np.asarray((L @ e) % 2, dtype=np.uint8)
            idx.append(k)
            failed.append(bool((pred != obs[k]).any()))
            seconds.append(dt)
            iters.append(it)
            conv.append(ok)
            preds.append(pred)
            on = e.astype(bool)
            n_faults.append(int(on.sum()))
            weights.append(float(llr[on].sum()))
        print(f"{name} {args.decoder} chunk {chunk}: shots {len(idx)}, failures {sum(failed)}, "
              f"mean {np.mean(seconds) * 1e3:.1f} ms, {time.time() - started:.0f} s", flush=True)
    np.savez(args.out, shot=np.array(idx), failed=np.array(failed), seconds=np.array(seconds),
             iterations=np.array(iters), converged=np.array(conv), predicted=np.array(preds, dtype=np.uint8),
             truth=obs[np.array(idx)], n_faults=np.array(n_faults, dtype=np.int32), weight=np.array(weights),
             data=args.data, data_sha256=sha256(args.data),
             p=p, basis=basis, chunks=args.chunks, chunk_size=args.chunk_size, started=started, ended=time.time(),
             pid=os.getpid(), **{f"desc_{k}": str(v) for k, v in desc.items()})
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
