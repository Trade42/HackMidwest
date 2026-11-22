"""Decode a range of shots of a loss dataset with relay-BP under several prior conditions.

usage (from challenge-1-ml-decoder-kit/):
    python -m loss.decode --data DATA.npz --conditions blind,starter --start 0 --stop 500 --workers 8 --out PART.npz

Every (shot, condition) is decoded by a fresh relay-BP decoder at the kit's settings (decoders.RELAY_SETTINGS:
stop after 5 solutions) with seed 1, so a result depends only on the shot and the condition: not on the order of
the shots, the chunking or the number of processes. A shot fails when any of the 12 predicted observables is
wrong; an unconverged shot is scored on the decoder's output like any other.

The decoder sees the basis detectors and the heralds. The true loss times (data_slot, check_k) are passed to the
prior builder only for the oracle conditions.
"""
import argparse
import multiprocessing as mp
import time

import numpy as np

from decoders import relay_decoder
from loss.lossmap import LossMap, PriorBuilder, parse_condition
from loss.lossmodel import load_dataset

_G = {}


def _init(data_path, conditions):
    data = load_dataset(data_path)
    meta = data["meta"]
    lmap = LossMap(meta["basis"], meta["p"])
    _G.update(data=data, lmap=lmap, builder=PriorBuilder(lmap, meta["s"]), conditions=conditions,
              specs=[parse_condition(c) for c in conditions], det=data["detectors"][:, lmap.rows])


def _decode_shot(k):
    g = _G
    lmap, data = g["lmap"], g["data"]
    oracle_inputs = (data["data_slot"][k], data["check_k"][k]) if "data_slot" in data else (None, None)
    rows = []
    for spec in g["specs"]:
        uses_truth = "oracle" in str(spec.get("data")) or spec.get("hook") == "oracle"
        truth = oracle_inputs if uses_truth else (None, None)
        t0 = time.perf_counter()
        pri = g["builder"].priors(spec, data["heralds"][k], *truth)
        t1 = time.perf_counter()
        dec = relay_decoder(lmap.H, pri, seed=1, **spec["relay"])
        t2 = time.perf_counter()
        r = dec.decode_detailed(g["det"][k])
        t3 = time.perf_counter()
        e = np.asarray(r.decoding, np.uint8)
        wrong = ((lmap.L @ e) % 2).astype(np.uint8) != data["observables"][k]
        rows.append((bool(wrong.any()), bool(r.success), int(r.iterations), int(np.packbits(wrong, bitorder="little").view(np.uint16)[0]),
                     1e3 * (t1 - t0), 1e3 * (t2 - t1), 1e3 * (t3 - t2)))
    return k, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    conditions = a.conditions.split(",")
    data = load_dataset(a.data)
    stop = a.stop or data["meta"]["shots"]
    shots = range(a.start, stop)
    print(f"decode {a.data} shots {a.start}..{stop} conditions {conditions} workers {a.workers}", flush=True)
    t0 = time.perf_counter()
    ctx = mp.get_context("fork")
    results = {}
    with ctx.Pool(a.workers, initializer=_init, initargs=(a.data, conditions)) as pool:
        for n, (k, rows) in enumerate(pool.imap_unordered(_decode_shot, shots, chunksize=2), 1):
            results[k] = rows
            if n % 50 == 0 or n == len(shots):
                done = np.array([results[i][c][0] for i in results for c in range(len(conditions))]).reshape(-1, len(conditions))
                print(f"  {n}/{len(shots)} shots, {time.perf_counter() - t0:.0f}s, failures so far "
                      + " ".join(f"{c}={int(done[:, i].sum())}" for i, c in enumerate(conditions)), flush=True)
    order = sorted(results)
    assert order == list(shots)
    arr = np.array([results[k] for k in order], dtype=np.float64)       # (shots, conditions, 7)
    np.savez_compressed(a.out, shots=np.array(order), conditions=np.array(conditions),
                        failed=arr[:, :, 0].astype(bool), converged=arr[:, :, 1].astype(bool),
                        iterations=arr[:, :, 2].astype(np.int32), wrong_mask=arr[:, :, 3].astype(np.uint16),
                        prior_ms=arr[:, :, 4], build_ms=arr[:, :, 5], decode_ms=arr[:, :, 6])
    print(f"wrote {a.out} in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
