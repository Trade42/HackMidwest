"""Decode a range of shots with relay-BP under loss-gates' conditions and under the tracker's variants.

usage (from challenge-1-ml-decoder-kit/):
    python -m loss_tracker.decode --data DATA.npz --post POST.npz --conditions starter,exact,tracker,oracle-round \
        --start 0 --stop 500 --workers 4 --out PART.npz

- A condition of `loss.lossmap.CONDITIONS` (`starter`, `exact`, the ORACLE `oracle-round`, with loss-gates'
  `@cap` and `/stop1` suffixes) is built by loss-gates' PriorBuilder, exactly as `loss.decode` does.
- `tracker...` (priors.parse_variant) takes the tracker's output from the file POST, written by track.py, which
  reads detectors and heralds only. A suffix `@cap` changes the cap on raised priors (default 0.5).
- `plumb-oracle-round` is an ORACLE for the plumbing check: this pipeline with the TRUE loss round put in.

As in loss.decode: every (shot, condition) gets a fresh relay-BP decoder at the kit's settings with seed 1, so a
result depends only on the shot and the condition; a shot fails when any of the 12 observables is wrong; an
unconverged shot is scored on the decoder's output. The part file has loss.decode's format, so loss.summarize
reads it. The true loss times are passed on only for the conditions that are ORACLES by name.
"""
import argparse
import multiprocessing as mp
import time

import numpy as np

from decoders import relay_decoder
from loss.lossmap import CONDITIONS, LossMap, PriorBuilder, parse_condition
from loss.lossmodel import load_dataset
from loss_tracker.priors import parse_variant, priors_from_slots, slot_dists, true_round_dists

_G = {}


def _init(data_path, post_path, conditions):
    data = load_dataset(data_path)
    meta = data["meta"]
    lmap = LossMap(meta["basis"], meta["p"])
    _G.update(data=data, lmap=lmap, builder=PriorBuilder(lmap, meta["s"]), conditions=conditions,
              det=data["detectors"][:, lmap.rows])
    if post_path:
        with np.load(post_path) as f:
            order = np.argsort(f["shot"], kind="stable")
            shot = f["shot"][order]
            _G.update(post_atom=f["atom"][order], post_ll=f["loglik"][order], log_prior=f["log_prior"],
                      post_lo=np.searchsorted(shot, np.arange(len(data["detectors"]))),
                      post_hi=np.searchsorted(shot, np.arange(len(data["detectors"])), side="right"))


def _priors(name, k):
    """(priors, relay-BP changes) of condition `name` for shot k."""
    g = _G
    data, builder = g["data"], g["builder"]
    heralds = data["heralds"][k]
    base, _, cap = name.partition("@")
    if base in CONDITIONS:
        spec = parse_condition(name)
        uses_truth = "oracle" in str(spec.get("data")) or spec.get("hook") == "oracle"
        truth = (data["data_slot"][k], data["check_k"][k]) if uses_truth else (None, None)
        return builder.priors(spec, heralds, *truth), spec["relay"]
    cap = float(cap) if cap else 0.5
    if base == "plumb-oracle-round":                         # ORACLE: reads the true loss slots
        return priors_from_slots(builder, heralds, true_round_dists(builder, data["data_slot"][k]), cap), {}
    variant = parse_variant(base)                            # the tracker: reads the file track.py wrote
    lo, hi = g["post_lo"][k], g["post_hi"][k]
    atoms = g["post_atom"][lo:hi]
    flagged = np.nonzero(heralds[builder.data_rec])[0]
    assert (np.sort(atoms) == flagged).all(), f"shot {k}: the tracker's atoms are not the flagged atoms"
    dists = slot_dists(builder, variant, atoms, g["post_ll"][lo:hi], g["log_prior"])
    return priors_from_slots(builder, heralds, dists, cap), {}


def _decode_shot(k):
    g = _G
    lmap, data = g["lmap"], g["data"]
    rows = []
    for name in g["conditions"]:
        t0 = time.perf_counter()
        pri, relay = _priors(name, k)
        t1 = time.perf_counter()
        dec = relay_decoder(lmap.H, pri, seed=1, **relay)
        t2 = time.perf_counter()
        r = dec.decode_detailed(g["det"][k])
        t3 = time.perf_counter()
        e = np.asarray(r.decoding, np.uint8)
        wrong = ((lmap.L @ e) % 2).astype(np.uint8) != data["observables"][k]
        rows.append((bool(wrong.any()), bool(r.success), int(r.iterations),
                     int(np.packbits(wrong, bitorder="little").view(np.uint16)[0]),
                     1e3 * (t1 - t0), 1e3 * (t2 - t1), 1e3 * (t3 - t2)))
    return k, rows


def decode(data_path, post_path, conditions, start, stop, workers, out, log=print):
    log(f"decode {data_path} shots {start}..{stop} conditions {conditions} workers {workers} post {post_path}")
    t0 = time.perf_counter()
    shots = range(start, stop)
    results = {}
    with mp.get_context("fork").Pool(workers, initializer=_init, initargs=(data_path, post_path, conditions)) as pool:
        for n, (k, rows) in enumerate(pool.imap_unordered(_decode_shot, shots, chunksize=2), 1):
            results[k] = rows
            if n % 100 == 0 or n == len(shots):
                failed = np.array([[r[0] for r in results[i]] for i in results])
                log(f"  {n}/{len(shots)} shots, {time.perf_counter() - t0:.0f}s, failures so far "
                    + " ".join(f"{c}={int(failed[:, i].sum())}" for i, c in enumerate(conditions)))
    order = sorted(results)
    assert order == list(shots)
    arr = np.array([results[k] for k in order], dtype=np.float64)       # (shots, conditions, 7)
    np.savez_compressed(out, shots=np.array(order), conditions=np.array(conditions),
                        failed=arr[:, :, 0].astype(bool), converged=arr[:, :, 1].astype(bool),
                        iterations=arr[:, :, 2].astype(np.int32), wrong_mask=arr[:, :, 3].astype(np.uint16),
                        prior_ms=arr[:, :, 4], build_ms=arr[:, :, 5], decode_ms=arr[:, :, 6])
    log(f"wrote {out} in {time.perf_counter() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--post", help="the tracker's output for this dataset (track.py); needed by `tracker...`")
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    conditions = a.conditions.split(",")
    assert a.post or not any(c.startswith("tracker") for c in conditions), "--post is needed"
    decode(a.data, a.post, conditions, a.start, a.stop, a.workers, a.out, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
