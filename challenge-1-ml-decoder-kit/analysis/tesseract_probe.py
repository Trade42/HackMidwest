"""G2 item 4: does the Tesseract search decoder (pip install tesseract-decoder) take this problem,
and decode one of relay-BP's failed shots at p = 0.004 in under 60 s?

  python -m analysis.tesseract_probe --data VALIDATION_X --pershot PERSHOT_NPZ --shots 3 --timeout 60 --out analysis/results/tesseract_probe.json

The DEM is built from the kit's H, L and priors, one error per column, so Tesseract decodes the
same 936-detector problem as the kit's decoders. Its correction is projected with L and scored
as the kit does. Each shot runs in a child process that is stopped at the timeout.
"""
import argparse
import time
from multiprocessing import get_context

import numpy as np

from analysis.common import TesseractOnMatrices, open_dataset, problem, sha256, write_json

_DEC = {}


def _init(path, config):
    ds = open_dataset(path)
    prob = problem(ds.basis, ds.p)
    t = time.perf_counter()
    _DEC["d"] = TesseractOnMatrices(prob, **config)
    _DEC["build"] = time.perf_counter() - t
    _DEC["ds"], _DEC["prob"] = ds, prob


def _decode(k):
    d, ds, prob = _DEC["d"], _DEC["ds"], _DEC["prob"]
    t = time.perf_counter()
    e = d.decode(ds.det[k])
    dt = time.perf_counter() - t
    pred = (prob.L @ e) % 2
    valid = bool((((prob.H @ e) % 2) == ds.det[k]).all())
    return dict(index=int(k), seconds=dt, build_seconds=_DEC["build"], wrong=bool((pred != ds.obs[k]).any()),
                valid=valid, confident=bool(d.converge), weight_llr=float(e @ prob.llr))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--pershot", required=True, help="G1's per-shot file of the same dataset")
    ap.add_argument("--shots", type=int, default=3, help="this many failed and this many right shots")
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    config = dict(det_beam=5, pqlimit=200_000, no_revisit_dets=True)
    with np.load(args.pershot) as z:
        failed = z["failed"]
        g1_wllr = z["wllr"]
    picks = [("relay-BP failed", int(i)) for i in np.flatnonzero(failed)[: args.shots]] + \
            [("relay-BP right", int(i)) for i in np.flatnonzero(~failed)[: args.shots]]
    rows = []
    ctx = get_context("fork")
    _init(args.data, config)          # build once here first: a broken DEM fails loudly instead of in a pool
    print(f"decoder built in {_DEC['build']:.1f} s", flush=True)
    pool = None
    for role, k in picks:
        if pool is None:
            pool = ctx.Pool(1, initializer=_init, initargs=(args.data, config))
        try:
            r = pool.apply_async(_decode, (k,)).get(timeout=args.timeout)
            r.update(role=role, timed_out=False, relay_weight_llr=float(g1_wllr[k]))
        except Exception as exc:   # the timeout, or a crash inside the decoder
            pool.terminate()
            pool.join()
            pool = None
            r = dict(index=k, role=role, timed_out=True, error=type(exc).__name__)
        print(r, flush=True)
        rows.append(r)
    if pool is not None:
        pool.close()
        pool.join()
    write_json(args.out, dict(dataset=dict(path=args.data, sha256=sha256(args.data)), config=config,
                              timeout_seconds=args.timeout, shots=rows))


if __name__ == "__main__":
    main()
