"""Checks that G1's per-shot outputs do not depend on the process count, and that the check can see a difference.

  python -m analysis.check_processes --data VALIDATION_X --cache DIR --chunks 2 3 4 5 --out analysis/results/check_processes.json

Three jobs run side by side, each in one process:
1. one process decodes the given chunks of G1's kit run again, one after the other, each on a
   fresh decoder. Every per-shot output except the time must equal G1's 8-process outputs;
2. the kit's plain decode() on chunk 0, on a fresh decoder, in order. The corrections must equal
   decode_detailed's, so the detailed path is the kit's path;
3. a control: one decoder for chunks 0 and 1 in a row, without the fresh start at shot 500.
   Chunk 0 must match; chunk 1 should differ, because relay-BP carries its random state over.
"""
import argparse
from multiprocessing import Pool

import numpy as np

from analysis.common import (CHUNK, chunk_tasks, collect, error_hash, machine, make_decoder, open_dataset,
                             problem, sha256, write_json)
from analysis.g1_relay_failures import KIT

FIELDS = ("failed", "converged", "iterations", "pred", "wcount", "wllr", "valid", "ehash")


def g1_tasks(path, n_shots):
    return chunk_tasks("g1_kit", KIT, path, sha256(path), n_shots)


def redo_chunks(args):
    path, chunks = args
    from analysis.common import decode_shots
    ds = open_dataset(path)
    prob = problem(ds.basis, ds.p)
    outs = []
    for c in chunks:
        idx = np.arange(c * CHUNK, (c + 1) * CHUNK)
        outs.append(decode_shots(KIT, prob, ds.det[idx], ds.obs[idx]))
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def plain_decode(args):
    path, = args
    ds = open_dataset(path)
    prob = problem(ds.basis, ds.p)
    dec = make_decoder("relay", prob, {}, 0)
    return np.array([error_hash(np.asarray(dec.decode(ds.det[k]), np.uint8)) for k in range(CHUNK)], np.uint64)


def continuous(args):
    path, = args
    ds = open_dataset(path)
    prob = problem(ds.basis, ds.p)
    dec = make_decoder("relay", prob, {}, 0)
    hashes, its = [], []
    for k in range(2 * CHUNK):
        r = dec.decode_detailed(ds.det[k])
        hashes.append(error_hash(np.asarray(r.decoding, np.uint8)))
        its.append(r.iterations)
    return np.array(hashes, np.uint64), np.array(its)


def _call(job):
    fn, args = job
    return fn(args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="the dataset G1 decoded (validation_X.npz at p = 0.004)")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--chunks", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    path = args.data
    with np.load(path) as z:
        n = int(z["detectors"].shape[0])
    tasks = g1_tasks(path, n)
    g1 = collect(tasks, args.cache)      # the outputs G1 wrote, from its multi-process run
    jobs = [(redo_chunks, (path, args.chunks)), (plain_decode, (path,)), (continuous, (path,))]
    with Pool(3) as pool:
        redo, plain_hashes, (cont_hashes, cont_its) = pool.map(_call, jobs)

    sel = np.concatenate([np.arange(c * CHUNK, (c + 1) * CHUNK) for c in args.chunks])
    per_field = {k: int((redo[k] != g1[k][sel]).sum()) for k in FIELDS}
    one_vs_many = dict(chunks=args.chunks, shots=len(sel), differing_shots_per_field=per_field,
                       identical=all(v == 0 for v in per_field.values()),
                       g1_failures_in_these_chunks=int(g1["failed"][sel].sum()),
                       redo_failures=int(redo["failed"].sum()))
    api = dict(shots=CHUNK, differing_corrections=int((plain_hashes != g1["ehash"][:CHUNK]).sum()))
    api["identical"] = api["differing_corrections"] == 0
    order = dict(chunk0_differing=int((cont_hashes[:CHUNK] != g1["ehash"][:CHUNK]).sum()),
                 chunk1_differing_corrections=int((cont_hashes[CHUNK:] != g1["ehash"][CHUNK:2 * CHUNK]).sum()),
                 chunk1_differing_iterations=int((cont_its[CHUNK:] != g1["iterations"][CHUNK:2 * CHUNK]).sum()))
    result = dict(dataset=dict(path=path, sha256=sha256(path)), machine=machine(1),
                  one_process_vs_g1=one_vs_many, decode_vs_decode_detailed=api, order_control=order)
    write_json(args.out, result)
    print(result)


if __name__ == "__main__":
    main()
