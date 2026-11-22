"""How much can per-shot priors help relay-BP at the kit's settings? An upper bound with an oracle.

Run from challenge-1-ml-decoder-kit/:

  python -m predecoder.oracle --p 0.004 --shots 300 --seed 11 --workers 4

DEM-sampled X shots (the full error model, the network's training distribution), each mapped
onto the columns of the kit's basis problem (mapping.py, checked on every shot). Every shot is
decoded by relay-BP at the kit's settings, a decoder built per shot, under these priors:

  kit            the kit's priors (the control)
  true at q      the kit's priors, with the shot's truly fired columns raised to q
  true and as many false at q   the same, plus as many columns that did NOT fire, also at q:
                 a calibrated predecoder that puts q = 0.5 on a column is wrong half the time

The oracle knows the truth by design: this is a diagnostic of the lever, never a decoder result.
"""
import argparse
import json
import multiprocessing
import time

import numpy as np
import torch

from circuit import dem_matrices
from make_circuit import memory_experiment
from neural.sampler import DemSampler
from predecoder.decode import decode_with_priors, paired, settings, wilson
from predecoder.mapping import column_map

_JOB = {}


def _start(H, L):
    _JOB["H"], _JOB["L"] = H, L


def _decode(task):
    index, name, priors, syndrome, truth = task
    result = decode_with_priors(_JOB["H"], _JOB["L"], priors, syndrome)
    return index, name, bool((result.predicted != truth).any()), result.converged, result.iterations


def variants(kit_priors, fired, rng, levels):
    """{name: priors} for one shot."""
    out = {"kit": kit_priors}
    on = np.nonzero(fired)[0]
    off = np.nonzero(fired == 0)[0]
    for q in levels:
        raised = kit_priors.copy()
        raised[on] = q
        out[f"true at {q:g}"] = raised
    for q in levels:
        raised = kit_priors.copy()
        raised[on] = q
        raised[rng.choice(off, size=len(on), replace=False)] = q
        out[f"true and as many false at {q:g}"] = raised
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p", type=float, default=0.004)
    parser.add_argument("--shots", type=int, default=300)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--levels", type=float, nargs="+", default=[0.05, 0.2, 0.5])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", help="a JSON file for the numbers")
    args = parser.parse_args(argv)

    start = time.time()
    sampler = DemSampler("gross", "X", None, "all", [args.p])
    mapping = column_map("gross", "X", None, args.p, sampler)
    detectors, observables, (shot, mechanism) = sampler.sample(args.shots, args.p, torch.Generator().manual_seed(args.seed),
                                                               return_faults=True)
    fired = mapping.columns_fired(shot, mechanism, args.shots).numpy()
    detectors, observables = detectors.numpy(), observables.numpy()
    mapping.check(fired, detectors, observables)          # raises if the mapping is wrong on any shot
    print(f"oracle: X memory, p={args.p:g}, {args.shots} DEM shots (seed {args.seed}); the mapping reproduces the basis "
          f"detectors and the observables of every shot; {fired.sum(axis=1).mean():.1f} fired columns per shot "
          f"(the kit's priors sum to {mapping.merged_priors(sampler.priors[args.p].double().numpy()).sum():.1f}; "
          f"merged priors equal the kit's to {mapping.prior_error:.1e})", flush=True)

    _, _, exact = dem_matrices(memory_experiment("gross", "X", None, args.p), "X")
    kit_priors = np.asarray(exact, np.float64)             # the kit's own priors of the basis problem
    rng = np.random.default_rng(args.seed)
    syndromes = detectors[:, mapping.basis_rows]
    tasks, names = [], None
    for k in range(args.shots):
        shot_variants = variants(kit_priors, fired[k], rng, args.levels)
        names = list(shot_variants)
        tasks.extend((k, name, priors, syndromes[k], observables[k]) for name, priors in shot_variants.items())

    rows = {name: np.zeros((args.shots, 3)) for name in names}
    with multiprocessing.get_context("spawn").Pool(args.workers, _start, (mapping.basis_H.tocsr(), mapping.basis_L)) as pool:
        for k, name, failed, converged, iterations in pool.imap_unordered(_decode, tasks, chunksize=4):
            rows[name][k] = failed, converged, iterations
    print(f"relay-BP at the kit's settings, a decoder built per shot (seed 0), {args.workers} processes, "
          f"{time.time() - start:.0f} s", flush=True)
    print(f"{'priors':34s} {'failures':>9s} {'95% interval':>17s} {'unconverged':>11s} {'mean iterations':>15s}   paired against kit")
    summary = {"settings": settings(), "p": args.p, "shots": args.shots, "seed": args.seed, "rows": {}}
    for name in names:
        failed, converged, iterations = rows[name][:, 0].astype(bool), rows[name][:, 1].astype(bool), rows[name][:, 2]
        low, high = wilson(failed.sum(), args.shots)
        both, only_kit, only_this, p_value = paired(rows["kit"][:, 0], failed)
        print(f"{name:34s} {int(failed.sum()):9d} {f'[{low:.4f}, {high:.4f}]':>17s} {int((~converged).sum()):11d} "
              f"{iterations.mean():15.0f}   kit only {only_kit}, this only {only_this}, p={p_value:.3g}", flush=True)
        summary["rows"][name] = {"failures": int(failed.sum()), "unconverged": int((~converged).sum()),
                                 "mean_iterations": float(iterations.mean()), "kit_only": only_kit, "this_only": only_this,
                                 "p": p_value}
    if args.out:
        with open(args.out, "w") as file:
            json.dump(summary, file, indent=1)


if __name__ == "__main__":
    main()
