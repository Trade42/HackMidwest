"""Sizing run for a hierarchical decoder on the Gross code (X memory only).

usage: python sizing_run.py KIT_DIR P N_SHOTS BUDGET_BP BUDGET_LEG BUDGET_R1 BUDGET_R5 OUT.npz
Every decoder walks the same shots in order until its time budget runs out.
"""
import sys, time
import numpy as np

kit, p, n, b_bp, b_leg, b_r1, b_r5, out = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), *map(float, sys.argv[4:8]), sys.argv[8]
sys.path.insert(0, kit)
from circuit import GROSS_CIRCUITS, dem_matrices, load_circuit, detector_ids, sample
from decoders import relay_decoder
from ldpc import BpDecoder

c = load_circuit(GROSS_CIRCUITS["X"], p)
H, L, pri = dem_matrices(c, "X")
det, obs = sample(c, n, 11)
det = det[:, detector_ids(c, "X")]
llr = np.log((1 - pri) / pri)


def run(decode_detailed, budget):
    rows, start = [], time.perf_counter()
    for k in range(n):
        if time.perf_counter() - start > budget:
            break
        t = time.perf_counter()
        ok, iters, err, post = decode_detailed(det[k])
        dt = time.perf_counter() - t
        err = np.asarray(err, np.uint8)
        fail = bool((((L @ err) % 2) != obs[k]).any())
        a = np.abs(post) if post is not None else np.zeros(1)
        rows.append((ok, iters, fail, dt, float(llr @ err), int(err.sum()), int(det[k].sum()),
                     float(a.mean()), float(np.percentile(a, 5)), float((a < 1).mean())))
    return np.array(rows, dtype=float)


def relay(**kw):
    d = relay_decoder(H, pri, **kw)
    def f(s):
        r = d.decode_detailed(s)
        return r.success, r.iterations, r.decoding, np.asarray(r.posterior_ratios)
    return f


bp = BpDecoder(H, error_channel=list(pri), max_iter=80, bp_method="minimum_sum", ms_scaling_factor=1.0)
def plain_bp(s):
    e = bp.decode(s)
    return bp.converge, bp.iter, e, np.asarray(bp.log_prob_ratios)

res = {}
for name, fn, budget in (("bp80", plain_bp, b_bp), ("leg1", relay(num_sets=0), b_leg),
                         ("relay1", relay(stop_nconv=1), b_r1), ("relay5", relay(), b_r5)):
    res[name] = run(fn, budget)
    r = res[name]
    print(f"p={p} {name:7s} shots {len(r):5d} converged {r[:,0].mean():.4f} fail {int(r[:,2].sum()):4d} "
          f"({r[:,2].mean():.4f}) fail|conv {r[r[:,0]==1][:,2].mean():.4f} iters mean {r[:,1].mean():7.1f} "
          f"med {np.median(r[:,1]):6.0f} p99 {np.percentile(r[:,1],99):7.0f} ms mean {1e3*r[:,3].mean():7.2f}", flush=True)
np.savez_compressed(out, p=p, **res)
