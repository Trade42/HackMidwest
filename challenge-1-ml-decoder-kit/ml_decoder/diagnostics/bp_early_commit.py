"""Non-ML predecoder check 2: run K iterations of BP on every shot, commit only the columns it is confident
about, decode the residual with relay-BP. usage: python bp_early_commit.py KIT_DIR P N_SHOTS CAP_SETS K"""
import sys
import numpy as np
kit, p, n, cap_sets, K = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
sys.path.insert(0, kit)
import relay_bp
from circuit import GROSS_CIRCUITS, dem_matrices, load_circuit
from decoders import relay_decoder
c = load_circuit(GROSS_CIRCUITS["X"], p)
H, L, pri = dem_matrices(c, "X")
Hr = H.tocsr().astype(np.int32)
rng = np.random.default_rng(5)
E = (rng.random((n, H.shape[1])) < pri).astype(np.uint8)
S = (np.asarray((Hr @ E.T.astype(np.int32)).T) % 2).astype(np.uint8)
OBS = (np.asarray((L.astype(np.int32) @ E.T.astype(np.int32)).T) % 2).astype(np.uint8)
short = relay_bp.MinSumBPDecoderF64(H.tocsr(), error_priors=pri, max_iter=K, alpha=1.0, gamma0=0.15)
leg = relay_decoder(H, pri, num_sets=0, seed=1)
capped = relay_decoder(H, pri, num_sets=cap_sets, stop_nconv=1, seed=1)
THR = (6.0, 12.0, 24.0, 48.0)
stats = {t: [] for t in ("none",) + THR}
sign = None
for k in range(n):
    r = leg.decode_detailed(S[k]); it = r.iterations; easy0 = bool(r.success)
    if not easy0:
        r = capped.decode_detailed(S[k]); it += r.iterations
    e = np.asarray(r.decoding, np.uint8)
    stats["none"].append((easy0, it, (not r.success) or bool((((L @ e) % 2) != OBS[k]).any()), np.nan, np.nan, 0))
    rs = short.decode_detailed(S[k])
    post = np.asarray(rs.posterior_ratios, float); fired = E[k] == 1
    if sign is None:
        sign = 1.0 if post[fired].mean() < post[~fired].mean() else -1.0
        el = -sign * post
        print(f"after {K} iterations: fired columns median {np.median(el[fired]):.1f}, others median {np.median(el[~fired]):.1f}, range {el.min():.1f} to {el.max():.1f}", flush=True)
    err_llr = -sign * post
    for t in THR:
        commit = err_llr >= t
        flip = commit.astype(np.uint8)
        s2 = (np.asarray(Hr @ (E[k] ^ flip).astype(np.int32)) % 2).astype(np.uint8)
        r1 = leg.decode_detailed(s2); it = K + r1.iterations; easy = bool(r1.success)
        if not easy:
            r1 = capped.decode_detailed(s2); it += r1.iterations
        tot = np.asarray(r1.decoding, np.uint8) ^ flip
        fail = (not r1.success) or bool((((L @ tot) % 2) != OBS[k]).any())
        stats[t].append((easy, it, fail, E[k][commit].sum() / max(1, fired.sum()), int(commit.sum() - E[k][commit].sum()), int(commit.sum())))
print(f"p={p} shots={n}, {K} BP iterations then commit, capped relay {cap_sets} legs", flush=True)
for t, v in stats.items():
    v = np.array(v, float)
    prec = 1 - np.nansum(v[:, 4]) / max(1, np.nansum(v[:, 5])) if t != "none" else float("nan")
    print(f"  threshold {str(t):5s} | faults committed {np.nanmean(v[:, 3]):5.1%} | precision {prec:.4f} | wrong commits per shot {np.nanmean(v[:, 4]):5.2f} "
          f"| first leg fails {1 - v[:, 0].mean():6.1%} | mean iterations {v[:, 1].mean():6.0f} | failures {int(v[:, 2].sum())}/{n}", flush=True)
