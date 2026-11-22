"""Which faults does a predecoder have to remove to help? (X memory, oracle study)
usage: python predecoder_which.py KIT_DIR P N_SHOTS CAP_SETS
BP confidence comes from 8 iterations of BP with memory. All variants below are 100% precise (true faults only):
  bp-sure        every true fault BP is already sure about            (BP's picks with its mistakes removed)
  iso, bp-sure   isolated faults BP is already sure about
  iso, bp-unsure isolated faults BP is NOT sure about
  isolated       all isolated faults
"""
import sys
import numpy as np
kit, p, n, cap_sets = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
sys.path.insert(0, kit)
import relay_bp
from circuit import GROSS_CIRCUITS, dem_matrices, load_circuit
from decoders import relay_decoder
c = load_circuit(GROSS_CIRCUITS["X"], p)
H, L, pri = dem_matrices(c, "X")
Hc, Hr = H.tocsc(), H.tocsr().astype(np.int32)
rng = np.random.default_rng(5)
E = (rng.random((n, H.shape[1])) < pri).astype(np.uint8)
counts = np.asarray((Hr @ E.T.astype(np.int32)).T)
S = (counts % 2).astype(np.uint8)
OBS = (np.asarray((L.astype(np.int32) @ E.T.astype(np.int32)).T) % 2).astype(np.uint8)
short = relay_bp.MinSumBPDecoderF64(H.tocsr(), error_priors=pri, max_iter=8, alpha=1.0, gamma0=0.15)
leg = relay_decoder(H, pri, num_sets=0, seed=1)
capped = relay_decoder(H, pri, num_sets=cap_sets, stop_nconv=1, seed=1)
T, sign, rows = 12.0, None, {}
for k in range(n):
    fired = np.nonzero(E[k])[0]
    post = np.asarray(short.decode_detailed(S[k]).posterior_ratios, float)
    if sign is None:
        sign = 1.0 if post[fired].mean() < post[E[k] == 0].mean() else -1.0
    sure = (-sign * post)[fired] >= T
    iso = np.array([(counts[k, Hc.indices[Hc.indptr[j]:Hc.indptr[j + 1]]] == 1).all() for j in fired], bool)
    variants = {"none": fired[:0], "bp-sure": fired[sure], "iso, bp-sure": fired[iso & sure],
                "iso, bp-unsure": fired[iso & ~sure], "isolated": fired[iso], "not isolated, bp-unsure": fired[~iso & ~sure]}
    for name, pre in variants.items():
        flip = np.zeros(H.shape[1], np.uint8); flip[pre] = 1
        s = (np.asarray(Hr @ (E[k] ^ flip).astype(np.int32)) % 2).astype(np.uint8)
        r = leg.decode_detailed(s); easy, it = bool(r.success), r.iterations
        if not easy:
            r = capped.decode_detailed(s); it += r.iterations
        tot = np.asarray(r.decoding, np.uint8) ^ flip
        fail = (not r.success) or bool((((L @ tot) % 2) != OBS[k]).any())
        rows.setdefault(name, []).append((easy, it, fail, len(pre) / max(1, len(fired))))
print(f"p={p} shots={n}", flush=True)
for name, v in rows.items():
    v = np.array(v, float)
    print(f"  {name:24s} faults removed {v[:,3].mean():5.1%} | first leg fails {1 - v[:,0].mean():6.1%} | mean iterations {v[:,1].mean():6.0f} | failures {int(v[:,2].sum())}/{n}", flush=True)
