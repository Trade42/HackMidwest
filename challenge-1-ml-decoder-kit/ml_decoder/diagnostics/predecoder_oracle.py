"""How much could a predecoder help relay-BP on the Gross code? (X memory, upper bounds with an oracle)

usage: python predecoder_oracle.py KIT_DIR P N_SHOTS CAP_SETS
Shots are sampled from the kit's own error model (each column of H fires independently with its prior), so the
true error is known. An "oracle predecoder" removes some of the true faults, the residual syndrome goes to
relay-BP's first leg (80 iterations), and shots that still fail go to relay-BP capped at CAP_SETS legs, stopping
at the first solution.
  random 25% / 50%   removes a random share of the faults (optimistic: no real predecoder is this even-handed)
  isolated           removes only faults whose detectors are touched by no other fault (what a perfect LOCAL pattern-matcher could find)
  isolated, half     finds half of those
  isolated + 1 wrong as "isolated", plus one wrong column per shot (cost of a predecoder mistake)
"""
import sys, time
import numpy as np
kit, p, n, cap_sets = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
sys.path.insert(0, kit)
from circuit import GROSS_CIRCUITS, dem_matrices, load_circuit
from decoders import relay_decoder

c = load_circuit(GROSS_CIRCUITS["X"], p)
H, L, pri = dem_matrices(c, "X")
Hc, Hr = H.tocsc(), H.tocsr().astype(np.int32)
rng = np.random.default_rng(5)
E = (rng.random((n, H.shape[1])) < pri).astype(np.uint8)
counts = np.asarray((Hr @ E.T.astype(np.int32)).T)
OBS = (np.asarray((L.astype(np.int32) @ E.T.astype(np.int32)).T) % 2).astype(np.uint8)
leg = relay_decoder(H, pri, num_sets=0, seed=1)
capped = relay_decoder(H, pri, num_sets=cap_sets, stop_nconv=1, seed=1)

def picks(k):
    fired = np.nonzero(E[k])[0]
    iso = np.array([j for j in fired if (counts[k, Hc.indices[Hc.indptr[j]:Hc.indptr[j + 1]]] == 1).all()], dtype=int)
    wrong = rng.choice(np.nonzero(E[k] == 0)[0], 1)
    return fired, {"none": np.array([], dtype=int), "random 25%": fired[rng.random(len(fired)) < 0.25],
                   "random 50%": fired[rng.random(len(fired)) < 0.5], "isolated": iso,
                   "isolated, half": iso[rng.random(len(iso)) < 0.5], "isolated + 1 wrong": np.concatenate([iso, wrong])}

rows = {}
t0 = time.perf_counter()
for k in range(n):
    fired, variants = picks(k)
    for name, pre in variants.items():
        flip = np.zeros(H.shape[1], np.uint8); flip[pre] = 1
        res = E[k] ^ flip                                    # what is left for the decoder to find
        s = (np.asarray(Hr @ res.astype(np.int32)) % 2).astype(np.uint8)
        r = leg.decode_detailed(s)
        easy, iters = bool(r.success), r.iterations
        if not easy:
            r = capped.decode_detailed(s); iters = r.iterations
        total = np.asarray(r.decoding, np.uint8) ^ flip
        fail = (not r.success) or bool((((L @ total) % 2) != OBS[k]).any())
        true_pos = int(E[k][pre].sum())
        rows.setdefault(name, []).append((easy, iters, fail, true_pos / max(1, len(fired)), int(s.sum()), bool(r.success)))
print(f"p={p} shots={n} capped relay: {cap_sets} legs, {time.perf_counter() - t0:.0f}s", flush=True)
for name, v in rows.items():
    v = np.array(v, float)
    print(f"  {name:20s} faults removed {v[:,3].mean():5.1%} | syndrome weight {v[:,4].mean():5.0f} | first leg fails {1 - v[:,0].mean():6.1%} "
          f"| mean iterations {v[:,1].mean():6.0f} | failures {int(v[:,2].sum()):3d}/{n} (unconverged {int((v[:,5] == 0).sum())})", flush=True)
