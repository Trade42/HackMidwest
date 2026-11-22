"""Second pass: is the trajectory model calibrated, and what does it buy at a threshold picked on validation shots?
usage: python predecoder_real2.py KIT_DIR P N_TRAIN N_VAL N_TEST CAP_SETS
"""
import sys
import time
import numpy as np

kit, p = sys.argv[1], float(sys.argv[2])
n_train, n_val, n_test, cap_sets = (int(v) for v in sys.argv[3:7])
sys.path.insert(0, kit)
import relay_bp
from circuit import GROSS_CIRCUITS, dem_matrices, load_circuit
from decoders import relay_decoder
from sklearn.ensemble import HistGradientBoostingClassifier

circ = load_circuit(GROSS_CIRCUITS["X"], p)
H, L, pri = dem_matrices(circ, "X")
pri = np.asarray(pri, float)
C, K = H.shape[1], 8
Hcsr, Hr, Li = H.tocsr(), H.tocsr().astype(np.int32), L.astype(np.int32)
HT = H.T.tocsr().astype(np.int32)
colw = np.asarray(H.sum(0)).ravel().astype(np.float32)
prior_logit = np.log(pri / (1 - pri)).astype(np.float32)
bps = [relay_bp.MinSumBPDecoderF64(Hcsr, error_priors=pri, max_iter=k, alpha=1.0, gamma0=0.15) for k in range(1, K + 1)]


def sample(n, seed):
    rng = np.random.default_rng(seed)
    E = (rng.random((n, C)) < pri).astype(np.uint8)
    S = (np.asarray((Hr @ E.T.astype(np.int32)).T) % 2).astype(np.uint8)
    OBS = (np.asarray((Li @ E.T.astype(np.int32)).T) % 2).astype(np.uint8)
    return E, S, OBS


def feats(s):
    T = np.empty((C, K), np.float32)
    for k, bp in enumerate(bps):
        T[:, k] = -np.asarray(bp.decode_detailed(s).posterior_ratios, np.float32)   # error LLR: large = fault likely
    T = np.clip(np.nan_to_num(T, nan=0.0, posinf=30, neginf=-30), -30, 30)
    lit = np.asarray(HT @ s.astype(np.int32)).ravel() / colw
    return np.column_stack([T, T.min(1), T.max(1), T[:, -1] - T[:, 0], prior_logit, colw, lit]), T[:, -1]


t0 = time.time()
Etr, Str, _ = sample(n_train, 101)
X = np.concatenate([feats(Str[k])[0] for k in range(n_train)])
y = Etr.reshape(-1)
assert X[y == 1, K - 1].mean() > X[y == 0, K - 1].mean(), "LLR sign is wrong"
# repeat-call check: the same syndrome must give the same features, or the BP objects carry state between shots
again = feats(Str[0])[0]
print(f"p={p}: rows {len(y):,}, faults {int(y.sum()):,}. Same syndrome twice gives the same features: {bool(np.allclose(again, X[:C]))}", flush=True)
# Faults are 0.5% of the rows, which makes boosting unstable. Train on all faults plus 5% of the rest, then shift the
# scores back by log(0.05) so the probabilities refer to the full population again.
KEEP = 0.05
rng0 = np.random.default_rng(7)
use = (y == 1) | (rng0.random(len(y)) < KEEP)
raw = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100,
                                     l2_regularization=1.0, random_state=0).fit(X[use], y[use])
print(f"model fitted on {int(use.sum()):,} rows, {raw.n_iter_} rounds ({time.time() - t0:.0f} s)", flush=True)


class model:                                            # same interface, with the prior shift applied
    @staticmethod
    def predict_proba(F):
        z = raw.decision_function(F) + np.log(KEEP)
        q = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - q, q])


ptr = model.predict_proba(X)[:, 1]
del X

Ev, Sv, _ = sample(n_val, 303)
pv, lv = [], []
for k in range(n_val):
    F, llr = feats(Sv[k]); pv.append(model.predict_proba(F)[:, 1]); lv.append(llr)
pv, lv, yv = np.concatenate(pv), np.concatenate(lv), Ev.reshape(-1)
print("reliability (share of true faults among columns the model scores in each band)")
for lo, hi in ((0.1, 0.5), (0.5, 0.9), (0.9, 0.99), (0.99, 0.999), (0.999, 1.01)):
    a, b = (ptr >= lo) & (ptr < hi), (pv >= lo) & (pv < hi)
    print(f"  P in [{lo}, {min(hi, 1)}): train {y[a].mean():.3f} of {a.sum():,} | validation {yv[b].mean():.3f} of {b.sum():,}")
unsure_v = (yv == 1) & (lv < 12)
order = np.argsort(-pv)
hits = np.cumsum(yv[order]); prec = hits / np.arange(1, len(order) + 1)
taus = {}
for target in (0.999, 0.995, 0.99):
    ok = np.nonzero(prec >= target)[0]
    cut = ok.max() if len(ok) else 0
    taus[target] = pv[order][cut]
    sel = pv >= taus[target]
    print(f"  validation: precision {target:.1%} is reached down to P = {taus[target]:.4f}: {sel.sum() / n_val:.1f} commits per shot, "
          f"{(sel & unsure_v).sum() / max(1, unsure_v.sum()):.1%} of BP-missed faults found, {(sel & (yv == 1)).sum() / yv.sum():.1%} of all faults")
print(f"  for reference, BP picks (LLR >= 12): precision {yv[lv >= 12].mean():.1%}, {(lv >= 12).sum() / n_val:.1f} commits per shot", flush=True)

E, S, OBS = sample(n_test, 202)
leg = relay_decoder(H, pri, num_sets=0, seed=1)
capped = relay_decoder(H, pri, num_sets=cap_sets, stop_nconv=1, seed=1)
rows = {}
for k in range(n_test):
    F, llr = feats(S[k])
    prob = model.predict_proba(F)[:, 1]
    unsure = (E[k] == 1) & (llr < 12)
    variants = {"no predecoder": (np.zeros(C, bool), 0), "bp picks, LLR >= 12": (llr >= 12, K)}
    for target, tau in taus.items():
        variants[f"model, {target:.1%} precise"] = (prob >= tau, K)
        variants[f"model, {target:.1%}, BP-missed only"] = ((prob >= tau) & (llr < 12), K)
    for name, (commit, extra) in variants.items():
        commit = commit.astype(np.uint8)
        s2 = S[k] ^ (np.asarray(Hr @ commit.astype(np.int32)) % 2).astype(np.uint8)
        r = leg.decode_detailed(s2); easy, it = bool(r.success), r.iterations
        if not easy:
            r = capped.decode_detailed(s2); it += r.iterations
        total = np.asarray(r.decoding, np.uint8) ^ commit
        fail = (not r.success) or bool((((L @ total) % 2) != OBS[k]).any())
        rows.setdefault(name, []).append((easy, it + extra, fail, commit.sum(), (commit & E[k]).sum(), (commit.astype(bool) & unsure).sum(), unsure.sum()))
print(f"{n_test} test shots, relay-BP capped at {cap_sets} legs ({time.time() - t0:.0f} s)")
for name, v in rows.items():
    v = np.array(v, float)
    made, right, got, uns = v[:, 3].sum(), v[:, 4].sum(), v[:, 5].sum(), v[:, 6].sum()
    print(f"  {name:32s} commits/shot {made / n_test:5.1f} | precision {right / max(1, made):6.1%} | BP-missed found {got / max(1, uns):6.1%}"
          f" | hard shots {1 - v[:, 0].mean():6.1%} | mean iterations {v[:, 1].mean():5.0f} | failures {int(v[:, 2].sum())}/{n_test}", flush=True)
