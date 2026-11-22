"""Loss-blind vs flag-aware priors for relay-BP on Clifft loss shots (Gross code, X memory).
usage: python -I starter_scripts/loss_starter.py KIT P P_LOSS SHOTS NUM_SETS     (from challenge-1-ml-decoder-kit/)
Needs Clifft (ml_decoder/requirements.txt). Claims no number: it is a starting point for a loss-aware decoder.
Blind = relay-BP with the kit's static priors. Flag-aware = per-shot priors, built from the shot's heralds:
  heralded X-check atom -> its measurement-flip column at 0.5;
  heralded Z-check atom -> the partial-check columns of that cycle (one per CX layer it may have been lost after),
                           each raised by 0.5 / 6;
  data atom heralded lost at the final readout -> its one flicker column per cycle, with a uniform prior on the loss
                           time, and its final-readout column at 0.5.
"""
import sys, re, time
import numpy as np
kit, p, p_loss, shots, num_sets = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
sys.path.insert(0, kit)
import stim
from clifft import noncomp
from circuit import GROSS_CIRCUITS, dem_matrices, load_circuit, detector_ids
from decoders import relay_decoder

circ = load_circuit(GROSS_CIRCUITS["X"], p)
H, L, pri = dem_matrices(circ, "X")
ids = detector_ids(circ, "X")
row_of = {int(d): r for r, d in enumerate(ids)}
Hc = H.tocsc()
col_of = {tuple(sorted(Hc.indices[Hc.indptr[j]:Hc.indptr[j + 1]].tolist())): j for j in range(Hc.shape[1])}
assert len(col_of) == Hc.shape[1], "two columns of H share a detector pattern: key the map on observables too"
N2, CYC = 72, 12

# ---- which column does a lost data atom excite in each cycle? probe a noiseless circuit with tagged Z errors
flat = circ.without_noise().flattened()
def probe(tick_index, qubits):
    out, t = stim.Circuit(), -1
    for inst in flat:
        out.append(inst)
        if inst.name == "TICK":
            t += 1
            if t == tick_index:
                for k, q in enumerate(qubits):
                    out.append("Z_ERROR", [q], 0.001 * (k + 1))
    cols = {}
    errors = [e for e in out.detector_error_model().flattened() if e.type == "error"]
    assert len(errors) == len(qubits), "two probes merged: tag them in separate calls"
    for e in errors:
        q = qubits[int(round(e.args_copy()[0] / 0.001)) - 1]
        rows = tuple(sorted(row_of[x.val] for x in e.targets_copy() if x.is_relative_detector_id() and x.val in row_of))
        cols[q] = col_of[rows]
    return cols
t0 = time.perf_counter()
left, right = list(range(N2, 2 * N2)), list(range(2 * N2, 3 * N2))
flicker = {q: [] for q in left + right}            # flicker[q][c] = column for cycle c
for c in range(CYC):
    for q, col in probe(1 + 8 * c + 4, left).items(): flicker[q].append(col)     # left data: X-type gates resume at layer 5
    for q, col in probe(1 + 8 * c + 1, right).items(): flicker[q].append(col)    # right data: X-type gates resume at layer 2
final = probe(1 + 8 * (CYC - 1) + 6, left + right)                               # random placeholder at the final readout
zanc = list(range(3 * N2, 4 * N2))
hooks = {(j, c): [] for j in zanc for c in range(CYC)}   # a Z-check atom lost after its CX in layer l leaves a partial check: one column per l
for c in range(CYC):
    for l in range(5):
        for j, col in probe(1 + 8 * c + l, zanc).items(): hooks[(j, c)].append(col)
assert all(len(v) == CYC for v in flicker.values()) and len(final) == 2 * N2
print(f"loss map built in {time.perf_counter() - t0:.1f}s", flush=True)

def inline_loss(text, q):
    out = []
    for line in text.splitlines():
        out.append(line)
        m = re.match(r"^(\s*)CX (.*)$", line)
        if m: out.append(f"{m.group(1)}LOSS({q}) {m.group(2)}")
    return "\n".join(out)
C = np.zeros((3, 5)); C[0, 0] = C[0, 2] = 1.0; C[1, 1] = C[1, 3] = 1.0; C[2, 4] = 1.0
model = noncomp.Model(classifier=noncomp.Classifier(C.tolist()), reset_restores_lost=True)
s = noncomp.compile(inline_loss(str(circ), p_loss), model, execution="online").sample(shots, seed=7)
det, obs, h = np.asarray(s.detectors, np.uint8)[:, ids], np.asarray(s.observables, np.uint8), np.asarray(s.heralds, np.uint8)

def bump(pr, col, v):
    pr[col] = pr[col] * (1 - v) + v * (1 - pr[col])
def aware_priors(hk):
    pr = pri.copy()
    for slot in np.nonzero(hk[: CYC * 2 * N2])[0]:
        c, i = divmod(int(slot), 2 * N2)
        if i >= N2:                                   # X-check atom: its outcome was a coin flip
            i -= N2
            bump(pr, col_of[(N2 * c + i, N2 * (c + 1) + i)], 0.5)
        elif USE_HOOKS:                               # Z-check atom: a partial check acted on the data
            for col in hooks[(3 * N2 + i, c)]:
                bump(pr, col, 0.5 / 6)
    for k in np.nonzero(hk[CYC * 2 * N2:])[0]:        # data atom found missing at the final readout
        q = N2 + int(k)
        for c in range(CYC):
            bump(pr, flicker[q][c], 0.5 * (c + 0.5) / CYC)   # TRACKER: replace (c + 0.5) / CYC by a learned P(lost by cycle c)
        bump(pr, final[q], 0.5)
    return np.minimum(pr, 0.5)

res = {}
for name in ("blind", "flag-aware"):
    USE_HOOKS = name == "flag-aware"
    conv = fail = it = 0
    static = relay_decoder(H, pri, num_sets=num_sets, stop_nconv=1, seed=1)
    for k in range(shots):
        d = static if name == "blind" or not h[k].any() else relay_decoder(H, aware_priors(h[k]), num_sets=num_sets, stop_nconv=1, seed=1)
        r = d.decode_detailed(det[k])
        e = np.asarray(r.decoding, np.uint8)
        conv += r.success; it += r.iterations
        fail += bool((((L @ e) % 2) != obs[k]).any())
    res[name] = (conv, fail, it)
    print(f"p={p} p_loss={p_loss:g} {name:10s} shots {shots} | data atoms lost {h[:, 1728:].sum(axis=1).mean():.2f}/shot, check heralds {h[:, :1728].sum(axis=1).mean():.2f}/shot "
          f"| X-check detector weight {det.sum(axis=1).mean():.0f} | converged {conv/shots:.3f} failed {fail}/{shots} mean iters {it/shots:.0f}", flush=True)
