"""Can the neural network choose among the classical candidates? X memory, p = 0.004, step-8b files.

  python -I -B analyze_network_select.py <step>

Joins, by shot index in file order, results_8b/selector_<stem>.npz (R = relay-BP, P1 and P2 = PriorTilt with two
seed schemes) with the network's per-shot logits from runs/cascade_X_s1/eval/step<step>_<stem>_p0.004.npz, for
stem = validation_X and train_X (25,000 shots). The truth is used only to score.

Rules (a candidate's network score is the sum over the 12 bits of log sigmoid(logit) for a predicted 1 and
log sigmoid(-logit) for a predicted 0; its weight is the sum of log((1 - prior) / prior) over its faults):
  a. network picks         the candidate (of R, P1, P2) with the highest network score
  b. network breaks ties   P1 where P1 and P2 agree, otherwise rule a
  c. weight plus network   as b, with the score network - weight / C, for C in 10, 30, 100 (C is fitted to these shots)
  d. network alone
beside R, P1, P2 and the step-8b rules. Time per shot = the sum of the decoders a rule runs; the network's time is
its batch-one latency from evaluate_stim, measured beside the trainer.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(os.path.dirname(HERE), "cascade_X_s1", "eval")
STEMS = ("validation_X", "train_X")
NAMES = ("R", "P1", "P2")


def log_sigmoid(x):
    return -np.logaddexp(0.0, -x)


def load(step, stems=STEMS):
    parts, latency = [], []
    for stem in stems:
        s = np.load(os.path.join(HERE, "results_8b", f"selector_{stem}.npz"))
        e = np.load(os.path.join(EVAL, f"step{step}_{stem}_p0.004.npz"))
        meta = json.load(open(os.path.join(EVAL, f"step{step}_{stem}_p0.004.json")))
        if not np.array_equal(s["truth"], e["observables"]):
            raise SystemExit(f"{stem}: the network's observables and the selector file's truth differ")
        parts.append({**{k: s[k] for k in s.files if not k.startswith(("settings", "data_sha"))},
                      "logits": e["logits"].astype(np.float64), "confidence": e["confidence"]})
        latency.append((meta["latency_ms_mean"], meta["latency_shots"]))
    joined = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    net_ms = sum(m * n for m, n in latency) / sum(n for _, n in latency)
    return joined, net_ms


def main():
    step = int(sys.argv[1])
    stems = tuple(sys.argv[2:]) or STEMS
    d, net_ms = load(step, stems)
    truth, logits = d["truth"].astype(np.uint8), d["logits"]
    n = len(truth)
    pred = np.stack([d[f"predicted_{k}"] for k in NAMES])                       # [3, n, 12]
    weight = np.stack([d[f"weight_{k}"] for k in NAMES])
    conv = np.stack([d[f"converged_{k}"] for k in NAMES])
    sec = np.stack([d[f"seconds_{k}"] for k in NAMES]) * 1e3                   # ms
    right = (pred == truth[None]).all(axis=2)                                   # [3, n]
    rows = np.arange(n)
    net_score = np.where(pred == 1, log_sigmoid(logits)[None], log_sigmoid(-logits)[None]).sum(axis=2)   # [3, n]
    eq = lambda a, b: (pred[a] == pred[b]).all(axis=1)
    p1_p2, r_p1, r_p2 = eq(1, 2), eq(0, 1), eq(0, 2)
    differ = ~(p1_p2 & r_p1)
    lowest = weight.argmin(axis=0)
    lowest_conv = np.where(conv.any(axis=0), np.where(conv, weight, np.inf).argmin(axis=0), lowest)
    majority = np.where(p1_p2 | r_p1, 1, np.where(r_p2, 0, lowest))
    pick_a = net_score.argmax(axis=0)
    choices = {
        "R alone": (np.zeros(n, int), sec[0]),
        "P1 alone": (np.ones(n, int), sec[1]),
        "P2 alone": (np.full(n, 2), sec[2]),
        "8b lowest weight": (lowest, sec.sum(0)),
        "8b majority (ties: lowest weight)": (majority, sec.sum(0)),
        "8b P1 unless P1 != P2, then lowest weight": (np.where(p1_p2, 1, lowest), sec[1] + sec[2] + np.where(p1_p2, 0, sec[0])),
        "8b extra: lowest weight among converged": (lowest_conv, sec.sum(0)),
        "a. network picks": (pick_a, sec.sum(0) + net_ms),
        "b. network breaks ties": (np.where(p1_p2, 1, pick_a), sec[1] + sec[2] + np.where(p1_p2, 0, sec[0] + net_ms)),
    }
    for c in (10, 30, 100):
        pick_c = (net_score - weight / c).argmax(axis=0)
        choices[f"c. weight plus network, C = {c}"] = (np.where(p1_p2, 1, pick_c), sec[1] + sec[2] + np.where(p1_p2, 0, sec[0] + net_ms))
    net_wrong = ((logits > 0).astype(np.uint8) != truth).any(axis=1)
    some_right = right.any(axis=0)
    print(f"X memory, p = 0.004, {n} shots ({' + '.join(stems)}); network = checkpoint_step{step}.pt; "
          f"network batch-one latency {net_ms:.1f} ms (beside the trainer)")
    print(f"not-all-equal set: {int(differ.sum())} shots; with at least one right candidate: {int((differ & some_right).sum())}; "
          f"all three wrong: {int((~some_right).sum())}")
    print(f"{'rule':44s} {'failures':>8s} {'error':>7s}  {'right pick when one existed (not-all-equal set)':>48s}  {'ms/shot':>8s}")
    for name, (idx, ms) in choices.items():
        chosen_right = right[idx, rows]
        fails = int((~chosen_right).sum())
        mask = differ & some_right
        print(f"{name:44s} {fails:8d} {fails / n:7.3%}  {int(chosen_right[mask].sum()):>22d} of {int(mask.sum()):<23d}  {ms.mean():8.1f}")
    print(f"{'d. network alone':44s} {int(net_wrong.sum()):8d} {net_wrong.mean():7.3%}  {'':48s}  {net_ms:8.1f}")
    all_wrong = ~some_right
    print(f"shots where R, P1 and P2 are all wrong: {int(all_wrong.sum())}; the network alone is right on "
          f"{int((~net_wrong & all_wrong).sum())} of them")
    print("inside the network's most confident shots (confidence = smallest |logit| of the 12):")
    order = np.argsort(-d["confidence"], kind="stable")
    for frac in (0.10, 0.25, 0.50):
        kept = order[: int(round(frac * n))]
        print(f"  top {frac:4.0%} ({len(kept)} shots): network fails {int(net_wrong[kept].sum())}, "
              + ", ".join(f"{k} fails {int((~right[i, kept]).sum())}" for i, k in enumerate(NAMES)))
    print("Constants C, and any threshold read here, are fitted to these shots.")


if __name__ == "__main__":
    main()
