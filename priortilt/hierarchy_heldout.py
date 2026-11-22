"""Held-out score of the hierarchy "network where it is confident, else the 571-rule", X memory, p = 0.005.

  python -I -B hierarchy_heldout.py <network step>

Base rule (the "571-rule"): P1 where P1 and P2 agree; otherwise R is run and the lowest weight among the converged
candidates of R, P1, P2 is taken (all three if none converged). Hierarchy(c): the network's answer where its confidence
(the smallest |logit| of its 12 outputs) is at least c, else the base rule's.

The threshold is fitted on train_X only: the smallest c at which the hierarchy adds no failure against the base rule on
train (no train shot answered by the network that the network gets wrong and the base rule gets right). It is then
applied unchanged to validation_X. Reported for validation (held out) and, labelled, for train (in sample):
failures, coverage, the paired discordant counts against the base rule and an exact two-sided McNemar test.
A looser variant, the smallest c at which the network's answers have no more failures than the base rule's on the
same train shots (net change <= 0), is printed too, also fitted on train only.
Inputs: results_p005main/selector_<stem>.npz and ../cascade_X_s1/eval/step<N>_<stem>_p0.005.npz (logits).
"""
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(os.path.dirname(HERE), "cascade_X_s1", "eval")


def load(stem, step):
    s = np.load(os.path.join(HERE, "results_p005main", f"selector_{stem}.npz"))
    e = np.load(os.path.join(EVAL, f"step{step}_{stem}_p0.005.npz"))
    meta = json.load(open(os.path.join(EVAL, f"step{step}_{stem}_p0.005.json")))
    truth = s["truth"]
    if not np.array_equal(truth, e["observables"]):
        sys.exit(f"{stem}: the network's observables differ from the selector file's truth")
    pred = np.stack([s[f"predicted_{k}"] for k in ("R", "P1", "P2")])
    weight = np.stack([s[f"weight_{k}"] for k in ("R", "P1", "P2")])
    conv = np.stack([s[f"converged_{k}"] for k in ("R", "P1", "P2")])
    sec = np.stack([s[f"seconds_{k}"] for k in ("R", "P1", "P2")]) * 1e3
    rows = np.arange(len(truth))
    p1_p2 = (pred[1] == pred[2]).all(axis=1)
    lowest = weight.argmin(axis=0)
    lowest_conv = np.where(conv.any(axis=0), np.where(conv, weight, np.inf).argmin(axis=0), lowest)
    base_pred = pred[np.where(p1_p2, 1, lowest_conv), rows]
    base_ms = sec[1] + sec[2] + np.where(p1_p2, 0, sec[0])
    base_wrong = (base_pred != truth).any(axis=1)
    net_pred = e["predictions"]
    net_wrong = (net_pred != truth).any(axis=1)
    return {"n": len(truth), "base_wrong": base_wrong, "net_wrong": net_wrong, "conf": e["confidence"].astype(np.float64),
            "base_ms": base_ms, "net_ms": meta["latency_ms_mean"], "sha": str(s["data_sha256"])}


def fit_strict(d):
    """Smallest c with no network-answered shot that is net-wrong and base-right."""
    bad = d["net_wrong"] & ~d["base_wrong"]
    if not bad.any():
        return float(d["conf"].min())
    return float(np.nextafter(d["conf"][bad].max(), np.inf))


def fit_net(d):
    """Smallest c with failures of the network <= failures of the base rule on the shots it answers (net change <= 0)."""
    order = np.argsort(-d["conf"], kind="stable")
    delta = np.cumsum(d["net_wrong"][order].astype(int) - d["base_wrong"][order].astype(int))
    conf_sorted = d["conf"][order]
    best = None
    for k in range(len(order)):           # accept the first k + 1 shots; ties in confidence must be accepted together
        if k + 1 < len(order) and conf_sorted[k + 1] == conf_sorted[k]:
            continue
        if delta[k] <= 0:
            best = conf_sorted[k]
    return float(best) if best is not None else float("inf")


def score(label, d, c):
    covered = d["conf"] >= c
    hier_wrong = np.where(covered, d["net_wrong"], d["base_wrong"])
    b = int((hier_wrong & ~d["base_wrong"]).sum())      # hierarchy wrong, base right
    a = int((~hier_wrong & d["base_wrong"]).sum())      # hierarchy right, base wrong
    p = binomtest(b, a + b, 0.5).pvalue if a + b else 1.0
    ms = d["net_ms"] + np.where(covered, 0.0, d["base_ms"])
    print(f"  {label}: {d['n']} shots | coverage {covered.mean():.2%} ({int(covered.sum())} answered by the network) | "
          f"failures hierarchy {int(hier_wrong.sum())} ({hier_wrong.mean():.3%}) against base {int(d['base_wrong'].sum())} "
          f"({d['base_wrong'].mean():.3%}) | discordant: hierarchy-only wrong {b}, base-only wrong {a}, exact two-sided "
          f"p = {p:.3g} | mean ms/shot {ms.mean():.1f} (base alone {d['base_ms'].mean():.1f}; network {d['net_ms']:.1f} on every shot)")


def main():
    step = int(sys.argv[1])
    train, val = load("train_X", step), load("validation_X", step)
    print(f"Hierarchy: network (checkpoint_step{step}.pt) where confidence >= c, else the 571-rule. X memory, p = 0.005.")
    print(f"  train_X sha256 {train['sha']}; validation_X sha256 {val['sha']}")
    for name, fit in (("strict: no added failure on train", fit_strict), ("loose: net change <= 0 on train", fit_net)):
        c = fit(train)
        print(f"threshold fitted on train_X only ({name}): c = {c:.6g}")
        score("validation_X (held out)", val, c)
        score("train_X (in sample, the fit)", train, c)
    print("The threshold is fitted on train_X and applied unchanged to validation_X; the network was never trained on Stim shots.")


if __name__ == "__main__":
    main()
