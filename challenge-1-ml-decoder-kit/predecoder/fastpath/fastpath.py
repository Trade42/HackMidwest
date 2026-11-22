"""tcbp-fastpath: answer easy shots with a cheap decoder, hard ones with the TCBP version.

Exploration, after the test results; a separate branch. Validation shots only (validation_X, p = 0.005).
The criterion was committed first: CRITERION.md. Nothing here decodes: it reads saved per-shot files.

    python predecoder/fastpath/fastpath.py --data data/p0.005/validation_X.npz \
        --job3 <server-logs: files/predecoder-tcbp/validation_X_p0.005_per_shot.npz> \
        --priortilt <laptop-artifacts: priortilt/results_p005main/selector_validation_X.npz> \
        --speed <server-logs: files/tcbp-speed/check_per_shot.npz> --out predecoder/fastpath/result.json
"""
import argparse
import hashlib
import json
from math import comb

import numpy as np

NET_MEAN_MS = 9.4  # network + head, A100 float32, one shot at a time (job 4, meta_fp32.json)
NET_P99_MS = 19.8


def exact_p(a, b):
    """Exact two-sided binomial test on the discordant shots (a only one side, b only the other)."""
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 5), round(c + h, 5)]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--job3", required=True)
    ap.add_argument("--priortilt", required=True)
    ap.add_argument("--speed", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if "test" in a.data:
        raise SystemExit("validation only: no test file is opened")

    data = np.load(a.data)
    truth = data["observables"]
    weight = data["detectors"][:, data["basis_detectors"]].sum(axis=1).astype(int)
    n = len(truth)
    data_sha = sha(a.data)

    j3 = np.load(a.job3)
    pt = np.load(a.priortilt)
    sp = np.load(a.speed)
    assert (j3["truth"] == truth).all() and (pt["truth"] == truth).all()
    assert str(pt["data_sha256"]) == data_sha
    assert (j3["predicted_fresh"] == j3["predicted_batch"]).all()  # F built once = built per shot

    th_pred, f_pred, p2_pred = j3["predicted_head"], j3["predicted_fresh"], pt["predicted_P2"]
    th_fail = (th_pred != truth).any(1)
    f_fail = (f_pred != truth).any(1)
    p2_fail = (p2_pred != truth).any(1)
    assert th_fail.sum() == j3["failed_head"].sum() and f_fail.sum() == j3["failed_fresh"].sum()

    f_ms = 1e3 * j3["seconds_decode_fresh"]
    p2_ms = 1e3 * pt["seconds_P2"]
    fields = list(sp["fields"])
    th_csr_ms = 1e3 * sp["rows_csr"][:, fields.index("seconds_total")]  # priors+build+decode+free, shots 0-999
    assert (sp["truth"] == truth[: len(th_csr_ms)]).all()
    assert (sp["predicted_code_csr"] == sp["predicted_code_dense"]).all()
    th_mean = float(th_csr_ms.mean())
    slow_mean = NET_MEAN_MS + th_mean

    def evaluate(name, easy, fast_ms):
        fail = np.where(easy, f_fail, th_fail)
        only_rule = int((fail & ~th_fail).sum())
        only_th = int((th_fail & ~fail).sum())
        per_shot = fast_ms + np.where(easy, 0.0, slow_mean)
        m = len(th_csr_ms)  # p99 on the shots with per-shot TH times (network at its mean)
        p99_shots = fast_ms[:m] + np.where(easy[:m], 0.0, NET_MEAN_MS + th_csr_ms)
        return {
            "rule": name,
            "fast_fraction": round(float(easy.mean()), 4),
            "fast_shots": int(easy.sum()),
            "failures": int(fail.sum()),
            "share_95": [round(float(fail.mean()), 5), wilson(int(fail.sum()), n)],
            "both_fail": int((fail & th_fail).sum()),
            "only_rule": only_rule,
            "only_TH": only_th,
            "exact_p": exact_p(only_rule, only_th),
            "passes_gate": only_rule == 0,
            "mean_ms": round(float(per_shot.mean()), 2),
            "p99_ms_shots_0_999": round(float(np.percentile(p99_shots, 99)), 1),
        }

    rows = []
    th_alone = evaluate("TH alone (the TCBP version)", np.zeros(n, bool), np.zeros(n))
    rows.append(th_alone)
    rows.append(evaluate("F alone (TCBP-JOBS, kit priors, built once)", np.ones(n, bool), f_ms))
    agree = (f_pred == p2_pred).all(1)
    rows.append(evaluate("C1: F agrees with P2 on the 12 bits -> F, else TH", agree, np.maximum(f_ms, p2_ms)))
    scan = []
    for w in range(int(weight.min()), int(weight.max()) + 1):
        # the weight is known before any decoding: F runs only on the easy shots, TH only on the rest
        r = evaluate(f"C2: syndrome weight <= {w} -> F, else TH", weight <= w, np.where(weight <= w, f_ms, 0.0))
        scan.append({"w": w, **{k: r[k] for k in ("fast_fraction", "failures", "only_rule", "only_TH", "mean_ms")}})
    passing_w = [s["w"] for s in scan if s["only_rule"] == 0]
    w_star = max(passing_w)
    rows.append(evaluate(f"C2: syndrome weight <= {w_star} -> F, else TH", weight <= w_star,
                         np.where(weight <= w_star, f_ms, 0.0)))

    # a fast path must also be faster than the TCBP version alone
    candidates = [r for r in rows[2:] if r["passes_gate"] and r["mean_ms"] < th_alone["mean_ms"]]
    chosen = min(candidates, key=lambda r: r["mean_ms"]) if candidates else None
    out = {
        "label": "exploration, after the test results; a separate branch (tcbp-fastpath); validation only",
        "data": {"path": a.data, "sha256": data_sha, "shots": n, "p": float(data["p"]), "seed": int(data["seed"])},
        "inputs_sha256": {"job3": sha(a.job3), "priortilt": sha(a.priortilt), "speed": sha(a.speed)},
        "counts": {"TH": int(th_fail.sum()), "F": int(f_fail.sum()), "P2": int(p2_fail.sum())},
        "times_ms": {
            "F_decode_mean": round(float(f_ms.mean()), 2), "F_decode_p99": round(float(np.percentile(f_ms, 99)), 1),
            "P2_mean_laptop_cpu": round(float(p2_ms.mean()), 2),
            "net_mean": NET_MEAN_MS, "net_p99": NET_P99_MS,
            "TH_csr_total_mean_shots_0_999": round(th_mean, 2),
            "TH_csr_total_p99_shots_0_999": round(float(np.percentile(th_csr_ms, 99)), 1),
            "slow_path_mean": round(slow_mean, 2),
        },
        "rows": rows,
        "c2_scan": scan,
        "weight": {"min": int(weight.min()), "median": float(np.median(weight)), "max": int(weight.max()),
                   "smallest_weight_where_F_fails_and_TH_right": int(weight[f_fail & ~th_fail].min())},
        "chosen": chosen,
    }
    json.dump(out, open(a.out, "w"), indent=1)
    for r in rows:
        print(f"{r['rule']:<55} fast {r['fast_fraction']:.3f}  fail {r['failures']:>4} "
              f"(both {r['both_fail']}, only rule {r['only_rule']}, only TH {r['only_TH']}, p {r['exact_p']:.3g})  "
              f"mean {r['mean_ms']:.1f} ms  p99 {r['p99_ms_shots_0_999']:.1f} ms  gate {'PASS' if r['passes_gate'] else 'fail'}")
    print("chosen:", chosen["rule"] if chosen else None)


if __name__ == "__main__":
    main()
