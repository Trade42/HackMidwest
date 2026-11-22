"""tcbp-fastpath: the one held-out check of the frozen rule C9 (CRITERION_4.md). No free parameter.

Exploration, after the test results; a separate branch. Fresh Stim shots at p = 0.005, seed 2026112251.

    PYTHONPATH=. python predecoder/fastpath/holdout.py --data <holdout_X_p0.005_s2026112251.npz> \
        --arms <server: files/tcbp-fastpath/holdout_X_p0.005_per_shot.npz> --logits <logits folder> \
        --out predecoder/fastpath/result_holdout_p0.005.json
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from predecoder.fastpath.fastpath2 import exact_p, wilson
from predecoder.tcbp import row_hash

T_C9 = 69.14651567152464  # bits; CRITERION_4.md, from result3_p0.005.json
DATA_SHA256 = "c62d6efb3a1a5b584522233314c60285298ae349f997adb380c323a76fb96b68"
NET_MS = 1.7              # network + head as a CUDA graph (dq-orch-6, 18:43 UTC)
NET_MS_ROUND2 = 9.4       # the network time used in rounds 1-3, for comparison
PRIORS_MS, BUILD_MS, FREE_MS = 0.16, 34.0, 4.6  # TH's fixed parts with H as CSR (tcbp-speed job 1; dq-orch-6)


def head_entropy_bits(logits, chunk=250):
    out = []
    for first in range(0, len(logits), chunk):
        h = 1.0 / (1.0 + np.exp(-np.asarray(logits[first:first + chunk]).astype(np.float16).astype(np.float64)))
        h = np.clip(h, 1e-300, 1 - 1e-16)
        out.append((-(h * np.log2(h)) - (1 - h) * np.log2(1 - h)).sum(1))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--logits", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data_sha = hashlib.sha256(open(a.data, "rb").read()).hexdigest()
    assert data_sha == DATA_SHA256, "not the declared held-out file"
    data = np.load(a.data)
    truth = data["observables"]
    n = len(truth)
    arms = np.load(a.arms)
    assert (arms["truth"] == truth).all()
    assert (arms["predicted_fresh"] == arms["predicted_batch"]).all(), "F built once differs from F built per shot"
    meta = json.load(open(Path(a.logits) / "meta.json"))
    assert meta["data"]["sha256"] == data_sha
    logits = np.load(Path(a.logits) / "logits.npy", mmap_mode="r")
    same = np.array([row_hash(logits[k]) for k in range(n)], np.uint64) == arms["logits_hash_head"]
    assert same.all(), f"logits rows differ from the head arm's on {int((~same).sum())} shots"

    s = head_entropy_bits(logits)
    easy = s <= T_C9
    th_fail = (arms["predicted_head"] != truth).any(1)
    f_fail = (arms["predicted_batch"] != truth).any(1)
    rule_fail = np.where(easy, f_fail, th_fail)
    added = int((rule_fail & ~th_fail).sum())
    only_th = int((th_fail & ~rule_fail).sum())
    p = exact_p(added, only_th)
    verdict = "holds" if added == 0 else ("holds with a cost" if p >= 0.05 else "fails")

    f_ms = 1e3 * arms["seconds_decode_fresh"]
    th_ms = PRIORS_MS + BUILD_MS + 1e3 * arms["seconds_decode_head"] + FREE_MS

    def times(net):
        rule = net + np.where(easy, f_ms, th_ms)
        alone = net + th_ms
        return {"network_ms": net, "rule_mean": round(float(rule.mean()), 2), "rule_p99": round(float(np.percentile(rule, 99)), 1),
                "TH_alone_mean": round(float(alone.mean()), 2), "TH_alone_p99": round(float(np.percentile(alone, 99)), 1)}

    out = {
        "label": "exploration, after the test results; a separate branch (tcbp-fastpath); the one held-out check of C9",
        "data": {"path": a.data, "sha256": data_sha, "shots": n, "p": float(data["p"]), "seed": int(data["seed"])},
        "rule": {"check": "C9 head entropy (bits)", "t": T_C9},
        "failures": {"rule": int(rule_fail.sum()), "TH": int(th_fail.sum()), "F": int(f_fail.sum())},
        "rule_share_95": [round(float(rule_fail.mean()), 5), wilson(int(rule_fail.sum()), n)],
        "TH_share_95": [round(float(th_fail.mean()), 5), wilson(int(th_fail.sum()), n)],
        "paired_vs_TH": {"both": int((rule_fail & th_fail).sum()), "added": added, "only_TH": only_th, "exact_p": p},
        "fast_fraction": round(float(easy.mean()), 4), "fast_shots": int(easy.sum()),
        "F_only_shots_total": int((f_fail & ~th_fail).sum()),
        "F_only_on_fast_path": added,
        # one-sided 95% upper bound (Clopper-Pearson; 3 / n for 0 seen). Reporting only: the k > 0 case was added
        # after the held-out result, which it does not change
        "added_rate_upper_95_on_fast_shots": round(float(beta.ppf(0.95, added + 1, int(easy.sum()) - added)), 5)
        if easy.sum() else None,
        "entropy_bits": {"min": float(s.min()), "median": float(np.median(s)), "max": float(s.max())},
        "times_today": times(NET_MS),
        "times_round2_network": times(NET_MS_ROUND2),
        "F_decode_ms": {"mean": round(float(f_ms.mean()), 2), "p99": round(float(np.percentile(f_ms, 99)), 1)},
        "TH_decode_ms": {"mean": round(float(1e3 * arms["seconds_decode_head"].mean()), 2)},
        "verdict": verdict,
    }
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
