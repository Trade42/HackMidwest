"""Step 2: how well does the tracker find the loss round? Scored against the probes, before any decoding.

usage (from challenge-1-ml-decoder-kit/):
    python -m loss_tracker.accuracy --data VAL.npz --post NAME=POST.npz [NAME=POST.npz ...] --main NAME \
        --out loss_tracker/results/accuracy_val

The tracker's outputs come from the files track.py wrote (it reads detectors and heralds only). This script
reads the truth (`data_slot`) of the same dataset to score them. Rounds are counted 1..12 in the tables
(round n = cycle n - 1 of the code).
"""
import argparse
import json

import numpy as np

from loss.lossmodel import CYCLES, load_dataset
from loss.summarize import wilson
from loss_tracker.tracker import TYPES, Geometry, softmax, to_rounds

BINS = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0000001]


def load_post(path):
    with np.load(path) as f:
        return {k: (json.loads(str(f[k])) if k == "meta" else f[k]) for k in f.files}


def scores(rounds, true_cycle, mask=None):
    """Accuracy numbers of round posteriors (n, 12) against the true cycles (n,)."""
    if mask is not None:
        rounds, true_cycle = rounds[mask], true_cycle[mask]
    n = len(true_cycle)
    if n == 0:
        return {"atoms": 0}
    est = rounds.argmax(axis=1)
    p_true = rounds[np.arange(n), true_cycle]
    near = np.abs(est - true_cycle) <= 1
    mass_near = sum(rounds[np.arange(n), np.clip(true_cycle + d, 0, CYCLES - 1)] * ((true_cycle + d >= 0) & (true_cycle + d < CYCLES))
                    for d in (-1, 0, 1))
    return {"atoms": n, "top1": float(np.mean(est == true_cycle)), "within1": float(near.mean()),
            "within1_wilson95": list(wilson(int(near.sum()), n)),
            "mean_p_true": float(p_true.mean()), "mean_p_within1": float(mass_near.mean()),
            "mean_log_p_true": float(np.log(np.maximum(p_true, 1e-300)).mean()),
            "mean_error_rounds": float(np.mean(est - true_cycle)), "mean_abs_error_rounds": float(np.mean(np.abs(est - true_cycle)))}


def reliability(prob, hit):
    """Rows (bin low, bin high, count, mean predicted, observed frequency) and the expected calibration error."""
    rows, ece = [], 0.0
    for lo, hi in zip(BINS[:-1], BINS[1:], strict=True):
        m = (prob >= lo) & (prob < hi)
        if m.any():
            rows.append({"from": lo, "to": min(hi, 1.0), "count": int(m.sum()), "mean_predicted": float(prob[m].mean()),
                         "observed": float(hit[m].mean())})
            ece += m.sum() * abs(prob[m].mean() - hit[m].mean())
    return rows, float(ece / len(prob))


def categories(geo, heralds, shot, atom):
    """Per entry, from the heralds alone: does the atom share a check with another flagged data atom (X, Z), and
    was one of its 6 check atoms flagged in some cycle?"""
    flagged = heralds[:, geo.data_rec].astype(bool)
    out = {}
    for T in TYPES:
        out[f"shares_{T}"] = flagged[shot[:, None, None], geo.co_atom[T][atom]].any(axis=(1, 2))
    nb = np.zeros(len(shot), bool)
    for T in TYPES:
        h = heralds[:, geo.check_rec[T]].astype(bool).any(axis=2)             # (shots, 72): flagged in some cycle
        nb |= h[shot[:, None], geo.chk[T][atom]].any(axis=1)
    out["flagged_check_neighbour"] = nb
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--post", required=True, nargs="+", help="NAME=FILE of track.py, one per tracker version")
    ap.add_argument("--main", required=True, help="the version that gets the detailed tables")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = load_dataset(a.data)
    meta = data["meta"]
    geo = Geometry(meta["basis"], meta["p"])
    out = {"dataset": {**meta, "file": a.data.split("/")[-1]}, "versions": {}}
    md = [f"Dataset {a.data.split('/')[-1]} (seed {meta['seed']}), rounds counted 1..12.", ""]
    md += ["| version | atoms | most probable round is the true one | within one round | mean P(true round) | mean log P(true round) | mean error (rounds) |",
           "|---|---|---|---|---|---|---|"]
    posts = dict(x.split("=", 1) for x in a.post)
    for name, path in posts.items():
        f = load_post(path)
        assert f["meta"]["data_sha256"] and f["meta"]["data_seed"] == meta["seed"]
        true_slot = data["data_slot"][f["shot"], f["atom"]].astype(np.int64)
        assert (true_slot > 0).all() and len(true_slot) == int((data["data_slot"][:f["meta"]["shots"]] > 0).sum())
        tc = (true_slot - 1) // 6
        rounds = to_rounds(softmax(f["log_prior"][None, :] + f["loglik"].astype(np.float64)))
        sc = scores(rounds, tc)
        out["versions"][name] = {"tracker": f["meta"], "all": sc,
                                 "rounds_1_to_9": scores(rounds, tc, tc <= 8), "rounds_2_to_10": scores(rounds, tc, (tc >= 1) & (tc <= 9))}
        md.append(f"| {name} | {sc['atoms']} | {sc['top1']:.3f} | {sc['within1']:.3f} | {sc['mean_p_true']:.3f} | "
                  f"{sc['mean_log_p_true']:.3f} | {sc['mean_error_rounds']:+.2f} |")
        if name != a.main:
            continue
        detail = out["main"] = {"version": name}
        detail["by_true_round"] = [{"round": c + 1, **scores(rounds, tc, tc == c)} for c in range(CYCLES)]
        cat = categories(geo, data["heralds"], f["shot"], f["atom"])
        alone = ~cat["shares_X"] & ~cat["shares_Z"]
        groups = {"shares no check with another flagged data atom": alone,
                  "shares an X check only": cat["shares_X"] & ~cat["shares_Z"],
                  "shares a Z check only": cat["shares_Z"] & ~cat["shares_X"],
                  "shares X and Z checks": cat["shares_X"] & cat["shares_Z"],
                  "alone, and none of its 6 check atoms was ever flagged": alone & ~cat["flagged_check_neighbour"]}
        detail["by_sharing"] = [{"group": g, "share_of_atoms": float(m.mean()), **scores(rounds, tc, m)} for g, m in groups.items()]
        hit = np.zeros_like(rounds, bool)
        hit[np.arange(len(tc)), tc] = True
        detail["reliability_all_rounds"], detail["ece_all_rounds"] = reliability(rounds.ravel(), hit.ravel())
        detail["reliability_top1"], detail["ece_top1"] = reliability(rounds.max(axis=1), rounds.argmax(axis=1) == tc)
        detail["temperature"] = []
        for T in (0.7, 0.85, 1.0, 1.25, 1.5, 2.0, 3.0):
            r_T = to_rounds(softmax(f["log_prior"][None, :] + f["loglik"].astype(np.float64) / T))
            detail["temperature"].append({"temperature": T, "mean_log_p_true": float(np.log(r_T[np.arange(len(tc)), tc]).mean()),
                                          "ece_top1": reliability(r_T.max(axis=1), r_T.argmax(axis=1) == tc)[1]})
        uniform = np.full_like(rounds, 1 / CYCLES)
        detail["uniform_guess"] = scores(uniform + 1e-9 * np.arange(CYCLES)[::-1], tc)
    k = out["versions"][a.main]
    md += ["", f"Kill line (version `{a.main}`): losses of rounds 1 to 9 within one round of the truth: "
           f"{k['rounds_1_to_9']['within1']:.3f} of {k['rounds_1_to_9']['atoms']} (rounds 2 to 10: {k['rounds_2_to_10']['within1']:.3f} of "
           f"{k['rounds_2_to_10']['atoms']}). The line is 0.5.", "",
           f"By true loss round (version `{a.main}`):", "",
           "| true round | atoms | most probable round is the true one | within one round (95% Wilson) | mean P(true round) | mean P(within one) | mean error (rounds) |",
           "|---|---|---|---|---|---|---|"]
    for r in out["main"]["by_true_round"]:
        lo, hi = r["within1_wilson95"]
        md.append(f"| {r['round']} | {r['atoms']} | {r['top1']:.3f} | {r['within1']:.3f} ({lo:.3f}–{hi:.3f}) | {r['mean_p_true']:.3f} | "
                  f"{r['mean_p_within1']:.3f} | {r['mean_error_rounds']:+.2f} |")
    md += ["", "By what the atom shares (from the heralds alone):", "",
           "| group | share of the flagged atoms | atoms | true round | within one round | mean log P(true round) |", "|---|---|---|---|---|---|"]
    for r in out["main"]["by_sharing"]:
        md.append(f"| {r['group']} | {r['share_of_atoms']:.3f} | {r['atoms']} | {r['top1']:.3f} | {r['within1']:.3f} | {r['mean_log_p_true']:.3f} |")
    md += ["", f"Reliability of the round posterior, every (atom, round) pair (expected calibration error {out['main']['ece_all_rounds']:.4f}):", "",
           "| posterior between | pairs | mean posterior | share that is the true round |", "|---|---|---|---|"]
    for r in out["main"]["reliability_all_rounds"]:
        md.append(f"| {r['from']:.2f} and {r['to']:.2f} | {r['count']} | {r['mean_predicted']:.3f} | {r['observed']:.3f} |")
    md += ["", f"Reliability of the most probable round (expected calibration error {out['main']['ece_top1']:.4f}):", "",
           "| its posterior between | atoms | mean posterior | share that is the true round |", "|---|---|---|---|"]
    for r in out["main"]["reliability_top1"]:
        md.append(f"| {r['from']:.2f} and {r['to']:.2f} | {r['count']} | {r['mean_predicted']:.3f} | {r['observed']:.3f} |")
    md += ["", "Temperature on the evidence (1 = as fitted; the tracker alone, before any decoding):", "",
           "| temperature | mean log P(true round) | calibration error of the most probable round |", "|---|---|---|"]
    for r in out["main"]["temperature"]:
        md.append(f"| {r['temperature']} | {r['mean_log_p_true']:.4f} | {r['ece_top1']:.4f} |")
    u = out["main"]["uniform_guess"]
    md += ["", f"For scale, the uniform guess: mean P(true round) {u['mean_p_true']:.3f}, mean log P {u['mean_log_p_true']:.3f}."]
    with open(a.out + ".json", "w") as fh:
        json.dump(out, fh, indent=1)
    with open(a.out + ".md", "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
