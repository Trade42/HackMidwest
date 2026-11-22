"""tcbp-fastpath, round 2: the checks C2-C7 of CRITERION_2.md, scored on saved per-shot files.

Exploration, after the test results; a separate branch. Validation only: a test_* or go_* file is refused.
Nothing here decodes. Truth is used only to score.

    python predecoder/fastpath/fastpath2.py --data data/p0.005/validation_X.npz --arms <job 3 per_shot.npz> \
        --logits <logits folder> --settings predecoder/logs/fit_tcbp_X_p0.005.json --speed <tcbp-speed check_per_shot.npz> \
        --time-model csr_total --alt job1=<job 1 fresh per_shot.npz>:fresh stored=<validation_X.tcbp.npz>:T \
        --out predecoder/fastpath/result2_p0.005.json
"""
import argparse
import hashlib
import json
from math import comb
from pathlib import Path

import numpy as np

from predecoder.decode import kit_problem
from predecoder.priors import load_settings, shot_priors
from predecoder.tcbp import capped, row_hash

NET_MEAN_MS = 9.4  # network + head, A100 float32, one shot at a time (server job 4, meta_fp32.json)


def exact_p(a, b):
    n = a + b
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(c - h), 5), round(float(c + h), 5)]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def logit(x):
    return np.log(x) - np.log1p(-x)


def prior_moves(logits, kit, setting, chunk=250):
    """Per shot, in log-odds: C3 the total move, C4 the largest, C5 the columns moved by more than 1 (CRITERION_2.md);
    C8 the total upward move; and C9 the head's own entropy in bits, from its raw probabilities (CRITERION_3.md)."""
    base = logit(kit)
    out = {k: [] for k in ("C3", "C4", "C5", "C8", "C9")}
    for first in range(0, len(logits), chunk):
        rows = np.asarray(logits[first:first + chunk])
        q = capped(shot_priors(rows, kit, setting["alpha"], setting["eps"], setting["mode"]))
        signed = logit(q) - base
        m = np.abs(signed)
        out["C3"].append(m.sum(1))
        out["C4"].append(m.max(1))
        out["C5"].append((m > 1).sum(1))
        out["C8"].append(np.maximum(signed, 0.0).sum(1))
        h = 1.0 / (1.0 + np.exp(-rows.astype(np.float16).astype(np.float64)))
        h = np.clip(h, 1e-300, 1 - 1e-16)
        out["C9"].append((-(h * np.log2(h)) - (1 - h) * np.log2(1 - h)).sum(1))
    return {k: np.concatenate(v).astype(float) for k, v in out.items()}


def largest_safe(s, bad):
    """The largest observed value t with no 'bad' shot at s <= t (None if every value is at or above a bad one)."""
    below = s[s < s[bad].min()] if bad.any() else s
    return None if below.size == 0 else float(below.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--arms", required=True, help="per_shot.npz with the head arm (TH), the fresh arm (F), batch check")
    ap.add_argument("--logits", required=True, help="predecoder.evaluate's folder (logits.npy, meta.json)")
    ap.add_argument("--settings", required=True)
    ap.add_argument("--speed", required=True, help="tcbp-speed's check_per_shot.npz (TH's CSR timers)")
    ap.add_argument("--time-model", choices=("csr_total", "decode_plus_csr_means"), required=True)
    ap.add_argument("--alt", nargs="*", default=[], help="name=<per_shot.npz>:<arm suffix>, another run of F")
    ap.add_argument("--round3", action="store_true", help="add C8 and C9 (CRITERION_3.md)")
    ap.add_argument("--carry", help="another p's result json: its t* applied unchanged to these shots")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.data).name.startswith(("test", "go_")):
        raise SystemExit("validation only: no test file is opened")

    data = np.load(a.data)
    truth = data["observables"]
    n = len(truth)
    p, basis = float(data["p"]), str(data["basis"])
    weight = data["detectors"][:, data["basis_detectors"]].sum(1).astype(float)
    data_sha = sha(a.data)

    arms = np.load(a.arms)
    assert (arms["truth"] == truth).all()
    assert (arms["predicted_fresh"] == arms["predicted_batch"]).all(), "F built once differs from F built per shot"
    th_fail = (arms["predicted_head"] != truth).any(1)
    f_fail = (arms["predicted_fresh"] != truth).any(1)
    bad = f_fail & ~th_fail  # the shots a fast path must not send to F

    meta = json.load(open(Path(a.logits) / "meta.json"))
    assert meta["data"]["sha256"] == data_sha
    logits = np.load(Path(a.logits) / "logits.npy", mmap_mode="r")
    assert logits.shape[0] == n
    same = np.array([row_hash(logits[k]) for k in range(n)], np.uint64) == arms["logits_hash_head"]
    assert same.all(), f"logits rows differ from the head arm's on {int((~same).sum())} shots"
    setting = load_settings(a.settings)
    kit = kit_problem(basis, p).priors
    moves = prior_moves(logits, kit, setting)

    f_ms = 1e3 * arms["seconds_decode_fresh"]
    sp = np.load(a.speed)
    fields = list(sp["fields"])
    csr = {k: 1e3 * sp["rows_csr"][:, fields.index(k)] for k in fields if k.startswith("seconds_")}
    if a.time_model == "csr_total":  # p = 0.005: TH's one-timer CSR total, per shot for shots 0-999
        th_ms = np.full(n, csr["seconds_total"].mean())
        th_ms[: len(csr["seconds_total"])] = csr["seconds_total"]
        timed = np.arange(n) < len(csr["seconds_total"])
    else:  # p = 0.004: this run's decode per shot + CSR priors, build and free at their means (p = 0.005)
        fixed = csr["seconds_priors"].mean() + csr["seconds_build"].mean() + csr["seconds_free"].mean()
        th_ms = 1e3 * arms["seconds_decode_head"] + fixed
        timed = np.ones(n, bool)
    th_mean = float(csr["seconds_total"].mean()) if a.time_model == "csr_total" else float(th_ms.mean())

    checks = {
        "C2 syndrome weight": (weight, "pre"),
        "C3 total prior move": (moves["C3"], "net"),
        "C4 largest prior move": (moves["C4"], "net"),
        "C5 columns moved > 1": (moves["C5"], "net"),
        "C6 F decode time (ms)": (f_ms, "F"),
        "C7 F correction weight": (arms["n_faults_fresh"].astype(float), "F"),
    }
    if a.round3:
        checks["C8 upward prior move"] = (moves["C8"], "net")
        checks["C9 head entropy (bits)"] = (moves["C9"], "net")

    # csr_total: the mean uses TH's mean on every shot, the p99 TH's per-shot times on shots 0-999 (CRITERION_2.md)
    th_for_mean = np.full(n, th_mean) if a.time_model == "csr_total" else th_ms

    def cost(easy, kind, th):
        if kind is None:  # TH alone
            return NET_MEAN_MS + th
        if kind == "pre":  # check before any decode: F on easy, network + TH on hard
            return np.where(easy, f_ms, NET_MEAN_MS + th)
        if kind == "net":  # network on every shot: then F on easy, TH (network already run) on hard
            return NET_MEAN_MS + np.where(easy, f_ms, th)
        return f_ms + np.where(easy, 0.0, NET_MEAN_MS + th)  # F first on every shot

    def row(name, easy, kind, t, check=None):
        fail = np.where(easy, f_fail, th_fail)
        only_rule, only_th = int((fail & ~th_fail).sum()), int((th_fail & ~fail).sum())
        per_shot = cost(easy, kind, th_ms)
        bound = round(3 / int(easy.sum()), 5) if only_rule == 0 and easy.sum() else None
        return {"rule": name, "check": check, "t": t, "added_rate_upper_95_on_fast_shots": bound,
                "TH_failure_rate": round(float(th_fail.mean()), 5), "fast_fraction": round(float(easy.mean()), 4), "fast_shots": int(easy.sum()),
                "failures": int(fail.sum()), "share_95": [round(float(fail.mean()), 5), wilson(int(fail.sum()), n)],
                "both_fail": int((fail & th_fail).sum()), "only_rule": only_rule, "only_TH": only_th,
                "exact_p": exact_p(only_rule, only_th), "passes_gate": only_rule == 0,
                "mean_ms": round(float(cost(easy, kind, th_for_mean).mean()), 2),
                "p99_ms": round(float(np.percentile(per_shot[timed], 99)), 1)}

    rows = [row("TH alone (the TCBP version)", np.zeros(n, bool), None, None),
            row("F alone (TCBP-JOBS, kit priors, built once)", np.ones(n, bool), "pre", None)]
    halves = {}
    alts = {}
    for spec in a.alt:
        name, rest = spec.split("=", 1)
        path, suffix = rest.rsplit(":", 1)
        d = np.load(path)
        assert (d["truth"] == truth).all()
        alts[name] = {"fail": (d[f"predicted_{suffix}"] != truth).any(1),
                      "C6 F decode time (ms)": 1e3 * d[f"seconds_decode_{suffix}"] if f"seconds_decode_{suffix}" in d.files
                      and np.isfinite(d[f"seconds_decode_{suffix}"]).all() else None,
                      "C7 F correction weight": d[f"n_faults_{suffix}"].astype(float) if f"n_faults_{suffix}" in d.files else None,
                      "failures": int((d[f"predicted_{suffix}"] != truth).any(1).sum())}
    even = np.arange(n) % 2 == 0
    for name, (s, kind) in checks.items():
        t = largest_safe(s, bad)
        easy = s <= t if t is not None else np.zeros(n, bool)
        rows.append(row(f"{name} <= {t:g}" if t is not None else f"{name}: no safe t", easy, kind, t, name))
        h = {}
        for pick, other in (("even", ~even), ("odd", even)):
            mask = even if pick == "even" else ~even
            tp = largest_safe(s[mask], bad[mask])
            e = (s <= tp) & other if tp is not None else np.zeros(n, bool)
            h[f"picked_on_{pick}"] = {"t": tp, "fast_fraction_other_half": round(float(e.sum() / other.sum()), 4),
                                      "added_on_other_half": int((e & bad).sum())}
        halves[name] = h
        for alt_name, alt in alts.items():
            s_alt = alt.get(name, s) if name in ("C6 F decode time (ms)", "C7 F correction weight") else s
            entry = alts[alt_name].setdefault("added_at_t_star", {})
            if s_alt is None or t is None:
                entry[name] = "unmeasured" if s_alt is None else 0
            else:
                entry[name] = int((alt["fail"] & ~th_fail & (s_alt <= t)).sum())

    # not declared, a diagnostic: C6 if F's decode could be stopped at t* (not tested on cudaq-qec 0.9.0)
    t6 = largest_safe(f_ms, bad)
    timeout = None
    if t6 is not None:
        easy6 = f_ms <= t6
        stop = np.minimum(f_ms, t6)
        timeout = {"t_ms": t6, "mean_ms": round(float((stop + np.where(easy6, 0.0, NET_MEAN_MS + th_for_mean)).mean()), 2),
                   "p99_ms": round(float(np.percentile((stop + np.where(easy6, 0.0, NET_MEAN_MS + th_ms))[timed], 99)), 1)}

    carried = None
    if a.carry:
        other = json.load(open(a.carry))
        carried = {"from": a.carry, "from_p": other["data"]["p"], "checks": {}}
        for r in other["rows"]:
            name = r.get("check")
            if name in checks and r["t"] is not None:
                s, _ = checks[name]
                easy = s <= r["t"]
                carried["checks"][name] = {"t": r["t"], "fast_fraction": round(float(easy.mean()), 4),
                                           "added": int((easy & bad).sum()), "fixed": int((easy & th_fail & ~f_fail).sum())}

    cands = [r for r in rows if r["passes_gate"]]
    chosen = min(cands, key=lambda r: r["mean_ms"])
    out = {
        "label": "exploration, after the test results; a separate branch (tcbp-fastpath); validation only; CRITERION_2.md",
        "data": {"path": a.data, "sha256": data_sha, "shots": n, "p": p, "basis": basis, "seed": int(data["seed"])},
        "inputs_sha256": {"arms": sha(a.arms), "logits.npy": sha(Path(a.logits) / "logits.npy"),
                          "settings": sha(a.settings), "speed": sha(a.speed)},
        "counts": {"TH": int(th_fail.sum()), "F": int(f_fail.sum()), "F_only": int(bad.sum()),
                   "TH_only": int((th_fail & ~f_fail).sum())},
        "times_ms": {"time_model": a.time_model, "net_mean": NET_MEAN_MS, "F_decode_mean": round(float(f_ms.mean()), 2),
                     "TH_mean": round(th_mean, 2), "TH_alone_mean": round(NET_MEAN_MS + th_mean, 2),
                     "p99_over_shots": int(timed.sum())},
        "check_ranges": {k: {"min": float(v[0].min()), "median": float(np.median(v[0])), "max": float(v[0].max()),
                             "smallest_on_an_F_only_failure": float(v[0][bad].min()) if bad.any() else None}
                         for k, v in checks.items()},
        "rows": rows,
        "half_split": halves,
        "other_F_runs": {k: {"failures": v["failures"], "added_at_t_star": v.get("added_at_t_star")} for k, v in alts.items()},
        "chosen": chosen,
        "diagnostic_not_declared_C6_with_F_stopped_at_t": timeout,
        "carried": carried,
    }
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{Path(a.data).name} p={p:g}: TH {out['counts']['TH']}, F {out['counts']['F']} (F only {out['counts']['F_only']}, "
          f"TH only {out['counts']['TH_only']}); time model {a.time_model}")
    for r in rows:
        print(f"{r['rule']:<46} fast {r['fast_fraction']:.3f}  fail {r['failures']:>3} (both {r['both_fail']}, only rule "
              f"{r['only_rule']}, only TH {r['only_TH']})  mean {r['mean_ms']:5.1f} ms  p99 {r['p99_ms']:6.1f} ms  "
              f"{'PASS' if r['passes_gate'] else 'fail'}")
    print("half split (t picked on one half, added failures on the other):")
    for k, h in halves.items():
        print(f"  {k:<26} " + "; ".join(f"{pk}: t {v['t']}, fast {v['fast_fraction_other_half']:.3f}, added {v['added_on_other_half']}"
                                         for pk, v in h.items()))
    for k, v in out["other_F_runs"].items():
        print(f"other F run {k} ({v['failures']} failures), added at t*: {v['added_at_t_star']}")
    print("chosen:", chosen["rule"], chosen["mean_ms"], "ms mean,", chosen["p99_ms"], "ms p99")
    print("not declared, a diagnostic: C6 with F stopped at t*:", timeout)
    for r in rows:
        if r["added_rate_upper_95_on_fast_shots"] is not None and r["fast_shots"]:
            print(f"  95% upper bound on the added-failure rate among fast shots, {r['rule']}: "
                  f"{r['added_rate_upper_95_on_fast_shots']} (TH's failure rate {r['TH_failure_rate']})")
    if carried:
        print(f"t* carried from p = {carried['from_p']:g}, unchanged, on these shots:")
        for k, v in carried["checks"].items():
            print(f"  {k:<26} t {v['t']:g}: fast {v['fast_fraction']:.3f}, added {v['added']}, fixed {v['fixed']}")


if __name__ == "__main__":
    main()
