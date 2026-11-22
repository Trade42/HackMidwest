"""G2: how much accuracy sits above relay-BP. Other decoders on the shots relay-BP failed in G1, and on a control.

  python -m analysis.g2_headroom select --pershot analysis/results/pershot --p 0.004 --out analysis/results/g2_selection_p0.004.npz
  python -m analysis.g2_headroom run --selection SEL.npz... --data FILE... --cache DIR --processes 8 --log LOG [--only NAME...] [--p P...] [--caps CAPS]
  python -m analysis.g2_headroom summarize --selection SEL.npz... --data FILE... --cache DIR --out analysis/results [--caps CAPS]

CAPS is a comma-separated list of variant@p=F/C/Fu: a seeded subsample of at most F failures and
C control shots per basis, and among the chosen failures at most Fu that did not converge in G1
('all' for no cap). Example: joint@0.004=50/100/all,cand16@0.005=all/all/10.

The sets, per p and basis:
- F: every shot the kit's relay-BP failed in G1 (optionally a seeded subsample, stated);
- C: as many shots relay-BP got right (or at least --control-min), drawn without replacement with a
  fixed seed.
Every decode here runs on a fresh decoder per shot (and per seed), so no result depends on
which other shots were decoded. The variants:
- kit_fresh: the kit's relay-BP again, seed 0, fresh decoder: the re-roll floor. Relay-BP is
  randomized, so some failures go away just by decoding the shot again;
- big: relay-BP with a larger budget, stop_nconv=20 and num_sets=600;
- cand16: relay-BP with stop_nconv=1 under seeds 1 to 16. Its converged candidates are picked
  by lowest weight, or by a vote by logical class (each distinct candidate adds exp(-weight));
  the weight is the sum of log((1 - prior) / prior). The truth only scores the picks;
- bposd10 and bposd_hi: the kit's BP-OSD, at its settings and at a higher OSD order;
- joint: the kit's relay-BP (RELAY_SETTINGS, seed 0) on the full problem, dem_matrices(circuit)
  with no basis: all 1728 detectors of the shot and every mechanism, the same 12 observables.
  The relay-BP authors' joint ("xyz") runs use these same settings;
- joint600: the same with num_sets=600. With the same seed its first 300 legs are joint's, so it
  can differ from joint only on shots where joint found fewer than 5 solutions in 300 legs.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from analysis.common import (Log, Task, clopper_pearson, collect, iteration_stats, machine, paired, run_tasks,
                             sha256, task_file, write_json)

CONTROL_SEED = 20261122
SUBSAMPLE_SEED = 20261123
BPOSD_HI_ORDER = 60   # OSD-CS cost grows slowly with order: 60 cost 1.9 times order 10 on 5 failed shots
VARIANTS = {
    "kit_fresh": {"kind": "relay", "kwargs": {}, "mode": "fresh", "seeds": [0]},
    "big": {"kind": "relay", "kwargs": {"stop_nconv": 20, "num_sets": 600}, "mode": "fresh", "seeds": [0]},
    "cand16": {"kind": "relay", "kwargs": {"stop_nconv": 1}, "mode": "fresh", "seeds": list(range(1, 17)),
               "seed_axis": True},
    "bposd10": {"kind": "bposd", "kwargs": {}, "mode": "fresh", "seeds": [0]},
    "bposd_hi": {"kind": "bposd", "kwargs": {"osd_order": BPOSD_HI_ORDER}, "mode": "fresh", "seeds": [0]},
    "joint": {"kind": "relay", "kwargs": {}, "mode": "fresh", "seeds": [0], "problem": "full"},
    "joint600": {"kind": "relay", "kwargs": {"num_sets": 600}, "mode": "fresh", "seeds": [0], "problem": "full"},
}
SHOTS_PER_TASK = {"kit_fresh": 5, "big": 3, "cand16": 1, "bposd10": 25, "bposd_hi": 25, "joint": 1, "joint600": 1}


def parse_caps(text):
    """'joint@0.004=50/100/all' -> {("joint", 0.004): (50, 100, None)}: caps on F, C and unconverged F."""
    caps = {}
    for item in filter(None, (text or "").split(",")):
        key, val = item.split("=")
        variant, p = key.split("@")
        caps[(variant, float(p))] = tuple(None if v == "all" else int(v) for v in val.split("/"))
    return caps


# ---------------------------------------------------------------- selection

def load_pershot(folder, p):
    """G1's committed per-shot files at p, grouped by basis: {basis: [(file name, sha256, arrays)]}."""
    out = {}
    for f in sorted(Path(folder).glob(f"g1_p{p}_*.npz")):
        with np.load(f) as z:
            arrays = {k: z[k] for k in z.files}
        basis = f.stem.split("_")[2]
        out.setdefault(basis, []).append((str(arrays["dataset_file"]), str(arrays["sha256"]), arrays))
    return out


def cmd_select(args):
    rows = []
    for p in args.p:
        for basis, blocks in sorted(load_pershot(args.pershot, p).items()):
            fail, right = [], []
            for b, (name, sha, a) in enumerate(blocks):
                idx = np.arange(len(a["failed"]))
                fail += [(b, i) for i in idx[a["failed"]]]
                right += [(b, i) for i in idx[~a["failed"]]]
            fail, right = np.array(fail), np.array(right)
            n_fail_total = len(fail)
            cap = args.max_failed.get(p) if args.max_failed else None
            if cap and len(fail) > cap:
                rng = np.random.default_rng([SUBSAMPLE_SEED, int(round(p * 1e4)), ord(basis)])
                fail = fail[np.sort(rng.choice(len(fail), cap, replace=False))]
            rng = np.random.default_rng([CONTROL_SEED, int(round(p * 1e4)), ord(basis)])
            n_ctrl = max(len(fail), (args.control_min or {}).get(p, 0))
            ctrl = right[np.sort(rng.choice(len(right), n_ctrl, replace=False))]
            for role, sel in (("F", fail), ("C", ctrl)):
                for b, i in sel:
                    name, sha, a = blocks[b]
                    rows.append((p, basis, role, name, sha, int(i), bool(a["converged"][i]), bool(a["leg1_converged"][i]),
                                 n_fail_total, len(right)))
            print(f"p={p} {basis}: relay-BP failed {n_fail_total} (selected {len(fail)}), right {len(right)}, control {len(ctrl)}")
    cols = list(zip(*rows))
    np.savez_compressed(args.out, p=np.array(cols[0]), basis=np.array(cols[1]), role=np.array(cols[2]),
                        dataset_file=np.array(cols[3]), sha256=np.array(cols[4]), index=np.array(cols[5]),
                        g1_converged=np.array(cols[6]), g1_leg1_converged=np.array(cols[7]),
                        failed_total=np.array(cols[8]), right_total=np.array(cols[9]),
                        control_seed=CONTROL_SEED, subsample_seed=SUBSAMPLE_SEED,
                        max_failed=str(args.max_failed or {}), control_min=str(args.control_min or {}))


PER_SHOT_KEYS = ("p", "basis", "role", "dataset_file", "sha256", "index", "g1_converged", "g1_leg1_converged",
                 "failed_total", "right_total")


def load_selection(paths):
    """One or more selection files, concatenated."""
    parts = []
    for path in paths:
        with np.load(path) as z:
            parts.append({k: z[k] for k in PER_SHOT_KEYS})
    sel = {k: np.concatenate([p[k] for p in parts]) for k in PER_SHOT_KEYS}
    sel["file"] = sel.pop("dataset_file")
    return sel


# ---------------------------------------------------------------- running

def _seeded_pick(take, members, cap, salt, p, basis):
    rng = np.random.default_rng([SUBSAMPLE_SEED + salt, int(round(p * 1e4)), ord(basis)])
    take[members[np.sort(rng.choice(len(members), min(cap, len(members)), replace=False))]] = True


def shots_for(sel, variant, caps, only_p=None):
    """Which selected shots a variant decodes: all, or seeded subsamples where CAPS has an entry."""
    take = np.zeros(len(sel["index"]), bool)
    F = sel["role"] == "F"
    for p in np.unique(sel["p"]):
        if only_p is not None and float(p) not in only_p:
            continue
        for basis in np.unique(sel["basis"]):
            g = (sel["p"] == p) & (sel["basis"] == basis)
            f_cap, c_cap, fu_cap = caps.get((variant, float(p)), (None, None, None))
            pick_f = np.zeros(len(take), bool)
            members = np.flatnonzero(g & F)
            if f_cap is None:
                pick_f[members] = True
            else:
                _seeded_pick(pick_f, members, f_cap, 2, float(p), str(basis))
            if fu_cap is not None:   # at most fu_cap of the chosen failures that G1 left unconverged
                unconv = np.flatnonzero(pick_f & ~sel["g1_converged"])
                pick_f[unconv] = False
                _seeded_pick(pick_f, unconv, fu_cap, 1, float(p), str(basis))
            members = np.flatnonzero(g & ~F)
            if c_cap is None:
                take[members] = True
            else:
                _seeded_pick(take, members, c_cap, 3, float(p), str(basis))
            take |= pick_f
    return take


FULL_BUDGET_FLOOR = 296 * 60   # a 300-leg run that never found 5 solutions has at least 296 legs of 60 iterations


def joint600_shots(sel, paths, caps, cache, only_p=None):
    """Where joint600 must run. Its first 300 legs are joint's (same seed, fresh decoder), so it can
    differ only where joint ran all 300 legs, which takes at least FULL_BUDGET_FLOOR iterations.
    Elsewhere joint600 equals joint by construction. Two seeded shots per p and basis below the
    floor are decoded anyway, to check that."""
    take = shots_for(sel, "joint600", caps, only_p)
    joint = gather(sel, paths, "joint", cache, caps)
    key = lambda j: (float(sel["p"][j]), str(sel["basis"][j]), str(sel["file"][j]), int(sel["index"][j]))
    have = np.array([key(j) in joint for j in range(len(take))])
    need = np.array([bool(take[j] and have[j] and joint[key(j)]["iterations"] >= FULL_BUDGET_FLOOR)
                     for j in range(len(take))])
    check = np.zeros(len(take), bool)
    for p in np.unique(sel["p"]):
        for basis in np.unique(sel["basis"]):
            g = (sel["p"] == p) & (sel["basis"] == basis)
            _seeded_pick(check, np.flatnonzero(g & take & have & ~need), 2, 4, float(p), str(basis))
    return need | check, take & have


def build_tasks(sel, paths, variants, caps, only_p=None, cache=None):
    """Tasks grouped by dataset file; the costliest shots (unconverged failures, high p) first."""
    tasks = []
    for v in variants:
        take = joint600_shots(sel, paths, caps, cache, only_p)[0] if v == "joint600" else shots_for(sel, v, caps, only_p)
        keys = sorted({(float(p), str(f)) for p, f, t in zip(sel["p"], sel["file"], take) if t}, key=lambda k: -k[0])
        for p, f in keys:
            for hard in (True, False):
                m = take & (sel["p"] == p) & (sel["file"] == f) & ((sel["role"] == "F") & ~sel["g1_converged"] if hard
                                                                   else ~((sel["role"] == "F") & ~sel["g1_converged"]))
                idx = np.unique(sel["index"][m])
                sha = str(sel["sha256"][np.flatnonzero(m)[0]]) if m.any() else None
                k = SHOTS_PER_TASK[v]
                for a in range(0, len(idx), k):
                    tasks.append(Task(f"g2_{v}", VARIANTS[v], paths[(p, f)], sha, idx[a:a + k]))
    return tasks


def dataset_paths(files):
    out = {}
    for f in files:
        with np.load(f) as z:
            out[(float(z["p"]), Path(f).name)] = str(f)
    return out


def cmd_run(args):
    log = Log(args.log)
    sel = load_selection(args.selection)
    paths = dataset_paths(args.data)
    for (p, name), path in paths.items():
        want = {str(s) for s, pp, f in zip(sel["sha256"], sel["p"], sel["file"]) if pp == p and f == name}
        if want and want != {sha256(path)}:
            raise SystemExit(f"{path}: sha256 differs from the selection's")
    variants = args.only or list(VARIANTS)
    if "joint600" in variants and "joint" in variants:
        variants = [v for v in variants if v != "joint600"]
        log("joint600 needs joint's results: running joint first")
        run_tasks(build_tasks(sel, paths, variants, parse_caps(args.caps), set(args.p) if args.p else None),
                  args.processes, args.cache, log)
        variants = ["joint600"]
    tasks = build_tasks(sel, paths, variants, parse_caps(args.caps), set(args.p) if args.p else None, args.cache)
    log("G2 run:", machine(args.processes), "variants", variants)
    run_tasks(tasks, args.processes, args.cache, log)
    log("G2 run done")


# ---------------------------------------------------------------- analysis

def gather(sel, paths, variant, cache, caps):
    """{(p, basis, file, index): per-shot output of the variant}."""
    out = {}
    for t in build_tasks(sel, paths, [variant], caps, cache=cache):
        if not task_file(cache, t).exists():
            continue
        d = collect([t], cache)
        with np.load(t.path) as z:
            p, basis = float(z["p"]), str(z["basis"])
        name = Path(t.path).name
        for j, i in enumerate(d["indices"]):
            out[(p, basis, name, int(i))] = {k: v[..., j] for k, v in d.items() if k != "indices"}
    return out


def truth_of(pershot_dir, p):
    """{(basis, file, index): (truth bits packed, G1 pred packed, G1 wllr)} from G1's per-shot files."""
    out = {}
    for basis, blocks in load_pershot(pershot_dir, p).items():
        for name, sha, a in blocks:
            out[(basis, name)] = a
    return out


def two_by_two(sel_g, wrong_other, n_right_total, n_fail_total):
    """relay-BP (G1) against another decoder on F and C, plus the control's break rate scaled to all shots."""
    relay_wrong = sel_g["role"] == "F"
    t = paired(relay_wrong, wrong_other)   # a = relay-BP, b = the other decoder
    nF, nC = int(relay_wrong.sum()), int((~relay_wrong).sum())
    fixes = int((relay_wrong & ~wrong_other).sum())
    breaks = int((~relay_wrong & wrong_other).sum())
    lo, hi = clopper_pearson(breaks, nC) if nC else (float("nan"), float("nan"))
    fix_rate = fixes / nF if nF else float("nan")
    flo, fhi = clopper_pearson(fixes, nF) if nF else (float("nan"), float("nan"))
    est_fail = n_fail_total * (1 - fix_rate) + n_right_total * breaks / nC
    est_lo = n_fail_total * (1 - fhi) + n_right_total * lo
    est_hi = n_fail_total * (1 - flo) + n_right_total * hi
    return dict(t, F=nF, C=nC, fixes=fixes, fix_rate=fix_rate, fix_rate_ci=(flo, fhi), breaks=breaks,
                break_rate=breaks / nC if nC else float("nan"), break_rate_ci=(lo, hi),
                control_fraction=nC / n_right_total, relay_failures_total=n_fail_total, relay_right_total=n_right_total,
                expected_breaks_all=n_right_total * breaks / nC if nC else float("nan"),
                expected_breaks_all_ci=(n_right_total * lo, n_right_total * hi),
                other_failures_all_estimate=est_fail, other_failures_all_ci=(est_lo, est_hi))


def candidate_picks(c, truth):
    """Picks among one shot's 16 candidates. c holds arrays with a seed axis."""
    conv = c["converged"].astype(bool)
    if not conv.any():
        return dict(n_converged=0, n_distinct=0, n_classes=0, any_right=False, lowest_right=False, vote_right=False,
                    lowest_tie=False)
    hashes, first = np.unique(c["ehash"][conv], return_index=True)
    preds, weights = c["pred"][conv][first], c["wllr"][conv][first]
    order = np.argsort(weights, kind="stable")
    lowest = preds[order[0]]
    tie = len(order) > 1 and np.isclose(weights[order[0]], weights[order[1]]) and preds[order[0]] != preds[order[1]]
    classes = np.unique(preds)
    scores = []
    for cl in classes:
        w = -weights[preds == cl]
        scores.append(w.max() + np.log(np.exp(w - w.max()).sum()))   # log of the sum of exp(-weight)
    vote = classes[int(np.argmax(scores))]
    return dict(n_converged=int(conv.sum()), n_distinct=len(hashes), n_classes=len(classes),
                any_right=bool((preds == truth).any()), lowest_right=bool(lowest == truth),
                vote_right=bool(vote == truth), lowest_tie=bool(tie))


def cmd_summarize(args):
    sel = load_selection(args.selection)
    paths = dataset_paths(args.data)
    out = dict(ms_label="ms per shot inside the decode call, a fresh decoder per shot (its construction is not timed), "
                        "8 processes side by side on luminaty (AMD Ryzen 7 7730U), nice 15, cores 8-15",
               selection=args.selection,
               control_seed=CONTROL_SEED, subsample_seed=SUBSAMPLE_SEED, bposd_hi_order=BPOSD_HI_ORDER,
               tables={}, candidates={})
    flat = []
    caps = parse_caps(args.caps)
    g1 = {p: truth_of(args.pershot, p) for p in np.unique(sel["p"])}
    for v in VARIANTS:
        res = gather(sel, paths, v, args.cache, caps)
        if not res:
            continue
        planned = shots_for(sel, v, caps)
        if v == "joint600":   # fill in joint's result where joint600 equals it by construction
            joint = gather(sel, paths, "joint", args.cache, caps)
            run_keys = set(res)
            same = [bool(np.array_equal(res[k]["ehash"], joint[k]["ehash"]) and res[k]["iterations"] == joint[k]["iterations"])
                    for k in run_keys if k in joint and joint[k]["iterations"] < FULL_BUDGET_FLOOR]
            out["joint600_check"] = dict(run=len(run_keys), below_floor_decoded=len(same), below_floor_identical=int(sum(same)),
                                         floor_iterations=FULL_BUDGET_FLOOR)
            for k, o in joint.items():
                if k not in res and o["iterations"] < FULL_BUDGET_FLOOR:
                    res[k] = o
        for p in np.unique(sel["p"]):
            for basis in np.unique(sel["basis"]):
                g = (sel["p"] == p) & (sel["basis"] == basis)
                if not g.any():
                    continue
                keys = [(float(p), str(basis), str(f), int(i)) for f, i in zip(sel["file"][g], sel["index"][g])]
                have = np.array([k in res for k in keys]) & planned[g]
                if not have.any():
                    continue
                sub = {k: sel[k][g][have] for k in ("role", "g1_converged", "g1_leg1_converged", "file", "index")}
                n_fail_total, n_right_total = int(sel["failed_total"][g][0]), int(sel["right_total"][g][0])
                tag = f"p{p}_{basis}"
                if v == "cand16":
                    out["candidates"][tag] = candidate_summary(sub, [res[k] for k, h in zip(keys, have) if h],
                                                               g1[p], basis, n_fail_total, n_right_total)
                    continue
                outs = [res[k] for k, h in zip(keys, have) if h]
                wrong = np.array([bool(o["failed"]) for o in outs])
                tab = two_by_two(sub, wrong, n_right_total, n_fail_total)
                its = np.array([int(o["iterations"]) for o in outs])
                ms = np.array([float(o["seconds"]) * 1e3 for o in outs])
                for role in ("F", "C"):
                    m = sub["role"] == role
                    if m.any():
                        tab[f"iterations_{role}"] = iteration_stats(its[m])
                        tab[f"ms_{role}"] = dict(mean=float(ms[m].mean()), median=float(np.median(ms[m])),
                                                 max=float(ms[m].max()))
                tab["unconverged_F"] = int(sum(1 for o, r in zip(outs, sub["role"]) if r == "F" and not o["converged"]))
                tab["unconverged_C"] = int(sum(1 for o, r in zip(outs, sub["role"]) if r == "C" and not o["converged"]))
                F = sub["role"] == "F"
                tab["fixes_of_unconverged"] = int((F & ~sub["g1_converged"] & ~wrong).sum())
                tab["F_unconverged"] = int((F & ~sub["g1_converged"]).sum())
                tab["fixes_of_converged_wrong"] = int((F & sub["g1_converged"] & ~wrong).sum())
                tab["F_converged_wrong"] = int((F & sub["g1_converged"]).sum())
                tab["complete"] = bool(have.sum() == planned[g].sum())
                out["tables"].setdefault(v, {})[tag] = tab
                flat.append(dict(variant=v, p=float(p), basis=str(basis),
                                 **{k: val for k, val in tab.items() if not isinstance(val, tuple)}))
    write_json(Path(args.out) / "g2_summary.json", out)
    if flat:
        keys = list(dict.fromkeys(k for f in flat for k in f))
        with open(Path(args.out) / "g2_tables.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(flat)
    print_tables(out)


def candidate_summary(sub, results, g1p, basis, n_fail_total, n_right_total):
    rows = []
    for j, c in enumerate(results):
        a = g1p[(basis, str(sub["file"][j]))]
        i = int(sub["index"][j])
        truth = a["truth"][i]
        rows.append(dict(role=str(sub["role"][j]), g1_converged=bool(sub["g1_converged"][j]),
                         leg1=bool(sub["g1_leg1_converged"][j]), **candidate_picks(c, truth)))
    summary = {}
    for name, keep in (("converged_wrong", lambda r: r["role"] == "F" and r["g1_converged"]),
                       ("unconverged_sample", lambda r: r["role"] == "F" and not r["g1_converged"]),
                       ("control", lambda r: r["role"] == "C")):
        rs = [r for r in rows if keep(r)]
        if not rs:
            continue
        block = dict(shots=len(rs))
        for key in ("any_right", "lowest_right", "vote_right", "lowest_tie"):
            block[key] = int(sum(r[key] for r in rs))
        block["none_converged"] = int(sum(r["n_converged"] == 0 for r in rs))
        block["one_distinct"] = int(sum(r["n_distinct"] == 1 for r in rs))
        block["distinct_median"] = float(np.median([r["n_distinct"] for r in rs]))
        block["classes_median"] = float(np.median([r["n_classes"] for r in rs]))
        for leg in (True, False):
            sub_rs = [r for r in rs if r["leg1"] == leg]
            block[f"leg1_{'converged' if leg else 'unconverged'}"] = dict(
                shots=len(sub_rs), any_right=int(sum(r["any_right"] for r in sub_rs)),
                lowest_right=int(sum(r["lowest_right"] for r in sub_rs)), vote_right=int(sum(r["vote_right"] for r in sub_rs)),
                one_distinct=int(sum(r["n_distinct"] == 1 for r in sub_rs)))
        summary[name] = block
    summary["relay_failures_total"], summary["relay_right_total"] = n_fail_total, n_right_total
    if "control" in summary:
        c = summary["control"]
        for key in ("lowest_right", "vote_right"):
            broken = c["shots"] - c[key]
            lo, hi = clopper_pearson(broken, c["shots"])
            c[f"{key.split('_')[0]}_breaks"] = broken
            c[f"{key.split('_')[0]}_break_rate_ci"] = (lo, hi)
    return summary


def print_tables(out):
    for v, tabs in out["tables"].items():
        for tag, t in tabs.items():
            print(f"{v:9s} {tag:9s} F {t['F']:4d} fixes {t['fixes']:4d} (unconv {t['fixes_of_unconverged']}/{t['F_unconverged']}, "
                  f"conv-wrong {t['fixes_of_converged_wrong']}/{t['F_converged_wrong']})  C {t['C']:4d} breaks {t['breaks']:3d}  "
                  f"paired p {t['mcnemar_p']:.2g}  est. failures on all {t['other_failures_all_estimate']:.0f} "
                  f"[{t['other_failures_all_ci'][0]:.0f}, {t['other_failures_all_ci'][1]:.0f}] vs relay {t['relay_failures_total']}"
                  f"{'' if t['complete'] else '  (INCOMPLETE)'}")
    for tag, c in out["candidates"].items():
        print("cand16", tag, {k: v for k, v in c.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--pershot", default="analysis/results/pershot")
    s.add_argument("--p", type=float, nargs="+", required=True)
    s.add_argument("--max-failed", type=lambda t: {float(k): int(v) for k, v in (kv.split(":") for kv in t.split(","))},
                   help="seeded subsample of F, for example 0.005:300")
    s.add_argument("--control-min", type=lambda t: {float(k): int(v) for k, v in (kv.split(":") for kv in t.split(","))},
                   help="at least this many control shots per basis, for example 0.003:300")
    s.add_argument("--out", required=True)
    for name in ("run", "summarize"):
        r = sub.add_parser(name)
        r.add_argument("--selection", nargs="+", required=True)
        r.add_argument("--data", nargs="+", required=True)
        r.add_argument("--cache", required=True)
        r.add_argument("--caps", default="", help="seeded subsamples, see above")
    sub.choices["run"].add_argument("--processes", type=int, default=8)
    sub.choices["run"].add_argument("--log", required=True)
    sub.choices["run"].add_argument("--only", nargs="+", choices=list(VARIANTS))
    sub.choices["run"].add_argument("--p", type=float, nargs="+", help="run only these p")
    sub.choices["summarize"].add_argument("--pershot", default="analysis/results/pershot")
    sub.choices["summarize"].add_argument("--out", default="analysis/results")
    args = ap.parse_args()
    {"select": cmd_select, "run": cmd_run, "summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
