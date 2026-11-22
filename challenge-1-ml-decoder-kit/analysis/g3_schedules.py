"""G3: where relay-BP's time goes, and which schedules are faster at no measurable cost.

  python -m analysis.g3_schedules tail --pershot analysis/results/pershot --out analysis/results
  python -m analysis.g3_schedules run --data FILE... --cache DIR --processes 8 --log LOG [--schedules A/B/C ...] [--max-shots N]
  python -m analysis.g3_schedules summarize --data FILE... --cache DIR --out analysis/results --tag NAME [--schedules ...] [--max-shots N]

tail: from G1's per-shot iterations, the share of all iterations spent in the slowest 1% and 10% of shots.
run / summarize: relay-BP schedules first leg / later legs / stop after k solutions, every other
setting the kit's (300 legs, seed 0), each 500-shot chunk in file order on a fresh decoder, exactly
as in G1. The kit's schedule 80 / 60 / 5 is G1's own run (the same tasks), so every schedule is
paired with the kit's on the same shots, decoded the same way.
"""
import argparse
import csv
import itertools
from pathlib import Path

import numpy as np

from analysis.common import (Log, chunk_tasks, collect, machine, paired, run_tasks, sha256,
                             wilson, write_json)
from analysis.g1_relay_failures import KIT

GRID = [(pre, leg, stop) for stop, pre, leg in itertools.product((1, 2, 3, 5), (40, 80), (30, 60))]
KIT_SCHEDULE = (80, 60, 5)


def parse_schedule(text):
    pre, leg, stop = (int(x) for x in text.split("/"))
    return pre, leg, stop


def spec_of(schedule):
    if schedule == KIT_SCHEDULE:
        return "g1_kit", KIT
    pre, leg, stop = schedule
    return f"g3_{pre}_{leg}_{stop}", {"kind": "relay", "kwargs": {"pre_iter": pre, "set_max_iter": leg, "stop_nconv": stop},
                                      "mode": "chunk", "seeds": [0]}


def dataset_infos(files):
    out = []
    for f in files:
        with np.load(f) as z:
            out.append(dict(path=str(f), file=Path(f).name, sha256=sha256(f), basis=str(z["basis"]), p=float(z["p"]),
                            shots=int(z["detectors"].shape[0]), observables=None))
    return out


def tasks_for(schedule, info, max_shots):
    run, spec = spec_of(schedule)
    n = info["shots"] if max_shots is None else min(max_shots, info["shots"])
    return chunk_tasks(run, spec, info["path"], info["sha256"], n)


def cmd_tail(args):
    rows = []
    for f in sorted(Path(args.pershot).glob("g1_p*.npz")):
        with np.load(f) as z:
            rows.append((float(f.stem.split("_")[1][1:]), f.stem.split("_")[2], z["iterations"].astype(np.int64),
                         z["failed"], z["converged"]))
    groups = {}
    for p, basis, its, failed, conv in rows:
        g = groups.setdefault((p, basis), [[], [], []])
        g[0].append(its), g[1].append(failed), g[2].append(conv)
    out = []
    for (p, basis), (its, failed, conv) in sorted(groups.items()):
        its, failed, conv = np.concatenate(its), np.concatenate(failed), np.concatenate(conv)
        s = np.sort(its)[::-1]
        total = s.sum()
        n = len(s)
        row = dict(p=p, basis=basis, shots=n, total_iterations=int(total), mean=float(its.mean()),
                   share_slowest_1pct=float(s[: max(1, n // 100)].sum() / total),
                   share_slowest_10pct=float(s[: max(1, n // 10)].sum() / total),
                   share_unconverged_shots=float(its[~conv].sum() / total), unconverged_shots=int((~conv).sum()),
                   share_failed_shots=float(its[failed].sum() / total), failed_shots=int(failed.sum()),
                   share_at_full_budget=float(its[its >= 80 + 300 * 60].sum() / total),
                   shots_at_full_budget=int((its >= 80 + 300 * 60).sum()))
        out.append(row)
        print(f"| {p} | {basis} | {n:,} | {row['mean']:.0f} | {row['share_slowest_1pct']:.1%} | {row['share_slowest_10pct']:.1%} | "
              f"{row['share_unconverged_shots']:.1%} ({row['unconverged_shots']}) | {row['share_failed_shots']:.1%} ({row['failed_shots']}) |")
    write_json(Path(args.out) / "g3_tail.json", out)


def schedules_from(args):
    return [parse_schedule(s) for s in args.schedules] if args.schedules else GRID


def cmd_run(args):
    log = Log(args.log)
    infos = dataset_infos(args.data)
    scheds = schedules_from(args)
    log("G3 run:", machine(args.processes), "schedules", scheds)
    # costliest first: stop after 5 before stop after 1
    tasks = [t for s in sorted(scheds, key=lambda s: (-s[2], -s[0], -s[1])) for info in infos
             for t in tasks_for(s, info, args.max_shots)]
    run_tasks(tasks, args.processes, args.cache, log)
    log("G3 run done")


def cmd_summarize(args):
    infos = dataset_infos(args.data)
    scheds = schedules_from(args)
    if KIT_SCHEDULE not in scheds:
        scheds = [KIT_SCHEDULE] + scheds
    per = {}   # (schedule, basis) -> arrays
    p_values = set()
    for info in infos:
        p_values.add(info["p"])
        for s in scheds:
            per[(s, info["basis"])] = collect(tasks_for(s, info, args.max_shots), args.cache)
    rows = []
    label = f"ms per shot inside decode_detailed, {args.processes} processes side by side on {args.machine}"
    for s in scheds:
        row = dict(schedule="/".join(map(str, s)), first_leg=s[0], later_legs=s[1], stop_after=s[2],
                   is_kit=s == KIT_SCHEDULE)
        pair_fail_kit, pair_fail = None, None
        for basis in sorted({i["basis"] for i in infos}):
            d, k = per[(s, basis)], per[(KIT_SCHEDULE, basis)]
            assert np.array_equal(d["indices"], k["indices"])
            n = len(d["failed"])
            t = paired(k["failed"], d["failed"])   # a = kit, b = this schedule
            ms = d["seconds"].astype(float) * 1e3
            row.update({f"{basis}_shots": n, f"{basis}_failures": int(d["failed"].sum()),
                        f"{basis}_unconverged": int((~d["converged"]).sum()),
                        f"{basis}_failed_unconverged": int((d["failed"] & ~d["converged"]).sum()),
                        f"{basis}_it_mean": float(d["iterations"].mean()),
                        f"{basis}_it_median": float(np.median(d["iterations"])),
                        f"{basis}_it_p99": float(np.percentile(d["iterations"], 99)),
                        f"{basis}_it_max": int(d["iterations"].max()),
                        f"{basis}_ms_mean": float(ms.mean()), f"{basis}_ms_median": float(np.median(ms)),
                        f"{basis}_ms_p99": float(np.percentile(ms, 99)),
                        f"{basis}_both_wrong": t["both_wrong"], f"{basis}_only_kit_wrong": t["only_a_wrong"],
                        f"{basis}_only_this_wrong": t["only_b_wrong"], f"{basis}_paired_p": t["mcnemar_p"]})
            pair_fail = d["failed"] if pair_fail is None else (pair_fail | d["failed"])
            pair_fail_kit = k["failed"] if pair_fail_kit is None else (pair_fail_kit | k["failed"])
        if len({i["basis"] for i in infos}) == 2:
            t = paired(pair_fail_kit, pair_fail)
            lo, hi = wilson(int(pair_fail.sum()), len(pair_fail))
            row.update(pair_shots=len(pair_fail), pair_failures=int(pair_fail.sum()), pair_ci=(lo, hi),
                       pair_only_kit_wrong=t["only_a_wrong"], pair_only_this_wrong=t["only_b_wrong"],
                       pair_paired_p=t["mcnemar_p"],
                       it_mean_both=float(np.mean([row[f"{b}_it_mean"] for b in "XZ"])),
                       speedup_vs_kit_mean_iterations=None)
        rows.append(row)
    if "it_mean_both" in rows[0]:
        kit_mean = next(r for r in rows if r["is_kit"])["it_mean_both"]
        for r in rows:
            r["speedup_vs_kit_mean_iterations"] = kit_mean / r["it_mean_both"]
    write_json(Path(args.out) / f"g3_{args.tag}.json", dict(p=sorted(p_values), datasets=infos, ms_label=label,
                                                           max_shots=args.max_shots, rows=rows))
    keys = list(dict.fromkeys(k for r in rows for k in r if not isinstance(r[k], tuple)))
    with open(Path(args.out) / f"g3_{args.tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print_table(rows)


def print_table(rows):
    print("| schedule | mean / median / p99 iterations X | Z | failures [X Z] | unconverged [X Z] | "
          "paired vs kit, only kit wrong / only this wrong [X Z] | pairs failed | speed-up (mean iterations) | ms mean [X Z] |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: r.get("speedup_vs_kit_mean_iterations") or 0, reverse=True):
        cells = [r["schedule"] + (" (kit)" if r["is_kit"] else "")]
        for b in "XZ":
            if f"{b}_shots" in r:
                cells.append(f"{r[f'{b}_it_mean']:.0f} / {r[f'{b}_it_median']:.0f} / {r[f'{b}_it_p99']:.0f}")
        cells.append("[" + " ".join(str(r[f"{b}_failures"]) for b in "XZ" if f"{b}_shots" in r) + "]")
        cells.append("[" + " ".join(str(r[f"{b}_unconverged"]) for b in "XZ" if f"{b}_shots" in r) + "]")
        cells.append("[" + " ".join(f"{r[f'{b}_only_kit_wrong']}/{r[f'{b}_only_this_wrong']} (p {r[f'{b}_paired_p']:.2g})"
                                    for b in "XZ" if f"{b}_shots" in r) + "]")
        cells.append(f"{r.get('pair_failures', '')} of {r.get('pair_shots', '')}")
        cells.append(f"{r['speedup_vs_kit_mean_iterations']:.2f}" if r.get("speedup_vs_kit_mean_iterations") else "")
        cells.append("[" + " ".join(f"{r[f'{b}_ms_mean']:.0f}" for b in "XZ" if f"{b}_shots" in r) + "]")
        print("| " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tail")
    t.add_argument("--pershot", default="analysis/results/pershot")
    t.add_argument("--out", default="analysis/results")
    for name in ("run", "summarize"):
        s = sub.add_parser(name)
        s.add_argument("--data", nargs="+", required=True)
        s.add_argument("--cache", required=True)
        s.add_argument("--schedules", nargs="+", help="first/later/stop, for example 40/30/1; default the 16-schedule grid")
        s.add_argument("--max-shots", type=int)
        s.add_argument("--processes", type=int, default=8)
    sub.choices["run"].add_argument("--log", required=True)
    sub.choices["summarize"].add_argument("--out", default="analysis/results")
    sub.choices["summarize"].add_argument("--tag", required=True)
    sub.choices["summarize"].add_argument("--machine", default="luminaty (AMD Ryzen 7 7730U), nice 15, cores 8-15 (4 physical cores, 2 threads each)")
    args = ap.parse_args()
    {"tail": cmd_tail, "run": cmd_run, "summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
