"""G1: how relay-BP fails. The kit's baseline relay-BP on every shot of the given datasets.

  python -m analysis.g1_relay_failures run --data FILE... --cache DIR --processes 12 --log LOG
  python -m analysis.g1_relay_failures summarize --data FILE... --cache DIR --out analysis/results

Decoder: decoders.relay_decoder(H, priors) with RELAY_SETTINGS unchanged (80 / 60 / stop at 5,
300 legs), seed 0 (the constructor's default, which the kit's baseline uses), through
decode_detailed. Each 500-shot chunk, in file order, is decoded in order on a fresh decoder.
A second pass decodes the same chunks with the first leg alone (num_sets=0, deterministic):
it says whether the first 80 iterations found a solution.
Scoring is the kit's (decoders.score): a shot fails when any of the 12 observables is wrong;
an unconverged shot is scored on the correction relay-BP returned.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from analysis.common import (Log, chunk_tasks, collect, iteration_stats, machine, open_dataset, pack_bits,
                             refuse_held_out, run_tasks, sha256, task_file, wilson, write_json)

KIT = {"kind": "relay", "kwargs": {}, "mode": "chunk", "seeds": [0]}
LEG1 = {"kind": "relay", "kwargs": {"num_sets": 0}, "mode": "chunk", "seeds": [0]}
KIT_REFERENCE_P004 = {"X": (45, 10_000), "Z": (60, 10_000)}   # the kit README's relay-BP row at p = 0.004


def dataset_info(path):
    refuse_held_out(path)
    with np.load(path) as z:
        return dict(path=str(path), file=Path(path).name, sha256=sha256(path), basis=str(z["basis"]),
                    p=float(z["p"]), seed=int(z["seed"]), shots=int(z["detectors"].shape[0]))


def split_name(info):
    return info["file"].split("_")[0]   # train, validation or explore


def tasks_for(info, max_shots=None):
    n = info["shots"] if max_shots is None else min(info["shots"], max_shots)
    kit = chunk_tasks("g1_kit", KIT, info["path"], info["sha256"], n)
    leg1 = chunk_tasks("g1_leg1", LEG1, info["path"], info["sha256"], n)
    return kit, leg1


def cmd_run(args):
    log = Log(args.log)
    infos = [dataset_info(f) for f in args.data]
    log("G1 run:", machine(args.processes))
    tasks = []
    for info in infos:
        log(f"dataset {info['file']} p={info['p']} {info['basis']} seed {info['seed']} shots {info['shots']} sha256 {info['sha256']}")
        kit, leg1 = tasks_for(info, args.max_shots)
        tasks += kit + leg1
    run_tasks(tasks, args.processes, args.cache, log)
    log("G1 run done")


def pershot(info, cache, max_shots=None):
    kit_tasks, leg1_tasks = tasks_for(info, max_shots)
    # use the finished prefix of chunks only, so a partial run is still a block of shots in file order
    prefix = 0
    while prefix < len(kit_tasks) and task_file(cache, kit_tasks[prefix]).exists() \
            and task_file(cache, leg1_tasks[prefix]).exists():
        prefix += 1
    if prefix == 0:
        return None
    kit = collect(kit_tasks[:prefix], cache)
    leg1 = collect(leg1_tasks[:prefix], cache)
    ds = open_dataset(info["path"])
    idx = kit["indices"]
    assert np.array_equal(idx, leg1["indices"]) and np.array_equal(idx, np.arange(len(idx)))
    out = dict(kit)
    out.update({f"leg1_{k}": v for k, v in leg1.items() if k != "indices"})
    out["truth"] = pack_bits(ds.obs[idx])
    return out


def summarize_block(d, seconds_label):
    n = len(d["failed"])
    failed, conv = d["failed"], d["converged"]
    k = int(failed.sum())
    lo, hi = wilson(k, n)
    leg1_conv = d["leg1_converged"]
    same_class = leg1_conv & (d["leg1_pred"] == d["pred"])
    same_corr = leg1_conv & (d["leg1_ehash"] == d["ehash"])
    ms = d["seconds"].astype(float) * 1e3
    row = dict(
        shots=n, failures=k, ler=k / n, ler_lo=lo, ler_hi=hi,
        failed_unconverged=int((failed & ~conv).sum()), failed_converged_wrong=int((failed & conv).sum()),
        unconverged=int((~conv).sum()), unconverged_but_right=int((~conv & ~failed).sum()),
        converged_but_invalid=int((conv & ~d["valid"]).sum()),
        iterations=iteration_stats(d["iterations"]),
        iterations_failed=iteration_stats(d["iterations"][failed]) if k else None,
        leg1_converged=int(leg1_conv.sum()), leg1_converged_share=float(leg1_conv.mean()),
        leg1_converged_share_ci=wilson(int(leg1_conv.sum()), n),
        leg1_alone_failures=int(d["leg1_failed"].sum()),
        leg1_conv_kit_same_class=int(same_class.sum()), leg1_conv_kit_same_correction=int(same_corr.sum()),
        leg1_conv_kit_failed=int((leg1_conv & failed).sum()),
        leg1_unconv_kit_failed=int((~leg1_conv & failed).sum()),
        leg1_conv_leg1_wrong=int((leg1_conv & d["leg1_failed"]).sum()),
        # failures if a decoder stopped at the first leg's answer whenever that leg converged, else ran the kit's
        stop_at_leg1_else_kit_failures=int(((leg1_conv & d["leg1_failed"]) | (~leg1_conv & failed)).sum()),
        failed_converged_at_full_budget=int((failed & conv & (d["iterations"] >= 296 * 60)).sum()),
        weight_count_median_right=float(np.median(d["wcount"][~failed])),
        weight_llr_median_right=float(np.median(d["wllr"][~failed])),
        weight_count_median_failed_conv=float(np.median(d["wcount"][failed & conv])) if (failed & conv).any() else None,
        weight_llr_median_failed_conv=float(np.median(d["wllr"][failed & conv])) if (failed & conv).any() else None,
        ms_mean=float(ms.mean()), ms_median=float(np.median(ms)), ms_p99=float(np.percentile(ms, 99)),
        ms_label=seconds_label,
    )
    return row


COMMIT_FIELDS = ("failed", "converged", "iterations", "pred", "truth", "wcount", "wllr", "valid",
                 "leg1_converged", "leg1_iterations", "leg1_failed", "leg1_pred")


def cmd_summarize(args):
    out = Path(args.out)
    (out / "pershot").mkdir(parents=True, exist_ok=True)
    scratch = Path(args.cache) / "pershot"
    scratch.mkdir(parents=True, exist_ok=True)
    label = f"ms per shot inside decode_detailed, {args.processes} processes side by side on {args.machine}"
    groups, rows, files = {}, [], []
    for f in args.data:
        info = dataset_info(f)
        d = pershot(info, args.cache, args.max_shots)
        if d is None:
            print(f"{info['file']} p={info['p']}: no finished chunk yet")
            continue
        tag = f"g1_p{info['p']}_{info['basis']}_{split_name(info)}"
        np.savez_compressed(scratch / f"{tag}.npz", sha256=info["sha256"], **d)
        small = {k: d[k] for k in COMMIT_FIELDS}
        small["wllr"] = small["wllr"].astype(np.float32)
        small["ms"] = (d["seconds"].astype(np.float64) * 1e3).astype(np.float16)
        np.savez_compressed(out / "pershot" / f"{tag}.npz", sha256=info["sha256"], dataset_file=info["file"], **small)
        files.append(dict(info, shots_decoded=len(d["failed"]), pershot=f"pershot/{tag}.npz",
                          pershot_full_scratch=str(scratch / f"{tag}.npz")))
        row = dict(p=info["p"], basis=info["basis"], block=info["file"], **summarize_block(d, label))
        rows.append(row)
        groups.setdefault((info["p"], info["basis"]), []).append(d)
    group_rows = []
    for (p, basis), parts in sorted(groups.items()):
        d = {k: np.concatenate([x[k] for x in parts]) for k in parts[0]}
        group_rows.append(dict(p=p, basis=basis, block="all", **summarize_block(d, label)))
    sanity = []
    for r in rows:
        if r["p"] == 0.004 and r["block"].startswith("validation"):
            k, n = KIT_REFERENCE_P004[r["basis"]]
            sanity.append(dict(basis=r["basis"], ours=(r["failures"], r["shots"]), ours_ci=(r["ler_lo"], r["ler_hi"]),
                               kit_reference=(k, n), kit_rate=k / n,
                               kit_rate_inside_our_interval=bool(r["ler_lo"] <= k / n <= r["ler_hi"]),
                               kit_reference_ci=wilson(k, n)))
    write_json(out / "g1_summary.json", dict(decoder="decoders.relay_decoder(H, priors), RELAY_SETTINGS, seed 0, "
                                                     "fresh decoder per 500-shot chunk in file order",
                                             datasets=files, groups=group_rows, blocks=rows, sanity_p004_validation=sanity))
    flat = []
    for r in group_rows + rows:
        f = {k: v for k, v in r.items() if not isinstance(v, (dict, tuple, list)) and v is not None}
        for key in ("iterations", "iterations_failed"):
            if r.get(key):
                f.update({f"{key}_{s}": v for s, v in r[key].items()})
        flat.append(f)
    keys = sorted({k for f in flat for k in f}, key=lambda k: (k not in ("p", "basis", "block"), k))
    with open(out / "g1_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)
    print_tables(group_rows, rows, sanity)


def print_tables(group_rows, rows, sanity):
    print("| p | basis | block | shots | failures | did not converge | converged wrong | rate per basis [95% Wilson] |"
          " first leg converged | iterations mean / median / p90 / p99 / max |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in group_rows + rows:
        it = r["iterations"]
        print(f"| {r['p']} | {r['basis']} | {r['block']} | {r['shots']:,} | {r['failures']} | {r['failed_unconverged']} | "
              f"{r['failed_converged_wrong']} | {r['ler']:.2e} [{r['ler_lo']:.2e}, {r['ler_hi']:.2e}] | "
              f"{r['leg1_converged_share']:.4f} | {it['mean']:.0f} / {it['median']:.0f} / {it['p90']:.0f} / "
              f"{it['p99']:.0f} / {it['max']:.0f} |")
    for s in sanity:
        print("sanity", s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "summarize"):
        s = sub.add_parser(name)
        s.add_argument("--data", nargs="+", required=True)
        s.add_argument("--cache", required=True, help="folder for the per-chunk outputs (scratch)")
        s.add_argument("--max-shots", type=int, help="decode only the first N shots of each file (whole chunks)")
        s.add_argument("--processes", type=int, default=8)
    sub.choices["run"].add_argument("--log", required=True)
    sub.choices["summarize"].add_argument("--out", default="analysis/results")
    sub.choices["summarize"].add_argument("--machine", default="luminaty (AMD Ryzen 7 7730U), nice 15, cores 8-15 (4 physical cores, 2 threads each)")
    args = ap.parse_args()
    {"run": cmd_run, "summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
