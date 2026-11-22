"""The network in fixed point at several widths, scored on the team's validation shots beside the float network.

Run from the top of the repository, with the kit's environment and torch (CPU build):

  python fpga_network/widths.py --checkpoint CKPT --out DIR --run float          # the float network, as evaluate_stim runs it
  python fpga_network/widths.py --checkpoint CKPT --out DIR --run w16 --processes 4
  python fpga_network/widths.py --checkpoint CKPT --out DIR --table              # the table, from what has run

DIR receives calibration.json (the ranges, from the first 200 shots of data/p0.004/train_X.npz, never from the
scored shots), <run>.npz (per shot, in file order: logits) and <run>.json (settings, saturations, checkpoint and data
hashes). The scored file is data/p0.004/validation_X.npz (5,000 Stim circuit shots, seed 201); test_* and go_* files
are refused. With --tie-break it also reads the laptop's candidates for the same file (hierarchy/README.md:
priortilt/results_8b/selector_validation_X.npz on laptop-artifacts) and counts the tie-break choices that change.

The runs (fixed_point.FixedNet): weights of W bits with one power of two per output channel; intermediate values of
the same width with one power of two per tensor; the residual stream and the small head at 16 bits in every run but
"w8s8"; SiLU as a table; one rounding rule (add half, round down, saturate). Ranges are the largest value seen on
the calibration shots, except the runs ending in "p": the 99.99th percentile, so the rarest 0.01% saturate.
"""
import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "challenge-1-ml-decoder-kit"
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixed_point as fp                                    # noqa: E402

DATA = KIT / "data" / "p0.004" / "validation_X.npz"
CALIBRATION = KIT / "data" / "p0.004" / "train_X.npz"
CALIBRATION_SHOTS = 200
RUNS = {   # name: (weight bits, value bits, stream bits, calibration, shots scored)
    "w16": (16, 16, 16, "max", 5000),
    "w12": (12, 12, 16, "max", 5000),
    "w10": (10, 10, 16, "max", 5000),
    "w12p": (12, 12, 16, "p99.99", 5000),
    "w10p": (10, 10, 16, "p99.99", 5000),
    "w8p": (8, 8, 16, "p99.99", 5000),
    "w6p": (6, 6, 16, "p99.99", 500),
    "w4p": (4, 4, 16, "p99.99", 500),
    "w8": (8, 8, 16, "max", 500),
    "w8s8": (8, 8, 8, "max", 500),
    "w6": (6, 6, 16, "max", 500),
    "w4": (4, 4, 16, "max", 500),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_shots(path, n=None):
    path = Path(path)
    if path.name.startswith(("test_", "go_")):
        raise SystemExit(f"{path} is a held-out file")
    with np.load(path) as d:
        det, obs = d["detectors"], d["observables"]
        assert str(d["basis"]) == "X"
    return (det if n is None else det[:n]), (obs if n is None else obs[:n])


class Percentile(fp.Float):
    """Calibration that records, per tensor, the 99.99th percentile of |value| of each batch (the largest over batches)."""

    def __init__(self, folded, q=0.9999):
        super().__init__(folded)
        self.q, self.pct = q, {}

    def act(self, name, x):
        a = x.abs().flatten().numpy()
        k = min(len(a) - 1, int(len(a) * self.q))
        self.pct[name] = max(self.pct.get(name, 0.0), float(np.partition(a, k)[k]))
        return super().act(name, x)


def calibrate(folded, out, checkpoint_sha, limit=None):
    path = out / "calibration.json"
    shots = limit or CALIBRATION_SHOTS
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("checkpoint_sha256") != checkpoint_sha or saved.get("count") != shots:
            raise SystemExit(f"{path} was made for another checkpoint or shot count: use another --out folder")
        return saved
    det, _ = load_shots(CALIBRATION, shots)
    q = Percentile(folded)
    with torch.no_grad():
        for start in range(0, len(det), 20):
            fp.forward(q, torch.from_numpy(det[start:start + 20]))
    result = {"shots": f"the first {len(det)} of {CALIBRATION.relative_to(ROOT)} (sha256 {sha256(CALIBRATION)})",
              "count": len(det), "checkpoint_sha256": checkpoint_sha, "max": q.ranges, "p99.99": q.pct}
    path.write_text(json.dumps(result, indent=1) + "\n")
    return result


def make_net(folded, ranges, run):
    """The FixedNet of a run. A tensor kept at 16 bits (the stream, the head) always takes the largest value as its
    range; the percentile, where a run asks for it, applies to the narrower tensors only."""
    wbits, abits, sbits, calib, _ = RUNS[run]
    net = fp.FixedNet(folded, wbits, ranges["max"], abits)
    for name in net.width:
        if re.fullmatch(r"s\d+", name):
            net.width[name] = sbits
    for name in net.ranges:
        if calib != "max" and net.width[name] < 16:
            net.ranges[name] = ranges[calib][name]
    net.fa = {name: fp.frac_bits(m, net.width[name]) for name, m in net.ranges.items()}
    return net


_STATE = {}


def _work(args):
    """One worker: one contiguous chunk of shots, one thread. Saves the chunk at once, so a cut-off run keeps it."""
    torch.set_num_threads(1)
    run, lo, hi, path = args
    net = make_net(_STATE["folded"], _STATE["ranges"], run)
    logits = fp.run(net, torch.from_numpy(_STATE["det"][lo:hi]), batch=10)
    np.savez(path, logits=logits, saturated=json.dumps(net.saturated))
    return lo


def shot_set(name, candidates=None, limit=None):
    """(detectors, observables, indices, description) of a scored set.

    "validation": every shot of validation_X.npz. "train_tiebreak": the shots of train_X.npz where the laptop's two
    PriorTilt answers differ (from its selector_train_X.npz), leaving out the calibration shots; these add tie-break
    shots, which validation_X has few of. They are train shots: the network never saw them (it trains on shots it
    draws itself), but the hierarchy's numbers in hierarchy/README.md include them."""
    if name == "validation":
        det, obs = load_shots(DATA, limit)
        return det, obs, np.arange(len(det)), str(DATA.relative_to(ROOT))
    with np.load(candidates) as c:
        differ = (c["predicted_P1"] != c["predicted_P2"]).any(axis=1)
    det, obs = load_shots(CALIBRATION)
    idx = np.flatnonzero(differ)
    idx = idx[idx >= CALIBRATION_SHOTS][:limit]
    return det[idx], obs[idx], idx, f"{CALIBRATION.relative_to(ROOT)}, the {len(idx)} shots where P1 != P2, after shot {CALIBRATION_SHOTS}"


def do_run(checkpoint, out, run, processes, limit=None, which="validation", candidates=None, shots_override=None):
    from neural.train import load_decoder
    model, saved = load_decoder(checkpoint)
    det, obs, idx, about = shot_set(which, candidates, limit)
    tag = run if which == "validation" else f"{run}.{which}"
    info = {"run": run, "set": which, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "step": saved["step"], "data": about, "data_sha256": sha256(DATA if which == "validation" else CALIBRATION),
            "machine": "luminaty (Ryzen 7 7730U)", "processes": processes, "indices": idx.tolist()}
    start = time.time()
    if run == "float":
        from neural.evaluate_stim import decode
        torch.set_num_threads(processes)
        logits, predictions = decode(model, np.ascontiguousarray(det, dtype=np.uint8), torch.device("cpu"))
        assert np.array_equal(predictions, (logits > 0).astype(np.uint8)), "an X model on the kit's logicals predicts logits > 0"
        info |= {"what": "the checkpoint's model in float32, as neural.evaluate_stim.decode runs it", "shots": len(det)}
    else:
        wbits, abits, sbits, calib, shots = RUNS[run]
        shots = min(shots_override or shots, len(det))
        folded = fp.fold(model)
        ranges = calibrate(folded, out, info["checkpoint_sha256"], limit)
        _STATE.update(folded=folded, ranges=ranges, det=det[:shots])
        parts = out / f"{tag}.parts"
        parts.mkdir(exist_ok=True)
        bounds = list(range(0, shots, 50)) + [shots]
        jobs = [(run, a, b, parts / f"{a:05d}-{b:05d}.npz") for a, b in zip(bounds[:-1], bounds[1:])]
        todo = [j for j in jobs if not j[3].exists()]
        print(f"{run}: {len(jobs) - len(todo)} of {len(jobs)} chunks of 50 shots already done", flush=True)
        import multiprocessing as mp
        with mp.get_context("fork").Pool(processes) as pool:
            for done, lo in enumerate(pool.imap_unordered(_work, todo), 1):
                print(f"  chunk from shot {lo} done ({done} of {len(todo)}, {time.time() - start:.0f} s)", flush=True)
        logits, saturated = [], {}
        for _, _, _, path in jobs:
            with np.load(path) as part:
                logits.append(part["logits"])
                for k, v in json.loads(str(part["saturated"])).items():
                    saturated[k] = saturated.get(k, 0) + v
        logits = np.concatenate(logits)
        info |= {"weight_bits": wbits, "value_bits": abits, "stream_bits": sbits, "head_bits": fp.FixedNet.HEAD_BITS,
                 "calibration": calib, "shots": shots, "saturated_values": saturated}
    info["seconds"] = round(time.time() - start, 1)
    np.savez_compressed(out / f"{tag}.npz", logits=logits.astype(np.float64))
    (out / f"{tag}.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "indices"}))


def wilson(k, n, z=1.959964):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (0.0 if k == 0 else (c - h) / (1 + z * z / n)), (1.0 if k == n else (c + h) / (1 + z * z / n))


def table(out, candidates, anchor=None, train_candidates=None):
    flo = np.load(out / "float.npz")["logits"]
    _, obs = load_shots(DATA, len(flo))
    finfo = json.loads((out / "float.json").read_text())
    pool = None
    if candidates:
        from hierarchy.rules import ORDER, agree, most_likely, tie_break
        with np.load(candidates) as c:
            assert np.array_equal(c["truth"][:len(obs)], obs), "the candidates are not of the scored shots"
            predicted = np.stack([c[f"predicted_{k}"][:len(obs)] for k in ORDER])
        differ = ~agree(predicted)
        pool = (predicted, differ, most_likely, tie_break)

    def network_rule(logits, n):
        predicted, differ, most_likely_, tie_break_ = pool
        p = predicted[:, :n]
        choice = tie_break_(p, most_likely_(p, logits[:n]))
        answer = p[choice, np.arange(n)]
        return choice, (answer != obs[:n]).any(axis=1)

    print("# The network in fixed point, on the team's validation shots\n")
    print(f"Made by `python fpga_network/widths.py --checkpoint <checkpoint> --out <folder> --table` from the runs listed "
          f"there. Checkpoint step {finfo['step']} (sha256 `{finfo['checkpoint_sha256'][:16]}...`). Shots: "
          f"`{finfo['data']}` (5,000 Stim circuit shots, X memory, p = 0.004, seed 201; sha256 `{finfo['data_sha256'][:16]}...`). "
          f"Ranges calibrated on the first {CALIBRATION_SHOTS} shots of `train_X.npz`. Machine: {finfo['machine']}, CPU. "
          f"Every number here is **simulated** in software (torch, float64 holding whole numbers exactly), not on an FPGA.\n")
    print("A shot fails when any of its 12 logical bits is wrong (evaluate_stim's number). \"Answers changed\" counts the shots "
          "whose 12 predicted bits differ from the float network's on the same shot. The tie-break columns use the laptop's "
          "three candidates per shot (relay-BP, PriorTilt twice) and the hierarchy's network rule, on the shots where the two "
          "PriorTilt answers differ.\n")
    head = "| run | weights | values | stream | ranges | shots | network fails alone | answers changed vs float | fails: both / float only / fixed only | largest, median logit change | largest float |logit| of a flipped bit |"
    if pool:
        head += " tie-break shots | choices changed | network-rule failures, float -> fixed |"
    print(head)
    print("|" + "---|" * (head.count("|") - 1))
    n = len(obs)
    fail_f = ((flo > 0) != obs).any(axis=1)
    lo, hi = wilson(int(fail_f.sum()), n)
    row = f"| float (evaluate_stim) | float32 | float32 | float32 | | {n:,} | {int(fail_f.sum())} ({fail_f.mean():.2%}, [{lo:.2%}, {hi:.2%}]) | | | | |"
    if pool:
        _, fr = network_rule(flo, n)
        row += f" {int(pool[1].sum())} | | {int(fr.sum())} |"
    print(row)
    footer = []
    if anchor:
        with np.load(anchor) as a:
            lap = a["logits"].astype(np.float64)[:len(flo)]
        footer.append(f"The float row against the laptop's GPU run of the same checkpoint on the same file (`{Path(anchor).name}`): "
                      f"largest |logit difference| {np.abs(lap - flo).max():.2e}, {int(((lap > 0) != (flo > 0)).sum())} of "
                      f"{lap.size:,} bits of different sign, failures {int(((lap > 0) != obs).any(axis=1).sum())} there and "
                      f"{int(fail_f.sum())} here.")
    if train_candidates and (out / "float.train_tiebreak.npz").exists():
        from hierarchy.rules import ORDER, most_likely
        finfo_t = json.loads((out / "float.train_tiebreak.json").read_text())
        idx = np.asarray(finfo_t["indices"])
        _, tobs = load_shots(CALIBRATION)
        tobs = tobs[idx]
        with np.load(train_candidates) as c:
            tpred = np.stack([c[f"predicted_{k}"][idx] for k in ORDER])
        def pick(lg):
            choice = most_likely(tpred, lg)
            return choice, (tpred[choice, np.arange(len(idx))] != tobs).any(axis=1)
        ft = np.load(out / "float.train_tiebreak.npz")["logits"]
        cf, rf = pick(ft)
        parts = [f"float: network-rule failures {int(rf.sum())}"]
        for run in RUNS:
            pth = out / f"{run}.train_tiebreak.npz"
            if pth.exists():
                cq, rq = pick(np.load(pth)["logits"])
                parts.append(f"{run}: choices changed {int((cf != cq).sum())}, failures {int(rq.sum())}")
        footer.append(f"Tie-break on {len(idx)} more shots, of `train_X.npz` where P1 != P2 (after the 200 calibration shots): "
                      + "; ".join(parts) + ".")
    for run, (wb, ab, sb, calib, shots) in RUNS.items():
        if not (out / f"{run}.npz").exists():
            print(f"| {run} | {wb} | {ab} | {sb} | {calib} | not run | | | | | |" + (" | | |" if pool else ""))
            continue
        lq = np.load(out / f"{run}.npz")["logits"]
        info = json.loads((out / f"{run}.json").read_text())
        m = len(lq)
        fq = ((lq > 0) != obs[:m]).any(axis=1)
        ff = fail_f[:m]
        changed = ((lq > 0) != (flo[:m] > 0)).any(axis=1)
        d = np.abs(lq - flo[:m])
        flipped = (lq > 0) != (flo[:m] > 0)
        worst_flip = f"{np.abs(flo[:m][flipped]).max():.3f}" if flipped.any() else "none"
        lo, hi = wilson(int(fq.sum()), m)
        row = (f"| {run} | {wb} | {ab} | {sb} | {'largest' if calib == 'max' else '99.99th pct'} | {m:,} | "
               f"{int(fq.sum())} ({fq.mean():.2%}, [{lo:.2%}, {hi:.2%}]) | {int(changed.sum())} | "
               f"{int((ff & fq).sum())} / {int((ff & ~fq).sum())} / {int((~ff & fq).sum())} | {d.max():.3f}, {np.median(d):.4f} | {worst_flip} |")
        if pool:
            cf, rf = network_rule(flo, m)
            cq, rq = network_rule(lq, m)
            row += f" {int(pool[1][:m].sum())} | {int((cf != cq).sum())} | {int(rf.sum())} -> {int(rq.sum())} |"
        print(row)
        if info.get("saturated_values"):
            print(f"|  | saturated values: {sum(info['saturated_values'].values()):,} ({', '.join(f'{k} {v}' for k, v in sorted(info['saturated_values'].items(), key=lambda kv: -kv[1])[:4])}) | | | | | | | | | |" + (" | | |" if pool else ""))
    _print_footer(footer)


def _print_footer(lines):
    for line in lines:
        print("\n" + line)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run", choices=["float", *RUNS])
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--limit", type=int, help="smoke test only: score and calibrate on this many shots")
    parser.add_argument("--shots", type=int, help="score the first N shots of the set (chunks are kept: a later, larger "
                                                  "N reuses them)")
    parser.add_argument("--tie-break", help="the laptop's selector_validation_X.npz")
    parser.add_argument("--tie-break-train", help="the laptop's selector_train_X.npz (for --set train_tiebreak)")
    parser.add_argument("--set", choices=["validation", "train_tiebreak"], default="validation")
    parser.add_argument("--anchor", help="the laptop's evaluate_stim --save .npz of the same checkpoint on validation_X")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.run:
        do_run(Path(args.checkpoint), out, args.run, args.processes, args.limit, args.set, args.tie_break_train, args.shots)
    if args.table:
        table(out, args.tie_break, args.anchor, args.tie_break_train)


if __name__ == "__main__":
    main()
