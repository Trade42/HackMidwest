"""Gate jobs that need only committed files, for another machine. Run from challenge-1-ml-decoder-kit/:

  PYTHONDONTWRITEBYTECODE=1 python -m analysis.remote g3 --data-dir DATA --out OUT --processes 10
  PYTHONDONTWRITEBYTECODE=1 python -m analysis.remote g2 --data-dir DATA --out OUT --processes 10

DATA is the team's p = 0.004 folder (main: challenge-1-ml-decoder-kit/data/p0.004); only its
train_* and validation_* files are opened. OUT receives the per-task cache, the log, and the
summaries; the final tables are printed to stdout.

- g3: G3 item 2, the 16-schedule frontier on validation_X.npz and validation_Z.npz (5,000 shots each),
  every schedule in 500-shot chunks on fresh decoders, seed 0, paired with the kit's 80 / 60 / 5.
- g2: G2 items 1 and 2 at p = 0.004 on the committed selection (results/g2_selection_p0.004.npz):
  relay-BP with stop_nconv=20 and num_sets=600, and 16 candidates (stop_nconv=1, seeds 1 to 16),
  at most 10 of G1's unconverged failures per basis for the candidates.

Both jobs start with a machine check: the kit's relay-BP on a fresh decoder, seed 0, on the first
10 shots of validation_X.npz, compared with luminaty's run (iterations, corrections and predicted
bits). If it matches, this machine's iterations and failures are luminaty's; its milliseconds are
its own and are labelled with its name.
"""
import argparse
import hashlib
import platform
import sys
from argparse import Namespace
from pathlib import Path


from analysis.common import Log, decode_shots, machine, open_dataset, problem, run_tasks, sha256
from analysis.g1_relay_failures import KIT

LUMINATY_CHECK = dict(shots=10, iterations_sum=19652,
                      sha256="8563551f648ee0be27f96ad9322347b153fc9f4ed5dabbb138aa066c4ec7d4f1")
P004_SHA256 = {"validation_X.npz": "b00ee09c35ed088a12bcba60b8b704b528c47fb156c7d09cf0f171f90a181f3d",
               "validation_Z.npz": "843c4daf24e2b56c5988670550bbcdbac8f50b18dbb9d24beb64969146cb8e20",
               "train_X.npz": "98c0afeb2a0b6b27f3b689e832160a060a4c0aeb37d229f84daa1995a9a0f7cc",
               "train_Z.npz": "14dce11586fa51256f476224d0fe3965e0562f2d8f68cba6192883a65fb6b24a"}
ANALYSIS = Path(__file__).resolve().parent


def data_files(data_dir, names):
    files = []
    for name in names:
        path = Path(data_dir) / name
        got = sha256(path)
        if got != P004_SHA256[name]:
            raise SystemExit(f"{path}: sha256 {got} is not the team's committed file {P004_SHA256[name]}")
        files.append(str(path))
    return files


def machine_check(data_dir, log):
    """The kit's relay-BP on the first 10 validation_X shots, fresh decoder, seed 0, against luminaty."""
    path = data_files(data_dir, ["validation_X.npz"])[0]
    ds = open_dataset(path)
    n = LUMINATY_CHECK["shots"]
    d = decode_shots(KIT, problem(ds.basis, ds.p), ds.det[:n], ds.obs[:n])
    h = hashlib.sha256(d["iterations"].astype("<i4").tobytes() + d["ehash"].astype("<u8").tobytes()
                       + d["pred"].astype("<u2").tobytes()).hexdigest()
    same = h == LUMINATY_CHECK["sha256"]
    log(f"machine check (kit relay-BP, fresh decoder, seed 0, validation_X shots 0-{n - 1}): "
        f"iterations {int(d['iterations'].sum())} (luminaty {LUMINATY_CHECK['iterations_sum']}), "
        + ("IDENTICAL to luminaty" if same else f"DIFFERS from luminaty: hash {h}"))
    return same


def cmd_g3(args):
    from analysis import g3_schedules as g3
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "g3_run.log")
    log("job g3 on", platform.node(), machine(args.processes))
    machine_check(args.data_dir, log)
    files = data_files(args.data_dir, ["validation_X.npz", "validation_Z.npz"])
    infos = g3.dataset_infos(files)
    scheds = [g3.parse_schedule(s) for s in args.schedules] if args.schedules else g3.GRID
    if g3.KIT_SCHEDULE not in scheds:
        scheds = [g3.KIT_SCHEDULE] + scheds
    tasks = [t for s in sorted(scheds, key=lambda s: (-s[2], -s[0], -s[1])) for info in infos
             for t in g3.tasks_for(s, info, args.max_shots)]
    run_tasks(tasks, args.processes, out / "cache", log)
    sys.stdout.flush()
    g3.cmd_summarize(Namespace(data=files, cache=str(out / "cache"), schedules=args.schedules, max_shots=args.max_shots,
                               processes=args.processes, out=str(out), tag="frontier_p0.004",
                               machine=f"{platform.node()} ({args.processes} processes)"))
    log(f"ms in this table: per shot inside decode_detailed, {args.processes} processes side by side on {platform.node()}; "
        "iterations and failures do not depend on the machine when the machine check above is IDENTICAL")
    log("job g3 done")


def cmd_g2(args):
    from analysis import g2_headroom as g2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "g2_run.log")
    log("job g2 on", platform.node(), machine(args.processes))
    machine_check(args.data_dir, log)
    files = data_files(args.data_dir, ["validation_X.npz", "validation_Z.npz", "train_X.npz", "train_Z.npz"])
    selection = [str(ANALYSIS / "results" / "g2_selection_p0.004.npz")]
    sel = g2.load_selection(selection)
    paths = g2.dataset_paths(files)
    caps = g2.parse_caps(args.caps)
    tasks = g2.build_tasks(sel, paths, args.variants, caps)
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]
    run_tasks(tasks, args.processes, out / "cache", log)
    sys.stdout.flush()
    g2.cmd_summarize(Namespace(selection=selection, data=files, cache=str(out / "cache"), caps=args.caps,
                               pershot=str(ANALYSIS / "results" / "pershot"), out=str(out)))
    log(f"ms in these tables: per shot inside the decode call, fresh decoder per shot, {args.processes} processes "
        f"side by side on {platform.node()}")
    log("job g2 done")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("g3", "g2"):
        s = sub.add_parser(name)
        s.add_argument("--data-dir", required=True, help="the p = 0.004 data folder")
        s.add_argument("--out", required=True, help="output folder")
        s.add_argument("--processes", type=int, default=10)
    sub.choices["g3"].add_argument("--schedules", nargs="+", help="default: the 16-schedule grid (for smoke tests only)")
    sub.choices["g3"].add_argument("--max-shots", type=int, help="first N shots per file (for smoke tests only)")
    sub.choices["g2"].add_argument("--variants", nargs="+", default=["big", "cand16"])
    sub.choices["g2"].add_argument("--caps", default="cand16@0.004=all/all/10")
    sub.choices["g2"].add_argument("--max-tasks", type=int, help="only the first N tasks (for smoke tests only)")
    args = ap.parse_args()
    {"g3": cmd_g3, "g2": cmd_g2}[args.cmd](args)


if __name__ == "__main__":
    main()
