"""Copy the result files the report reads into report/data/, and write report/data/SOURCES.md.

This is the team's own tool, not a step of the reproduction: the judges get report/data/ as
committed. It exists so that no file is copied by hand and every copy has its origin, its
sha256 and its time on record.

Run from challenge-1-ml-decoder-kit/, inside a clone that has the side branches fetched:

  python -m report.sync_data            # copy what exists now, list what is still missing
  python -m report.sync_data --check    # copy nothing; verify every file against its record

A source is a file on a branch of this repository (`origin/server-logs`, `origin/laptop-artifacts`)
or a file of the working tree (results the team already committed elsewhere in the kit). A file
that was copied before and whose source now has other bytes stops the run, unless the source is
a log that grows (marked below) or --accept-change names it.
"""
import argparse
import fnmatch
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data"
RECORD = DATA / "sources.json"

NETWORK_STEP = 120000   # the newest snapshot of the X network whose logits are copied; older copies are removed by hand

# (destination under report/data/, where from, path or pattern there, grows?)
# "tree" is the working tree of this branch, relative to challenge-1-ml-decoder-kit/.
# A pattern with * is matched against every file under its fixed prefix; the part matched from
# the first * on is appended to the destination.
SOURCES = [
    # the timed rows on the saved TEST shots, from the team's server (evaluation.evaluate, one process per row)
    # only the registered row folders (eval_p<p>_<row>, tcbp_p<p>, ensemble_p<p>): a rehearsal on validation shots is never copied here
    ("test/", "origin/server-logs", "files/clean/eval_p*/summary.json", False),
    ("test/", "origin/server-logs", "files/clean/eval_p*/summary.csv", False),
    ("test/", "origin/server-logs", "files/clean/eval_p*/*_shots.npz", False),
    ("test/", "origin/server-logs", "files/clean/tcbp_p*/test_*.tcbp.npz", False),
    ("test/", "origin/server-logs", "files/clean/ensemble_p*/test_*.npz", False),
    ("test/", "origin/server-logs", "files/clean/ensemble_p*/test_*.json", False),
    ("test/cpu_monitor.out", "origin/server-logs", "cpu_monitor.out", True),
    ("test/test_rows.out", "origin/server-logs", "test_rows.out", True),
    # the ensemble on exploration shots (validation files): the members' answers combined offline, and TCBP-JOBS's answers
    ("exploration/ensemble/", "origin/server-logs", "files/ens_p*/validation_*.npz", False),
    ("exploration/ensemble/", "origin/server-logs", "files/ens_p*/validation_*.json", False),
    ("test/clean_rows.out", "origin/server-logs", "clean_rows.out", True),
    ("test/runner_cpu.out", "origin/server-logs", "runner_cpu.out", True),
    ("test/runner_gpu.out", "origin/server-logs", "runner_gpu.out", True),
    ("test/runner_ctl.out", "origin/server-logs", "runner_ctl.out", True),
    # the teammates' earlier rows on the same test shots (other machines, other load): a cross-check only
    ("teammates_earlier/icrn_eval_p0.003/", "tree", "results/icrn/eval_p0.003/summary.*", False),
    ("teammates_earlier/icrn_eval_p0.003/", "tree", "results/icrn/eval_p0.003/02_relay-bp_shots.npz", False),
    ("teammates_earlier/icrn_router_eval_p0.003/", "tree", "results/icrn/router_eval_p0.003_portable/summary.*", False),
    ("teammates_earlier/icrn_gain_p0.003/", "tree", "results/icrn/gain_p0.003_portable/comparison.json", False),
    ("teammates_earlier/mac_eval_p0.004/", "tree", "results/mac/eval_p0.004/summary.*", False),
    ("teammates_earlier/mac_eval_p0.004/", "tree", "results/mac/eval_p0.004/02_relay-bp_shots.npz", False),
    ("teammates_earlier/mac_router_eval_p0.004/", "tree", "results/mac/router_eval_p0.004/summary.*", False),
    ("teammates_earlier/mac_gain_p0.004/", "tree", "results/mac/gain_p0.004/comparison.json", False),
    # TCBP-JOBS as the teammates ran it: its own sampled shots
    ("tcbp_own_shots/", "tree", "results/tcbp-jobs_p0.003_shots10000_seed1_20261122_005215_437826+0000.log", False),
    ("tcbp_own_shots/", "tree", "results/tcbp-jobs_p0.004_shots10000_seed1_20261122_043426_147901+0000.log", False),
    ("tcbp_own_shots/", "tree", "results/tcbp-jobs_p0.005_shots10000_seed1_20261122_050203_539870+0000.log", False),
    # exploration shots (train and validation files): the laptop's per-shot decodes and tables
    ("exploration/priortilt/", "origin/laptop-artifacts", "priortilt/results_8b/selector_*.npz", False),
    ("exploration/priortilt/", "origin/laptop-artifacts", "priortilt/results_8b.txt", True),   # the laptop adds rules to it
    ("exploration/priortilt/", "origin/laptop-artifacts", "priortilt/results_network_select_step*.txt", False),
    ("exploration/priortilt/", "origin/laptop-artifacts", "priortilt/results_step9_p0.005_X.txt", False),
    ("exploration/priortilt_p0.005/", "origin/laptop-artifacts", "priortilt/results_p005main/selector_*.npz", False),
    ("exploration/priortilt_p0.005/", "origin/laptop-artifacts", "priortilt/results_p005main_*.txt", True),
    # TCBP-JOBS on the same exploration shots at p = 0.004 (batch mode on a shared GPU: answers only, no times)
    ("exploration/tcbp_p0.004/", "origin/server-logs", "files/tcbp_p0.004/*.tcbp.npz", False),
    ("exploration/cascade_X_s1/", "origin/laptop-artifacts", "cascade_X_s1/SCORES.md", True),
    # the X network's per-shot logits on the exploration shots, for ONE snapshot (NETWORK_STEP): the tie-break is recomputed from them
    ("exploration/cascade_X_s1/", "origin/laptop-artifacts", f"cascade_X_s1/eval/step{NETWORK_STEP}_validation_X_p0.004.npz", False),
    ("exploration/cascade_X_s1/", "origin/laptop-artifacts", f"cascade_X_s1/eval/step{NETWORK_STEP}_train_X_p0.004.npz", False),
    # the networks' training logs
    ("networks/cascade_X_s1.train.log", "origin/laptop-artifacts", "cascade_X_s1/train.log", True),
    ("networks/cascade_Z_s1.train.log", "origin/server-logs", "cascade_Z_s1.train.log", True),
    # loss: this PC's tables, and the server's jobs
    ("loss/pc/", "tree", "loss/results/L0_X_p0.004.json", False),
    ("loss/pc/", "tree", "loss/results/L1_X_p0.004_s*.json", False),
    ("loss/pc/", "tree", "loss/results/L3_*.json", False),
    ("loss/pc/", "tree", "loss/results/L4_timing.json", False),
    # the loss-time tracker: its run on its test split (this PC, seed 7300), the replication (server, seed 7310), the pooled table
    ("loss/tracker/", "tree", "loss_tracker/results/test.*", False),
    ("loss/tracker/", "tree", "loss_tracker/results/replication_seed7310.*", False),
    ("loss/tracker/", "tree", "loss_tracker/results/pooled_7300_7310.*", False),
    ("loss/tracker/", "tree", "loss_tracker/results/accuracy_test.*", False),
    ("loss/tracker/", "tree", "loss_tracker/results/test_secondary.*", False),
    ("loss/tracker/", "tree", "loss_tracker/results/timing.*", False),
    ("loss/tracker/", "tree", "loss_tracker/results/accuracy_val_learned.*", False),
    ("loss/server/L1_s2e-4_seed5100/tables.md", "origin/server-logs", "files/loss/L1_s2e-4_seed5100/tables.md", False),
    ("loss/server/L1_s5e-4_seed5200/tables.md", "origin/server-logs", "files/loss/L1_s5e-4_seed5200/tables.md", False),
    # two jobs wrote tables.md of the same folder: job A's tables (the oracles) are pinned at the commit that held them, job D's (the pieces of the map) are the file now
    ("loss/server/L3_s1e-3_seed5300/tables.md", "origin/server-logs@e08c3c3c9390e542cfe435e68b1750f5210bac50", "files/loss/L3_s1e-3_seed5300/tables.md", False),
    ("loss/server/L3_s1e-3_seed5300_jobD/tables.md", "origin/server-logs", "files/loss/L3_s1e-3_seed5300/tables.md", False),
    # the machines: what the server printed about itself
    ("machines/server_nvidia-smi.txt", "origin/server-logs", "nvidia-smi.txt", True),
    ("machines/cascade_X_s1_SHA256SUMS", "origin/laptop-artifacts", "cascade_X_s1/SHA256SUMS", True),
    # the measurements behind the design, and what did not work (exploration shots)
    ("background/", "tree", "analysis/results/g2_server_p0.004_summary.json", False),
    ("background/", "tree", "analysis/results/g2_summary.json", False),
    ("background/", "tree", "analysis/results/g3_frontier_p0.004.csv", False),
    ("background/", "tree", "ml_decoder/logs/relay_schedule_paired_X_p0.004_5000_seed1.log", False),
    ("background/", "tree", "ml_decoder/logs/predecoder_real2_p0.004_300_300_300_20.log", False),
    ("background/", "tree", "ml_decoder/logs/train_X_p0.0001_seed1.log", False),
    # "network first": the network answers where it is confident, else the hierarchy (p = 0.005, threshold fitted on train_X)
    ("background/", "origin/laptop-artifacts", f"priortilt/results_hierarchy_heldout_step{NETWORK_STEP}.txt", False),
    # the laptop's early plain-CNN runs: its table of the runs, and the logs of the two runs the report names
    ("background/plain_cnn_runs/", "origin/laptop-artifacts", "plain_cnn_runs/README.md", False),
    ("background/plain_cnn_runs/", "origin/laptop-artifacts", "plain_cnn_runs/x_4h.log", False),
    ("background/plain_cnn_runs/", "origin/laptop-artifacts", "plain_cnn_runs/x_4h.ckpt_0.83h.eval_lowp.log", False),
    ("background/plain_cnn_runs/", "origin/laptop-artifacts", "plain_cnn_runs/x_1h_w32.log", False),
    ("loss/server/", "origin/server-logs", "files/loss/*/*.json", False),
]


def git(*arguments, binary=False):
    result = subprocess.run(["git", "-C", str(KIT), *arguments], capture_output=True)
    if result.returncode:
        raise RuntimeError(f"git {' '.join(arguments)}: {result.stderr.decode().strip()}")
    return result.stdout if binary else result.stdout.decode()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def expand(where, pattern):
    """Every (source path, part to append to the destination) the pattern names."""
    if "*" not in pattern:
        return [(pattern, Path(pattern).name)]
    prefix = pattern[: pattern.index("*")]
    folder = prefix.rsplit("/", 1)[0] if "/" in prefix else ""
    if where == "tree":
        listed = [str(path.relative_to(KIT)) for path in sorted((KIT / folder).rglob("*")) if path.is_file()]
    else:
        try:
            listed = git("ls-tree", "-r", "--full-tree", "--name-only", where.split("@")[-1], *([folder] if folder else [])).splitlines()
        except RuntimeError:
            return []
    found = []
    for name in listed:
        if fnmatch.fnmatchcase(name, pattern) and name.count("/") == pattern.count("/"):
            found.append((name, name[len(folder) + 1:] if folder else name))
    return found


def read_source(where, path):
    """(bytes, the commit the copy is taken from) or None if the file is not there yet."""
    if where == "tree":
        file = KIT / path
        if not file.is_file():
            return None
        commit = git("log", "-1", "--format=%H", "--", path).strip() or "uncommitted"
        return file.read_bytes(), commit
    try:
        commit = git("rev-parse", where.split("@")[-1]).strip()     # "branch@commit" pins a source at one commit of its branch
        return git("show", f"{commit}:{path}", binary=True), commit
    except RuntimeError:
        return None


def describe(where, path, commit):
    if where == "tree":
        return f"this branch, `challenge-1-ml-decoder-kit/{path}` (last changed in commit `{commit[:12]}`)"
    return f"branch `{where.split('@')[0].removeprefix('origin/')}` at commit `{commit[:12]}`, `{path}`"


def write_sources_md(records):
    lines = ["# Where every file in this folder came from", "",
             "Written by `python -m report.sync_data`; do not edit by hand. The tables and figures of the report read",
             "only the files listed here. `server-logs` and `laptop-artifacts` are data-only branches of this repository:",
             "the first receives result files from the team's server, the second from the laptop that trained the X network.",
             "A file marked *grows* is a log that was still being written when it was copied; its line is the last copy.", "",
             "| File in `report/data/` | From | sha256 | Bytes | Copied (UTC) |", "|---|---|---|---|---|"]
    for record in sorted(records.values(), key=lambda r: r["file"]):
        grows = " *(grows)*" if record["grows"] else ""
        lines.append(f"| `{record['file']}`{grows} | {record['from']} | `{record['sha256']}` | {record['bytes']} | {record['copied_utc']} |")
    (DATA / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="copy nothing; verify every recorded file's sha256")
    parser.add_argument("--accept-change", nargs="*", default=[], help="files under report/data/ whose source may have changed")
    args = parser.parse_args(argv)
    records = json.loads(RECORD.read_text()) if RECORD.exists() else {}

    if args.check:
        bad = [name for name, record in records.items()
               if not (DATA / name).is_file() or sha256((DATA / name).read_bytes()) != record["sha256"]]
        unlisted = [str(path.relative_to(DATA)) for path in sorted(DATA.rglob("*"))
                    if path.is_file() and path.name not in ("SOURCES.md", "sources.json") and str(path.relative_to(DATA)) not in records]
        print(f"{len(records)} recorded files, {len(bad)} differ from their record, {len(unlisted)} files without a record")
        for name in bad + unlisted:
            print("  ", name)
        return 1 if bad or unlisted else 0

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    copied, changed, waiting, stopped = [], [], [], []
    for destination, where, pattern, grows in SOURCES:
        matches = expand(where, pattern)
        if not matches:
            waiting.append(f"{where}: {pattern}")
            continue
        for path, tail in matches:
            name = destination + tail if destination.endswith("/") else destination
            got = read_source(where, path)
            if got is None:
                waiting.append(f"{where}: {path}")
                continue
            data, commit = got
            digest = sha256(data)
            old = records.get(name)
            if old and old["sha256"] == digest and (DATA / name).is_file():
                continue
            if old and not grows and name not in args.accept_change:
                stopped.append(f"{name}: recorded {old['sha256'][:12]}, the source now has {digest[:12]} ({describe(where, path, commit)})")
                continue
            target = DATA / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            records[name] = {"file": name, "from": describe(where, path, commit), "branch": where, "commit": commit,
                             "path": path, "sha256": digest, "bytes": len(data), "copied_utc": now, "grows": grows}
            (changed if old else copied).append(name)
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(records, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    write_sources_md(records)
    print(f"{len(copied)} new, {len(changed)} updated, {len(records)} in all; {len(waiting)} sources not there yet")
    for name in copied + changed:
        print("  copied", name)
    for name in waiting:
        print("  waiting for", name)
    for line in stopped:
        print("  STOPPED, the source changed:", line)
    return 1 if stopped else 0


if __name__ == "__main__":
    sys.exit(main())
