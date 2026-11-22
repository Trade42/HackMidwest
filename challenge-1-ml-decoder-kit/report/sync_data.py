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

NETWORK_STEP = 120000
TRACKER_PARTS = "/home/sal/hackMidwest/orchestration/scratch/loss-tracker/parts"   # copied once; the copies under data/ are the record   # the newest snapshot of the X network whose logits are copied; older copies are removed by hand

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
    ("test/", "origin/server-logs", "files/test/net_p*/*.json", False),      # the final networks alone on the test files (neural.evaluate_stim)
    ("test/test_rows.out", "origin/server-logs", "test_rows.out", True),
    ("test/test_run.out", "origin/server-logs", "test_run.out", True),       # the test job's printout: the frozen commit and both checkpoints' sha256
    # the ensemble on exploration shots (validation files): the members' answers combined offline, and TCBP-JOBS's answers
    ("exploration/ensemble/", "origin/server-logs", "files/ens_p*/validation_*.npz", False),
    ("exploration/ensemble/", "origin/server-logs", "files/ens_p*/validation_*.json", False),
    ("test/clean_rows.out", "origin/server-logs", "clean_rows.out", True),
    ("test/clean_retime.out", "origin/server-logs", "clean_retime.out", True),   # the p = 0.003 baselines decoded again in a quiet window
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
    # the tracker's per-shot results on its test split (seed 7300), kept until now only in its task's working folder on the team's PC
    ("loss/tracker/parts/test/", f"file:{TRACKER_PARTS}", "test/*.npz", False),
    ("loss/tracker/parts/test_coupled2/", f"file:{TRACKER_PARTS}", "test_coupled2/*.npz", False),
    ("loss/tracker/", "tree", "loss_tracker/results/accuracy_val_learned.*", False),
    ("loss/server/L1_s2e-4_seed5100/tables.md", "origin/server-logs", "files/loss/L1_s2e-4_seed5100/tables.md", False),
    ("loss/server/L1_s5e-4_seed5200/tables.md", "origin/server-logs", "files/loss/L1_s5e-4_seed5200/tables.md", False),
    # two jobs wrote tables.md of the same folder: job A's tables (the oracles) are pinned at the commit that held them, job D's (the pieces of the map) are the file now
    ("loss/server/L3_s1e-3_seed5300/tables.md", "origin/server-logs@e08c3c3c9390e542cfe435e68b1750f5210bac50", "files/loss/L3_s1e-3_seed5300/tables.md", False),
    ("loss/server/L3_s1e-3_seed5300_jobD/tables.md", "origin/server-logs", "files/loss/L3_s1e-3_seed5300/tables.md", False),
    # the machines: what the server printed about itself
    ("machines/server_nvidia-smi.txt", "origin/server-logs", "nvidia-smi.txt", True),
    ("machines/cascade_X_s1_SHA256SUMS", "origin/laptop-artifacts", "cascade_X_s1/SHA256SUMS", True),
    # the ablation at scale: 80,000 fresh X shots at p = 0.004, never used for a fit (branch hierarchy, commit 534d428)
    ("exploration/ablation_80k/summary.json", "origin/hierarchy@534d428136a2d6d43886f675a71e2c91ee69b86f",
     "challenge-1-ml-decoder-kit/hierarchy/results/ablation_p0.004_X/summary.json", False),
    ("exploration/ablation_80k/per_shot.npz", "origin/hierarchy@534d428136a2d6d43886f675a71e2c91ee69b86f",
     "challenge-1-ml-decoder-kit/hierarchy/results/ablation_p0.004_X/per_shot.npz", False),
    ("exploration/ablation_80k/select_printout.txt", "origin/hierarchy@534d428136a2d6d43886f675a71e2c91ee69b86f",
     "challenge-1-ml-decoder-kit/hierarchy/results/ablation_p0.004_X/select_printout.txt", False),
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
    # the folders merged at the end: the learned selector (its one test run is written in its README) and the neutral-atom noise study
    ("background/selector_README.md", "tree", "selector/README.md", False),
    ("background/neutral_atom_noise_RESULTS.md", "tree", "neutral_atom_noise/RESULTS.md", False),
    # the learned predecoder (predecoder/, merged into integration at 8ddf14e): its runs on the saved TEST shots (summary and the
    # secondary rows beside TCBP-JOBS and the evaluator's relay-BP; the per-shot file stays out of git), and the declared gate on exploration shots
    ("test/predecoder/", "tree", "predecoder/logs/test_run_p*_summary.json", False),
    ("test/predecoder/", "tree", "predecoder/logs/test_run_p*_secondary.json", False),
    ("exploration/predecoder/", "tree", "predecoder/logs/compare_validation_X_summary.json", False),
    # the fallback rule (H where its relay-BP converged, else TCBP-JOBS), designed after the first test result: its runs on the test files, and its validation
    ("test/predecoder/", "tree", "predecoder/logs/fallback_test_*.json", False),
    ("test/predecoder/", "tree", "predecoder/logs/fallback_test_*.log", False),
    ("exploration/predecoder/", "tree", "predecoder/logs/fallback_validation_*.json", False),
    # the predecoder's clean latency: the team's server, one process, one thread, the first 1,000 shots of VALIDATION_X at p = 0.005 (server job cpu/050)
    ("exploration/predecoder/latency_validation_X_p0.005/", "origin/server-logs", "files/predecoder/latency_validation_X_p0.005/*", False),
    ("exploration/predecoder/job_cpu_050-predecoder-latency.out", "origin/server-logs", "job_cpu_050-predecoder-latency.out", False),
    ("exploration/predecoder/latency_validation_X_p0.004/", "origin/server-logs", "files/predecoder/latency_validation_X_p0.004/*", False),
    ("exploration/predecoder/latency_validation_X_p0.003/", "origin/server-logs", "files/predecoder/latency_validation_X_p0.003/*", False),
    ("exploration/predecoder/job_cpu_060-predecoder-latency-p0.003-p0.004.out", "origin/server-logs", "job_cpu_060-predecoder-latency-p0.003-p0.004.out", False),
    # TCBP-JOBS with the head's priors (branch predecoder-tcbp, pinned): its validation score, and its test scores when they land
    ("exploration/predecoder_tcbp/", "origin/predecoder-tcbp@dad4fd50073bc18f582565da403867fb6d9afb5d",
     "challenge-1-ml-decoder-kit/predecoder/logs/tcbp_validation_X_p0.005_score.json", False),
    ("exploration/predecoder_tcbp/", "origin/predecoder-tcbp@dad4fd50073bc18f582565da403867fb6d9afb5d",
     "challenge-1-ml-decoder-kit/predecoder/logs/tcbp_validation_X_p0.005_score.log", False),
    ("exploration/predecoder_tcbp/", "origin/predecoder-tcbp@dad4fd50073bc18f582565da403867fb6d9afb5d",
     "challenge-1-ml-decoder-kit/predecoder/logs/tcbp_job3_validation_summary.json", False),
    ("test/predecoder_tcbp/", "tree", "predecoder/logs/tcbp_test_*_score.json", False),
    ("test/predecoder_tcbp/", "tree", "predecoder/logs/tcbp_test_*_summary.json", False),
    ("test/predecoder_tcbp/", "tree", "predecoder/logs/tcbp_pair_test_*.json", False),
    # its time per X shot: the network on the A100 (job 4, validation shots) and TCBP-JOBS's build and decode on the test X shots at p = 0.005 (job 5)
    ("exploration/predecoder_tcbp/gpu_network_decode_val1000_fp32_summary.json", "origin/server-logs",
     "files/predecoder-tcbp/gpu_network/decode_val1000_fp32_summary.json", False),
    ("test/predecoder_tcbp/job5_test_X_p0.005_summary.json", "origin/server-logs", "files/predecoder-tcbp/test_X_p0.005/summary.json", False),
    # its speed work (branch tcbp-speed, merged into predecoder/; the first 1,000 validation X shots at p = 0.005, the A100, separate timers):
    # job 1 builds TCBP-JOBS with H as a scipy CSR matrix beside the dense path, job 3 runs the network as a CUDA graph beside eager
    ("exploration/predecoder_tcbp/speed/", "tree", "predecoder/logs/tcbp_speed_job1_summary.json", False),
    ("exploration/predecoder_tcbp/speed/", "tree", "predecoder/logs/tcbp_speed_job1_report_vs_job3.json", False),
    ("exploration/predecoder_tcbp/speed/", "tree", "predecoder/logs/tcbp_speed_job3_network_summary.json", False),
    ("exploration/predecoder_tcbp/speed/", "tree", "predecoder/logs/tcbp_speed_job3_analysis.json", False),
    # a fast path for easy shots (branch tcbp-fastpath, merged into predecoder/fastpath/; validation X at p = 0.005): rounds 1 to 3 in one file
    ("exploration/predecoder_tcbp/fastpath/", "tree", "predecoder/fastpath/result3_p0.005.json", False),
    ("test/predecoder/", "tree", "predecoder/logs/pair_test_p*.json", False),
    ("test/predecoder/", "tree", "predecoder/logs/pair_test_p*.log", False),
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
    if where.startswith("file:"):
        base = Path(where.removeprefix("file:"))
        listed = [str(path.relative_to(base)) for path in sorted((base / folder).rglob("*")) if path.is_file()] if (base / folder).is_dir() else []
    elif where == "tree":
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
    if where.startswith("file:"):
        file = Path(where.removeprefix("file:")) / path
        return (file.read_bytes(), "not in git") if file.is_file() else None
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
    if where.startswith("file:"):
        return (f"the loss-tracker task's working folder on the team's PC, `{where.removeprefix('file:')}/{path}` "
                "(not in git elsewhere; the test run of `loss_tracker/` at commit 0e71370, Clifft seed 7300)")
    if where == "tree":
        return f"this branch, `challenge-1-ml-decoder-kit/{path}` (last changed in commit `{commit[:12]}`)"
    return f"branch `{where.split('@')[0].removeprefix('origin/')}` at commit `{commit[:12]}`, `{path}`"


STAGES = {"agree": 0, "weight": 1, "converged": 2, "network": 3}
WINNERS = {"P1": 0, "P2": 1, "R": 2}


def derive_plugin_metrics(records, now):
    """Our plugin's per-shot record (*_metrics.json, about 9 MB a row) reduced to a small .npz of what the report reads:
    per basis, whether the two fast decodes agreed, the stage that answered, the winning candidate, and each stage's time.
    The source file's branch, commit and sha256 are recorded beside the derived file's own sha256."""
    import io
    import numpy as np
    derived = []
    for path, tail in expand("origin/server-logs", "files/clean/eval_p*/*_metrics.json"):
        if "hierarchy" not in path:
            continue
        name = "test/" + tail.rsplit("/", 1)[0] + "/plugin_shots.npz"
        got = read_source("origin/server-logs", path)
        if got is None:
            continue
        data, commit = got
        source_sha = sha256(data)
        if name in records and records[name].get("source_sha256") == source_sha and (DATA / name).is_file():
            continue
        metrics = json.loads(data)
        arrays = {}
        for basis in "XZ":
            shots = metrics[basis]
            arrays[f"agreed_{basis}"] = np.array([shot["agreed"] for shot in shots], bool)
            arrays[f"stage_{basis}"] = np.array([STAGES[shot["stage"]] for shot in shots], np.int8)
            arrays[f"winner_{basis}"] = np.array([WINNERS.get(shot.get("winner"), -1) for shot in shots], np.int8)
            arrays[f"any_converged_{basis}"] = np.array([shot.get("any_converged", True) for shot in shots], bool)
            for stage in ("P1", "P2", "R", "network"):
                arrays[f"seconds_{stage}_{basis}"] = np.array([shot.get(f"seconds_{stage}", np.nan) for shot in shots], np.float64)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, stage_codes=np.array(list(STAGES)), winner_codes=np.array(list(WINNERS)), **arrays)
        target = DATA / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(buffer.getvalue())
        records[name] = {"file": name, "from": f"derived by `report/sync_data.py` from {describe('origin/server-logs', path, commit)} "
                                              f"(sha256 `{source_sha[:16]}…`)", "branch": "origin/server-logs", "commit": commit, "path": path,
                         "source_sha256": source_sha, "sha256": sha256(target.read_bytes()), "bytes": target.stat().st_size,
                         "copied_utc": now, "grows": False}
        derived.append(name)
    return derived


TCBP_FITS = [("stored", "files/tcbp_p0.005/validation_X.tcbp.npz", "failed_T", "predicted_T"),
             ("job1", "files/predecoder-tcbp/validation_X_p0.005_fresh_per_shot.npz", "failed_fresh", "predicted_fresh"),
             ("job3", "files/predecoder-tcbp/validation_X_p0.005_per_shot.npz", "failed_fresh", "predicted_fresh")]


def derive_tcbp_fits(records, now):
    """TCBP-JOBS's model fit varies from run to run: three fits with the kit's priors on the same validation X shots at p = 0.005.
    Only the kit-prior arm of each file is kept (job 3's file also holds an arm of another branch, which stays out); the three files must
    hold the same shots (equal truth arrays), and job 3's batch decode is checked against its per-shot decode."""
    import io
    import numpy as np
    name = "exploration/tcbp_fits/validation_X_p0.005_fits.npz"
    got = {key: read_source("origin/server-logs", path) for key, path, _, _ in TCBP_FITS}
    if any(value is None for value in got.values()):
        return []
    shas = {key: sha256(data) for key, (data, _) in got.items()}
    if name in records and records[name].get("source_sha256") == shas and (DATA / name).is_file():
        return []
    arrays, truth = {}, None
    for key, path, failed, predicted in TCBP_FITS:
        with np.load(io.BytesIO(got[key][0]), allow_pickle=False) as file:
            if truth is None:
                truth = file["truth"]
            elif not np.array_equal(truth, file["truth"]):
                raise RuntimeError(f"{path}: other shots than {TCBP_FITS[0][1]}")
            arrays[f"failed_{key}"] = file[failed].astype(bool)
            if key == "job3":
                arrays["job3_batch_equals_per_shot"] = np.array(bool(np.array_equal(file["predicted_batch"], file[predicted])))
    with np.load(io.BytesIO(got["stored"][0]), allow_pickle=False) as file:
        arrays["data_sha256"] = file["data_sha256"]
    buffer = io.BytesIO()
    np.savez_compressed(buffer, truth_sha256=np.array(sha256(truth.tobytes())), **arrays)
    target = DATA / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(buffer.getvalue())
    commit = got["stored"][1]
    records[name] = {"file": name, "from": "derived by `report/sync_data.py` from branch `server-logs` at commit `" + commit[:12] + "`: "
                     + "; ".join(f"`{path}` (sha256 `{shas[key][:16]}…`, its `{failed}`)" for key, path, failed, _ in TCBP_FITS),
                     "branch": "origin/server-logs", "commit": commit, "path": "; ".join(path for _, path, _, _ in TCBP_FITS),
                     "source_sha256": shas, "sha256": sha256(target.read_bytes()), "bytes": target.stat().st_size, "copied_utc": now, "grows": False}
    return [name]


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
    parser.add_argument("--only", nargs="*", default=None, help="copy only the files whose name under report/data/ starts with one of these")
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
        if args.only is not None and not any(destination.startswith(only) or only.startswith(destination) for only in args.only):
            continue
        matches = expand(where, pattern)
        if not matches:
            waiting.append(f"{where}: {pattern}")
            continue
        for path, tail in matches:
            name = destination + tail if destination.endswith("/") else destination
            if args.only is not None and not name.startswith(tuple(args.only)):
                continue
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
    if args.only is None:
        copied += derive_plugin_metrics(records, now)
    if args.only is None or any("exploration/tcbp_fits/".startswith(only) or only.startswith("exploration/tcbp_fits/") for only in args.only):
        copied += derive_tcbp_fits(records, now)
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
