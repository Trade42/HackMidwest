"""Read the result files under report/data/ and check them while reading.

Every loader takes paths under report/data/ and returns plain dicts. Nothing here imports the
kit or a decoder: the report must be reproducible from the copied files alone.
"""
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from report import stats

# REPORT_DATA points the loaders at another folder: used by the tests only.
DATA = Path(os.environ.get("REPORT_DATA") or Path(__file__).resolve().parent / "data").resolve()


class Mismatch(Exception):
    """A file contradicts itself or another file. The report stops instead of guessing."""


# ---------- rows of the team's evaluator (evaluation.evaluate) ----------

def site_of(metadata):
    """Which machine an evaluator row ran on, from what its summary.json records."""
    machine = metadata.get("machine", {})
    path = metadata["datasets"]["test"]["X"]["path"]
    if path.startswith("/home/blakew3/"):
        name = "ICRN server (AMD EPYC-Milan, 12 CPUs by quota)"
    elif path.startswith("/Users/"):
        name = "a teammate's Mac (arm64)"
    else:
        name = f"{machine.get('system', '?')} {machine.get('machine', '')}".strip()
    return name, int(machine.get("workers", 1))


def evaluator_rows(folder):
    """Every decoder of one evaluator output folder, audited against its per-shot file.

    Returns {decoder name: row}. A row holds the counts, the rate per cycle with the kit's
    error, the three latencies per pair in ms (from seconds_total_pair: the whole decode_shot
    call, with every stage and rerun), the per-shot failure mask of the pairs (or None when the
    per-shot file was not copied), and the identity of the test files.
    """
    folder = Path(folder)
    metadata = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if metadata.get("cycles") != stats.CYCLES:
        raise Mismatch(f"{folder}: {metadata.get('cycles')} cycles, the kit's experiment has {stats.CYCLES}")
    site, workers = site_of(metadata)
    rows = {}
    for entry in metadata["decoders"]:
        summary = entry["summary"]
        shots = summary["paired_shots"]
        # what our plugin recorded about itself: the tie-break really used per basis, and where the network ran
        settings = entry.get("configuration") or {}
        rule = ({basis: settings[basis]["rule"] for basis in "XZ"}
                if all(isinstance(settings.get(basis), dict) and "rule" in settings[basis] for basis in "XZ") else None)
        where_network = str((metadata.get("plugin_options") or {}).get("device", "cpu")).lower()
        device = "CPU" if not rule or "network" not in rule.values() or where_network == "cpu" else f"CPU, the network on {where_network}"
        row = {"name": entry["name"], "p": metadata["p"], "shots": shots,
               "failures_X": summary["failures_X"], "failures_Z": summary["failures_Z"], "failures_pair": summary["failures_pair"],
               "mean_ms": summary["mean_pair_total_ms"], "median_ms": summary["median_pair_total_ms"], "p99_ms": summary["p99_pair_total_ms"],
               "test_sha256": {basis: metadata["datasets"]["test"][basis]["sha256"] for basis in "XZ"},
               "circuit_sha256": {basis: metadata["circuits"][basis]["sha256"] for basis in "XZ"},
               "site": site, "workers": workers, "device": device, "rule": rule, "failed_pair": None, "seconds_pair": None,
               "timer": "the whole decode of a pair (evaluation.evaluate)", "wall_seconds": summary.get("evaluation_wall_seconds"),
               "network": {basis: settings[basis].get("network") for basis in "XZ"} if rule else None,
               "peak_rss_mb": (entry.get("peak_parent_rss_bytes_after") or 0) / 2**20 or None,
               "setup_seconds": summary.get("setup_seconds"),
               "settings": entry.get("configuration"), "plugin_options": metadata.get("plugin_options"),
               "source": str(folder.relative_to(DATA)) + "/summary.json", "per_shot_file": None}
        rate, error = stats.per_cycle_rate(summary["failures_pair"], shots)
        if not (math.isclose(rate, summary["per_cycle_rate"], rel_tol=1e-9, abs_tol=1e-15)
                and math.isclose(error, summary["standard_error"], rel_tol=1e-9, abs_tol=1e-15)):
            raise Mismatch(f"{folder}, {entry['name']}: the rate per cycle in summary.json is not 1 - (1 - P)^(1/12) of its pair count")
        row["rate"], row["error"] = rate, error
        artifact = folder / summary["artifacts"]
        if artifact.is_file():
            with np.load(artifact, allow_pickle=False) as file:
                failed_x, failed_z, failed_pair = file["failed_X"].astype(bool), file["failed_Z"].astype(bool), file["failed_pair"].astype(bool)
                seconds = file["seconds_total_pair"].astype(float)
            checks = [(len(failed_pair) == shots, "the number of pairs"),
                      (int(failed_x.sum()) == summary["failures_X"], "the X failures"),
                      (int(failed_z.sum()) == summary["failures_Z"], "the Z failures"),
                      (np.array_equal(failed_pair, failed_x | failed_z), "pair = X or Z"),
                      (int(failed_pair.sum()) == summary["failures_pair"], "the pair failures"),
                      (math.isclose(seconds.mean() * 1e3, summary["mean_pair_total_ms"], rel_tol=1e-9), "the mean time per pair")]
            wrong = [what for passed, what in checks if not passed]
            if wrong:
                raise Mismatch(f"{artifact}: the per-shot file disagrees with summary.json on {', '.join(wrong)}")
            row |= {"failed_pair": failed_pair, "seconds_pair": seconds, "per_shot_file": str(artifact.relative_to(DATA)),
                    "median_ms": float(np.median(seconds) * 1e3), "p99_ms": float(np.percentile(seconds, 99) * 1e3)}
            if not (math.isclose(row["median_ms"], summary["median_pair_total_ms"], rel_tol=1e-9)
                    and math.isclose(row["p99_ms"], summary["p99_pair_total_ms"], rel_tol=1e-9)):
                raise Mismatch(f"{artifact}: median or p99 time per pair disagrees with summary.json")
        rows[entry["name"]] = row
    return rows


def clean_rows_log(path):
    """The server job's START and END lines: {(p, row name): start, end, exit code, rows beside it}.

    "beside" lists the other evaluator rows whose run overlapped this one, so that every latency
    can say what else was running on the machine.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    runs = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        found = re.match(r"(\S+Z) (START|END) p=(\S+) (\S+)(?: exit (\d+))?", line)
        if not found:
            continue
        when = datetime.strptime(found.group(1), "%Y-%m-%dT%H:%M:%SZ")
        run = runs.setdefault((found.group(3), found.group(4)), {"start": None, "end": None, "exit": None})
        if found.group(2) == "START":
            run["start"] = when
        else:
            run["end"], run["exit"] = when, int(found.group(5)) if found.group(5) else None
    for key, run in runs.items():
        run["beside"] = sorted(f"{other[1]} p={other[0]}" for other, r in runs.items()
                               if other != key and run["start"] and r["start"]
                               and r["start"] < (run["end"] or datetime.max) and (r["end"] or datetime.max) > run["start"])
    return runs


def runner_logs(paths):
    """The server runner's lines: {job: start, end}. A job without an END line was still running when the log was copied."""
    jobs = {}
    for path in paths:
        if not Path(path).is_file():
            continue
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            found = re.match(r"(\S+Z) (START|END) (\S+)", line)
            if found:
                when = datetime.strptime(found.group(1), "%Y-%m-%dT%H:%M:%SZ")
                jobs.setdefault(found.group(3), {"start": None, "end": None})["start" if found.group(2) == "START" else "end"] = when
    return jobs


# ---------- TCBP-JOBS ----------

def tcbp_log(path):
    """The one result line of the teammates' tcbp_jobs_baseline.py log (its own sampled shots)."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    found = re.search(r"tcbp-jobs\s+failures \[(\d+) (\d+)\] of (\d+)\s+per cycle (\S+) \+- (\S+)\s+mean (\S+) ms\s+p99 (\S+) ms", text)
    if not found:
        raise Mismatch(f"{path}: no result line")
    started = re.search(r"Started: (\S+)", text)
    host = re.search(r"Host: (\S+)", text)
    seed = re.search(r"seed(\d+)", Path(path).name)
    return {"failures_X": int(found.group(1)), "failures_Z": int(found.group(2)), "shots": int(found.group(3)),
            "rate_printed": float(found.group(4)), "error_printed": float(found.group(5)),
            "mean_ms": float(found.group(6)), "p99_ms": float(found.group(7)),
            "started": started.group(1) if started else "", "host": host.group(1) if host else "",
            "seed": seed.group(1) if seed else "", "serial": "serial decode" in text,
            "source": str(Path(path).relative_to(DATA))}


def tcbp_member(folder, p):
    """TCBP-JOBS on the saved test shots: the two files of hierarchy.tcbp_member, as one row."""
    folder = Path(folder)
    files = {basis: folder / f"test_{basis}.tcbp.npz" for basis in "XZ"}
    if not all(file.is_file() for file in files.values()):
        return None
    loaded = {}
    for basis, file in files.items():
        with np.load(file, allow_pickle=False) as data:
            loaded[basis] = {key: data[key] for key in ("failed_T", "seconds_T", "data_sha256", "mode", "machine") if key in data.files}
    failed = {basis: loaded[basis]["failed_T"].astype(bool) for basis in "XZ"}
    if len(failed["X"]) != len(failed["Z"]):
        raise Mismatch(f"{folder}: X and Z files hold different numbers of shots")
    pair = failed["X"] | failed["Z"]
    seconds = loaded["X"]["seconds_T"].astype(float) + loaded["Z"]["seconds_T"].astype(float)
    timed = not np.isnan(seconds).any()
    rate, error = stats.per_cycle_rate(int(pair.sum()), len(pair))
    return {"name": "tcbp-jobs", "p": float(p), "shots": len(pair),
            "failures_X": int(failed["X"].sum()), "failures_Z": int(failed["Z"].sum()), "failures_pair": int(pair.sum()),
            "rate": rate, "error": error, "failed_pair": pair, "seconds_pair": seconds if timed else None,
            "mean_ms": float(seconds.mean() * 1e3) if timed else None,
            "median_ms": float(np.median(seconds) * 1e3) if timed else None,
            "p99_ms": float(np.percentile(seconds, 99) * 1e3) if timed else None,
            "test_sha256": {basis: str(loaded[basis]["data_sha256"]) for basis in "XZ"},
            "site": "ICRN server, A100 slice (20 GB)", "workers": 1, "device": "GPU", "rule": None,
            "timer": "the decode call alone (hierarchy.tcbp_member), not the evaluator's whole decode", "wall_seconds": None,
            "mode": str(loaded["X"].get("mode", "")), "peak_rss_mb": None,
            "source": str(folder.relative_to(DATA)) + "/test_{X,Z}.tcbp.npz", "per_shot_file": str(files["X"].relative_to(DATA))}


# ---------- loss ----------

def loss_pc(folder):
    """This PC's loss tables (loss/results/*.json): {(s, condition): row}, each with its dataset."""
    rows = {}
    for file in sorted(Path(folder).glob("L[13]_*.json")):
        table = json.loads(file.read_text(encoding="utf-8"))
        dataset = table["dataset"]
        for row in table["rows"]:
            key = (float(dataset["s"]), row["condition"])
            kept = {"s": float(dataset["s"]), "condition": row["condition"], "shots": row["shots"], "failures": row["failures"],
                    "machine": "this PC", "seed": dataset["seed"], "dataset_sha256": dataset["sha256"],
                    "lost_data_atoms": dataset["lost_data_atoms_per_shot"], "source": str(file.relative_to(DATA))}
            if table["reference"] != row["condition"] and "ref_only" in row:
                kept["paired"] = {"reference": table["reference"], "shots": row["paired_shots"], "reference_only": row["ref_only"],
                                  "row_only": row["cond_only"], "p": row["sign_test_p"], "ratio": row["ratio_ref_over_cond"],
                                  "ratio95": row["ratio95"]}
            if key not in rows or kept["shots"] > rows[key]["shots"]:
                kept["paired_by_reference"] = dict(rows.get(key, {}).get("paired_by_reference", {}))
                rows[key] = kept
            if "paired" in kept:
                rows[key].setdefault("paired_by_reference", {})[table["reference"]] = kept["paired"]
    return rows


def loss_server(folder):
    """The server's loss jobs (tables.md of loss.job): {(s, seed, condition): row}."""
    rows = {}
    for file in sorted(Path(folder).glob("*/tables.md")):
        s = seed = reference = None
        for line in file.read_text(encoding="utf-8").splitlines():
            head = re.match(r"### X memory, p = 0\.004, s = (\S+), seed (\d+): reference (\S+)", line)
            if head:
                s, seed, reference = float(head.group(1)), int(head.group(2)), head.group(3)
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            count = re.fullmatch(r"(\d+) / (\d+)", cells[1]) if len(cells) > 7 else None
            if s is None or not count:
                continue
            key = (s, seed, cells[0])
            row = rows.setdefault(key, {"s": s, "seed": seed, "condition": cells[0], "failures": int(count.group(1)),
                                        "shots": int(count.group(2)), "machine": "ICRN server", "paired_by_reference": {},
                                        "source": str(file.relative_to(DATA))})
            only = re.match(r"(\d+) / (\d+) \(p=(\S+)\)", cells[7])
            ratio = re.match(r"(\S+) \((\S+?)–(\S+)\)", cells[8]) if len(cells) > 8 else None
            if only and ratio:
                row["paired_by_reference"][reference] = {
                    "reference": reference, "shots": int(cells[6].split(":")[0]), "reference_only": int(only.group(1)),
                    "row_only": int(only.group(2)), "p": float(only.group(3)), "ratio": float(ratio.group(1)),
                    "ratio95": [float(ratio.group(2)), float(ratio.group(3))]}
    return rows


# ---------- the networks ----------

def train_log(path):
    """A neural.train log: its header facts and, per report, the share of shots wrong at each p.

    The report shots are drawn from the error model with fixed seeds (a training diagnostic),
    not Stim circuit shots.
    """
    path = Path(path)
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    reports, step = [], None
    for line in text.splitlines():
        head = re.match(r"step (\d+) of (\d+) \| training noise p=(\S+)", line)
        if head:
            step = {"step": int(head.group(1)), "of": int(head.group(2)), "trained_at": float(head.group(3)), "wrong": {}}
            reports.append(step)
            continue
        share = re.match(r"\s+p=(\S+)\s+shots wrong (\S+) \(always-0 (\S+)\)", line)
        if share and step is not None:
            step["wrong"][f"{float(share.group(1)):.3f}"] = float(share.group(2))
    by_step = {}
    for report in reports:          # a resumed run prints a step again: keep the last print
        if report["wrong"]:
            by_step[report["step"]] = report
    model = re.search(r"model: width (\d+), depth (\d+), .*?(\d+) parameters \| batch (\d+)", text)
    basis = re.search(r"gross code, ([XZ]) memory", text)
    speeds = [float(value) for value in re.findall(r"speed: (\d+) shots/s in training", text)]
    report_shots = re.findall(r"averaged weights on (\d+) error-model shots", text)
    return {"basis": basis.group(1) if basis else "?", "reports": [by_step[step] for step in sorted(by_step)],
            "width": int(model.group(1)) if model else None, "depth": int(model.group(2)) if model else None,
            "parameters": int(model.group(3)) if model else None, "batch": int(model.group(4)) if model else None,
            "shots_per_second": speeds, "report_shots": sorted({int(value) for value in report_shots}),
            "warm_start": "initial weights: mirrored from X memory" in text, "source": str(path.relative_to(DATA))}


def scores_md(path):
    """The laptop's SCORES.md: the X network on Stim shot files, {(step, file): (share wrong, low, high, shots)}."""
    path = Path(path)
    if not path.is_file():
        return {}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("|")]
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    scores = {}
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        for name, cell in zip(header[1:], cells[1:]):
            found = re.match(r"(\S+)% \[(\S+)%, (\S+)%\] of (\d+)", cell)
            if found:
                scores[(int(cells[0]), name)] = (float(found.group(1)) / 100, float(found.group(2)) / 100,
                                                 float(found.group(3)) / 100, int(found.group(4)))
    return scores


# ---------- exploration shots: the three candidates of every shot ----------

def selector_files(folder, names):
    """The laptop's per-shot files (relay-BP R, PriorTilt P1 and P2 on the same shots), joined in the order given."""
    parts = []
    for name in names:
        file = Path(folder) / name
        if not file.is_file():
            return None
        with np.load(file, allow_pickle=False) as data:
            parts.append({key: data[key] for key in data.files if key.split("_")[0] in ("truth", "predicted", "failed", "converged", "weight", "seconds")}
                         | {"data_sha256": str(data["data_sha256"])})
    joined = {key: np.concatenate([part[key] for part in parts]) for key in parts[0] if key != "data_sha256"}
    joined["data_sha256"] = [part["data_sha256"] for part in parts]
    joined["source"] = [str((Path(folder) / name).relative_to(DATA)) for name in names]
    return joined


def weight_rule(shots):
    """The hierarchy without a network, scored offline on per-shot candidates.

    P1 where P1 and P2 predict the same 12 bits; otherwise the candidate of lowest weight among
    relay-BP, P1 and P2 (the weight is the sum of log((1 - prior) / prior) over a candidate's
    faults: lower is a more probable explanation). Returns (failed, agreed, seconds): the time
    is the sum of the decode calls the rule makes, so relay-BP counts only where it runs.
    """
    agreed = (shots["predicted_P1"] == shots["predicted_P2"]).all(axis=1)
    weights = np.stack([shots["weight_R"], shots["weight_P1"], shots["weight_P2"]], axis=1)
    failed_each = np.stack([shots["failed_R"], shots["failed_P1"], shots["failed_P2"]], axis=1).astype(bool)
    pick = np.argmin(weights, axis=1)
    failed = np.where(agreed, shots["failed_P1"].astype(bool), failed_each[np.arange(len(pick)), pick])
    seconds = shots["seconds_P1"] + shots["seconds_P2"] + np.where(agreed, 0.0, shots["seconds_R"])
    return failed, agreed, seconds


def network_rule(shots, logits):
    """The hierarchy with the neural tie-break, scored offline: P1 where P1 and P2 agree; otherwise
    the candidate among relay-BP, P1 and P2 whose 12 bits the network finds most likely.

    A candidate's score is the sum over the 12 bits of log sigmoid(logit) where it says 1 and
    log sigmoid(-logit) where it says 0 (hierarchy/rules.py, network_score). Ties go to the first
    of relay-BP, P1, P2. Returns the per-shot failure mask.
    """
    agreed = (shots["predicted_P1"] == shots["predicted_P2"]).all(axis=1)
    predicted = np.stack([shots["predicted_R"], shots["predicted_P1"], shots["predicted_P2"]])
    logits = np.asarray(logits, np.float64)
    score = np.where(predicted == 1, -np.logaddexp(0.0, -logits), -np.logaddexp(0.0, logits)).sum(axis=-1)
    pick = score.argmax(axis=0)
    failed_each = np.stack([shots["failed_R"], shots["failed_P1"], shots["failed_P2"]], axis=1).astype(bool)
    return np.where(agreed, shots["failed_P1"].astype(bool), failed_each[np.arange(len(pick)), pick])


def network_logits(folder, names, truth):
    """The X network's per-shot logits for the same shot files, joined in the order given; None if a file is missing."""
    parts = []
    for name in names:
        file = Path(folder) / name
        if not file.is_file():
            return None
        with np.load(file, allow_pickle=False) as data:
            parts.append((data["logits"], data["observables"]))
    logits, observables = (np.concatenate([part[index] for part in parts]) for index in (0, 1))
    if observables.shape != truth.shape or not np.array_equal(observables, truth):
        raise Mismatch(f"{folder}: the network's logits are not for the same shots as the candidates")
    return logits


def network_rule_table(path):
    """The laptop's selection table for one checkpoint: {row label: (failures, ms per shot or None)}, and its header line."""
    path = Path(path)
    if not path.is_file():
        return None
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    rows = {}
    for line in lines[3:]:
        found = re.match(r"(.+?)\s{2,}(\d+)\s+(\S+)%\s+(?:(\d+) of (\d+)\s+)?(\S+)\s*$", line)
        if found:
            rows[found.group(1).strip()] = (int(found.group(2)), float(found.group(6)))
    return {"header": lines[0], "rows": rows, "source": str(path.relative_to(DATA))}
