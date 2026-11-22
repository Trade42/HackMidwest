"""Tests of the report's arithmetic and of the two scripts.

Run from challenge-1-ml-decoder-kit/:  python -m pytest -p no:cacheprovider report

The scripts are also run on a stand-in data folder in which the teammates' committed evaluator
rows (results/icrn, results/mac) play the part of the timed rows, so that the tables and
figures are exercised with numbers in them. Nothing made from the stand-in is kept.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from report import stats
from report.rows import TEST_SHA256

KIT = Path(__file__).resolve().parent.parent
REPORT = KIT / "report"


def test_per_cycle_rate_is_the_kits():
    # the kit's README: relay-BP at p = 0.004 failed [45 60] of 10,000, "per cycle 8.71e-04 +- 8.5e-05";
    # its run had 104 failed pairs or so. Here: the formula on a known count.
    rate, error = stats.per_cycle_rate(5, 10000)
    assert rate == pytest.approx(4.167621832917856e-05, rel=1e-12)      # results/icrn/eval_p0.003, relay-bp
    assert error == pytest.approx(1.8639694121413974e-05, rel=1e-12)
    assert stats.per_cycle(1 - (1 - 0.01) ** 12) == pytest.approx(0.01)
    assert stats.per_cycle_rate(0, 100) == (0.0, 0.0)


def test_wilson_interval():
    low, high = stats.wilson(122, 25000)          # hierarchy/README.md: 0.488% [0.409%, 0.582%]
    assert round(low * 100, 3) == 0.409 and round(high * 100, 3) == 0.582
    low, high = stats.wilson(0, 500)              # neural/README.md: 0 of 500 -> [0, 0.0076]
    assert low == 0 and round(high, 4) == 0.0076


def test_paired_counts():
    reference = np.array([1, 1, 0, 0, 1, 0], bool)
    row = np.array([1, 0, 1, 0, 0, 0], bool)
    assert stats.paired(reference, row) == {"both": 1, "reference_only": 2, "row_only": 1, "p": pytest.approx(1.0)}
    assert stats.paired(reference, reference)["p"] == 1.0
    with pytest.raises(ValueError):
        stats.paired(reference, row[:3])


@pytest.mark.parametrize("reference, candidate, gain", [
    ("results/icrn/eval_p0.003/02_relay-bp_shots.npz", "results/icrn/router_eval_p0.003_portable/01_evaluation_router_make_tiered_decoder_shots.npz",
     "results/icrn/gain_p0.003_portable/comparison.json"),
    ("results/mac/eval_p0.004/02_relay-bp_shots.npz", "results/mac/router_eval_p0.004/01_evaluation_router_make_tiered_decoder_shots.npz",
     "results/mac/gain_p0.004/comparison.json"),
])
def test_paired_counts_agree_with_the_teams_compare_gain(reference, candidate, gain):
    theirs = json.loads((KIT / gain).read_text())["paired_failures"]
    mine = stats.paired(np.load(KIT / reference)["failed_pair"], np.load(KIT / candidate)["failed_pair"])
    assert (mine["both"], mine["reference_only"], mine["row_only"]) == (theirs["both_fail"], theirs["reference_only_fail"], theirs["candidate_only_fail"])
    assert mine["p"] == pytest.approx(theirs["exact_discordant_sign_test_p"])


def test_the_pinned_test_files_are_the_repositorys():
    for p, by_basis in TEST_SHA256.items():
        for basis, expected in by_basis.items():
            file = KIT / "data" / f"p{p}" / f"test_{basis}.npz"
            assert hashlib.sha256(file.read_bytes()).hexdigest() == expected      # hashed, never opened


def test_every_copied_file_matches_its_record():
    records = json.loads((REPORT / "data" / "sources.json").read_text())
    for name, record in records.items():
        assert hashlib.sha256((REPORT / "data" / name).read_bytes()).hexdigest() == record["sha256"], name


def test_the_top_readme_holds_the_generated_section():
    """Once our section is appended to the top-level README.md, it must be the generated one, and nothing above it may change."""
    from report.make_tables import README_HEADING
    readme = (KIT.parent / "README.md").read_text(encoding="utf-8")
    if README_HEADING not in readme:
        pytest.skip("our section is not appended yet")
    section = (REPORT / "tables" / "readme_section.md").read_text(encoding="utf-8")
    assert readme.endswith("\n\n" + section)
    assert len(section.rstrip("\n").split("\n")) <= 25


def stand_in(folder):
    """A copy of report/data/ whose test/ rows are the teammates' committed evaluator runs."""
    data = folder / "data"
    shutil.copytree(REPORT / "data", data, ignore=lambda folder, names: ["test"] if Path(folder) == REPORT / "data" else [])
    log = []
    for p, base, router in (("0.003", "results/icrn/eval_p0.003", "results/icrn/router_eval_p0.003_portable"),
                            ("0.004", "results/mac/eval_p0.004", "results/mac/router_eval_p0.004")):
        for source, name, row in ((base, "bposd", "bposd"), (base, "relay-bp", "relay-bp"), (router, "evaluation.router:make_tiered_decoder", "router")):
            metadata = json.loads((KIT / source / "summary.json").read_text())
            metadata["decoders"] = [entry for entry in metadata["decoders"] if entry["name"] == name]
            target = data / "test" / f"eval_p{p}_{row}"
            target.mkdir(parents=True)
            (target / "summary.json").write_text(json.dumps(metadata))
            artifact = metadata["decoders"][0]["summary"]["artifacts"]
            shutil.copy(KIT / source / artifact, target / artifact)
            state = "[load 6.50, evaluators 5, joint 0, loss.job 0, candidates 0, throttled so far %s s]"
            log += [f"2026-11-22T08:00:00Z START p={p} {row} " + state % "100.0", f"2026-11-22T09:00:00Z END p={p} {row} exit 0: x " + state % "161.0"]
    (data / "test" / "clean_rows.out").write_text("\n".join(log) + "\n")
    (data / "test" / "cpu_monitor.out").write_text(
        "# one line per minute (the quota is 12)\n# time_utc cpus thr_s thr_n | evaluator joint loss_job candidates tcbp_member evaluate_stim python_all | load_1m\n"
        + "".join(f"2026-11-22T08:{minute:02d}:30Z 6.0 {100 + minute} 5 | 6 0 0 1 0 0 9 | 6.5\n" for minute in range(0, 60, 10)))
    # the offline combination at p = 0.003: made-up rules on real per-shot outcomes, only to exercise the tables
    relay, other = (np.load(KIT / folder / name) for folder, name in (
        ("results/icrn/eval_p0.003", "02_relay-bp_shots.npz"), ("results/icrn/router_eval_p0.003_portable", "01_evaluation_router_make_tiered_decoder_shots.npz")))
    target = data / "test" / "ensemble_p0.003"
    target.mkdir()
    for basis in "XZ":
        r, first = relay[f"failed_{basis}"], other[f"failed_{basis}"]
        later = (r != first) | (np.arange(len(r)) < 30)
        weight = np.where(later, r, first)
        network = weight.copy()
        network[np.flatnonzero(weight)[:1]] = False
        arrays = {"failed_R": r, "failed_P1": first, "failed_P2": first, "failed_aw": weight, "later_aw": later, "failed_a": network, "later_a": later,
                  "failed_oracle_R_P1_P2": r & first}
        np.savez(target / f"test_{basis}.npz", data_sha256=TEST_SHA256["0.003"][basis], basis=basis, p=0.003, rule="a", **arrays)
        rows = [{"key": key.removeprefix("failed_"), "failures": int(value.sum()), "ms_per_shot": 40.0} for key, value in arrays.items() if key.startswith("failed_")]
        (target / f"test_{basis}.json").write_text(json.dumps({"rows": rows, "marked_rule": "a"}))
    return data


def run(module, data, out):
    environment = os.environ | {"REPORT_DATA": str(data), "REPORT_OUT": str(out), "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([sys.executable, "-m", module], cwd=KIT, env=environment, capture_output=True, text=True)


def test_scripts_on_a_stand_in(tmp_path):
    data = stand_in(tmp_path)
    done = run("report.make_tables", data, tmp_path)
    assert done.returncode == 0, done.stderr
    table = (tmp_path / "tables" / "test_p0.003.md").read_text()
    assert "| relay-BP (kit baseline) | the kit | 2, 3 | 5 of 10,000 | 4.17e-05 ± 1.9e-05 |" in table
    assert "4 / 2 / 1 (p = 1.00)" in table                       # the router against relay-BP, as in the team's comparison.json
    assert "our hierarchy, neural tie-break | ours | not measured" in table
    assert "PriorTilt, one run (from per-shot files, not a timed evaluator row)" in table      # the offline combination stands in
    assert "6.0 CPUs busy on average" in table and "throttled the notebook for 61 s summed over its CPUs during the row's 3,600 s, at most 0.1% of one process's time" in table
    assert "time not clean" not in table
    ladder = (tmp_path / "tables" / "test_ladder.md").read_text()
    assert "ORACLE: a perfect tie-break" in ladder and "| 0.004 | every rung | not measured" in ladder
    facts = (tmp_path / "tables" / "facts.md").read_text()
    assert "`framing` | On the test shots (one offline decode of each shot" in facts and "no better than the hand-written rule" in facts
    assert "not measured" in (tmp_path / "tables" / "test_p0.005.md").read_text()
    assert "the same outcome on every one of the 10,000 pairs" in (tmp_path / "tables" / "checks.md").read_text()
    assert "not measured" in (tmp_path / "tables" / "test_after_swap.md").read_text()      # the stand-in has no rule d
    done = run("report.make_plots", data, tmp_path)
    assert done.returncode == 0, done.stderr
    first = {file.name: file.read_bytes() for file in (tmp_path / "figures").iterdir()}
    assert len(first) == 12
    run("report.make_plots", data, tmp_path)                     # a second run gives the same files, byte for byte
    assert first == {file.name: file.read_bytes() for file in (tmp_path / "figures").iterdir()}


def test_a_retimed_row(tmp_path):
    data = stand_in(tmp_path)
    first, again = data / "test" / "eval_p0.003_relay-bp", data / "test" / "eval_p0.003_relay-bp_retime"
    shutil.copytree(first, again)
    metadata = json.loads((again / "summary.json").read_text())
    summary = metadata["decoders"][0]["summary"]
    arrays = dict(np.load(again / summary["artifacts"]))
    for key in ("seconds_total_X", "seconds_total_Z", "seconds_total_pair"):
        arrays[key] = arrays[key] / 2                       # the same answers, decoded twice as fast
    np.savez(again / summary["artifacts"], **arrays)
    for key in ("mean_pair_total_ms", "median_pair_total_ms", "p99_pair_total_ms"):
        summary[key] /= 2
    (again / "summary.json").write_text(json.dumps(metadata))
    done = run("report.make_tables", data, tmp_path)
    assert done.returncode == 0, done.stderr
    table = (tmp_path / "tables" / "test_p0.003.md").read_text()
    assert "| 42.8 / 26.3 / 278 |" in table and "the first run's mean was 85.7 ms" in table
    arrays["failed_X"][0] = ~arrays["failed_X"][0]           # other answers: the report must stop
    arrays["failed_pair"] = arrays["failed_X"] | arrays["failed_Z"]
    np.savez(again / summary["artifacts"], **arrays)
    summary["failures_X"], summary["failures_pair"] = int(arrays["failed_X"].sum()), int(arrays["failed_pair"].sum())
    summary["per_cycle_rate"], summary["standard_error"] = stats.per_cycle_rate(summary["failures_pair"], 10000)
    (again / "summary.json").write_text(json.dumps(metadata))
    done = run("report.make_tables", data, tmp_path)
    assert done.returncode == 1 and "differ from the first run" in done.stderr


def test_other_shots_are_refused(tmp_path):
    data = stand_in(tmp_path)
    file = data / "test" / "eval_p0.003_bposd" / "summary.json"
    metadata = json.loads(file.read_text())
    metadata["datasets"]["test"]["X"]["sha256"] = "0" * 64
    file.write_text(json.dumps(metadata))
    done = run("report.make_tables", data, tmp_path)
    assert done.returncode == 1 and "not the team's test shots" in done.stderr


def test_a_per_shot_file_that_disagrees_is_refused(tmp_path):
    data = stand_in(tmp_path)
    folder = data / "test" / "eval_p0.003_relay-bp"
    file = folder / "02_relay-bp_shots.npz"
    arrays = dict(np.load(file))
    arrays["failed_pair"][0] = ~arrays["failed_pair"][0]
    np.savez(file, **arrays)
    done = run("report.make_tables", data, tmp_path)
    assert done.returncode == 1 and "disagrees with summary.json" in done.stderr


def test_the_documents_hold_the_generated_tables():
    """Each table block of the committed documents holds exactly the table file make_tables wrote."""
    import re
    for name in ("REPORT.md", "APPENDIX.md", "SLIDES.md"):
        text = (REPORT / name).read_text(encoding="utf-8")
        blocks = re.findall(r"<!-- BEGIN table:(\S+) -->\n(.*?)<!-- END table:\1 -->", text, re.S)
        assert len(blocks) == text.count("<!-- BEGIN table:"), name
        for table, body in blocks:
            assert body.strip(), f"{name}: the table {table} is empty"
            assert body.rstrip("\n") == (REPORT / "tables" / f"{table}.md").read_text(encoding="utf-8").rstrip("\n"), f"{name}: {table}"


def test_the_rows_run_after_the_test_results_are_labelled_and_recomputed():
    """The swap (TCBP-JOBS in place of relay-BP, the ensemble's rule d) is labelled as an ablation and re-counted from the saved answers."""
    from report.make_tables import AFTER_TEST
    from report.rows import P_VALUES
    for name in ("test_after_swap", "test_after_swap_short"):
        assert (REPORT / "tables" / f"{name}.md").read_text(encoding="utf-8").startswith(f"**{AFTER_TEST[0].upper() + AFTER_TEST[1:]}.**")
    table = (REPORT / "tables" / "test_after_swap.md").read_text(encoding="utf-8")
    for p in P_VALUES:
        files = [np.load(REPORT / "data" / "test" / f"ensemble_p{p}" / f"test_{basis}.npz") for basis in "XZ"]
        pair = {key: files[0][f"failed_{key}"] | files[1][f"failed_{key}"] for key in ("d", "T", "a")}
        for other in ("T", "a"):
            paired = stats.paired(pair[other], pair["d"])
            assert (f"| {int(pair['d'].sum())} against {int(pair[other].sum())}; {paired['both']} / {paired['row_only']} / "
                    f"{paired['reference_only']} (p = {stats.p_text(paired['p'])}) |") in table, (p, other)
    facts = (REPORT / "tables" / "facts.md").read_text(encoding="utf-8")
    statement = next(line for line in facts.splitlines() if line.startswith("| `swap_statement` |"))
    assert AFTER_TEST in statement and "our result" not in statement


def test_the_predecoder_rows_are_labelled_and_match_their_files():
    """The predecoder's test rows keep the audit's labels and the run's own counts; they never enter a pair-scored table."""
    from report.rows import PREDECODER_LABEL
    table = (REPORT / "tables" / "test_predecoder.md").read_text(encoding="utf-8")
    assert table.startswith(f"**The learned predecoder on the test shots: {PREDECODER_LABEL}.**")
    for file in sorted((REPORT / "data" / "test" / "predecoder").glob("test_run_p*_summary.json")):
        summary = json.loads(file.read_text())
        pair = summary["pairs"]["head_vs_fresh"]
        assert f"| {summary['arms']['head']['failures']} of {summary['shots']:,} |" in table
        assert f"| {pair['both']} / {pair['only_head']} / {pair['only_fresh']} (p = {stats.p_text(pair['p'])}) |" in table
    for name in ("test_sweep", "test_p0.005", "test_comparisons", "readme_section"):
        assert "head's priors" not in (REPORT / "tables" / f"{name}.md").read_text(encoding="utf-8")


def test_rows_run_after_the_test_results_are_labelled_where_they_appear():
    """A timed row is labelled an ablation run after the test results exactly when its START is after the TEST RUN END line."""
    from report import loaders
    from report.make_tables import AFTER_SHORT, AFTER_TEST, load_test_rows
    from report.rows import DECODERS, P_VALUES
    measured, _, _ = load_test_rows()
    end = loaders.test_run_end(REPORT / "data" / "test" / "test_rows.out")
    assert end is not None
    sweep = (REPORT / "tables" / "test_sweep.md").read_text(encoding="utf-8")
    for p in P_VALUES:
        table = (REPORT / "tables" / f"test_p{p}.md").read_text(encoding="utf-8")
        for key, row in measured[p].items():
            if row is None or row.get("offline") or row.get("retimed") or not (row.get("run") or {}).get("start"):
                continue                   # a re-timed row's run is its second, timing-only run; its answers are the first run's
            after = row["run"]["start"] > end
            assert row["after_test"] == after and (AFTER_TEST in row["label_note"]) == after, (p, key)
            if after:
                assert f"| {DECODERS[key].label}{row['label_note']} |" in table, (p, key)
                line = next(line for line in sweep.splitlines() if line.startswith(f"| {DECODERS[key].label} |"))
                assert line.split(" | ")[1 + P_VALUES.index(p)].endswith(f"**{AFTER_SHORT}**") or line.split(" | ")[1 + P_VALUES.index(p)].endswith(f"**{AFTER_SHORT}** |"), (p, key)
    for p in ("0.003", "0.004"):            # the pre-registered arm of the test run itself
        assert measured[p]["hierarchy-converged"] is not None and not measured[p]["hierarchy-converged"]["after_test"]
