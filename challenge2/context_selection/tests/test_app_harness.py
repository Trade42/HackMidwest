"""The whole pipeline on three real questions with the stub reader (no downloads)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from qctx.config import DEFAULTS, deep_merge, load_config
from qctx.demo import format_rows, run_demo
from qctx.data import read_questions
from qctx.figures import make_figures
from qctx.harness import Experiment, lane_label, results_table, run_experiment
from qctx.hypotheses import evaluate_hypotheses, hypotheses_markdown
from qctx.studies import cz_count, run_studies

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    work = tmp_path_factory.mktemp("exp")
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    cfg["data"] = {**cfg["data"], "questions": str(ROOT / "data" / "demo.jsonl"),
                   "calibration": str(ROOT / "data" / "calibration.jsonl"), "demo": str(ROOT / "data" / "demo.jsonl")}
    shutil.copytree(ROOT / "scores", work / "scores")
    cfg["scores"] = {**cfg["scores"], "dir": str(work / "scores")}
    cfg["output"] = str(work / "results")
    cfg["figures"] = str(work / "figures")
    cfg["qaoa"] = {**cfg["qaoa"], "shots": 300}
    return cfg


@pytest.fixture(scope="module")
def summary(cfg):
    return run_experiment(cfg, log=lambda *_: None)


def test_config_merges_over_defaults(tmp_path):
    path = tmp_path / "mine.yaml"
    path.write_text("qaoa: {shots: 123}\nlanes: ['1a']\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg["qaoa"]["shots"] == 123 and cfg["qaoa"]["top"] == DEFAULTS["qaoa"]["top"]
    assert cfg["name"] == "mine" and cfg["output"] == "results/mine" and cfg["lanes"] == ["1a"]
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 5}}) == {"a": {"b": 5, "c": 2}}


def test_every_lane_reports_every_question(cfg, summary):
    lanes = summary["lanes"]
    for lane in ("1a", "1b", "2a", "2b", "2c", "rand", "3a", "3a-p2", "3b", "3c", "3c-direct", "4", "3b@32", "3c@32"):
        assert lane in lanes, lane
        assert lanes[lane]["n"] == 3
        path = Path(cfg["output"]) / f"answers_{lane.replace('@', '_at_')}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 3 and all(r["answer_single"] is not None for r in rows)
    assert lanes["1a"]["n_selected"]["mean"] == 10
    assert lanes["2a"]["n_selected"]["mean"] < 10
    assert lanes["2a"]["tokens_single"]["mean"] < lanes["1a"]["tokens_single"]["mean"]
    assert lane_label("3c@64").endswith("64 shots")


def test_solver_metrics_are_ordered_sensibly(summary):
    lanes = summary["lanes"]
    assert lanes["3a"]["kept_share"]["mean"] == 1.0 and lanes["3b"]["kept_share"]["mean"] == 1.0
    assert 0.05 < lanes["3c"]["kept_share"]["mean"] < 1.0                 # error detection discards shots
    assert lanes["3c"]["two_qubit_gates"]["mean"] > lanes["3b"]["two_qubit_gates"]["mean"]
    assert lanes["3a"]["approx_ratio"]["mean"] > lanes["rand"]["approx_ratio"]["mean"]
    assert lanes["3a-p2"]["approx_ratio"]["mean"] >= lanes["3a"]["approx_ratio"]["mean"] - 0.01
    assert lanes["2a"]["gold_recall"]["mean"] >= 0.5
    assert lanes["3c"]["qubits"]["mean"] == 18


def test_exact_and_ideal_lanes_agree_on_the_top_selection(cfg):
    out = Path(cfg["output"])
    exact = [json.loads(line) for line in (out / "answers_2a.jsonl").read_text(encoding="utf-8").splitlines()]
    ideal = [json.loads(line) for line in (out / "answers_3a-p2.jsonl").read_text(encoding="utf-8").splitlines()]
    same = sum(a["selected"] == b["selected"] for a, b in zip(exact, ideal))
    assert same >= 2


def test_reports_are_written(cfg, summary):
    out = Path(cfg["output"])
    for name in ("summary.json", "hypotheses.md", "results_table.md", "per_question.json", "angles.json"):
        assert (out / name).exists(), name
    ids = [h["id"] for h in summary["hypotheses"]]
    assert ids == ["H1", "H2", "H3", "H4", "H5"]
    assert "| 3c |" in results_table(summary)
    assert "H3" in (out / "hypotheses.md").read_text(encoding="utf-8")


def test_rerun_uses_cached_raw_counts(cfg, summary):
    again = Experiment(cfg, log=lambda *_: None)
    again.ensure_scores()
    again.build_problems()
    before = {p: p.stat().st_mtime_ns for p in (Path(cfg["output"]) / "raw").rglob("*.json.gz")}
    again.select_lane("3c")
    after = {p: p.stat().st_mtime_ns for p in (Path(cfg["output"]) / "raw").rglob("*.json.gz")}
    assert before == after and len(before) >= 15


def test_studies_and_figures(cfg, summary, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)                       # the hand-over file is written relative to the cwd
    cfg = {**cfg, "circuit_stats": "circuits/stats.json"}   # only configs that name it write it
    studies = run_studies(cfg, quick=True, log=lambda *_: None)
    budget = studies["fault_budget"]["tiers"]
    assert budget["bus"]["undetected_share"] < budget["direct"]["undetected_share"] < budget["bare"]["undetected_share"]
    assert budget["bus"]["ar_loss_slope"] < budget["bare"]["ar_loss_slope"]
    blocks = studies["fault_budget"]["bus_blocks"]
    assert blocks["init"]["undetected_error"] == 0 and blocks["syndrome"]["undetected_error"] == 0
    assert blocks["measure"]["undetected_error"] == 0
    stats = studies["compile_stats"]["mean"]
    assert stats["bare_grid"] >= stats["bare_all_to_all"]
    assert stats["direct_grid"] > stats["direct_all_to_all"]
    assert (tmp_path / "circuits" / "stats.json").exists()
    road = studies["roadmap"]["rows"]
    assert [r["paragraphs"] for r in road] == [10, 14, 18]
    assert road[0]["bus_fits_6x4"] and not road[-1]["bus_fits_6x4"]
    figures = make_figures(cfg)
    assert len(figures) == 8 and all(p.exists() and p.stat().st_size > 10_000 for p in figures)


def test_cz_count_counts_decompositions():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3)
    qc.h(0); qc.cx(0, 1); qc.rzz(0.3, 1, 2); qc.swap(0, 2); qc.cz(0, 1)
    assert cz_count(qc) == 1 + 2 + 3 + 1


def test_hypothesis_verdicts_follow_the_registered_rules():
    n = 200
    rng = np.random.default_rng(1)
    base = (rng.random(n) < 0.4).astype(float)
    better = np.clip(base + (rng.random(n) < 0.3), 0, 1)
    table = {
        "1a": {"em_single": base, "em_vote": base},
        "2a": {"em_single": better, "em_vote": better},
        "3a-p2": {"em_single": better, "em_vote": better, "hit_optimum": np.ones(n)},
        "3b": {"em_single": better, "em_vote": better, "approx_ratio": np.full(n, 0.90) + rng.normal(0, 0.01, n)},
        "3c": {"em_single": better, "em_vote": better, "approx_ratio": np.full(n, 0.92) + rng.normal(0, 0.01, n)},
    }
    verdict = {h["id"]: h["verdict"] for h in evaluate_hypotheses(table, n_boot=500)}
    assert verdict == {"H1": "supported", "H2": "supported", "H3": "supported", "H4": "supported", "H5": "not supported"}
    table["3c"]["approx_ratio"] = table["3b"]["approx_ratio"] - 0.02
    worse = {h["id"]: h["verdict"] for h in evaluate_hypotheses(table, n_boot=500)}
    assert worse["H3"] == "contradicted" and worse["H4"] == "not applicable"
    missing = evaluate_hypotheses({"1a": table["1a"]}, n_boot=100)
    assert all(h["verdict"] == "not evaluated" for h in missing)
    assert "not evaluated" in hypotheses_markdown(missing)


def test_demo_runs_every_lane_on_one_question(cfg, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    question = read_questions(cfg["data"]["demo"])[0]
    rows = run_demo(cfg, question, fresh=True)
    assert [r["lane"] for r in rows] == ["1a", "1b", "2a", "2b", "3a", "3b", "3c"]
    assert rows[0]["tokens"] > rows[2]["tokens"]
    text = format_rows(question, rows)
    assert question.question in text and "error-detected" in text


def test_oracle_lane_reads_the_gold_paragraphs_and_changes_nothing_else(cfg, tmp_path):
    def run(lanes, name):
        c = {**cfg, "lanes": lanes, "output": str(tmp_path / name), "qaoa": {**cfg["qaoa"], "shot_budgets": []}}
        return run_experiment(c, log=lambda *_: None)

    with_oracle = run(["1a", "2a", "oracle"], "with")
    without = run(["1a", "2a"], "without")
    lane = with_oracle["lanes"]["oracle"]
    assert lane["label"] == "Gold paragraphs (oracle)" and lane["n"] == 3
    assert lane["both_gold"]["mean"] == 1.0 and lane["gold_recall"]["mean"] == 1.0
    assert lane["tokens_vote"]["mean"] == lane["tokens_single"]["mean"]     # one selection, one call
    rows = [json.loads(line) for line in (tmp_path / "with" / "answers_oracle.jsonl").read_text(encoding="utf-8").splitlines()]
    gold = {q.qid: sorted(q.gold) for q in read_questions(cfg["data"]["questions"])[:3]}
    assert all(r["selected"] == gold[r["qid"]] and "solver" not in r for r in rows)
    assert "| oracle | Gold paragraphs (oracle) |" in results_table(with_oracle)
    for name in ("1a", "2a"):
        for metric in ("em_single", "em_vote", "f1_single", "gold_recall", "tokens_single"):
            assert with_oracle["lanes"][name][metric] == without["lanes"][name][metric]
    assert with_oracle["hypotheses"] == without["hypotheses"]


def test_equal_kept_shots_gives_the_error_detected_tier_matching_accepted_shots(cfg):
    from qctx.studies import equal_kept_shots, load_instances

    shots = 300
    res = equal_kept_shots(cfg, load_instances(cfg, 2), shots=shots, n_boot=200)
    bare, raw, kept = (res["tiers"][k] for k in ("bare", "bus_equal_raw", "bus_equal_kept"))
    assert bare["mean_raw_shots"] == bare["mean_kept_shots"] == shots and bare["raw_shot_multiplier"] == 1.0
    assert raw["mean_raw_shots"] == shots and raw["mean_kept_shots"] < shots
    assert kept["raw_shot_multiplier"] > 1.0 and abs(kept["mean_kept_shots"] - shots) < 0.2 * shots
    for row in res["tiers"].values():
        assert 0.0 < row["mean_approx_ratio"] <= 1.0 and 0.0 <= row["optimum_in_sample_share"] <= 1.0
    assert len(res["per_question"]) == 2 and "bus_equal_kept_minus_bare_approx_ratio" in res["paired_differences"]
