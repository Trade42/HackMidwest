"""End-to-end evaluation: every lane, every tier, one results folder.

Stages, each timed and each restartable from its files on disk:

1. scores      relevance and overlap per question            -> ``scores/{qid}.npz``
2. problems    one QUBO per question, angles tuned once on the ideal tier
3. selections  every lane picks its ranked selections         -> ``raw/{lane}/{qid}.json.gz``
4. answers     the language model answers from each selection -> ``answers_{lane}.jsonl``
5. summary     metrics with 95% intervals, hypothesis verdicts -> ``summary.json`` etc.

Lanes (the plan's table, plus two controls):

    1a  all paragraphs            2a  brute force (exact)      3a  QAOA, ideal tier
    1b  top-k by similarity       2b  greedy                   3b  QAOA, physical tier (bare)
    rand  uniform random sampler  2c  simulated annealing      3c  QAOA, error-detected tier
    3a-p2  ideal tier at p = 2 (hypothesis H2)                 3c-direct  textbook encoding
    3d-b / 3d-c  physical and error-detected tiers on the hardware backend
    4   router decides per question
    oracle  the gold supporting paragraphs (upper-bound diagnostic; no hypothesis uses it)
    <lane>@S  the same lane restricted to S raw shots (equal-shot comparison)
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import iceberg
from qsel import (Angles, QaoaSimulator, RankedSelection, SelectionResult, build_qubo, make_backend,
                  prepare_circuit, rank_selections, route, sample_stats, select_qubo, subsample_counts,
                  tune_angles)
from qsel.bits import selected
from qsel.classical import greedy

from .data import Question, read_questions
from .hypotheses import evaluate_hypotheses, hypotheses_markdown
from .lm import AnswerCache, make_lm
from .metrics import bootstrap_ci, both_gold, exact_match, f1_score, gold_recall
from .scoring import load_scores, make_scorer, score_path, score_questions
from .vote import majority_vote

CLASSICAL = {"1b": "top_k", "2a": "brute_force", "2b": "greedy", "2c": "sim_anneal"}
SAMPLED = ("rand", "3a", "3a-p2", "3b", "3c", "3c-direct", "3d-b", "3d-c")
LANE_LABELS = {
    "1a": "All 10 paragraphs", "1b": "Top-k by similarity", "2a": "Brute force (exact)",
    "2b": "Greedy", "2c": "Simulated annealing", "rand": "Random sampler (control)",
    "3a": "QAOA ideal", "3a-p2": "QAOA ideal, p=2", "3b": "QAOA bare, noisy",
    "3c": "QAOA error-detected, noisy", "3c-direct": "QAOA textbook Iceberg, noisy",
    "3d-b": "QAOA bare, Superstaq target", "3d-c": "QAOA error-detected, Superstaq target", "4": "Routed",
    "oracle": "Gold paragraphs (oracle)",
}


def lane_label(lane: str) -> str:
    base, _, budget = lane.partition("@")
    label = LANE_LABELS.get(base, base)
    return f"{label}, {budget} shots" if budget else label


# ------------------------------------------------------------------ small io
def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)


def _write_gz(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(obj, fh)


def _read_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _mean_ci(values: Sequence[float], n_boot: int, seed: int) -> dict[str, float]:
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    mean, lo, hi = bootstrap_ci(v, n_boot=n_boot, seed=seed)
    return {"mean": float(mean), "lo": float(lo), "hi": float(hi), "n": int(len(v))}


# ---------------------------------------------------------------- experiment
class Experiment:
    """One configured run. ``Experiment(cfg).run()`` does everything."""

    def __init__(self, cfg: Mapping[str, Any], *, log=print):
        self.cfg = cfg
        self.log = log
        self.out = Path(cfg["output"])
        self.out.mkdir(parents=True, exist_ok=True)
        self.seed = int(cfg["seed"])
        self.questions: list[Question] = read_questions(cfg["data"]["questions"])
        if cfg["data"].get("limit"):
            self.questions = self.questions[: int(cfg["data"]["limit"])]
        self.by_qid = {q.qid: q for q in self.questions}
        self.timings: dict[str, dict[str, float]] = {q.qid: {} for q in self.questions}
        self.qubos: dict[str, Any] = {}
        self.sims: dict[str, QaoaSimulator] = {}
        self.angles: dict[tuple[str, int], Angles] = {}
        self._backends: dict[str, Any] = {}
        self._lm = None
        self._cache: AnswerCache | None = None

    # ------------------------------------------------------------- stage 1
    def ensure_scores(self) -> None:
        sdir = Path(self.cfg["scores"]["dir"])
        missing = [q for q in self.questions if not score_path(q.qid, sdir).exists()]
        per_question = 0.0
        if missing:
            scorer = make_scorer(self.cfg["scores"]["scorer"])
            info = score_questions(missing, scorer, sdir)
            per_question = info["seconds_per_question"]
            self.log(f"scored {len(missing)} questions with {scorer.name} ({per_question * 1000:.0f} ms each)")
        timing_file = sdir / "timing.json"
        if missing:
            _write_json(timing_file, {"seconds_per_question": per_question})
        elif timing_file.exists():
            per_question = json.loads(timing_file.read_text(encoding="utf-8")).get("seconds_per_question", 0.0)
        for q in self.questions:
            self.timings[q.qid]["embedding"] = float(per_question)

    # ------------------------------------------------------------- stage 2
    def build_problems(self) -> None:
        qp = dict(self.cfg["qubo"])
        qp = {k: v for k, v in qp.items() if v is not None or k == "max_edges"}
        sdir = self.cfg["scores"]["dir"]
        depths = [int(self.cfg["qaoa"]["p"])]
        extra = self.cfg["qaoa"].get("p_ideal_extra")
        if extra and "3a-p2" in self.lanes():
            depths.append(int(extra))
        store = self.out / "angles.json"
        saved = json.loads(store.read_text(encoding="utf-8")) if store.exists() else {}
        dirty = False
        for q in self.questions:
            t0 = time.perf_counter()
            r, s = load_scores(q.qid, sdir)
            qubo = build_qubo(r, s, **qp)
            self.timings[q.qid]["build"] = time.perf_counter() - t0
            self.qubos[q.qid] = qubo
            self.sims[q.qid] = QaoaSimulator(qubo)
            for depth in depths:
                key = f"{q.qid}|p{depth}"
                if key not in saved:
                    t0 = time.perf_counter()
                    tuned = tune_angles(qubo, p=depth, seed=self.seed, simulator=self.sims[q.qid])
                    saved[key] = {**tuned.to_dict(), "seconds": time.perf_counter() - t0}
                    dirty = True
                self.angles[(q.qid, depth)] = Angles.from_dict(saved[key])
                if depth == depths[0]:
                    self.timings[q.qid]["tune"] = float(saved[key]["seconds"])
        if dirty:
            _write_json(store, saved)
        edges = [self.qubos[q.qid].num_edges for q in self.questions]
        self.log(f"built {len(self.questions)} QUBOs: {np.mean(edges):.1f} redundant edges on average "
                 f"(min {min(edges)}, max {max(edges)})")

    # ------------------------------------------------------------ backends
    def backend(self, role: str) -> Any:
        if role not in self._backends:
            spec = self.cfg["backends"].get(role)
            if not spec:
                raise ValueError(f"no backend configured for role {role!r}")
            spec = {k: v for k, v in dict(spec).items() if k != "limit"}
            if spec.get("kind") == "superstaq":
                spec.setdefault("cache_dir", str(self.out / "superstaq_cache"))
            if spec.get("kind") in ("local-pauli", "local-aer", "local-noisy", "ideal"):
                spec.setdefault("seed", self.seed)
            self._backends[role] = make_backend(spec)
        return self._backends[role]

    def lanes(self) -> list[str]:
        lanes = list(self.cfg["lanes"])
        if "3a" in lanes and self.cfg["qaoa"].get("p_ideal_extra") and "3a-p2" not in lanes:
            lanes.insert(lanes.index("3a") + 1, "3a-p2")
        return lanes

    def _lane_questions(self, lane: str) -> list[Question]:
        if lane.startswith("3d"):
            limit = (self.cfg["backends"].get("hardware") or {}).get("limit")
            return self.questions[: int(limit)] if limit else self.questions
        return self.questions

    # ------------------------------------------------------------- stage 3
    def _quantum_spec(self, lane: str) -> tuple[str, int, Any, dict | None]:
        """(tier, depth, backend, encoder options) of a sampled quantum lane."""
        p = int(self.cfg["qaoa"]["p"])
        ice = dict(self.cfg.get("iceberg") or {})
        if lane == "3a":
            return "ideal", p, make_backend({"kind": "ideal", "seed": self.seed}), None
        if lane == "3a-p2":
            return "ideal", int(self.cfg["qaoa"]["p_ideal_extra"]), make_backend({"kind": "ideal", "seed": self.seed + 1}), None
        if lane == "3b":
            return "physical", p, self.backend("physical"), None
        if lane == "3c":
            return "iceberg", p, self.backend("physical"), ice
        if lane == "3c-direct":
            return "iceberg", p, self.backend("physical"), {**ice, "style": "direct"}
        if lane == "3d-b":
            return "physical", p, self.backend("hardware"), None
        if lane == "3d-c":
            return "iceberg", p, self.backend("hardware"), ice
        raise ValueError(f"{lane!r} is not a quantum lane")

    def _raw_path(self, lane: str, qid: str) -> Path:
        return self.out / "raw" / lane.replace("@", "_at_") / f"{qid}.json.gz"

    def sample_lane(self, lane: str) -> None:
        """Run a sampling lane for every question that has no raw counts yet."""
        questions = self._lane_questions(lane)
        todo = [q for q in questions if not self._raw_path(lane, q.qid).exists()]
        if not todo:
            return
        shots = int(self.cfg["qaoa"]["shots"])
        if lane == "rand":
            rng = np.random.default_rng(self.seed + 7919)
            for q in todo:
                n = self.qubos[q.qid].n
                draws = rng.integers(0, 1 << n, size=shots)
                values, counts = np.unique(draws, return_counts=True)
                _write_gz(self._raw_path(lane, q.qid), {
                    "counts": {format(int(v), f"0{n}b"): int(c) for v, c in zip(values, counts)},
                    "shots": shots, "tier": "ideal", "backend": "uniform-random", "seconds": 0.0})
            return
        tier, depth, backend, enc = self._quantum_spec(lane)
        t0 = time.perf_counter()
        circuits = [prepare_circuit(self.qubos[q.qid], tier, self.angles[(q.qid, depth)], encoder_opts=enc) for q in todo]
        build = (time.perf_counter() - t0) / len(todo)
        t0 = time.perf_counter()
        counts_list = backend.run(circuits, shots)
        execute = (time.perf_counter() - t0) / len(todo)
        stats = getattr(backend, "last_stats", None) or [{}] * len(todo)
        for q, circuit, counts, st in zip(todo, circuits, counts_list, stats):
            meta = dict(circuit.metadata or {})
            _write_gz(self._raw_path(lane, q.qid), {
                "counts": {k: int(v) for k, v in counts.items()}, "shots": shots, "tier": tier,
                "backend": getattr(backend, "name", type(backend).__name__), "depth": depth,
                "iceberg": meta.get("iceberg"), "compiled": dict(st),
                "circuit_qubits": circuit.num_qubits, "seconds_build": build, "seconds": execute})
        self.log(f"lane {lane}: sampled {len(todo)} questions on {getattr(backend, 'name', '?')} "
                 f"({execute:.2f} s per question)")

    def _result_from_raw(self, lane: str, q: Question, budget: int | None = None) -> SelectionResult:
        raw = _read_gz(self._raw_path(lane, q.qid))
        counts = raw["counts"]
        if budget is not None and budget < raw["shots"]:
            counts = subsample_counts(counts, budget, seed=self.seed + int(q.qid[-6:], 16) % 100003)
        qa = self.cfg["qaoa"]
        qubo = self.qubos[q.qid]
        post = None
        logical, raw_shots = counts, None
        if raw.get("iceberg"):
            logical, ps = iceberg.postselect(counts, {"iceberg": raw["iceberg"]})
            post, raw_shots = ps.to_dict(), ps.shots
        selections = rank_selections(qubo, logical, top=int(qa["top"]), min_select=int(qa["min_select"])) if logical else []
        stats = sample_stats(qubo, logical, raw_shots=raw_shots, min_select=int(qa["min_select"]))
        res = SelectionResult(tier=raw["tier"], backend=raw["backend"], selections=selections, stats=stats,
                              postselect=post)
        res.tier = raw["tier"]
        res.timings.update({"execute": float(raw.get("seconds", 0.0)), "circuit_build": float(raw.get("seconds_build", 0.0))})
        res.extra["compiled"] = raw.get("compiled") or {}
        res.extra["circuit_qubits"] = raw.get("circuit_qubits")
        return res

    def select_lane(self, lane: str) -> dict[str, SelectionResult | None]:
        """Ranked selections of one lane for every question (``None`` for lane 1a)."""
        base, _, budget = lane.partition("@")
        questions = self._lane_questions(base)
        qa = self.cfg["qaoa"]
        out: dict[str, SelectionResult | None] = {}
        if base == "1a":
            return {q.qid: None for q in questions}
        if base in CLASSICAL:
            annealer = None
            if base == "2c" and (self.cfg["backends"].get("annealer") or {}).get("kind") == "superstaq":
                annealer = self.backend("annealer")
            for q in questions:
                t0 = time.perf_counter()
                res = select_qubo(self.qubos[q.qid], CLASSICAL[base], annealer, top=int(qa["top"]),
                                  seed=self.seed, min_select=int(qa["min_select"]), k=int(self.cfg["top_k"]))
                res.timings.setdefault("execute", time.perf_counter() - t0)
                out[q.qid] = res
            return out
        if base == "4":
            return self._routed(questions)
        if base == "oracle":
            return {q.qid: self._oracle(q) for q in questions}
        if base not in SAMPLED:
            raise ValueError(f"unknown lane {lane!r}")
        self.sample_lane(base)
        for q in questions:
            out[q.qid] = self._result_from_raw(base, q, int(budget) if budget else None)
        return out

    def _oracle(self, q: Question) -> SelectionResult:
        """The gold supporting paragraphs as the only selection (upper-bound diagnostic)."""
        qubo = self.qubos[q.qid]
        gold = tuple(sorted(int(i) for i in q.gold))
        x = tuple(int(i in gold) for i in range(qubo.n))
        ranked = [RankedSelection(x, gold, float(qubo.energy(x)), 0)]
        return SelectionResult(tier="oracle", backend="gold-labels", selections=ranked, stats=None)

    def _routed(self, questions: Sequence[Question]) -> dict[str, SelectionResult | None]:
        """Router lane: per question, classical, bare or error-detected."""
        noise = (self.cfg["backends"].get("physical") or {}).get("noise") or {}
        rcfg = dict(self.cfg.get("router") or {})
        out: dict[str, SelectionResult | None] = {}
        decisions: dict[str, Any] = {}
        mapping = {"physical": "3b", "iceberg": "3c"}
        for q in questions:
            decision = route(self.qubos[q.qid], p=int(self.cfg["qaoa"]["p"]), p2=float(noise.get("p2", 0.005)),
                             p_meas=float(noise.get("p_meas", 0.005)),
                             style=(self.cfg.get("iceberg") or {}).get("style", "bus"), **rcfg)
            decisions[q.qid] = decision.to_dict()
            if decision.tier in mapping:
                self.sample_lane(mapping[decision.tier])
                res = self._result_from_raw(mapping[decision.tier], q)
            else:
                res = select_qubo(self.qubos[q.qid], decision.tier, top=int(self.cfg["qaoa"]["top"]),
                                  seed=self.seed, min_select=int(self.cfg["qaoa"]["min_select"]))
            res.extra["route"] = decision.to_dict()
            out[q.qid] = res
        _write_json(self.out / "routes.json", decisions)
        return out

    # ------------------------------------------------------------- stage 4
    @property
    def lm(self):
        if self._lm is None:
            self._lm = make_lm(self.cfg["lm"])
            self._cache = AnswerCache(self.out / "answer_cache.jsonl")
        return self._lm

    def answer_lane(self, lane: str, selections: Mapping[str, SelectionResult | None]) -> list[dict[str, Any]]:
        """Language-model answers for one lane: single selection and 3-selection vote."""
        lm = self.lm
        records = []
        for qid, res in selections.items():
            q = self.by_qid[qid]
            qubo = self.qubos[qid]
            fallback = False
            if res is None:                                     # lane 1a
                ranked = [RankedSelection(tuple([1] * len(q.paragraphs)), tuple(range(len(q.paragraphs))), 0.0, 0)]
            else:
                ranked = list(res.selections)
                if not ranked:                                  # every shot was discarded
                    x = greedy(qubo)
                    ranked = [RankedSelection(x, tuple(selected(x)), qubo.energy(x), 0)]
                    fallback = True
            answers = [self._cache.answer(lm, q, sel.indices) for sel in ranked]
            single = answers[0]
            t0 = time.perf_counter()
            voted, info = majority_vote([a.answer for a in answers], [sel.energy for sel in ranked])
            t_vote = time.perf_counter() - t0
            top = ranked[0]
            rec: dict[str, Any] = {
                "qid": qid, "lane": lane, "type": q.type, "gold_answer": q.answer,
                "selections": [s.to_dict() for s in ranked], "selected": list(top.indices),
                "n_selected": len(top.indices), "fallback": fallback,
                "gold_recall": gold_recall(top.indices, q.gold), "both_gold": both_gold(top.indices, q.gold),
                "answer_single": single.answer, "em_single": exact_match(single.answer, q.answer),
                "f1_single": f1_score(single.answer, q.answer), "tokens_single": single.prompt_tokens,
                "answers": [a.answer for a in answers], "answer_vote": voted,
                "em_vote": exact_match(voted, q.answer), "f1_vote": f1_score(voted, q.answer),
                "tokens_vote": int(sum(a.prompt_tokens for a in answers)), "vote": info,
                "timings": {**self.timings[qid], "lm_single": single.latency_s,
                            "lm_vote": float(sum(a.latency_s for a in answers)), "vote": t_vote},
            }
            if res is not None:
                rec["timings"].update({k: float(v) for k, v in res.timings.items()})
                if res.stats is not None:
                    rec["solver"] = res.stats.to_dict()
                    sim = self.sims[qid]
                    rec["solver"]["ideal_approx_ratio"] = sim.approx_ratio(float(sim.energies.mean()))
                rec["circuit"] = {**res.circuit, **(res.extra.get("compiled") or {})}
                if res.extra.get("circuit_qubits"):
                    rec["circuit"]["qubits"] = res.extra["circuit_qubits"]
                if res.postselect:
                    rec["postselect"] = res.postselect
                if res.extra.get("route"):
                    rec["route"] = res.extra["route"]
            records.append(rec)
        with (self.out / f"answers_{lane.replace('@', '_at_')}.jsonl").open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
        return records

    # ------------------------------------------------------------- stage 5
    def summarise(self, lane_records: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
        nb, bs = int(self.cfg["bootstrap"]["n_boot"]), int(self.cfg["bootstrap"]["seed"])
        summary: dict[str, Any] = {"experiment": self.cfg["name"], "questions": len(self.questions), "lanes": {}}
        table: dict[str, dict[str, list[float]]] = {}
        order = [q.qid for q in self.questions]
        for lane, recs in lane_records.items():
            by = {r["qid"]: r for r in recs}

            def col(path: str, by=by) -> list[float]:
                out = []
                for qid in order:
                    node: Any = by.get(qid)
                    for part in path.split("."):
                        node = node.get(part) if isinstance(node, dict) else None
                    out.append(float(node) if isinstance(node, (int, float, bool)) else float("nan"))
                return out

            metrics = {
                "em_single": col("em_single"), "em_vote": col("em_vote"), "f1_single": col("f1_single"),
                "f1_vote": col("f1_vote"), "gold_recall": col("gold_recall"), "both_gold": col("both_gold"),
                "n_selected": col("n_selected"), "tokens_single": col("tokens_single"),
                "tokens_vote": col("tokens_vote"), "approx_ratio": col("solver.approx_ratio"),
                "feasibility": col("solver.feasibility"), "p_optimum": col("solver.p_optimum"),
                "kept_share": col("solver.kept_share"), "diversity_per_1000": col("solver.diversity_per_1000"),
                "hit_optimum": col("solver.hit_optimum"), "two_qubit_gates": col("circuit.two_qubit_gates"),
                "qubits": col("circuit.qubits"), "fallback": col("fallback"),
            }
            for stage in ("embedding", "build", "tune", "execute", "lm_single", "lm_vote", "vote"):
                metrics[f"t_{stage}"] = col(f"timings.{stage}")
            table[lane] = metrics
            summary["lanes"][lane] = {"label": lane_label(lane), "n": len(recs),
                                      **{name: _mean_ci(vals, nb, bs) for name, vals in metrics.items()}}
        hyp = evaluate_hypotheses(table, n_boot=nb, seed=bs)
        summary["hypotheses"] = hyp
        if any(lane.startswith("3d") for lane in table):
            summary["hypotheses_hardware"] = evaluate_hypotheses(
                table, lanes={"bare": "3d-b", "iceberg": "3d-c"}, n_boot=nb, seed=bs)
        _write_json(self.out / "summary.json", summary)
        (self.out / "hypotheses.md").write_text(
            hypotheses_markdown(hyp, f"Hypothesis report: {self.cfg['name']}"), encoding="utf-8")
        (self.out / "results_table.md").write_text(results_table(summary), encoding="utf-8")
        _write_json(self.out / "per_question.json", {"qids": order, "lanes": table})
        return summary

    # ------------------------------------------------------------------ run
    def run(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        self.ensure_scores()
        self.build_problems()
        lanes = self.lanes()
        budgets = [int(b) for b in (self.cfg["qaoa"].get("shot_budgets") or [])]
        expanded: list[str] = []
        for lane in lanes:
            expanded.append(lane)
            if lane in SAMPLED and lane != "3a-p2":
                expanded.extend(f"{lane}@{b}" for b in budgets)
        records: dict[str, list[dict[str, Any]]] = {}
        for lane in expanded:
            t1 = time.perf_counter()
            selections = self.select_lane(lane)
            records[lane] = self.answer_lane(lane, selections)
            em = np.mean([r["em_single"] for r in records[lane]])
            self.log(f"lane {lane:<10} {lane_label(lane):<34} EM {em:.3f}  ({time.perf_counter() - t1:.1f} s)")
        summary = self.summarise(records)
        summary["seconds_total"] = time.perf_counter() - t0
        _write_json(self.out / "config_used.json", {k: v for k, v in self.cfg.items()})
        self.log(f"done in {summary['seconds_total']:.0f} s; results in {self.out}")
        return summary


def results_table(summary: Mapping[str, Any]) -> str:
    """Markdown table of the headline metrics per lane."""
    def fmt(cell: Mapping[str, float], digits: int = 3, ci: bool = True) -> str:
        if not cell or cell["n"] == 0:
            return ""
        if ci:
            return f"{cell['mean']:.{digits}f} [{cell['lo']:.{digits}f}, {cell['hi']:.{digits}f}]"
        return f"{cell['mean']:.{digits}f}"

    lines = [f"# Results: {summary['experiment']} ({summary['questions']} questions)", "",
             "Exact match with 95% bootstrap intervals. Tokens are prompt tokens per question.", "",
             "| Lane | Selector | EM single | EM vote | F1 single | Gold recall | Paragraphs | Tokens single | Tokens vote | Approx. ratio | Shots kept |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for lane, s in summary["lanes"].items():
        lines.append("| " + " | ".join([
            lane, s["label"], fmt(s["em_single"]), fmt(s["em_vote"]), fmt(s["f1_single"], ci=False),
            fmt(s["gold_recall"], ci=False), fmt(s["n_selected"], 2, False), fmt(s["tokens_single"], 0, False),
            fmt(s["tokens_vote"], 0, False), fmt(s["approx_ratio"], 4, False), fmt(s["kept_share"], 3, False)]) + " |")
    lines.append("")
    return "\n".join(lines)


def run_experiment(cfg: Mapping[str, Any], *, log=print) -> dict[str, Any]:
    """Run every stage of an experiment and return its summary."""
    return Experiment(cfg, log=log).run()
