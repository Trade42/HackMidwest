"""Live demo: one question in, every lane side by side.

    python -m qctx.demo --config configs/offline.yaml              # first demo question
    python -m qctx.demo --config configs/offline.yaml --index 2
    python -m qctx.demo --config configs/offline.yaml --question my_question.json

``my_question.json`` holds ``{"question": ..., "answer": ..., "paragraphs": [{"title":
..., "text": ...}, ...]}`` (``answer`` and ``gold`` are optional). Results are cached
under ``results/demo/``, so a rehearsed question replays instantly; pass ``--fresh`` to
recompute.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from .config import load_config
from .data import Question, read_questions, write_questions
from .harness import Experiment, lane_label

DEMO_LANES = ["1a", "1b", "2a", "2b", "3a", "3b", "3c"]


def _load_question(args: argparse.Namespace, cfg: dict[str, Any]) -> Question:
    if args.question:
        raw = json.loads(Path(args.question).read_text(encoding="utf-8"))
        return Question(qid=str(raw.get("qid", "custom")), question=raw["question"], answer=str(raw.get("answer", "")),
                        type=str(raw.get("type", "")), level=str(raw.get("level", "")),
                        paragraphs=tuple(raw["paragraphs"]), gold=tuple(raw.get("gold", ())))
    questions = read_questions(cfg["data"]["demo"])
    return questions[args.index % len(questions)]


def run_demo(cfg: dict[str, Any], question: Question, *, fresh: bool = False, log=print) -> list[dict[str, Any]]:
    """Run the demo lanes on one question and return one row per lane."""
    out = Path("results") / "demo" / question.qid
    if fresh and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    write_questions(out / "question.jsonl", [question])
    demo_cfg = copy.deepcopy(cfg)
    demo_cfg.update({"name": f"demo-{question.qid}", "output": str(out)})
    demo_cfg["data"] = {**cfg["data"], "questions": str(out / "question.jsonl"), "limit": None}
    demo_cfg["scores"] = {**cfg["scores"], "dir": str(out / "scores")}
    demo_cfg["lanes"] = [lane for lane in DEMO_LANES if lane in cfg["lanes"] or lane in ("1a", "2a")]
    demo_cfg["qaoa"] = {**cfg["qaoa"], "p_ideal_extra": None, "shot_budgets": []}
    demo_cfg["bootstrap"] = {"n_boot": 10, "seed": 0}
    exp = Experiment(demo_cfg, log=lambda *_: None)
    exp.ensure_scores()
    exp.build_problems()
    rows = []
    for lane in exp.lanes():
        rec = exp.answer_lane(lane, exp.select_lane(lane))[0]
        t = rec["timings"]
        rows.append({
            "lane": lane, "label": lane_label(lane),
            "paragraphs": [question.paragraphs[i]["title"] for i in rec["selected"]],
            "answer": rec["answer_single"], "vote": rec["answer_vote"],
            "correct": bool(rec["em_single"]) if question.answer else None,
            "tokens": rec["tokens_single"], "tokens_vote": rec["tokens_vote"],
            "kept": (rec.get("solver") or {}).get("kept_share"),
            "approx_ratio": (rec.get("solver") or {}).get("approx_ratio"),
            "cz": (rec.get("circuit") or {}).get("two_qubit_gates"),
            "qubits": (rec.get("circuit") or {}).get("qubits"),
            "seconds_select": float(t.get("execute", 0.0)), "seconds_lm": float(t.get("lm_single", 0.0)),
        })
    return rows


def format_rows(question: Question, rows: list[dict[str, Any]]) -> str:
    lines = [f"Question: {question.question}"]
    if question.answer:
        gold = ", ".join(question.paragraphs[i]["title"] for i in question.gold)
        lines.append(f"Gold answer: {question.answer}    (gold paragraphs: {gold})")
    lines.append("")
    head = f"{'lane':<5}{'selector':<28}{'answer':<26}{'ok':<4}{'vote':<22}{'tokens':>7}{'kept':>7}{'CZ':>5}{'select s':>10}{'LM s':>7}"
    lines += [head, "-" * len(head)]
    for r in rows:
        ok = "" if r["correct"] is None else ("yes" if r["correct"] else "no")
        kept = "" if r["kept"] is None else f"{100 * r['kept']:.0f}%"
        cz = "" if not r["cz"] else f"{r['cz']:.0f}"
        lines.append(f"{r['lane']:<5}{r['label'][:27]:<28}{r['answer'][:25]:<26}{ok:<4}{r['vote'][:21]:<22}"
                     f"{r['tokens']:>7}{kept:>7}{cz:>5}{r['seconds_select']:>10.2f}{r['seconds_lm']:>7.2f}")
        lines.append(f"     reads: {'; '.join(r['paragraphs']) if len(r['paragraphs']) < 10 else 'all 10 paragraphs'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/offline.yaml")
    ap.add_argument("--index", type=int, default=0, help="which demo question to use")
    ap.add_argument("--question", help="JSON file with a custom question and paragraphs")
    ap.add_argument("--fresh", action="store_true", help="ignore cached results")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    question = _load_question(args, cfg)
    print(format_rows(question, run_demo(cfg, question, fresh=args.fresh)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
