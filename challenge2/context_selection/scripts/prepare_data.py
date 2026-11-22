"""Fix the evaluation, calibration and demo questions with seeds.

    python scripts/prepare_data.py                      # official HotpotQA dev distractor set
    python scripts/prepare_data.py --source my.json     # any file in HotpotQA format

Three disjoint files are written:

* ``data/questions.jsonl``   150 evaluation questions (seed 0)
* ``data/calibration.jsonl``  50 questions for tuning tau and theta (seed 1)
* ``data/demo.jsonl``          3 short bridge questions for the live demo

Scores are normally written by ``reproduce.py`` with the scorer named in the experiment
config. Pass ``--scorer`` (a kind such as ``tfidf`` or a JSON object such as
``'{"kind": "tfidf", "overlap": "coverage", "hop": 3}'``) to write ``scores/{qid}.npz`` here.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qctx.data import Question, load_hotpot, sample_questions, to_question, write_questions  # noqa: E402


def pick_demo(raw: list[dict], n: int, seed: int, exclude: set[str]) -> list[Question]:
    """Short, clear bridge questions make the best live demo."""
    pool = sample_questions(raw, n=_count(raw, exclude, ("bridge",)), seed=seed, exclude=exclude, types=("bridge",))
    short = [q for q in pool
             if len(q.question) <= 110 and 1 <= len(q.answer.split()) <= 3 and " or " not in f" {q.answer} "]
    ranked = sorted(short or pool, key=lambda q: (sum(len(p["text"]) for p in q.paragraphs), q.qid))
    if len(ranked) < n:
        raise ValueError(f"only {len(ranked)} demo candidates, need {n}")
    return ranked[:n]


def _count(raw: list[dict], exclude: set[str], types: tuple[str, ...] | None) -> int:
    seen: set[str] = set()
    for rec in raw:
        q = to_question(rec)
        if q.qid in exclude or q.qid in seen or len(q.paragraphs) != 10 or len(q.gold) < 2:
            continue
        if types is not None and q.type not in types:
            continue
        seen.add(q.qid)
    return len(seen)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=None, help="HotpotQA-format JSON file or URL (default: official dev distractor)")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/questions.jsonl")
    ap.add_argument("--calibration-n", type=int, default=50)
    ap.add_argument("--calibration-seed", type=int, default=1)
    ap.add_argument("--calibration-out", default="data/calibration.jsonl")
    ap.add_argument("--demo-n", type=int, default=3)
    ap.add_argument("--demo-seed", type=int, default=2)
    ap.add_argument("--demo-out", default="data/demo.jsonl")
    ap.add_argument("--types", nargs="*", default=None, help="restrict to question types, e.g. bridge")
    ap.add_argument("--scorer", default=None, help="also write scores: a scorer kind or a JSON scorer config")
    ap.add_argument("--scores-dir", default="scores")
    args = ap.parse_args(argv)

    raw = load_hotpot(args.source)
    types = tuple(args.types) if args.types else None

    evaluation = sample_questions(raw, args.n, args.seed, types=types)
    used = {q.qid for q in evaluation}
    calibration: list[Question] = []
    if args.calibration_n > 0:
        calibration = sample_questions(raw, args.calibration_n, args.calibration_seed, exclude=used, types=types)
        used |= {q.qid for q in calibration}
    demo: list[Question] = []
    if args.demo_n > 0:
        demo = pick_demo(raw, args.demo_n, args.demo_seed, used)

    for path, questions, label in (
        (args.out, evaluation, "evaluation"),
        (args.calibration_out, calibration, "calibration"),
        (args.demo_out, demo, "demo"),
    ):
        if not questions:
            continue
        write_questions(ROOT / path if not Path(path).is_absolute() else Path(path), questions)
        kinds = dict(sorted(Counter(q.type for q in questions).items()))
        print(f"{label:12} {len(questions):4d} questions -> {path}  {kinds}")

    if args.scorer:
        from qctx.scoring import make_scorer, score_questions

        import json

        spec = json.loads(args.scorer) if args.scorer.strip().startswith("{") else {"kind": args.scorer}
        scorer = make_scorer(spec)
        out_dir = ROOT / args.scores_dir if not Path(args.scores_dir).is_absolute() else Path(args.scores_dir)
        info = score_questions(evaluation + calibration + demo, scorer, out_dir, overwrite=True)
        print(f"scores       {int(info['written'])} files -> {args.scores_dir}  "
              f"({scorer.name}, {info['seconds_per_question'] * 1000:.1f} ms per question)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
