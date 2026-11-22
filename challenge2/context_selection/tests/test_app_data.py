"""Data loading, sampling and storage."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from qctx.data import (
    Question,
    load_hotpot,
    normalize_record,
    read_questions,
    sample_questions,
    to_question,
    write_questions,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "hotpot_mini.json"


@pytest.fixture(scope="module")
def raw() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _wide_records(count: int, paragraphs: int = 10) -> list[dict]:
    """Synthetic records with the distractor setting's 10 paragraphs."""
    recs = []
    for i in range(count):
        context = [[f"Title {i}-{j}", [f"Sentence about item {i}-{j}. ", "Second sentence."]] for j in range(paragraphs)]
        recs.append({
            "_id": f"wide{i:03d}",
            "question": f"Short question number {i}?",
            "answer": f"item {i}",
            "type": "bridge" if i % 4 else "comparison",
            "level": "hard",
            "supporting_facts": [[f"Title {i}-1", 0], [f"Title {i}-7", 1]],
            "context": context,
        })
    return recs


def test_to_question_joins_sentences_and_finds_gold(raw):
    q = to_question(raw[0])
    assert q.qid == "synthetic00"
    assert len(q.paragraphs) == 4
    assert q.paragraphs[0] == {"title": "Alder Bridge",
                               "text": "Alder Bridge is a stone bridge in the town of Alder. It crosses the River Alda."}
    assert q.gold == (0, 2)
    assert q.answer == "40 kilometres"


def test_gold_ignores_titles_missing_from_context(raw):
    rec = dict(raw[0])
    rec["supporting_facts"] = [["Alder Bridge", 0], ["Not In Context", 0], ["Alder Bridge", 1]]
    assert to_question(rec).gold == (0,)


def test_hugging_face_schema_is_normalised(raw):
    r = raw[1]
    hf = {
        "id": r["_id"],
        "question": r["question"],
        "answer": r["answer"],
        "type": r["type"],
        "level": r["level"],
        "supporting_facts": {"title": [f[0] for f in r["supporting_facts"]],
                             "sent_id": [f[1] for f in r["supporting_facts"]]},
        "context": {"title": [c[0] for c in r["context"]], "sentences": [c[1] for c in r["context"]]},
    }
    assert normalize_record(hf) == normalize_record(r)
    assert to_question(hf) == to_question(r)


def test_sampling_is_deterministic_and_order_independent(raw):
    a = sample_questions(raw, 4, seed=0, num_paragraphs=4)
    b = sample_questions(list(reversed(raw)), 4, seed=0, num_paragraphs=4)
    c = sample_questions(raw, 4, seed=1, num_paragraphs=4)
    assert [q.qid for q in a] == [q.qid for q in b]
    assert len({q.qid for q in a}) == 4
    assert [q.qid for q in a] != [q.qid for q in c]


def test_sampling_filters(raw):
    assert len(sample_questions(raw, 8, seed=0, num_paragraphs=4)) == 8
    with pytest.raises(ValueError):
        sample_questions(raw, 1, seed=0)                       # nothing has 10 paragraphs
    with pytest.raises(ValueError):
        sample_questions(raw, 9, seed=0, num_paragraphs=4)     # more than available
    first = sample_questions(raw, 5, seed=0, num_paragraphs=4)
    rest = sample_questions(raw, 3, seed=0, num_paragraphs=4, exclude={q.qid for q in first})
    assert not {q.qid for q in first} & {q.qid for q in rest}
    comparison = sample_questions(raw, 2, seed=0, num_paragraphs=4, types=("comparison",))
    assert {q.type for q in comparison} == {"comparison"}
    one_gold = [dict(r, supporting_facts=r["supporting_facts"][:1]) for r in raw]
    with pytest.raises(ValueError):
        sample_questions(one_gold, 1, seed=0, num_paragraphs=4)
    assert len(sample_questions(one_gold, 2, seed=0, num_paragraphs=4, require_gold=1)) == 2


def test_duplicate_ids_are_dropped(raw):
    assert len(sample_questions(raw + raw, 8, seed=0, num_paragraphs=4)) == 8


def test_write_read_roundtrip_keeps_unicode(tmp_path, raw):
    qs = sample_questions(raw, 3, seed=0, num_paragraphs=4)
    special = Question("q-ü", "Où est Jesús García?", "Liga MX – México", "bridge", "hard",
                       ({"title": "T", "text": "naïve café"},), (0,))
    path = tmp_path / "nested" / "questions.jsonl"
    write_questions(path, qs + [special])
    back = read_questions(path)
    assert back == qs + [special]
    assert "Jesús" in path.read_text(encoding="utf-8")
    assert Question.from_dict(special.to_dict()) == special


def test_load_hotpot_from_local_file(tmp_path, raw):
    path = tmp_path / "dev.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    loaded = load_hotpot(path)
    assert len(loaded) == len(raw)
    assert loaded[0]["_id"] == "synthetic00"
    assert loaded[0]["context"][0][0] == "Alder Bridge"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_hotpot(bad)


def test_prepare_data_cli_writes_disjoint_files(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps(_wide_records(40)), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("prepare_data", ROOT / "scripts" / "prepare_data.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["prepare_data"] = module
    spec.loader.exec_module(module)
    out = {name: tmp_path / f"{name}.jsonl" for name in ("eval", "calib", "demo")}
    code = module.main([
        "--source", str(source), "--n", "12", "--out", str(out["eval"]),
        "--calibration-n", "6", "--calibration-out", str(out["calib"]),
        "--demo-n", "2", "--demo-out", str(out["demo"]),
        "--scorer", "tfidf", "--scores-dir", str(tmp_path / "scores"),
    ])
    assert code == 0
    sets = {name: {q.qid for q in read_questions(path)} for name, path in out.items()}
    assert [len(sets[n]) for n in ("eval", "calib", "demo")] == [12, 6, 2]
    assert not sets["eval"] & sets["calib"] and not sets["eval"] & sets["demo"] and not sets["calib"] & sets["demo"]
    assert len(list((tmp_path / "scores").glob("*.npz"))) == 20
    assert all(q.type == "bridge" for q in read_questions(out["demo"]))


def test_committed_question_sets_are_consistent():
    paths = [ROOT / "data" / name for name in ("questions.jsonl", "calibration.jsonl", "demo.jsonl")]
    if not all(p.exists() for p in paths):
        pytest.skip("data files not prepared")
    evaluation, calibration, demo = (read_questions(p) for p in paths)
    assert len(evaluation) == 150 and len(calibration) == 50 and len(demo) == 3
    ids = [{q.qid for q in qs} for qs in (evaluation, calibration, demo)]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    for q in evaluation + calibration + demo:
        assert len(q.paragraphs) == 10
        assert len(q.gold) == 2 and all(0 <= g < 10 for g in q.gold)
        assert q.question and q.answer
