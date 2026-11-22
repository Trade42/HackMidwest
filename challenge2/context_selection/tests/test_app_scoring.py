"""Relevance and overlap scorers and their on-disk format."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from qctx.data import Question, read_questions
from qctx.scoring import (
    OnnxT5Scorer,
    Scorer,
    SentenceTransformerScorer,
    TfidfScorer,
    load_scores,
    make_scorer,
    paragraph_text,
    recall_at_k,
    score_path,
    score_questions,
    scorer_name,
)

ROOT = Path(__file__).resolve().parents[1]
T5_DIR = Path(os.environ.get("QCTX_T5_DIR", ROOT / "models" / "t5"))

QUESTION = "How long is the river crossed by Alder Bridge?"
PARAGRAPHS = [
    "Alder Bridge. Alder Bridge is a stone bridge in the town of Alder. It crosses the River Alda.",
    "Alder Museum. The Alder Museum shows local pottery. It opened on a Tuesday.",
    "River Alda. The River Alda is 40 kilometres long. It flows past Alder toward the sea.",
    "Cheese Fair. The fair is held every autumn and features a cheese contest.",
    "Alder Bridge. Alder Bridge is a stone bridge in the town of Alder. It crosses the River Alda.",
]


def _check_shapes(r: np.ndarray, s: np.ndarray, n: int) -> None:
    assert r.shape == (n,) and s.shape == (n, n)
    assert r.dtype == np.float64 and s.dtype == np.float64
    assert np.allclose(s, s.T)
    assert np.allclose(np.diag(s), 1.0)
    assert np.all(np.abs(r) <= 1.0 + 1e-12) and np.all(np.abs(s) <= 1.0 + 1e-12)


def test_paragraph_text():
    assert paragraph_text({"title": "River Alda", "text": "It is long."}) == "River Alda. It is long."


def test_tfidf_scores_are_cosines_with_sensible_ranking():
    scorer = TfidfScorer()
    assert isinstance(scorer, Scorer)
    r, s = scorer.score(QUESTION, PARAGRAPHS)
    _check_shapes(r, s, 5)
    assert int(np.argmax(r)) in (0, 4)             # the bridge paragraph is the closest match
    assert r[3] == pytest.approx(0.0)              # the fair shares no words with the question
    assert s[0, 4] == pytest.approx(1.0)           # identical paragraphs are fully redundant
    assert s[0, 3] == pytest.approx(0.0) and s[0, 2] > 0.1     # shares "River Alda" with the river paragraph
    r2, s2 = scorer.score(QUESTION, PARAGRAPHS)
    assert np.array_equal(r, r2) and np.array_equal(s, s2)


def test_tfidf_edge_cases():
    scorer = TfidfScorer()
    r, s = scorer.score("anything", [])
    assert r.shape == (0,) and s.shape == (0, 0)
    r, s = scorer.score("the of and", ["the and of", "of the"])       # stop words only
    assert np.array_equal(r, np.zeros(2)) and np.array_equal(s, np.eye(2))
    r, s = scorer.score("river", ["one river paragraph"])
    _check_shapes(r, s, 1)


class _FakeEmbedder:
    """Stands in for ``SentenceTransformer``: bag-of-letters vectors, records its calls."""

    def __init__(self):
        self.calls: list[dict] = []

    def encode(self, texts, **kwargs):
        self.calls.append({"n": len(texts), **kwargs})
        vecs = np.zeros((len(texts), 26), dtype=np.float32)
        for i, text in enumerate(texts):
            for ch in text.lower():
                if "a" <= ch <= "z":
                    vecs[i, ord(ch) - 97] += 1.0
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def test_sentence_transformer_scorer_uses_the_encode_api():
    fake = _FakeEmbedder()
    scorer = SentenceTransformerScorer("fake/model", batch_size=7, model=fake)
    assert scorer.name == "st:fake/model"
    r, s = scorer.score(QUESTION, PARAGRAPHS)
    _check_shapes(r, s, 5)
    assert fake.calls == [{"n": 6, "normalize_embeddings": True, "batch_size": 7, "show_progress_bar": False}]
    assert s[0, 4] == pytest.approx(1.0)


def test_make_scorer():
    assert make_scorer({"kind": "tfidf"}).name == "tfidf"
    assert make_scorer({}).name == "tfidf"
    with pytest.raises(ValueError):
        make_scorer({"kind": "quantum-magic"})


def _questions() -> list[Question]:
    paragraphs = tuple({"title": t.split(". ")[0], "text": t.split(". ", 1)[1]} for t in PARAGRAPHS)
    return [Question(f"q{i}", QUESTION, "40 kilometres", "bridge", "easy", paragraphs, (0, 2)) for i in range(3)]


def test_score_files_roundtrip(tmp_path):
    scorer = TfidfScorer()
    qs = _questions()
    info = score_questions(qs, scorer, tmp_path / "scores")
    assert info["written"] == 3 and info["skipped"] == 0 and info["questions"] == 3
    assert info["seconds_total"] >= 0 and info["seconds_per_question"] >= 0
    r, s = load_scores("q1", tmp_path / "scores")
    expected_r, expected_s = scorer.score(QUESTION, [paragraph_text(p) for p in qs[1].paragraphs])
    assert np.array_equal(r, expected_r) and np.array_equal(s, expected_s)
    assert scorer_name("q1", tmp_path / "scores") == "tfidf"
    again = score_questions(qs, scorer, tmp_path / "scores")
    assert again["written"] == 0 and again["skipped"] == 3
    forced = score_questions(qs, scorer, tmp_path / "scores", overwrite=True)
    assert forced["written"] == 3
    with pytest.raises(FileNotFoundError):
        load_scores("missing", tmp_path / "scores")
    assert recall_at_k(qs, tmp_path / "scores", k=3)["recall"] == 1.0


def test_score_path_is_safe_on_every_platform(tmp_path):
    assert score_path("a/b:c*?", tmp_path).name == "a_b_c__.npz"


def test_committed_scores_cover_every_question():
    files = [ROOT / "data" / name for name in ("questions.jsonl", "calibration.jsonl", "demo.jsonl")]
    if not all(f.exists() for f in files) or not (ROOT / "scores").exists():
        pytest.skip("data or scores not prepared")
    for f in files:
        for q in read_questions(f):
            r, s = load_scores(q.qid, ROOT / "scores")
            _check_shapes(r, s, 10)


@pytest.mark.skipif(not (T5_DIR / "t5-encoder-12.onnx").exists(), reason="offline T5 files not downloaded")
def test_onnx_t5_scorer():
    scorer = make_scorer({"kind": "onnx-t5", "dir": str(T5_DIR)})
    assert isinstance(scorer, OnnxT5Scorer)
    r, s = scorer.score(QUESTION, PARAGRAPHS)
    _check_shapes(r, s, 5)
    assert s[0, 4] == pytest.approx(1.0, abs=1e-5)


# ------------------------------------------------------- question-coverage overlap
def test_coverage_overlap_separates_complementary_from_redundant_paragraphs():
    """Paragraphs matching the same part of the question are redundant; different parts are not."""
    from qctx.scoring import TfidfScorer, coverage_overlap, question_terms

    question = "Which river flows through the capital of Austria?"
    paragraphs = [
        "Vienna. Vienna is the capital of Austria and its largest city.",          # covers capital/Austria
        "Austria. The capital of Austria is Vienna, in the east of the country.",  # same part again
        "Danube. The Danube river flows through ten countries.",                   # covers river/flows
    ]
    plain_r, plain_s = TfidfScorer(overlap="paragraph").score(question, paragraphs)
    cov_r, cov_s = TfidfScorer(overlap="coverage").score(question, paragraphs)
    assert np.allclose(plain_r, cov_r)                  # relevance does not depend on the overlap mode
    assert cov_s[0, 1] > 0.8                            # the two "capital of Austria" paragraphs repeat
    assert cov_s[0, 2] < 0.2 and cov_s[1, 2] < 0.2      # the river paragraph covers something else
    assert np.allclose(cov_s, cov_s.T) and np.allclose(np.diag(cov_s), 1.0)
    assert TfidfScorer(overlap="coverage").name == "tfidf+coverage"
    assert question_terms(question) == ["river", "flows", "through", "capital", "Austria"]
    assert coverage_overlap(np.zeros((3, 0))).tolist() == np.eye(3).tolist()
    with pytest.raises(ValueError):
        TfidfScorer(overlap="nope")


def test_sentence_transformer_coverage_uses_question_terms():
    from qctx.scoring import SentenceTransformerScorer

    class Fake:
        def __init__(self):
            self.texts = None

        def encode(self, texts, **kwargs):
            self.texts = list(texts)
            rng = np.random.default_rng(0)
            vecs = rng.normal(size=(len(texts), 8))
            return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    fake = Fake()
    scorer = SentenceTransformerScorer("fake-model", overlap="coverage", model=fake)
    r, s = scorer.score("Who founded the company?", ["A. first", "B. second", "C. third"])
    assert fake.texts[0] == "Who founded the company?" and fake.texts[4:] == ["founded", "company"]
    assert r.shape == (3,) and s.shape == (3, 3)
    assert np.allclose(s, s.T) and np.allclose(np.diag(s), 1.0)
    assert scorer.name.endswith("+coverage")


def test_second_hop_boosts_the_linked_paragraph():
    """The bridge paragraph barely matches the question but is linked to one that does."""
    from qctx.scoring import TfidfScorer, apply_hop

    question = "What sport does the director of the film Silver Tide play?"
    paragraphs = [
        "Silver Tide. Silver Tide is a 2009 film directed by Marta Olsen.",        # matches the question
        "Marta Olsen. Marta Olsen is a Norwegian filmmaker and curling champion.",  # the bridge
        "Rugby. Rugby is a team sport played with an oval ball.",                  # distractor
    ]
    r0, _ = TfidfScorer().score(question, paragraphs)
    r2, _ = TfidfScorer(hop=2.0).score(question, paragraphs)
    assert r2[1] - r0[1] > r2[2] - r0[2]            # the linked paragraph gains the most
    assert np.argsort(-r2)[:2].tolist() == [0, 1]
    assert TfidfScorer(overlap="coverage", hop=2.0).name == "tfidf+coverage+hop2"
    assert np.allclose(apply_hop(r0, np.eye(3), 5.0), r0)       # no links, no change
    assert np.allclose(apply_hop(r0, np.ones((3, 3)), 0.0), r0)
