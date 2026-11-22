"""Relevance and overlap scores that feed the selection QUBO.

For one question with ``n`` paragraphs a scorer returns

* ``r`` (``n``): cosine similarity of each paragraph to the question;
* ``s`` (``n x n``): overlap between every pair of paragraphs.

Two definitions of overlap are available (``overlap=`` on every scorer):

* ``"paragraph"``: cosine similarity between the two paragraphs. This is the literal
  reading of the plan. On HotpotQA it works against multi-hop questions: the two gold
  paragraphs share the bridge entity, so they are among the *most* similar pairs and a
  redundancy edge between them forbids selecting both.
* ``"coverage"``: cosine similarity between the two paragraphs' *question-coverage
  profiles* (how strongly each paragraph matches each part of the question). Two
  paragraphs are redundant when they answer the same part of the question, and
  complementary when they cover different parts. Gold pairs come out among the least
  redundant pairs, which is what the independent-set constraint needs.

``hop=lambda`` (default 0, off) adds a second-hop term to relevance:
``r_i <- r_i + lambda * max_j r_j * sim(i, j)``. A paragraph that is strongly linked to a
relevant paragraph becomes relevant itself, which is how the second paragraph of a
bridge question is found: it rarely matches the question directly.

Scores are written to ``scores/{qid}.npz`` so the solver side never has to load
an embedding model.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable

import numpy as np


def paragraph_text(p: dict) -> str:
    """Text of one paragraph as the scorers see it."""
    return f"{p['title']}. {p['text']}"


@runtime_checkable
class Scorer(Protocol):
    """Anything that turns a question and its paragraphs into ``(r, s)``."""

    name: str

    def score(self, question: str, paragraphs: list[str]) -> tuple[np.ndarray, np.ndarray]:
        ...


def _cosine_scores(question_vec: np.ndarray, para_vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cosine relevance and overlap from raw vectors (rows are paragraphs)."""
    q = np.asarray(question_vec, dtype=np.float64).reshape(-1)
    p = np.asarray(para_vecs, dtype=np.float64)
    if p.ndim != 2:
        p = p.reshape(len(p), -1)
    qn = np.linalg.norm(q)
    pn = np.linalg.norm(p, axis=1, keepdims=True)
    q = q / qn if qn > 0 else q
    p = np.divide(p, pn, out=np.zeros_like(p), where=pn > 0)
    r = p @ q
    s = p @ p.T
    s = 0.5 * (s + s.T)
    np.fill_diagonal(s, 1.0)
    return np.clip(r, -1.0, 1.0), np.clip(s, -1.0, 1.0)


OVERLAPS = ("paragraph", "coverage")


def coverage_overlap(profiles: np.ndarray, *, center: bool = False) -> np.ndarray:
    """Overlap between paragraphs from their question-coverage profiles.

    ``profiles[i, t]`` says how strongly paragraph ``i`` matches part ``t`` of the
    question. Rows are compared by cosine similarity; ``center`` first removes each
    part's mean over paragraphs (needed for dense embeddings, whose similarities share
    a large common offset).
    """
    c = np.asarray(profiles, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] == 0:
        return np.eye(len(c))
    if center:
        c = c - c.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(c, axis=1, keepdims=True)
    c = np.divide(c, norm, out=np.zeros_like(c), where=norm > 0)
    s = c @ c.T
    s = 0.5 * (s + s.T)
    np.fill_diagonal(s, 1.0)
    return np.clip(s, -1.0, 1.0)


def apply_hop(r: np.ndarray, paragraph_similarity: np.ndarray, lam: float) -> np.ndarray:
    """Second-hop relevance: boost paragraphs linked to a relevant paragraph."""
    if lam <= 0 or len(r) < 2:
        return np.asarray(r, dtype=np.float64)
    link = np.array(paragraph_similarity, dtype=np.float64)
    np.fill_diagonal(link, 0.0)
    return np.asarray(r, dtype=np.float64) + float(lam) * np.max(link * np.asarray(r)[None, :], axis=1)


def _name(base: str, overlap: str, hop: float) -> str:
    out = base if overlap == "paragraph" else f"{base}+{overlap}"
    return out if hop <= 0 else f"{out}+hop{hop:g}"


_STOP = frozenset("""a an and are as at be by did do does for from had has have how in is it its of on or that the
their this to was were what when where which who whom whose why will with would""".split())


def question_terms(question: str) -> list[str]:
    """Content words of the question, in order, without duplicates."""
    seen: dict[str, None] = {}
    for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", question):
        if word.lower() not in _STOP and len(word) > 1:
            seen.setdefault(word, None)
    return list(seen)


def _check_overlap(overlap: str) -> str:
    if overlap not in OVERLAPS:
        raise ValueError(f"overlap must be one of {OVERLAPS}")
    return overlap


# --------------------------------------------------------------------- TF-IDF
class TfidfScorer:
    """Bag-of-words scorer fitted per question. No downloads; the offline fallback."""

    def __init__(self, ngram_range: tuple[int, int] = (1, 2), sublinear_tf: bool = True,
                 overlap: str = "paragraph", hop: float = 0.0):
        self.ngram_range = tuple(ngram_range)
        self.sublinear_tf = bool(sublinear_tf)
        self.overlap = _check_overlap(overlap)
        self.hop = float(hop)
        self.name = _name("tfidf", overlap, self.hop)

    def score(self, question: str, paragraphs: list[str]) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.feature_extraction.text import TfidfVectorizer

        n = len(paragraphs)
        if n == 0:
            return np.zeros(0), np.zeros((0, 0))
        vec = TfidfVectorizer(stop_words="english", sublinear_tf=self.sublinear_tf,
                              ngram_range=self.ngram_range)
        try:
            mat = vec.fit_transform(list(paragraphs) + [question]).toarray()
        except ValueError:      # nothing but stop words
            return np.zeros(n), np.eye(n)
        r, s = _cosine_scores(mat[-1], mat[:-1])
        r = apply_hop(r, s, self.hop)
        if self.overlap == "coverage":
            # profile = the paragraph's weights on the terms the question contains
            s = coverage_overlap(mat[:-1][:, mat[-1] > 0])
        return r, s


# ------------------------------------------------------- sentence-transformers
class SentenceTransformerScorer:
    """Small embedding model from Hugging Face (the plan's default scorer)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str | None = None,
                 batch_size: int = 32, overlap: str = "paragraph", hop: float = 0.0, *,
                 model: Any | None = None):
        self.model_name = model_name
        self.overlap = _check_overlap(overlap)
        self.hop = float(hop)
        self.name = _name(f"st:{model_name}", overlap, self.hop)
        self.batch_size = int(batch_size)
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "the sentence-transformers scorer needs `pip install sentence-transformers`; "
                    "use {'kind': 'tfidf'} to run without it"
                ) from exc
            model = SentenceTransformer(model_name, device=device)
        self.model = model

    def score(self, question: str, paragraphs: list[str]) -> tuple[np.ndarray, np.ndarray]:
        n = len(paragraphs)
        if n == 0:
            return np.zeros(0), np.zeros((0, 0))
        terms = question_terms(question) if self.overlap == "coverage" else []
        emb = self.model.encode([question] + list(paragraphs) + terms, normalize_embeddings=True,
                                batch_size=self.batch_size, show_progress_bar=False)
        emb = np.asarray(emb, dtype=np.float64)
        r, s = _cosine_scores(emb[0], emb[1:n + 1])
        r = apply_hop(r, s, self.hop)
        if terms:
            # profile = similarity of the paragraph to each content word of the question.
            # Not validated offline (no model access here): check it with scripts/calibrate.py.
            s = coverage_overlap(emb[1:n + 1] @ emb[n + 1:].T, center=True)
        return r, s


# ------------------------------------------------------------------- ONNX T5
class OnnxT5Scorer:
    """Mean-pooled T5 encoder states: "the language model's own embeddings".

    This is the plan's fallback when no embedding model is available. T5 was not
    trained for similarity, so the pooled vectors share a large common component;
    ``center=True`` removes the per-question mean before taking cosines.
    """

    def __init__(self, encoder_path: str | Path, tokenizer_path: str | Path, *,
                 max_tokens: int = 512, center: bool = True, threads: int = 2):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ImportError("the onnx-t5 scorer needs `pip install onnxruntime tokenizers`") from exc
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(threads)
        self.session = ort.InferenceSession(str(encoder_path), opts, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.max_tokens = int(max_tokens)
        self.center = bool(center)
        self.name = "t5-base-onnx-meanpool" + ("-centered" if center else "")

    def _embed(self, text: str) -> np.ndarray:
        ids = self.tokenizer.encode(text).ids[: self.max_tokens]
        hidden = self.session.run(None, {"input_ids": np.asarray([ids], dtype=np.int64)})[0]
        return hidden[0].mean(axis=0).astype(np.float64)

    def score(self, question: str, paragraphs: list[str]) -> tuple[np.ndarray, np.ndarray]:
        n = len(paragraphs)
        if n == 0:
            return np.zeros(0), np.zeros((0, 0))
        emb = np.stack([self._embed(t) for t in [question] + list(paragraphs)])
        if self.center and n > 1:
            emb = emb - emb[1:].mean(axis=0, keepdims=True)
        return _cosine_scores(emb[0], emb[1:])


# -------------------------------------------------------------------- factory
def make_scorer(cfg: dict) -> Scorer:
    """Build a scorer from a config dict.

    ``{"kind": "tfidf"}``, ``{"kind": "sentence-transformers", "model": ...}`` or
    ``{"kind": "onnx-t5", "encoder": ..., "tokenizer": ...}``.
    """
    cfg = dict(cfg or {})
    kind = str(cfg.pop("kind", "tfidf")).lower().replace("_", "-")
    if kind == "tfidf":
        return TfidfScorer(**cfg)
    if kind in ("sentence-transformers", "st", "sbert"):
        if "model" in cfg:
            cfg["model_name"] = cfg.pop("model")
        return SentenceTransformerScorer(**cfg)
    if kind == "onnx-t5":
        if "dir" in cfg:
            base = Path(cfg.pop("dir"))
            cfg.setdefault("encoder", base / "t5-encoder-12.onnx")
            cfg.setdefault("tokenizer", base / "tokenizer.json")
        return OnnxT5Scorer(cfg.pop("encoder"), cfg.pop("tokenizer"), **cfg)
    raise ValueError(f"unknown scorer kind {kind!r}; use tfidf, sentence-transformers or onnx-t5")


# ----------------------------------------------------------------- persistence
def score_path(qid: str, scores_dir: str | Path) -> Path:
    """File that holds the scores of question ``qid``."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(qid))
    return Path(scores_dir) / f"{safe}.npz"


def score_questions(questions: Iterable[Any], scorer: Scorer, out_dir: str | Path, *,
                    overwrite: bool = False) -> dict[str, float]:
    """Score every question and write ``out_dir/{qid}.npz`` (arrays ``r``, ``s``, ``scorer``).

    Existing files are kept unless ``overwrite`` is set. Returns timing info:
    ``seconds_total``, ``seconds_per_question`` (over the questions actually
    scored), ``questions``, ``written`` and ``skipped``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    t0 = time.perf_counter()
    for q in questions:
        path = score_path(q.qid, out)
        if path.exists() and not overwrite:
            skipped += 1
            continue
        r, s = scorer.score(q.question, [paragraph_text(p) for p in q.paragraphs])
        np.savez(path, r=np.asarray(r, dtype=np.float64), s=np.asarray(s, dtype=np.float64),
                 scorer=np.array(scorer.name))
        written += 1
    total = time.perf_counter() - t0
    return {
        "seconds_total": float(total),
        "seconds_per_question": float(total / written) if written else 0.0,
        "questions": float(written + skipped),
        "written": float(written),
        "skipped": float(skipped),
    }


def load_scores(qid: str, scores_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read ``(r, s)`` for one question."""
    path = score_path(qid, scores_dir)
    if not path.exists():
        raise FileNotFoundError(f"no scores for question {qid!r} in {scores_dir}; run the scoring step first")
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["r"], dtype=np.float64), np.asarray(data["s"], dtype=np.float64)


def scorer_name(qid: str, scores_dir: str | Path) -> str:
    """Name of the scorer that produced a score file."""
    with np.load(score_path(qid, scores_dir), allow_pickle=False) as data:
        return str(data["scorer"])


def recall_at_k(questions: Sequence[Any], scores_dir: str | Path, k: int = 3) -> dict[str, float]:
    """Gold-paragraph recall of the top-``k`` paragraphs by relevance (a scorer check)."""
    rec, both = [], []
    for q in questions:
        r, _ = load_scores(q.qid, scores_dir)
        top = set(np.argsort(-r, kind="stable")[:k].tolist())
        gold = set(q.gold)
        rec.append(len(top & gold) / len(gold) if gold else 0.0)
        both.append(float(gold <= top) if gold else 0.0)
    return {"recall": float(np.mean(rec)) if rec else 0.0, "both": float(np.mean(both)) if both else 0.0, "k": float(k)}
