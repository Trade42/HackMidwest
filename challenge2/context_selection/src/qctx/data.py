"""HotpotQA (distractor setting) loading, sampling and storage.

Every question comes with 10 paragraphs, of which 2 hold the facts needed for
the answer. The evaluation set is fixed by a seed so every lane and every tier
sees the same questions.

Stored format, one JSON object per line in ``data/questions.jsonl``::

    {"qid": ..., "question": ..., "answer": ..., "type": "bridge" | "comparison",
     "level": ..., "paragraphs": [{"title": ..., "text": ...}, ...], "gold": [i, j]}
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

HF_REPO = "hotpotqa/hotpot_qa"
HF_CONFIG = "distractor"
HF_PARQUET = "distractor/validation-00000-of-00001.parquet"
CMU_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
CACHE_DIR = Path("data") / "cache"


@dataclass(frozen=True)
class Question:
    """One multi-hop question with its candidate paragraphs."""

    qid: str
    question: str
    answer: str
    type: str
    level: str
    paragraphs: tuple[dict, ...]   #: each {"title": str, "text": str}
    gold: tuple[int, ...]          #: indices of the supporting paragraphs

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "question": self.question,
            "answer": self.answer,
            "type": self.type,
            "level": self.level,
            "paragraphs": [dict(p) for p in self.paragraphs],
            "gold": list(self.gold),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Question":
        return cls(
            qid=str(d["qid"]),
            question=str(d["question"]),
            answer=str(d["answer"]),
            type=str(d.get("type", "")),
            level=str(d.get("level", "")),
            paragraphs=tuple({"title": str(p["title"]), "text": str(p["text"])} for p in d["paragraphs"]),
            gold=tuple(int(i) for i in d["gold"]),
        )


# ------------------------------------------------------------------- loading
def load_hotpot(source: str | Path | None = None) -> list[dict]:
    """Load HotpotQA records in the original (raw) JSON format.

    Args:
        source: a local JSON file or an http(s) URL holding a list of records. When
            omitted, the official dev distractor set is fetched: first through the
            ``datasets`` library, then as a parquet file from the Hugging Face hub,
            then as the original JSON from the HotpotQA server.

    Returns:
        Records with keys ``_id``, ``question``, ``answer``, ``type``, ``level``,
        ``supporting_facts`` (``[[title, sent_id], ...]``) and ``context``
        (``[[title, [sentences]], ...]``).
    """
    if source is not None:
        text = str(source)
        if text.startswith(("http://", "https://")):
            return _read_json(_download(text, CACHE_DIR / Path(text.split("?")[0]).name))
        return _read_json(Path(source))

    errors: list[str] = []
    for name, loader in (
        ("datasets.load_dataset", _load_with_datasets),
        ("huggingface_hub parquet", _load_hf_parquet),
        ("HotpotQA server JSON", lambda: _read_json(_download(CMU_URL, CACHE_DIR / Path(CMU_URL).name))),
    ):
        try:
            return loader()
        except Exception as exc:  # each route is optional; report all if none works
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "could not load HotpotQA dev distractor from any source. Download "
        f"{CMU_URL} yourself and pass its path as `source`.\n  " + "\n  ".join(errors)
    )


def normalize_record(rec: dict) -> dict:
    """Bring one record to the raw HotpotQA format (accepts the Hugging Face schema)."""
    context = rec["context"]
    if isinstance(context, dict):
        context = [[str(t), [str(x) for x in sents]]
                   for t, sents in zip(context["title"], context["sentences"])]
    else:
        context = [[str(c[0]), [str(x) for x in c[1]]] for c in context]
    facts = rec.get("supporting_facts", [])
    if isinstance(facts, dict):
        facts = [[str(t), int(i)] for t, i in zip(facts["title"], facts["sent_id"])]
    else:
        facts = [[str(f[0]), int(f[1])] for f in facts]
    return {
        "_id": str(rec.get("_id", rec.get("id", ""))),
        "question": str(rec["question"]),
        "answer": str(rec.get("answer", "")),
        "type": str(rec.get("type", "")),
        "level": str(rec.get("level", "")),
        "supporting_facts": facts,
        "context": context,
    }


def _read_json(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not hold a list of HotpotQA records")
    return [normalize_record(r) for r in data]


def _download(url: str, dest: Path) -> Path:
    """Fetch ``url`` into ``dest`` unless it is already there."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as fh:
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            fh.write(block)
    tmp.replace(dest)
    return dest


def _load_with_datasets() -> list[dict]:
    try:
        import datasets
    except ImportError as exc:
        raise ImportError("install `datasets` (pip install datasets) to use this route") from exc
    ds = datasets.load_dataset(HF_REPO, HF_CONFIG, split="validation")
    return [normalize_record(dict(r)) for r in ds]


def _load_hf_parquet() -> list[dict]:
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("install `huggingface_hub`, `pandas` and `pyarrow` to use this route") from exc
    path = hf_hub_download(repo_id=HF_REPO, filename=HF_PARQUET, repo_type="dataset")
    frame = pd.read_parquet(path)
    return [normalize_record(_plain(r)) for r in frame.to_dict(orient="records")]


def _plain(obj: Any) -> Any:
    """Turn numpy containers from parquet into plain lists and dicts."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return [_plain(v) for v in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# ---------------------------------------------------------------- conversion
def to_question(raw: dict) -> Question:
    """Raw record -> :class:`Question` (paragraph text is its sentences joined)."""
    rec = normalize_record(raw)
    paragraphs = tuple({"title": title, "text": "".join(sentences).strip()} for title, sentences in rec["context"])
    titles = [p["title"] for p in paragraphs]
    gold = sorted({titles.index(title) for title, _ in rec["supporting_facts"] if title in titles})
    return Question(
        qid=rec["_id"],
        question=rec["question"].strip(),
        answer=rec["answer"].strip(),
        type=rec["type"],
        level=rec["level"],
        paragraphs=paragraphs,
        gold=tuple(gold),
    )


def sample_questions(
    raw: Sequence[dict],
    n: int,
    seed: int,
    *,
    num_paragraphs: int = 10,
    require_gold: int = 2,
    exclude: Iterable[str] = frozenset(),
    types: tuple[str, ...] | None = None,
) -> list[Question]:
    """Draw ``n`` questions deterministically.

    Keeps records with exactly ``num_paragraphs`` paragraphs and at least
    ``require_gold`` gold paragraphs, sorts them by id, then draws without
    replacement with ``numpy.random.default_rng(seed)``. The result does not
    depend on the order of ``raw``.
    """
    skip = set(exclude)
    pool: list[Question] = []
    seen: set[str] = set()
    for rec in raw:
        q = to_question(rec)
        if q.qid in skip or q.qid in seen:
            continue
        if len(q.paragraphs) != num_paragraphs or len(q.gold) < require_gold:
            continue
        if types is not None and q.type not in types:
            continue
        seen.add(q.qid)
        pool.append(q)
    pool.sort(key=lambda q: q.qid)
    if n > len(pool):
        raise ValueError(f"asked for {n} questions but only {len(pool)} pass the filters")
    rng = np.random.default_rng(seed)
    picks = sorted(int(i) for i in rng.choice(len(pool), size=n, replace=False))
    return [pool[i] for i in picks]


# ------------------------------------------------------------------- storage
def write_questions(path: str | Path, questions: Iterable[Question]) -> None:
    """Write questions as JSON lines (utf-8, one question per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for q in questions:
            fh.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")


def read_questions(path: str | Path) -> list[Question]:
    """Read questions written by :func:`write_questions`."""
    out: list[Question] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Question.from_dict(json.loads(line)))
    return out
