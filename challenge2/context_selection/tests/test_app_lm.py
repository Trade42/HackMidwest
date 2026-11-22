"""Language-model backends, prompt building and the answer cache."""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

from qctx.data import Question
from qctx.lm import (
    NO_CONTEXT,
    SYSTEM_PROMPT,
    AnswerCache,
    AnswerResult,
    HFChatLM,
    LanguageModel,
    OllamaLM,
    OnnxT5LM,
    StubLM,
    build_context,
    build_user_prompt,
    clean_answer,
    make_lm,
)

ROOT = Path(__file__).resolve().parents[1]
T5_DIR = Path(os.environ.get("QCTX_T5_DIR", ROOT / "models" / "t5"))
HAVE_T5 = all((T5_DIR / n).exists() for n in
              ("t5-encoder-12.onnx", "t5-decoder-with-lm-head-12.onnx", "tokenizer.json"))

PARAGRAPHS = [
    {"title": "Hamlet", "text": "Hamlet is a tragedy written by William Shakespeare around 1600."},
    {"title": "Cheese Fair", "text": "The fair is held every autumn."},
]
QUESTION = "Who wrote the tragedy Hamlet?"


# ------------------------------------------------------------ prompt helpers
def test_build_context_numbers_the_paragraphs():
    assert build_context(PARAGRAPHS) == (
        "[1] Hamlet: Hamlet is a tragedy written by William Shakespeare around 1600.\n"
        "[2] Cheese Fair: The fair is held every autumn."
    )
    assert build_context([]) == ""
    prompt = build_user_prompt(QUESTION, [])
    assert NO_CONTEXT in prompt and QUESTION in prompt


@pytest.mark.parametrize("raw, expected", [
    ("William Shakespeare", "William Shakespeare"),
    ("  Answer: William Shakespeare.\nBecause the context says so.", "William Shakespeare"),
    ("\n\nThe answer is \"Liga MX\".", "Liga MX"),
    ("Final answer: **yes**", "yes"),
    ("A: 'Sam Raimi'", "Sam Raimi"),
    ("a-ha", "a-ha"),
    ("Buddha's delight", "Buddha's delight"),
    ("answer: Answer: 2015.", "2015"),
    ("", ""),
    ("   \n  ", ""),
])
def test_clean_answer(raw, expected):
    assert clean_answer(raw) == expected


# --------------------------------------------------------------------- stub
def test_stub_is_deterministic_and_handles_no_context():
    stub = StubLM()
    assert isinstance(stub, LanguageModel)
    a = stub.answer(QUESTION, PARAGRAPHS)
    assert a.answer == "Hamlet" and a == AnswerResult("Hamlet", "Hamlet", a.prompt_tokens, a.latency_s)
    assert a.prompt_tokens == len(f"{QUESTION} {build_context(PARAGRAPHS)}".split())
    assert stub.answer(QUESTION, PARAGRAPHS).answer == a.answer
    empty = stub.answer(QUESTION, [])
    assert empty.answer == "" and empty.prompt_tokens == len(QUESTION.split())


# ----------------------------------------------------------- Hugging Face LM
class _FakeTokenizer:
    """Whitespace tokenizer with a chat template, mimicking the calls HFChatLM makes."""

    pad_token_id = None
    eos_token_id = 2
    chat_template = "fake"

    def __init__(self, system_role: bool = True):
        self.system_role = system_role
        self.vocab: list[str] = []
        self.last_prompt = ""
        self.last_kwargs: dict = {}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False and add_generation_prompt is True
        if not self.system_role and any(m["role"] == "system" for m in messages):
            raise ValueError("system role not supported")
        return "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in messages) + "<|assistant|>\n"

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        self.last_prompt = text
        self.last_kwargs = {"return_tensors": return_tensors, "add_special_tokens": add_special_tokens}
        self.vocab = text.split()
        return {"input_ids": np.arange(len(self.vocab))[None, :]}

    def decode(self, ids, skip_special_tokens=False):
        assert skip_special_tokens is True
        return " Answer: William Shakespeare.\nExplanation follows" if len(ids) else ""


class _FakeModel:
    device = None

    def __init__(self):
        self.kwargs: dict = {}

    def generate(self, input_ids=None, **kwargs):
        self.kwargs = kwargs
        new = np.array([[101, 102, 103]])
        return np.concatenate([input_ids, new], axis=1)


def test_hf_chat_lm_builds_prompt_and_decodes_only_new_tokens():
    tok, model = _FakeTokenizer(), _FakeModel()
    lm = HFChatLM("fake/instruct-0.5b", max_new_tokens=16, tokenizer=tok, model=model)
    assert lm.name == "fake/instruct-0.5b"
    res = lm.answer(QUESTION, PARAGRAPHS)
    assert res.answer == "William Shakespeare"
    assert res.raw.startswith(" Answer: William Shakespeare.")
    assert res.prompt_tokens == len(tok.last_prompt.split())
    assert res.latency_s >= 0
    # prompt: system instruction, numbered context, question, generation marker
    assert SYSTEM_PROMPT in tok.last_prompt
    assert "[1] Hamlet: Hamlet is a tragedy" in tok.last_prompt and "[2] Cheese Fair" in tok.last_prompt
    assert QUESTION in tok.last_prompt and tok.last_prompt.endswith("<|assistant|>\n")
    assert tok.last_kwargs == {"return_tensors": "pt", "add_special_tokens": False}
    # randomness off
    assert model.kwargs["do_sample"] is False
    assert model.kwargs["max_new_tokens"] == 16
    assert model.kwargs["temperature"] is None and model.kwargs["top_p"] is None
    assert model.kwargs["pad_token_id"] == 2          # falls back to the end-of-sequence id


def test_hf_chat_lm_without_system_role_or_context():
    tok = _FakeTokenizer(system_role=False)
    lm = HFChatLM("fake/no-system", tokenizer=tok, model=_FakeModel())
    res = lm.answer(QUESTION, [])
    assert res.answer == "William Shakespeare"
    assert tok.last_prompt.startswith("<|user|>\n" + SYSTEM_PROMPT)
    assert NO_CONTEXT in tok.last_prompt and "<|system|>" not in tok.last_prompt


# ------------------------------------------------------------------- Ollama
class _OllamaHandler(BaseHTTPRequestHandler):
    seen: list[dict] = []

    def do_POST(self):  # noqa: N802 (http.server API)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
        type(self).seen.append({"path": self.path, **body})
        payload = {"message": {"role": "assistant", "content": "Answer: William Shakespeare."}, "done": True}
        if "Hamlet" in body["messages"][-1]["content"]:
            payload["prompt_eval_count"] = 57
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def test_ollama_lm_against_a_local_fake_server():
    _OllamaHandler.seen = []
    server = HTTPServer(("127.0.0.1", 0), _OllamaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        lm = make_lm({"kind": "ollama", "model": "qwen2.5:0.5b", "host": f"http://127.0.0.1:{server.server_port}/"})
        assert isinstance(lm, OllamaLM) and lm.name == "ollama:qwen2.5:0.5b"
        res = lm.answer(QUESTION, PARAGRAPHS)
        other = lm.answer("Unrelated question?", [])
    finally:
        server.shutdown()
        server.server_close()
    assert res.answer == "William Shakespeare" and res.prompt_tokens == 57
    assert other.prompt_tokens > 0                    # falls back to a word count when Ollama omits it
    req = _OllamaHandler.seen[0]
    assert req["path"] == "/api/chat" and req["model"] == "qwen2.5:0.5b" and req["stream"] is False
    assert req["options"]["temperature"] == 0
    assert req["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert QUESTION in req["messages"][1]["content"]


# ------------------------------------------------------------------ factory
def test_make_lm():
    assert isinstance(make_lm({"kind": "stub"}), StubLM)
    with pytest.raises(ValueError):
        make_lm({"kind": "oracle"})


# -------------------------------------------------------------------- cache
class _CountingLM:
    name = "counting"

    def __init__(self):
        self.calls: list[list[str]] = []

    def answer(self, question, paragraphs):
        self.calls.append([p["title"] for p in paragraphs])
        return AnswerResult("+".join(p["title"] for p in paragraphs), "raw", 10 * len(paragraphs), 0.25)


def _question() -> Question:
    paragraphs = tuple({"title": f"T{i}", "text": f"text {i}"} for i in range(5))
    return Question("q-1", "Which one?", "T3", "bridge", "hard", paragraphs, (1, 3))


def test_answer_cache_reads_through_and_persists(tmp_path):
    path = tmp_path / "cache" / "answers.jsonl"
    lm, q = _CountingLM(), _question()
    cache = AnswerCache(path)
    first = cache.answer(lm, q, [3, 1])
    assert first.answer == "T1+T3"                      # paragraphs are passed in original order
    assert cache.answer(lm, q, (1, 3)) == first         # same selection, different order: cache hit
    assert cache.answer(lm, q, [1, 3, 4]).answer == "T1+T3+T4"
    assert cache.answer(lm, q, []).answer == ""         # empty selection still answers
    assert len(lm.calls) == 3 and cache.hits == 1 and cache.misses == 3 and len(cache) == 3

    reopened = AnswerCache(path)                        # a new run resumes from disk
    assert reopened.get("counting", "q-1", [3, 1]) == first
    assert reopened.get("other-model", "q-1", [1, 3]) is None
    assert reopened.answer(lm, q, [1, 3]) == first and len(lm.calls) == 3

    with path.open("a", encoding="utf-8") as fh:        # interrupted write: half a line
        fh.write('{"lm": "counting", "qid": "q-1", "ind')
    assert len(AnswerCache(path)) == 3


def test_answer_cache_put_and_get(tmp_path):
    cache = AnswerCache(tmp_path / "a.jsonl")
    res = AnswerResult("Liga MX", "Liga MX – México", 123, 1.5)
    cache.put("m", "q", [2, 0], res)
    assert cache.get("m", "q", [0, 2]) == res
    assert AnswerCache(tmp_path / "a.jsonl").get("m", "q", (0, 2)) == res


# ------------------------------------------------------------------ ONNX T5
@pytest.fixture(scope="module")
def t5() -> OnnxT5LM:
    return make_lm({"kind": "onnx-t5", "dir": str(T5_DIR)})


@pytest.mark.skipif(not HAVE_T5, reason="offline T5 files not downloaded")
def test_onnx_t5_answers_from_context(t5):
    assert t5.name == "t5-base-onnx"
    res = t5.answer(QUESTION, PARAGRAPHS)
    assert "shakespeare" in res.answer.lower()
    assert res.prompt_tokens > 20 and res.latency_s > 0
    again = t5.answer(QUESTION, PARAGRAPHS)
    assert again.answer == res.answer and again.prompt_tokens == res.prompt_tokens     # greedy: deterministic
    assert isinstance(t5.answer(QUESTION, []).answer, str)                              # no context still works
