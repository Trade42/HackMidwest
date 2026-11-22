"""Language models that answer a question from the selected paragraphs.

Every backend exposes ``answer(question, paragraphs) -> AnswerResult`` and runs
with randomness off (greedy decoding, temperature 0), so a rerun gives the same
answers. ``AnswerCache`` stores answers per (model, question, selection); many
lanes pick the same paragraphs, so the model is called once per distinct
selection.
"""

from __future__ import annotations

import contextlib
import json
import re
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable

SYSTEM_PROMPT = (
    "You answer questions using only the given context. "
    "Reply with the short answer only: a few words, a name, a number, or yes/no. "
    "Do not explain."
)
NO_CONTEXT = "(no context)"


@dataclass(frozen=True)
class AnswerResult:
    answer: str          #: cleaned short answer
    raw: str             #: text exactly as the model produced it
    prompt_tokens: int   #: tokens the model read for this call
    latency_s: float     #: wall-clock seconds for this call

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class LanguageModel(Protocol):
    """Anything that can answer a question from a list of paragraphs."""

    name: str

    def answer(self, question: str, paragraphs: list[dict]) -> AnswerResult:
        ...


# ------------------------------------------------------------ prompt helpers
def build_context(paragraphs: Sequence[dict]) -> str:
    """Numbered context block: one ``[i] Title: text`` line per paragraph."""
    return "\n".join(f"[{i}] {p['title']}: {p['text']}" for i, p in enumerate(paragraphs, start=1))


def build_user_prompt(question: str, paragraphs: Sequence[dict]) -> str:
    """User turn for chat models."""
    context = build_context(paragraphs) if paragraphs else NO_CONTEXT
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer with the short answer only."


_PREFIX = re.compile(r"^\s*(?:final\s+answer|short\s+answer|answer|a)\s*:\s*", re.IGNORECASE)
_LEAD_IN = re.compile(r"^\s*the\s+(?:short\s+)?answer\s+is\s*:?\s*", re.IGNORECASE)
_QUOTES = "\"'`*“”‘’"


def clean_answer(raw: str) -> str:
    """Reduce model output to the short answer.

    Keeps the first non-empty line, drops ``Answer:`` style prefixes, surrounding
    quotes and one trailing period.
    """
    text = ""
    for line in str(raw).splitlines():
        if line.strip():
            text = line.strip()
            break
    previous = None
    while previous != text:
        previous = text
        text = _PREFIX.sub("", text)
        text = _LEAD_IN.sub("", text)
        text = text.strip().strip(_QUOTES).strip()
    if text.endswith(".") and not text.endswith(".."):
        text = text[:-1].rstrip()
    return text.strip(_QUOTES).strip()


# ----------------------------------------------------------- Hugging Face LM
class HFChatLM:
    """A small instruct model from Hugging Face, run locally with greedy decoding.

    ``tokenizer`` and ``model`` can be injected (tests do this with tiny fakes);
    otherwise both are loaded with ``transformers``.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str | None = None,
                 max_new_tokens: int = 32, dtype: Any | None = None, *,
                 tokenizer: Any | None = None, model: Any | None = None):
        self.name = model_name
        self.max_new_tokens = int(max_new_tokens)
        self._torch = None
        if tokenizer is None or model is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "the Hugging Face model needs `pip install torch transformers accelerate`"
                ) from exc
            self._torch = torch
            if device is None:
                if torch.cuda.is_available():
                    device = "cuda"
                elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            if dtype is None:
                dtype = torch.float16 if device == "cuda" else torch.float32
            elif isinstance(dtype, str):
                dtype = getattr(torch, dtype)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            try:
                model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
            except TypeError:    # newer transformers renamed the argument
                model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
            model = model.to(device)
            model.eval()
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    # -- prompt
    def build_prompt(self, question: str, paragraphs: Sequence[dict]) -> str:
        """Full prompt text, using the model's chat template when it has one."""
        user = build_user_prompt(question, paragraphs)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        if getattr(self.tokenizer, "chat_template", None) is None and not hasattr(self.tokenizer, "apply_chat_template"):
            return f"{SYSTEM_PROMPT}\n\n{user}\nAnswer:"
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # templates without a system role (some Gemma builds): fold it into the user turn
            merged = [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user}"}]
            try:
                return self.tokenizer.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)
            except Exception:
                return f"{SYSTEM_PROMPT}\n\n{user}\nAnswer:"

    # -- generation
    def answer(self, question: str, paragraphs: list[dict]) -> AnswerResult:
        prompt = self.build_prompt(question, paragraphs)
        t0 = time.perf_counter()
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        n_in = int(enc["input_ids"].shape[-1])
        if hasattr(enc, "to") and getattr(self.model, "device", None) is not None:
            enc = enc.to(self.model.device)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", None)
        no_grad = self._torch.no_grad() if self._torch is not None else contextlib.nullcontext()
        with no_grad:
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,          # temperature 0: greedy, deterministic
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=pad_id,
            )
        new_tokens = out[0][n_in:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return AnswerResult(clean_answer(raw), raw, n_in, time.perf_counter() - t0)


# ------------------------------------------------------------------ ONNX T5
class OnnxT5LM:
    """T5-base exported to ONNX: the offline stand-in reader.

    T5 was trained on SQuAD in the ``question: ... context: ...`` format, so it
    answers from context without Hugging Face access. It is a stand-in for the
    planned instruct model, and results from it are labelled as such.
    """

    name = "t5-base-onnx"

    def __init__(self, encoder_path: str | Path, decoder_path: str | Path, tokenizer_path: str | Path,
                 max_new_tokens: int = 24, threads: int = 2, max_input_tokens: int = 4096):
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ImportError("the ONNX T5 reader needs `pip install onnxruntime tokenizers`") from exc
        self._np = np
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(threads)
        self.encoder = ort.InferenceSession(str(encoder_path), opts, providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(str(decoder_path), opts, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.max_new_tokens = int(max_new_tokens)
        self.max_input_tokens = int(max_input_tokens)

    @staticmethod
    def build_prompt(question: str, paragraphs: Sequence[dict]) -> str:
        context = " ".join(f"{p['title']}: {p['text']}" for p in paragraphs)
        return f"question: {question} context: {context}"

    def answer(self, question: str, paragraphs: list[dict]) -> AnswerResult:
        np = self._np
        t0 = time.perf_counter()
        ids = self.tokenizer.encode(self.build_prompt(question, paragraphs)).ids
        if len(ids) > self.max_input_tokens:
            ids = ids[: self.max_input_tokens - 1] + [1]       # keep the end-of-sequence token
        hidden = self.encoder.run(None, {"input_ids": np.asarray([ids], dtype=np.int64)})[0]
        out = [0]                                               # decoder start token
        for _ in range(self.max_new_tokens):
            logits = self.decoder.run(None, {"input_ids": np.asarray([out], dtype=np.int64),
                                             "encoder_hidden_states": hidden})[0]
            nxt = int(logits[0, -1].argmax())
            if nxt == 1:                                        # end of sequence
                break
            out.append(nxt)
        raw = self.tokenizer.decode(out[1:], skip_special_tokens=True)
        return AnswerResult(clean_answer(raw), raw, len(ids), time.perf_counter() - t0)


# ------------------------------------------------------------------- Ollama
class OllamaLM:
    """A model served by a local Ollama instance (optional backend, standard library only)."""

    def __init__(self, model: str = "qwen2.5:0.5b", host: str = "http://localhost:11434",
                 max_new_tokens: int = 32, timeout: float = 300.0):
        self.model = model
        self.name = f"ollama:{model}"
        self.host = host.rstrip("/")
        self.max_new_tokens = int(max_new_tokens)
        self.timeout = float(timeout)

    def answer(self, question: str, paragraphs: list[dict]) -> AnswerResult:
        user = build_user_prompt(question, paragraphs)
        body = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            "options": {"temperature": 0, "seed": 0, "num_predict": self.max_new_tokens},
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        raw = str(payload.get("message", {}).get("content", ""))
        # Ollama omits prompt_eval_count when the prompt was served from its cache.
        tokens = int(payload.get("prompt_eval_count") or len(f"{SYSTEM_PROMPT} {user}".split()))
        return AnswerResult(clean_answer(raw), raw, tokens, time.perf_counter() - t0)


# --------------------------------------------------------------------- stub
class StubLM:
    """Deterministic test double for CI. It is NOT a language model.

    It returns the title of the paragraph that shares the most words with the
    question. It exists so the pipeline can be exercised without model weights and
    is never used for reported numbers.
    """

    name = "stub"

    def answer(self, question: str, paragraphs: list[dict]) -> AnswerResult:
        t0 = time.perf_counter()
        prompt = f"{question} {build_context(paragraphs)}"
        q_words = set(re.findall(r"\w+", question.lower()))
        best_title, best_overlap = "", -1
        for p in paragraphs:
            words = set(re.findall(r"\w+", f"{p['title']} {p['text']}".lower()))
            overlap = len(q_words & words)
            if overlap > best_overlap:
                best_title, best_overlap = str(p["title"]), overlap
        return AnswerResult(best_title, best_title, len(prompt.split()), time.perf_counter() - t0)


# ------------------------------------------------------------------ factory
def make_lm(cfg: dict) -> LanguageModel:
    """Build a language model from a config dict.

    Kinds: ``hf`` (``model``, ``device``, ``max_new_tokens``, ``dtype``), ``onnx-t5``
    (``dir`` or ``encoder``/``decoder``/``tokenizer``), ``ollama`` (``model``, ``host``)
    and ``stub``.
    """
    cfg = dict(cfg or {})
    kind = str(cfg.pop("kind", "hf")).lower().replace("_", "-")
    if kind == "hf":
        if "model" in cfg:
            cfg["model_name"] = cfg.pop("model")
        return HFChatLM(**cfg)
    if kind == "onnx-t5":
        if "dir" in cfg:
            base = Path(cfg.pop("dir"))
            cfg.setdefault("encoder", base / "t5-encoder-12.onnx")
            cfg.setdefault("decoder", base / "t5-decoder-with-lm-head-12.onnx")
            cfg.setdefault("tokenizer", base / "tokenizer.json")
        return OnnxT5LM(cfg.pop("encoder"), cfg.pop("decoder"), cfg.pop("tokenizer"), **cfg)
    if kind == "ollama":
        return OllamaLM(**cfg)
    if kind == "stub":
        return StubLM()
    raise ValueError(f"unknown language model kind {kind!r}; use hf, onnx-t5, ollama or stub")


# -------------------------------------------------------------------- cache
class AnswerCache:
    """Answers on disk, keyed by (model name, question id, sorted selected indices).

    One JSON object per line. Every write is flushed, so an interrupted run
    resumes where it stopped.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._items: dict[tuple[str, str, tuple[int, ...]], AnswerResult] = {}
        self.hits = 0
        self.misses = 0
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue            # a half-written last line from an interrupted run
                    key = self._key(row["lm"], row["qid"], row["indices"])
                    self._items[key] = AnswerResult(str(row["answer"]), str(row["raw"]),
                                                    int(row["prompt_tokens"]), float(row["latency_s"]))

    @staticmethod
    def _key(lm_name: str, qid: str, indices: Iterable[int]) -> tuple[str, str, tuple[int, ...]]:
        return (str(lm_name), str(qid), tuple(sorted(int(i) for i in indices)))

    def __len__(self) -> int:
        return len(self._items)

    def get(self, lm_name: str, qid: str, indices: Iterable[int]) -> AnswerResult | None:
        return self._items.get(self._key(lm_name, qid, indices))

    def put(self, lm_name: str, qid: str, indices: Iterable[int], result: AnswerResult) -> None:
        key = self._key(lm_name, qid, indices)
        self._items[key] = result
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"lm": key[0], "qid": key[1], "indices": list(key[2]), **result.to_dict()}
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()

    def answer(self, lm: LanguageModel, question: Any, indices: Iterable[int]) -> AnswerResult:
        """Answer ``question`` from the paragraphs at ``indices`` (in original order), reading through the cache."""
        idx = tuple(sorted(int(i) for i in indices))
        hit = self.get(lm.name, question.qid, idx)
        if hit is not None:
            self.hits += 1
            return hit
        self.misses += 1
        result = lm.answer(question.question, [dict(question.paragraphs[i]) for i in idx])
        self.put(lm.name, question.qid, idx, result)
        return result
