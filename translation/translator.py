"""MarianMT candidate generation and attention-based terminology projection."""

from __future__ import annotations

from dataclasses import dataclass, field
import re


MODEL_NAME = "Helsinki-NLP/opus-mt-en-es"
EN_STOP = set("a an and are as at be by for from he her his in is it its of on or she that the their them they this to was were with".split())
ES_STOP = set("a al de del el ella en es esta este fue la las le lo los para por que se su sus un una y".split())


@dataclass
class Candidate:
    text: str
    mean_log_prob: float
    terms: dict[str, str] = field(default_factory=dict)
    beam_score: float | None = None


def split_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph.strip()) if s.strip()]


def repeated_terms(sentences: list[str]) -> set[str]:
    words = [set(re.findall(r"[a-z]+", sentence.lower())) - EN_STOP for sentence in sentences]
    return {word for i, left in enumerate(words) for right in words[i + 1:] for word in left & right}


def _word_spans(tokens: list[str], stopwords: set[str]) -> list[tuple[str, list[int]]]:
    """Group SentencePiece tokens into words while retaining token positions."""
    groups: list[tuple[str, list[int]]] = []
    word, positions = "", []
    for i, token in enumerate(tokens):
        if token in {"</s>", "<pad>", "<unk>"}:
            continue
        if token.startswith("▁") and positions:
            clean = re.sub(r"[^a-záéíóúüñ]", "", word.lower())
            if clean and clean not in stopwords:
                groups.append((clean, positions))
            word, positions = "", []
        word += token.lstrip("▁")
        positions.append(i)
    if positions:
        clean = re.sub(r"[^a-záéíóúüñ]", "", word.lower())
        if clean and clean not in stopwords:
            groups.append((clean, positions))
    return groups


class MarianTranslator:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        import torch
        from transformers import AutoTokenizer, MarianMTModel

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name, attn_implementation="eager").eval()

    def generate(self, sentences: list[str], count: int = 2) -> list[list[Candidate]]:
        if not 2 <= count <= 4:
            raise ValueError("candidate count must be 2–4")
        terms = repeated_terms(sentences)
        all_candidates: list[list[Candidate]] = []
        for sentence in sentences:
            source = self.tokenizer(sentence, return_tensors="pt")
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **source, num_beams=max(6, count * 2), num_return_sequences=count,
                    max_new_tokens=80, early_stopping=True,
                    return_dict_in_generate=True, output_scores=True,
                )
            unique: dict[str, Candidate] = {}
            for ids, beam_score in zip(generated.sequences, generated.sequences_scores):
                # Marian sequences begin with the decoder start (pad) token.
                target = ids[1:].unsqueeze(0)
                text = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
                if not text or text.casefold() in unique:
                    continue
                with self.torch.inference_mode():
                    output = self.model(**source, labels=target, output_attentions=bool(terms))
                    log_probs = output.logits.log_softmax(dim=-1)
                    token_scores = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                    mask = target.ne(self.tokenizer.pad_token_id)
                    score = (token_scores * mask).sum().item() / mask.sum().item()
                projected = self._project_terms(source["input_ids"][0], target[0], output.cross_attentions, terms) if terms else {}
                unique[text.casefold()] = Candidate(text, score, projected, float(beam_score))
            if not unique:
                raise RuntimeError(f"MarianMT produced no translation for {sentence!r}")
            all_candidates.append(list(unique.values()))
        return all_candidates

    def _project_terms(self, source_ids, target_ids, cross_attentions, terms: set[str]) -> dict[str, str]:
        if not cross_attentions:
            return {}
        source_words = _word_spans(self.tokenizer.convert_ids_to_tokens(source_ids.tolist()), EN_STOP)
        target_words = _word_spans(self.tokenizer.convert_ids_to_tokens(target_ids.tolist()), ES_STOP)
        if not target_words:
            return {}
        # Mean over decoder layers and heads; attention is a heuristic alignment, not a linguistic annotation.
        attention = self.torch.stack([layer[0].mean(dim=0) for layer in cross_attentions]).mean(dim=0)
        mapped: dict[str, str] = {}
        for term in terms:
            spans = [span for word, span in source_words if word == term]
            if not spans:
                continue
            source_positions = [position for span in spans for position in span]
            best_word = max(
                target_words,
                key=lambda item: attention[item[1], :][:, source_positions].sum().item() / len(item[1]),
            )
            mapped[term] = best_word[0]
        return mapped
