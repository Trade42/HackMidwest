"""Frozen, revision-pinned DistilBERT embeddings; labels never enter encoding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from qfp import RobustAngleScaler

MODEL_ID = 'distilbert/distilbert-base-uncased'
MODEL_REVISION = '12040accade4e8a0f71eabdb258fecc2e7e948be'
ENCODING_VERSION = 'sentence-pair-candidates-masked-mean-v1'


def text_pair(row):
    """Use text and ordered candidates only, excluding labels and metadata."""
    return (row['sentence'], f"Candidate 0: {row['candidate_0']}. Candidate 1: {row['candidate_1']}.")


def input_fingerprint(rows, specification):
    payload = {'specification': specification, 'inputs': [text_pair(row) for row in rows]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def masked_mean(hidden, attention_mask, special_tokens_mask):
    """Pool content tokens; padding and inserted special tokens have no weight."""
    mask = attention_mask * (1 - special_tokens_mask)
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)


class FrozenDistilBertEncoder:
    def __init__(self, *, model_id=MODEL_ID, revision=MODEL_REVISION,
                 batch_size=32, threads=4, max_length=128):
        self.model_id, self.revision = model_id, revision
        self.batch_size, self.threads, self.max_length = batch_size, threads, max_length
        self.model = self.tokenizer = None

    @property
    def specification(self):
        return dict(model_id=self.model_id, revision=self.revision, pooling=ENCODING_VERSION,
                    max_length=self.max_length, device='cpu', dtype='float32')

    def load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(self.threads)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        self.model = AutoModel.from_pretrained(self.model_id, revision=self.revision,
                                              use_safetensors=True, attn_implementation='eager')
        self.model.eval()
        self.model.requires_grad_(False)

    def encode(self, rows, *, cache_dir=None, progress=None):
        fingerprint = input_fingerprint(rows, self.specification)
        cache = Path(cache_dir) / f'{fingerprint}.npz' if cache_dir is not None else None
        start = time.perf_counter()
        if cache is not None and cache.exists():
            with np.load(cache, allow_pickle=False) as data:
                values = data['embeddings']
                metadata = json.loads(str(data['metadata']))
            return values, {**metadata, 'cache_hit': True, 'current_seconds': time.perf_counter()-start}
        self.load()
        import torch
        chunks, longest = [], 0
        pairs = [text_pair(row) for row in rows]
        with torch.inference_mode():
            for offset in range(0, len(rows), self.batch_size):
                batch = pairs[offset:offset+self.batch_size]
                tokens = self.tokenizer([p[0] for p in batch], [p[1] for p in batch],
                                        padding=True, truncation=False, return_tensors='pt',
                                        return_special_tokens_mask=True)
                length = tokens['input_ids'].shape[1]
                longest = max(longest, length)
                if length > self.max_length:
                    raise ValueError(f'{length} tokens exceeds {self.max_length}; refusing silent truncation')
                special = tokens.pop('special_tokens_mask')
                tokens.pop('token_type_ids', None)  # DistilBERT does not use segment embeddings.
                hidden = self.model(**tokens).last_hidden_state
                chunks.append(masked_mean(hidden, tokens['attention_mask'], special).cpu().numpy())
                if progress is not None:
                    progress(min(offset+self.batch_size, len(rows)), len(rows))
        values = np.concatenate(chunks) if chunks else np.empty((0, self.model.config.dim), np.float32)
        elapsed = time.perf_counter()-start
        metadata = dict(specification=self.specification, input_fingerprint=fingerprint,
                        embedding_width=values.shape[1], examples=len(values), longest_tokens=longest,
                        generation_seconds=elapsed, cache_hit=False, current_seconds=elapsed,
                        frozen=True, training=False, requires_grad=False)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, embeddings=values, metadata=json.dumps(metadata))
        return values, metadata


class TrainingOnlyReducer:
    """Fit PCA and angle scaling on the current labeled training subset only."""
    def __init__(self, dimensions=8, max_angle=.55):
        self.dimensions, self.max_angle = dimensions, max_angle

    def fit(self, embeddings):
        if len(embeddings) <= self.dimensions:
            raise ValueError('PCA requires more training rows than components')
        self.pca = PCA(n_components=self.dimensions, svd_solver='full')
        with threadpool_limits(limits=1):
            reduced = self.pca.fit_transform(np.asarray(embeddings, dtype=float))
        self.scaler = RobustAngleScaler(max_angle=self.max_angle).fit(reduced)
        self.n_fit_rows = len(embeddings)
        return self

    def transform(self, embeddings):
        with threadpool_limits(limits=1):
            return self.scaler.transform(self.pca.transform(np.asarray(embeddings, dtype=float)))
