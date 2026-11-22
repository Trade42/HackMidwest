"""Bit-ordering helpers shared by every solver tier.

One convention everywhere, agreed between the QAOA and Iceberg code:

* paragraph ``i`` is logical qubit ``i`` and classical bit ``i``;
* ``x[i] == 1`` means paragraph ``i`` is selected (qubit measured in ``|1>``);
* a selection is indexed by the integer ``sum_i x[i] << i``;
* count keys follow Qiskit: bit ``i`` is the ``i``-th character from the right.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

Selection = tuple[int, ...]


def index_to_x(index: int, n: int) -> Selection:
    """Integer index -> selection tuple of length ``n``."""
    return tuple((index >> i) & 1 for i in range(n))


def x_to_index(x: Sequence[int]) -> int:
    """Selection -> integer index."""
    return int(sum(int(b) << i for i, b in enumerate(x)))


def key_to_x(key: str, n: int | None = None) -> Selection:
    """Qiskit-style count key (bit 0 rightmost) -> selection tuple."""
    key = key.replace(" ", "")
    bits = tuple(int(c) for c in reversed(key))
    if n is not None:
        if len(bits) < n:
            bits = bits + (0,) * (n - len(bits))
        bits = bits[:n]
    return bits


def x_to_key(x: Sequence[int]) -> str:
    """Selection tuple -> Qiskit-style count key."""
    return "".join(str(int(b)) for b in reversed(tuple(x)))


def selected(x: Sequence[int]) -> list[int]:
    """Indices of the selected paragraphs."""
    return [i for i, b in enumerate(x) if b]


def from_indices(indices: Iterable[int], n: int) -> Selection:
    """Paragraph indices -> selection tuple."""
    chosen = set(int(i) for i in indices)
    return tuple(1 if i in chosen else 0 for i in range(n))


def all_bit_table(n: int) -> np.ndarray:
    """``(2**n, n)`` uint8 table whose row ``k`` is ``index_to_x(k, n)``."""
    idx = np.arange(1 << n, dtype=np.int64)
    return ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.uint8)


def counts_to_index_array(counts: dict[str, int], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Counts dict -> (indices, multiplicities) arrays."""
    idx = np.fromiter((int(k.replace(" ", ""), 2) for k in counts), dtype=np.int64, count=len(counts))
    mult = np.fromiter((int(v) for v in counts.values()), dtype=np.int64, count=len(counts))
    mask = (1 << n) - 1
    return idx & mask, mult
