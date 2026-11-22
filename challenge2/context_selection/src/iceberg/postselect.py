"""Discard shots where the Iceberg code caught an error and decode the rest.

A shot is kept when every enabled check passes:

* ``S_Z``: the measured code bits (data, ``t``, ``b``) have even parity;
* ``S_X``: the ancilla ``a`` reads 0;
* flags: every flag qubit reads 0.

For kept shots, logical bit ``i`` is ``bit(d_i) XOR bit(b)``. The returned counts use
the bare circuit's classical-bit order, so downstream code cannot tell the tiers apart.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from .layout import IcebergLayout, layout_of


@dataclass(frozen=True)
class PostselectStats:
    shots: int
    kept: int
    fail_sz: int      #: shots with odd code parity
    fail_sx: int      #: shots where the S_X ancilla fired
    fail_flag: int    #: shots where the flag fired

    @property
    def kept_share(self) -> float:
        return self.kept / self.shots if self.shots else 0.0

    @property
    def discarded(self) -> int:
        return self.shots - self.kept

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "kept_share": self.kept_share}


def _check_bits(layout: IcebergLayout) -> tuple[int, int, int]:
    """Bit masks: code parity mask, S_X ancilla bit, flag bit (0 when absent)."""
    code_mask = sum(1 << j for j in layout.code_qubits)
    sx_mask = (1 << layout.sx_ancilla) if (layout.check_x and layout.sx_ancilla is not None) else 0
    flag_mask = sum(1 << j for j in layout.flags)
    return code_mask, sx_mask, flag_mask


def postselect(
    counts: Mapping[str, int],
    layout: IcebergLayout | Mapping[str, Any] | Any | None = None,
) -> tuple[dict[str, int], PostselectStats]:
    """Keep the shots that pass every check and decode them to logical bitstrings.

    Args:
        counts: Qiskit-style counts of the encoded circuit (bit 0 rightmost).
        layout: the encoded circuit, its metadata dict, or an :class:`IcebergLayout`.
            Required: the two encodings cannot be told apart from the bitstrings.

    Returns:
        ``(logical_counts, stats)``. Keys of ``logical_counts`` are plain bitstrings over
        the bare circuit's classical bits (bit 0 rightmost, no register spaces).
    """
    if layout is None:
        raise ValueError("postselect needs the encoded circuit (or its layout): "
                         "iceberg.postselect(counts, encoded_circuit)")
    lay = layout_of(layout)
    if not counts:
        return {}, PostselectStats(0, 0, 0, 0, 0)
    code_mask, sx_mask, flag_mask = _check_bits(lay)
    b_bit = lay.bottom
    width = lay.num_bare_clbits

    out: dict[str, int] = {}
    shots = kept = fail_sz = fail_sx = fail_flag = 0
    for key, c in counts.items():
        c = int(c)
        bits = str(key).replace(" ", "")
        if len(bits) != lay.num_qubits or set(bits) - {"0", "1"}:
            raise ValueError(f"count key {key!r} is not a {lay.num_qubits}-bit string; "
                             "expected Qiskit-style binary keys of the encoded circuit")
        m = int(bits, 2)
        shots += c
        bad_z = lay.check_z and (bin(m & code_mask).count("1") & 1)
        bad_x = bool(m & sx_mask)
        bad_f = bool(m & flag_mask)
        fail_sz += c * bool(bad_z)
        fail_sx += c * bad_x
        fail_flag += c * bad_f
        if bad_z or bad_x or bad_f:
            continue
        kept += c
        zb = (m >> b_bit) & 1
        logical = 0
        for i, cl in enumerate(lay.logical_clbits):
            if cl >= 0 and (((m >> i) & 1) ^ zb):
                logical |= 1 << cl
        lk = format(logical, f"0{width}b")
        out[lk] = out.get(lk, 0) + c
    return out, PostselectStats(shots, kept, fail_sz, fail_sx, fail_flag)


class ProbabilityDecoder:
    """Vectorised post-selection of a full outcome distribution.

    Used by the exact checks (ideal equivalence, fault enumeration), where the
    encoded circuit's outcome probabilities are available for all bitstrings.
    """

    def __init__(self, layout: IcebergLayout | Any):
        lay = layout_of(layout)
        self.layout = lay
        m = np.arange(1 << lay.num_qubits, dtype=np.int64)
        code_mask, sx_mask, flag_mask = _check_bits(lay)
        parity = np.zeros(len(m), dtype=np.int64)
        for j in lay.code_qubits:
            parity ^= (m >> j) & 1
        self.fail_sz = (parity == 1) if lay.check_z else np.zeros(len(m), dtype=bool)
        self.fail_sx = (m & sx_mask) != 0
        self.fail_flag = (m & flag_mask) != 0
        self.keep = ~(self.fail_sz | self.fail_sx | self.fail_flag)
        zb = (m >> lay.bottom) & 1
        logical = np.zeros(len(m), dtype=np.int64)
        for i, cl in enumerate(lay.logical_clbits):
            if cl >= 0:
                logical |= (((m >> i) & 1) ^ zb) << cl
        self.logical_index = logical
        self.size = 1 << lay.num_bare_clbits

    def __call__(self, probs: np.ndarray) -> tuple[np.ndarray, float]:
        """Return ``(logical_mass, kept)``: unnormalised logical distribution and kept mass."""
        kept = probs[self.keep]
        mass = np.bincount(self.logical_index[self.keep], weights=kept, minlength=self.size)
        return mass, float(kept.sum())

    def breakdown(self, probs: np.ndarray) -> dict[str, float]:
        return {
            "kept": float(probs[self.keep].sum()),
            "fail_sz": float(probs[self.fail_sz].sum()),
            "fail_sx": float(probs[self.fail_sx].sum()),
            "fail_flag": float(probs[self.fail_flag].sum()),
        }


def postselect_probs(probs: np.ndarray, layout: IcebergLayout | Any) -> tuple[np.ndarray, float]:
    """Post-select an exact outcome distribution; see :class:`ProbabilityDecoder`."""
    return ProbabilityDecoder(layout)(np.asarray(probs, dtype=float))
