"""Iceberg ``[[k+2, k, 2]]`` error-detecting code for QAOA-style circuits.

Two encoders share one layout convention and one post-selection routine:

* :func:`encode_bus` (default): nearest-neighbour only, hub gates routed through a row
  of bus ancillas, 18 qubits for 10 paragraphs. This is the one to run on Sqale.
* :func:`encode_direct`: the textbook mapping with direct hub gates, 14 qubits for 10
  paragraphs, all-to-all connectivity assumed. Reference point and ablations.
"""

from __future__ import annotations

from typing import Any

from qiskit import QuantumCircuit

from .bus import encode_bus, grid_coupling, place
from .encode import UnsupportedGateError, encode_direct, layout_from_circuit
from .layout import IcebergLayout, layout_of
from .postselect import PostselectStats, ProbabilityDecoder, postselect, postselect_probs

STYLES = ("bus", "direct")


def encode(circuit: QuantumCircuit, *, style: str = "bus", **options: Any) -> QuantumCircuit:
    """Iceberg-encode a QAOA-style circuit.

    Args:
        circuit: bare circuit: ``H`` on every qubit, then ``rz``, ``rzz``, ``rx`` layers,
            then terminal measurements (what :func:`qsel.qaoa_circuit` builds).
        style: ``"bus"`` (grid-native, recommended) or ``"direct"`` (textbook).
        options: passed to :func:`encode_bus` or :func:`encode_direct`.

    Returns the encoded circuit. Feed its counts and the circuit itself to
    :func:`postselect` to discard flagged shots and recover logical bitstrings.
    """
    if style == "bus":
        checks = options.pop("checks", "both")
        if checks != "both":
            raise ValueError('the bus style always runs both checks; use style="direct" for checks="z" or "x"')
        if "sx_flag" in options:
            raise TypeError('option sx_flag only applies to style="direct"; the bus style always flags its S_X check')
        return encode_bus(circuit, **options)
    if style == "direct":
        bad = sorted(set(options) & {"placement", "edge_mode", "precopy"})
        if bad:
            raise TypeError(f'option(s) {", ".join(bad)} only apply to style="bus"')
        options.pop("seed", None)          # the direct mapping has no random choices
        return encode_direct(circuit, **options)
    raise ValueError(f"style must be one of {STYLES}")


__all__ = [
    "IcebergLayout",
    "PostselectStats",
    "ProbabilityDecoder",
    "STYLES",
    "UnsupportedGateError",
    "encode",
    "encode_bus",
    "encode_direct",
    "grid_coupling",
    "layout_from_circuit",
    "layout_of",
    "place",
    "postselect",
    "postselect_probs",
]
