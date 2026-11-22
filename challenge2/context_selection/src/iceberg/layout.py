"""Where everything lives in an Iceberg-encoded circuit.

The ``[[k+2, k, 2]]`` Iceberg code stores ``k`` logical qubits (``k`` even) in the
``k`` data qubits plus two extra qubits, ``t`` (top) and ``b`` (bottom):

    stabilizers   S_X = X^(k+2)      S_Z = Z^(k+2)
    logical X_i = X_t X_i            logical Z_i = Z_i Z_b

Physical qubit order in the encoded circuit (classical bit ``j`` reads qubit ``j``):

    0 .. K-1    data qubits d_0 .. d_{K-1}   (K = k rounded up to even)
    K           t
    K+1         b
    then        a  (ancilla that measures S_X),  f  (flag)   when present

The grid-native "bus" style (:func:`iceberg.encode_bus`) replaces ``a`` and ``f`` by a
row of bus ancillas: one of them reads out S_X and the others act as flags. A shot is
kept only if the S_X qubit and every flag read 0. Qubit indices follow the order above,
not grid order; ``positions`` gives each qubit's ``(row, col)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CHECKS = ("both", "z", "x")


@dataclass(frozen=True)
class IcebergLayout:
    k: int                        #: logical qubits in the bare circuit
    k_code: int                   #: logical qubits in the code (even)
    checks: str                   #: "both", "z" (parity only) or "x" (ancilla only)
    init: str                     #: "plus" or "zero": the logical starting state
    ghz_init: bool                #: True when the entangled starting state is prepared
    top: int
    bottom: int
    sx_ancilla: int | None        #: qubit whose readout is the S_X syndrome
    flags: tuple[int, ...]        #: qubits that must read 0
    init_flag: bool               #: a flag checks the state preparation
    sx_flag: bool                 #: a flag guards the S_X measurement against hook errors
    logical_clbits: tuple[int, ...]   #: bare classical bit of logical qubit i, -1 if unmeasured
    num_bare_clbits: int
    measured: bool
    style: str = "direct"         #: "direct" (hub gates) or "bus" (grid-native)
    positions: tuple[tuple[int, int], ...] | None = None   #: (row, col) per qubit, bus style

    # ----------------------------------------------------------- derived views
    @property
    def n_code(self) -> int:
        """Physical qubits in the code block."""
        return self.k_code + 2

    @property
    def data(self) -> tuple[int, ...]:
        return tuple(range(self.k_code))

    @property
    def code_qubits(self) -> tuple[int, ...]:
        return tuple(range(self.k_code + 2))

    @property
    def num_qubits(self) -> int:
        return self.n_code + (self.sx_ancilla is not None) + len(self.flags)

    @property
    def flag(self) -> int | None:
        """The single flag qubit of the direct style (``None`` when absent)."""
        return self.flags[0] if self.flags else None

    @property
    def check_z(self) -> bool:
        return self.checks in ("both", "z")

    @property
    def check_x(self) -> bool:
        return self.checks in ("both", "x")

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["logical_clbits"] = list(self.logical_clbits)
        d["flags"] = list(self.flags)
        d["positions"] = [list(p) for p in self.positions] if self.positions is not None else None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "IcebergLayout":
        d = dict(d)
        d["logical_clbits"] = tuple(d["logical_clbits"])
        d["flags"] = tuple(d.get("flags", ()))
        pos = d.get("positions")
        d["positions"] = tuple(tuple(p) for p in pos) if pos is not None else None
        return cls(**d)

    @classmethod
    def default(cls, k: int, *, checks: str = "both", init_flag: bool = True,
                sx_flag: bool = True, init: str = "plus") -> "IcebergLayout":
        """Layout produced by :func:`iceberg.encode_direct` for a fully measured ``k``-qubit circuit."""
        return make_layout(k, checks=checks, init=init, init_flag=init_flag, sx_flag=sx_flag,
                           logical_clbits=tuple(range(k)), num_bare_clbits=k, measured=True)


def make_layout(k: int, *, checks: str, init: str, init_flag: bool, sx_flag: bool,
                logical_clbits: tuple[int, ...], num_bare_clbits: int, measured: bool) -> IcebergLayout:
    if checks not in CHECKS:
        raise ValueError(f"checks must be one of {CHECKS}")
    if init not in ("plus", "zero"):
        raise ValueError("init must be 'plus' or 'zero'")
    if k < 1:
        raise ValueError("need at least one logical qubit")
    k_code = k + (k % 2)
    check_z = checks in ("both", "z")
    check_x = checks in ("both", "x")
    # The entangled (GHZ-type) start is needed only when the stabilizer that the
    # product state leaves undefined is one we intend to check.
    ghz_init = check_z if init == "plus" else check_x
    use_init_flag = bool(init_flag and ghz_init)
    use_sx_flag = bool(sx_flag and check_x)
    nxt = k_code + 2
    sx_ancilla = None
    if check_x:
        sx_ancilla, nxt = nxt, nxt + 1
    flags = (nxt,) if (use_init_flag or use_sx_flag) else ()
    return IcebergLayout(
        k=k, k_code=k_code, checks=checks, init=init, ghz_init=ghz_init,
        top=k_code, bottom=k_code + 1, sx_ancilla=sx_ancilla, flags=flags,
        init_flag=use_init_flag, sx_flag=use_sx_flag,
        logical_clbits=tuple(logical_clbits), num_bare_clbits=num_bare_clbits, measured=measured,
    )


def layout_of(obj: Any) -> IcebergLayout:
    """Recover the layout from a layout, a dict, or an encoded circuit's metadata."""
    if isinstance(obj, IcebergLayout):
        return obj
    if isinstance(obj, dict):
        return IcebergLayout.from_dict(obj.get("iceberg", obj))
    meta = getattr(obj, "metadata", None)
    if isinstance(meta, dict) and "iceberg" in meta:
        return IcebergLayout.from_dict(meta["iceberg"])
    raise ValueError("no Iceberg layout found; pass the encoded circuit or its layout")
