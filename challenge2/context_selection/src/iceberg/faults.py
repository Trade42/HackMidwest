"""Exhaustive single-fault analysis of a circuit.

Every place a fault can occur gets every Pauli it can produce:

* after each one-qubit gate: X, Y, Z on that qubit;
* after each two-qubit gate: the 15 non-identity two-qubit Paulis on its qubits;
* before each measurement: a bit flip.

Each fault is propagated exactly (statevector, no sampling) and the checks are applied
to the resulting outcome distribution. Per fault we record the probability that the
shot is discarded and how far the kept shots' logical distribution moved from the
ideal one. This answers "which errors do the checks catch" without statistical noise,
and gives a one-fault model of the kept share and of the surviving logical error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Iterable

import numpy as np
from qiskit import QuantumCircuit

from .layout import layout_of
from .postselect import ProbabilityDecoder
from .sim import Program, apply, apply_pauli, outcome_probabilities, prefix_states, to_program

ONE_QUBIT = ("X", "Y", "Z")
TWO_QUBIT = tuple("".join(p) for p in product("IXYZ", repeat=2) if "".join(p) != "II")
TOL = 1e-9


@dataclass(frozen=True)
class Fault:
    kind: str            #: "gate1", "gate2" or "readout"
    op_index: int        #: index into the flattened program (-1 for readout)
    gate: str
    block: str
    qubits: tuple[int, ...]
    pauli: str
    p_detect: float      #: probability the shot is discarded
    tvd: float           #: total variation distance of the kept logical distribution from ideal
    d_energy: float      #: shift of the kept shots' mean energy (0 when no energies are given)

    @property
    def outcome(self) -> str:
        if self.p_detect > 1 - TOL:
            return "detected"
        if self.tvd < TOL:
            return "harmless" if self.p_detect < TOL else "partly detected, harmless"
        return "undetected error" if self.p_detect < TOL else "partly detected"

    @property
    def accepted_error(self) -> float:
        """Logical error that survives post-selection: ``(1 - p_detect) * tvd``."""
        return (1.0 - self.p_detect) * self.tvd


@dataclass
class FaultReport:
    faults: list[Fault]
    ideal: np.ndarray
    num_qubits: int
    locations: dict[str, int] = field(default_factory=dict)   #: fault locations per kind
    ideal_energy: float = 0.0

    # ------------------------------------------------------------ summaries
    def by(self, key: Callable[[Fault], Any]) -> dict[Any, dict[str, float]]:
        """Aggregate over faults sharing ``key(fault)``."""
        groups: dict[Any, list[Fault]] = {}
        for f in self.faults:
            groups.setdefault(key(f), []).append(f)
        out = {}
        for name, fs in groups.items():
            out[name] = {
                "faults": len(fs),
                "detected": sum(f.outcome == "detected" for f in fs),
                "harmless": sum(f.outcome == "harmless" for f in fs),
                "undetected_error": sum(f.outcome == "undetected error" for f in fs),
                "partial": sum(f.outcome.startswith("partly") for f in fs),
                "mean_p_detect": float(np.mean([f.p_detect for f in fs])),
                "mean_accepted_error": float(np.mean([f.accepted_error for f in fs])),
            }
        return out

    def by_block(self) -> dict[str, dict[str, float]]:
        return self.by(lambda f: f.block)

    def by_kind(self) -> dict[str, dict[str, float]]:
        return self.by(lambda f: f.kind)

    def totals(self) -> dict[str, float]:
        return self.by(lambda f: "all")["all"]

    # ------------------------------------------------------- one-fault model
    def predict(self, p1: float, p2: float, p_meas: float) -> dict[str, float]:
        """Kept share and surviving error if at most one fault occurs.

        Each one-qubit gate fails with probability ``p1``, each two-qubit gate with
        ``p2`` (uniform over its Paulis), each measurement flips with ``p_meas``.
        ``p_multi`` is the probability of two or more faults, which this model
        does not resolve; it bounds how far the prediction can be off.
        """
        rate = {"gate1": p1 / 3.0, "gate2": p2 / 15.0, "readout": p_meas}
        loc_p = {"gate1": p1, "gate2": p2, "readout": p_meas}
        p_none = 1.0
        for kind, count in self.locations.items():
            p_none *= (1.0 - loc_p[kind]) ** count
        kept = p_none
        err = 0.0
        d_e = 0.0
        p_single = 0.0
        for f in self.faults:
            pf = p_none * rate[f.kind] / max(1.0 - loc_p[f.kind], 1e-12)
            p_single += pf
            kept += pf * (1.0 - f.p_detect)
            err += pf * (1.0 - f.p_detect) * f.tvd
            d_e += pf * (1.0 - f.p_detect) * f.d_energy
        return {
            "p_no_fault": p_none,
            "p_multi": max(0.0, 1.0 - p_none - p_single),
            "kept_share": kept,
            "logical_tvd": err / kept if kept else 0.0,
            "d_energy": d_e / kept if kept else 0.0,
        }

    def coefficients(self) -> dict[str, float]:
        """Low-noise slopes per unit two-qubit error rate, with ``p1 = p_meas = 0``.

        ``discard`` is d(discarded share)/d(p2) and ``error`` is d(surviving TVD)/d(p2)
        at ``p2 -> 0``. Bare and encoded circuits can be compared on ``error`` alone.
        """
        two = [f for f in self.faults if f.kind == "gate2"]
        return {
            "two_qubit_gates": self.locations.get("gate2", 0),
            "discard": float(sum(f.p_detect for f in two) / 15.0),
            "error": float(sum(f.accepted_error for f in two) / 15.0),
            "d_energy": float(sum((1.0 - f.p_detect) * f.d_energy for f in two) / 15.0),
        }


def _decoder_for(circuit: QuantumCircuit, program: Program) -> Callable[[np.ndarray], tuple[np.ndarray, float]]:
    """Checks + decoding for an encoded circuit, identity for a bare one."""
    try:
        lay = layout_of(circuit)
    except ValueError:
        return lambda probs: (probs, float(probs.sum()))
    return ProbabilityDecoder(lay)


def enumerate_faults(
    circuit: QuantumCircuit,
    *,
    energies: np.ndarray | None = None,
    virtual_gates: Iterable[str] = (),
    include_readout: bool = True,
) -> FaultReport:
    """Inject every single fault into ``circuit`` and score what the checks do.

    Args:
        circuit: bare or Iceberg-encoded circuit with terminal measurements and bound
            parameters. Gates may act on at most two qubits.
        energies: optional objective value per logical outcome; enables ``d_energy``.
        virtual_gates: gate names treated as error-free (for example ``("rz",)`` when
            Z rotations are frame changes).
        include_readout: also flip each measured bit.
    """
    program = to_program(circuit)
    n = program.num_qubits
    decode = _decoder_for(circuit, program)
    states = prefix_states(program)
    skip = set(virtual_gates)

    def score(psi: np.ndarray) -> tuple[np.ndarray, float]:
        mass, kept = decode(outcome_probabilities(program, psi))
        return mass, kept

    ideal_mass, ideal_kept = score(states[-1])
    if ideal_kept < 1 - 1e-7:
        raise ValueError(f"the noiseless circuit fails its own checks (kept {ideal_kept:.6f})")
    ideal = ideal_mass / ideal_kept
    e0 = float(ideal @ energies) if energies is not None else 0.0

    def record(kind: str, gi: int, gate: str, block: str, qubits: tuple[int, ...], pauli: str,
               mass: np.ndarray, kept: float) -> Fault:
        if kept > 1e-14:
            dist = mass / kept
            tvd = 0.5 * float(np.abs(dist - ideal).sum())
            d_e = float(dist @ energies) - e0 if energies is not None else 0.0
        else:
            tvd, d_e = 0.0, 0.0
        return Fault(kind, gi, gate, block, qubits, pauli, float(min(max(1.0 - kept, 0.0), 1.0)), tvd, d_e)

    faults: list[Fault] = []
    locations = {"gate1": 0, "gate2": 0, "readout": 0}
    for gi, op in enumerate(program.ops):
        if op.name in skip:
            continue
        kind = "gate1" if len(op.qubits) == 1 else "gate2"
        locations[kind] += 1
        paulis = ONE_QUBIT if kind == "gate1" else TWO_QUBIT
        after = states[gi + 1]
        for pauli in paulis:
            psi = apply_pauli(after, pauli, op.qubits, n)
            for nxt in program.ops[gi + 1:]:
                psi = apply(psi, nxt.matrix, nxt.qubits, n)
            mass, kept = score(psi)
            faults.append(record(kind, gi, op.name, op.block, op.qubits, pauli, mass, kept))

    if include_readout:
        final = states[-1]
        for qb, _ in program.measures:
            locations["readout"] += 1
            psi = apply_pauli(final, "X", (qb,), n)
            mass, kept = score(psi)
            faults.append(record("readout", -1, "measure", "measure", (qb,), "X", mass, kept))

    return FaultReport(faults, ideal, n, locations, e0)


def format_report(report: FaultReport, title: str = "") -> str:
    """Plain-text table of a :class:`FaultReport`, grouped by block."""
    lines = []
    if title:
        lines.append(title)
    head = f"{'block':<10}{'faults':>8}{'detected':>10}{'harmless':>10}{'undetected':>12}{'partial':>9}{'mean P(discard)':>17}"
    lines.append(head)
    rows = report.by_block()
    rows["TOTAL"] = report.totals()
    for name, r in rows.items():
        lines.append(
            f"{name:<10}{r['faults']:>8}{r['detected']:>10}{r['harmless']:>10}"
            f"{r['undetected_error']:>12}{r['partial']:>9}{r['mean_p_detect']:>17.3f}"
        )
    return "\n".join(lines)
