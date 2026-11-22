"""Exhaustive single-fault enumeration with my own simulator and my own Pauli tracker.

For every fault two independent things are computed:
  (A) exact physical simulation -> kept mass and kept logical distribution;
  (B) Pauli-frame prediction: push the fault to the end of the circuit (exact Heisenberg
      identity  R(theta) P = P R(+-theta)), read off flipped measurement bits, and build
      the logical circuit with the sign-flipped rotations.
(A) is ground truth for "discarded / harmless / harmful".  (B) is used for the logical
weight of the fault and is validated against (A) for every fault.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import stim

import iceberg
from vlib import (PAULI, apply, block_labels, gate_matrix, kron2, parse, qubit_probs, run_ops,
                  zero_state, tv)

ROT = {"rz": "Z", "rx": "X", "rzz": "ZZ", "rxx": "XX"}
TWO_PAULIS = ["".join(p) for p in itertools.product("IXYZ", repeat=2) if "".join(p) != "II"]


def popc(x: int) -> int:
    return bin(x).count("1")


# ------------------------------------------------------------------ Pauli tracker
def push(x: int, z: int, name: str, qs) -> tuple[int, int]:
    """Conjugate Pauli (x,z bitmasks) by a Clifford gate (phases ignored)."""
    if name == "h":
        q = qs[0]
        bx, bz = (x >> q) & 1, (z >> q) & 1
        if bx != bz:
            x ^= 1 << q
            z ^= 1 << q
    elif name == "cx":
        c, t = qs
        if (x >> c) & 1:
            x ^= 1 << t
        if (z >> t) & 1:
            z ^= 1 << c
    elif name in ("x", "y", "z", "id"):
        pass
    else:
        raise ValueError(name)
    return x, z


def generator(name: str, qs) -> tuple[int, int]:
    m = sum(1 << q for q in qs)
    return (0, m) if ROT[name][0] == "Z" else (m, 0)


def anticommute(x1, z1, x2, z2) -> bool:
    return (popc(x1 & z2) + popc(z1 & x2)) & 1 == 1


def pauli_masks(pstr: str, qs) -> tuple[int, int]:
    x = z = 0
    for ch, q in zip(pstr, qs):
        if ch in "XY":
            x |= 1 << q
        if ch in "YZ":
            z |= 1 << q
    return x, z


# ------------------------------------------------------------------ analysis object
@dataclass
class Fault:
    kind: str            # "gate", "prep", "meas"
    op_index: int        # index into ops (gate), qubit (prep / meas)
    data_index: int      # index into enc.data
    gate: str
    qubits: tuple
    pauli: str
    block: str
    kept: float = 0.0
    tv: float = 0.0      # TV(kept normalised logical dist, ideal)
    status: str = ""     # discarded / harmless / harmful / partial
    pred_kept: int = -1
    pred_ok: bool = True
    shift: int = 0       # logical flip pattern at readout (from tracker)
    lx: int = 0          # logical X pattern of the residual just before the syndrome block
    lz: int = 0          # logical Z pattern there
    anc_x: int = 0       # X part left on ancilla/bus qubits just before the syndrome block
    flipped: tuple = ()  # rotations (op indices) whose sign is flipped
    pure_shift: int = -1  # pattern s with kept dist == ideal shifted by s (exact sim), else -1

    @property
    def lweight(self) -> int:
        return popc(self.lx | self.lz)


class Analysis:
    def __init__(self, enc, tol=1e-9):
        self.enc = enc
        self.lay = lay = iceberg.layout_of(enc)
        self.n = n = enc.num_qubits
        self.ops, self.meas = parse(enc)
        assert [(q, c) for _, q, c in self.meas] == [(j, j) for j in range(n)], "expect clbit j = qubit j"
        self.labels = block_labels(enc)
        assert len(self.labels) == len(enc.data)
        self.op_block = [self.labels[idx] for idx, *_ in self.ops]
        self.mats = [gate_matrix(name, params) for _, name, _, params in self.ops]
        self.tol = tol
        # index of the first op of the syndrome block
        syn = [i for i, b in enumerate(self.op_block) if b == "syndrome"]
        self.syn_start = syn[0] if syn else len(self.ops)
        assert all(b == "syndrome" for b in self.op_block[self.syn_start:]), "syndrome block is not a suffix"
        # keep rule / decoding, written from the layout
        m = np.arange(1 << n, dtype=np.int64)
        self.code_mask = sum(1 << j for j in lay.code_qubits)
        parity = np.zeros_like(m)
        for j in lay.code_qubits:
            parity ^= (m >> j) & 1
        self.check_mask = (0 if lay.sx_ancilla is None else 1 << lay.sx_ancilla) | sum(1 << f for f in lay.flags)
        self.use_parity = lay.checks in ("both", "z")
        keep = (m & self.check_mask) == 0
        if self.use_parity:
            keep &= parity == 0
        self.keep = keep
        b = (m >> lay.bottom) & 1
        logical = np.zeros_like(m)
        for i, cl in enumerate(lay.logical_clbits):
            if cl >= 0:
                logical |= (((m >> i) & 1) ^ b) << cl
        self.logical_index = logical
        self.nb = lay.num_bare_clbits
        # prefix states
        self.states = [zero_state(n)]
        for (idx, name, qs, params), mat in zip(self.ops, self.mats):
            self.states.append(apply(self.states[-1], mat, qs, n))
        self.ideal_q = qubit_probs(self.states[-1])
        self.ideal, kept = self.decode(self.ideal_q)
        assert abs(kept - 1) < 1e-9, f"ideal circuit keeps {kept}"
        self.ideal = self.ideal / kept
        self._build_logical_model()

    # -------------------------------------------------------------- decoding
    def decode(self, qprobs):
        kept = qprobs[self.keep]
        return np.bincount(self.logical_index[self.keep], weights=kept, minlength=1 << self.nb), float(kept.sum())

    def flips_kept(self, fx: int) -> bool:
        if fx & self.check_mask:
            return False
        if self.use_parity and popc(fx & self.code_mask) & 1:
            return False
        return True

    def flips_to_shift(self, fx: int) -> int:
        lay = self.lay
        b = (fx >> lay.bottom) & 1
        s = 0
        for i, cl in enumerate(lay.logical_clbits):
            if cl >= 0 and (((fx >> i) & 1) ^ b):
                s |= 1 << cl
        return s

    def logical_decode(self, x: int, z: int):
        """Logical X / Z patterns (bit i = logical qubit i, code numbering) of a Pauli on the
        code qubits that commutes with both stabilizers; None if it does not commute."""
        lay = self.lay
        xc, zc = x & self.code_mask, z & self.code_mask
        if popc(xc) & 1 or popc(zc) & 1:
            return None
        xb = (xc >> lay.bottom) & 1
        zt = (zc >> lay.top) & 1
        lx = lz = 0
        for i in range(lay.k_code):
            if ((xc >> i) & 1) ^ xb:
                lx |= 1 << i
            if ((zc >> i) & 1) ^ zt:
                lz |= 1 << i
        return lx, lz

    # -------------------------------------------------------------- tracker
    def propagate(self, x: int, z: int, start: int):
        """Push a Pauli sitting after op start-1 to the end.  Returns
        (x_pre, z_pre, x_end, z_end, flipped) with *_pre taken just before the syndrome block."""
        flipped = []
        x_pre = z_pre = None
        for j in range(start, len(self.ops)):
            if j == self.syn_start:
                x_pre, z_pre = x, z
            _, name, qs, _ = self.ops[j]
            if name in ROT:
                gx, gz = generator(name, qs)
                if anticommute(x, z, gx, gz):
                    flipped.append(j)
            else:
                x, z = push(x, z, name, qs)
        if x_pre is None:
            x_pre, z_pre = (x, z) if start <= self.syn_start else (None, None)
        return x_pre, z_pre, x, z, tuple(flipped)

    def _build_logical_model(self):
        """Logical generator (with sign, via stim) of every rotation, pushed to the end of the
        logical part through the Clifford skeleton."""
        n, lay = self.n, self.lay
        self.rot = []          # (op index, theta, sign, lx, lz)
        self.model_ok = True
        self.model_notes = []
        for j, (idx, name, qs, params) in enumerate(self.ops):
            if name not in ROT:
                continue
            assert j < self.syn_start, "rotation inside the syndrome block"
            ps = stim.PauliString(n)
            for q in qs:
                ps[q] = ROT[name][0]
            circ = stim.Circuit()
            for _, nm, q2, _ in self.ops[j + 1:self.syn_start]:
                if nm in ROT:
                    continue
                circ.append({"h": "H", "cx": "CX", "x": "X", "y": "Y", "z": "Z"}[nm], q2)
            if len(circ):
                ps = ps.after(circ)
            xs, zs = ps.to_numpy()
            x = sum(1 << q for q in range(n) if xs[q])
            z = sum(1 << q for q in range(n) if zs[q])
            sign = ps.sign
            assert sign in (1, -1), f"imaginary sign {sign}"
            anc = ((1 << n) - 1) ^ self.code_mask
            if x & anc:
                self.model_ok = False
                self.model_notes.append(f"rotation op {j} leaves X on ancilla")
            dec = self.logical_decode(x, z)
            if dec is None:
                self.model_ok = False
                self.model_notes.append(f"rotation op {j} generator does not commute with stabilizers")
                dec = (0, 0)
            # a pure X^(K+2) or Z^(K+2) factor is a stabilizer; logical_decode quotients it.
            # Y-type overlaps would need a sign fix; generators here are pure X or pure Z type.
            if x & self.code_mask and z & self.code_mask:
                self.model_notes.append(f"rotation op {j} generator is mixed X/Z")
            self.rot.append((j, float(params[0]), int(sign.real), dec[0], dec[1]))
        # tracker cross-check against stim (phases ignored) for the same generators
        for j, theta, sign, lx, lz in self.rot:
            _, name, qs, _ = self.ops[j]
            gx, gz = generator(name, qs)
            xp, zp, *_ = self.propagate(gx, gz, j + 1)
            assert self.logical_decode(xp, zp) == (lx, lz), "own tracker disagrees with stim"

    def logical_dist(self, flipped=(), shift=0) -> np.ndarray:
        """Distribution of the logical circuit with given sign-flipped rotations, then XOR shift.
        Logical start state: |+>^K (init plus) or |0>^K."""
        K = self.lay.k_code
        psi = zero_state(K)
        if self.lay.init == "plus":
            for i in range(K):
                psi = apply(psi, gate_matrix("h", []), [i], K)
        flipped = set(flipped)
        for j, theta, sign, lx, lz in self.rot:
            th = theta * sign * (-1 if j in flipped else 1)
            # exp(-i th/2 P) with P = prod X^lx Z^lz (lx&lz == 0 here)
            assert lx & lz == 0
            qs = [i for i in range(K) if (lx | lz) >> i & 1]
            if not qs:
                continue        # generator is a stabilizer: global phase
            mats = [PAULI["X"] if (lx >> i) & 1 else PAULI["Z"] for i in qs]
            c, s = math.cos(th / 2), math.sin(th / 2)
            out = psi * c
            tmp = psi
            for i, m in zip(qs, mats):
                tmp = apply(tmp, m, [i], K)
            psi = out - 1j * s * tmp
        p = qubit_probs(psi)
        # logical qubit i -> bare clbit logical_clbits[i]
        idx = np.arange(1 << K, dtype=np.int64)
        cl = np.zeros_like(idx)
        for i, c in enumerate(self.lay.logical_clbits):
            if c >= 0:
                cl |= ((idx >> i) & 1) << c
        d = np.bincount(cl, weights=p, minlength=1 << self.nb)
        if shift:
            d = d[np.arange(len(d)) ^ shift]
        return d

    # -------------------------------------------------------------- enumeration
    def _finish(self, f: Fault, qprobs, x, z, start):
        logical, kept = self.decode(qprobs)
        f.kept = kept
        if kept < self.tol:
            f.status = "discarded"
        else:
            norm = logical / kept
            f.tv = tv(norm, self.ideal)
            if abs(kept - 1) > 1e-7:
                f.status = "partial-harmful" if f.tv > 1e-7 else "partial-harmless"
            else:
                f.status = "harmful" if f.tv > 1e-7 else "harmless"
            # pure XOR-shift test on the exact distribution
            idx = np.arange(len(norm))
            for s in range(len(norm)):
                if np.abs(norm[idx ^ s] - self.ideal).max() < 1e-9:
                    f.pure_shift = s
                    break
        # tracker prediction
        if x is not None:
            x_pre, z_pre, x_end, z_end, flipped = self.propagate(x, z, start)
            f.flipped = flipped
            f.pred_kept = int(self.flips_kept(x_end))
            f.shift = self.flips_to_shift(x_end)
            ok = (f.pred_kept == 1 and abs(kept - 1) < 1e-7) or (f.pred_kept == 0 and kept < self.tol)
            if ok and f.pred_kept:
                pred = self.logical_dist(flipped, f.shift)
                ok = np.abs(pred - logical / kept).max() < 1e-8
            f.pred_ok = bool(ok)
            if x_pre is not None:
                anc = ((1 << self.n) - 1) ^ self.code_mask
                f.anc_x = x_pre & anc
                dec = self.logical_decode(x_pre, z_pre)
                if dec is not None:
                    f.lx, f.lz = dec
                else:
                    f.lx = f.lz = -1
        return f

    def enumerate(self, prep=True, meas=True):
        n = self.n
        out = []
        if prep:      # X on a qubit before the first gate (bad reset)
            for q in range(n):
                psi = apply(self.states[0], PAULI["X"], [q], n)
                for (_, name, qs, _), mat in zip(self.ops, self.mats):
                    psi = apply(psi, mat, qs, n)
                f = Fault("prep", q, -1, "reset", (q,), "X", "init")
                out.append(self._finish(f, qubit_probs(psi), 1 << q, 0, 0))
        for li, (idx, name, qs, params) in enumerate(self.ops):
            plist = ["X", "Y", "Z"] if len(qs) == 1 else TWO_PAULIS
            for p in plist:
                psi = self.states[li + 1]
                for ch, q in zip(p, qs):
                    if ch != "I":
                        psi = apply(psi, PAULI[ch], [q], n)
                for (_, nm, q2, _), mat in zip(self.ops[li + 1:], self.mats[li + 1:]):
                    psi = apply(psi, mat, q2, n)
                x, z = pauli_masks(p, qs)
                f = Fault("gate", li, idx, name, tuple(qs), p, self.op_block[li])
                out.append(self._finish(f, qubit_probs(psi), x, z, li + 1))
        if meas:
            m = np.arange(1 << n)
            for idx, q, c in self.meas:
                qp = self.ideal_q[m ^ (1 << c)]
                f = Fault("meas", q, idx, "measure", (q,), "flip", "measure")
                self._finish(f, qp, None, None, None)
                f.pred_kept = int(self.flips_kept(1 << c))
                f.pred_ok = (f.pred_kept == 1) == (f.kept > 0.5)
                out.append(f)
        return out


def qname(enc, q):
    lay = iceberg.layout_of(enc)
    K = lay.k_code
    if q < K:
        return f"d{q}"
    if q == lay.top:
        return "t"
    if q == lay.bottom:
        return "b"
    if lay.style == "bus":
        return f"c{q - K - 2}"
    if q == lay.sx_ancilla:
        return "a"
    return "f"


def describe(enc, f: Fault) -> str:
    qs = ",".join(qname(enc, q) for q in f.qubits)
    return (f"[{f.block}] data[{f.data_index}] {f.gate}({qs}) fault {f.pauli}: kept={f.kept:.3f} TV={f.tv:.4f} "
            f"status={f.status} L_x={f.lx:b} L_z={f.lz:b} w={f.lweight} shift={f.shift:b} pure_shift={f.pure_shift}")


def summarize(enc, faults, title):
    from collections import Counter
    print(f"== {title}: {len(faults)} faults ({sum(f.kind=='gate' for f in faults)} gate, "
          f"{sum(f.kind=='prep' for f in faults)} prep, {sum(f.kind=='meas' for f in faults)} meas); "
          f"{enc.num_qubits} qubits")
    c = Counter((f.block, f.status) for f in faults)
    blocks = []
    for f in faults:
        if f.block not in blocks:
            blocks.append(f.block)
    for b in blocks:
        row = {s: c[(b, s)] for s in ("discarded", "harmless", "harmful", "partial-harmful", "partial-harmless") if c[(b, s)]}
        print(f"   {b:9s} {row}")
    bad_pred = [f for f in faults if not f.pred_ok]
    print(f"   tracker prediction matches exact simulation for {len(faults) - len(bad_pred)}/{len(faults)} faults")
    return c
