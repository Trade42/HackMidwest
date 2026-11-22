"""Tracker-only enumeration (no statevector) for larger codes, k = 8, 10, 12.
The tracker was validated against exact simulation on every fault of every k<=6 run.
'potentially harmful' = kept and (readout shift != 0 or some rotation sign flipped): a superset of harmful."""
import time
from collections import Counter
import numpy as np
import iceberg, qsel
from qsel import Angles
from vlib import *
from faultlib import *

class TrackerOnly(Analysis):
    def __init__(self, enc):
        self.enc = enc
        self.lay = lay = iceberg.layout_of(enc)
        self.n = n = enc.num_qubits
        self.ops, self.meas = parse(enc)
        assert [(q, c) for _, q, c in self.meas] == [(j, j) for j in range(n)]
        self.labels = block_labels(enc)
        self.op_block = [self.labels[idx] for idx, *_ in self.ops]
        syn = [i for i, b in enumerate(self.op_block) if b == "syndrome"]
        self.syn_start = syn[0] if syn else len(self.ops)
        self.code_mask = sum(1 << j for j in lay.code_qubits)
        self.check_mask = (0 if lay.sx_ancilla is None else 1 << lay.sx_ancilla) | sum(1 << f for f in lay.flags)
        self.use_parity = lay.checks in ("both", "z")
        self.nb = lay.num_bare_clbits
        self._build_logical_model()
        assert self.model_ok, self.model_notes

    def enumerate(self):
        out = []
        rot = {j: (lx, lz) for j, th, s, lx, lz in self.rot}
        def finish(f, x, z, start):
            x_pre, z_pre, x_end, z_end, flipped = self.propagate(x, z, start)
            f.pred_kept = int(self.flips_kept(x_end))
            f.shift = self.flips_to_shift(x_end)
            f.flipped = flipped
            if x_pre is not None:
                f.anc_x = x_pre & (((1 << self.n) - 1) ^ self.code_mask)
                dec = self.logical_decode(x_pre, z_pre)
                f.lx, f.lz = dec if dec is not None else (-1, -1)
            if not f.pred_kept:
                f.status = "discarded"
            elif f.shift or flipped:
                f.status = "harmful"      # potentially
            else:
                f.status = "harmless"
            out.append(f)
        for q in range(self.n):
            finish(Fault("prep", q, -1, "reset", (q,), "X", "init"), 1 << q, 0, 0)
        for li, (idx, name, qs, params) in enumerate(self.ops):
            for p in (["X", "Y", "Z"] if len(qs) == 1 else TWO_PAULIS):
                x, z = pauli_masks(p, qs)
                finish(Fault("gate", li, idx, name, tuple(qs), p, self.op_block[li]), x, z, li + 1)
        for idx, q, c in self.meas:
            f = Fault("meas", q, idx, "measure", (q,), "flip", "measure")
            f.pred_kept = int(self.flips_kept(1 << c))
            f.status = "discarded" if not f.pred_kept else "harmful"
            out.append(f)
        return out

def fp(an, f):
    rot = {j: (lx, lz) for j, th, s, lx, lz in an.rot}
    if f.anc_x == 0 and f.lx >= 0:
        return popc(f.lx | f.lz), True
    sup = 0
    for j in f.flipped:
        sup |= rot[j][0] | rot[j][1]
    for i, cl in enumerate(an.lay.logical_clbits):
        if cl >= 0 and (f.shift >> cl) & 1:
            sup |= 1 << i
    return popc(sup), False

if __name__ == "__main__":
    rng = np.random.default_rng(99)
    t0 = time.time(); total = 0
    for k, ef, seed, p in [(7, 0.4, 1, 2), (8, 0.3, 2, 1), (8, 0.4, 3, 2), (9, 0.3, 4, 2), (10, 0.2, 5, 1), (10, 0.3, 6, 2), (10, 1.0, 7, 2), (12, 0.25, 8, 2)]:
        q = qsel.random_qubo(k, edge_fraction=ef, seed=seed)
        bare = qsel.qaoa_circuit(q, Angles(tuple(rng.uniform(0.3, 1, p)), tuple(rng.uniform(0.3, 1, p))))
        print(f"--- k={k} p={p} E={q.num_edges}")
        for name, enc in (("direct", iceberg.encode_direct(bare)), ("bus[direct]", iceberg.encode_bus(bare)), ("bus[bus]", iceberg.encode_bus(bare, edge_mode="bus"))):
            an = TrackerOnly(enc)
            faults = an.enumerate()
            total += len(faults)
            harmful = [f for f in faults if f.status == "harmful"]
            part = [f for f in harmful if f.block in ("init", "syndrome", "measure")]
            clean, dirty = Counter(), Counter()
            for f in harmful:
                w, c = fp(an, f)
                (clean if c else dirty)[w] += 1
            print(f"   {name:12s} n={enc.num_qubits:2d} faults={len(faults):5d} potentially-harmful undetected={len(harmful):4d} in init/syndrome/measure={len(part)} "
                  f"clean weights={dict(sorted(clean.items()))} bus-residual footprints={dict(sorted(dirty.items()))}")
            for f in part[:3]:
                print("       PART:", describe(enc, f))
    print(f"total faults {total}, {time.time()-t0:.0f}s")
