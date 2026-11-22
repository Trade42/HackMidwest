"""Tracker-only enumeration for k = 7..12 with the init-state refinement:
a kept fault in the init block whose residual right after init is a pure X-type logical
operator (times stabilizers, times Z on |0> ancillas) fixes the logical |+>^K state exactly -> harmless."""
import time, sys
from collections import Counter
import numpy as np
import iceberg, qsel
from qsel import Angles
from vlib import *
from faultlib import *
from tracker_large import TrackerOnly, fp

def refine(an, faults):
    assert an.lay.init == "plus"
    for f in faults:
        if f.status == "harmful" and f.block == "init" and f.kind != "meas" and f.anc_x == 0 and f.lx >= 0 and f.lz == 0:
            f.status = "harmless"
    return faults

if __name__ == "__main__":
    rng = np.random.default_rng(99)
    t0 = time.time(); total = 0
    for k, ef, seed, p in [(7, 0.4, 1, 2), (8, 0.3, 2, 1), (8, 0.4, 3, 2), (9, 0.3, 4, 2), (10, 0.2, 5, 1), (10, 0.3, 6, 2), (10, 1.0, 7, 2), (12, 0.25, 8, 2)]:
        q = qsel.random_qubo(k, edge_fraction=ef, seed=seed)
        bare = qsel.qaoa_circuit(q, Angles(tuple(rng.uniform(0.3, 1, p)), tuple(rng.uniform(0.3, 1, p))))
        print(f"--- k={k} p={p} E={q.num_edges}")
        for name, enc in (("direct", iceberg.encode_direct(bare)), ("bus[direct]", iceberg.encode_bus(bare)), ("bus[bus]", iceberg.encode_bus(bare, edge_mode="bus"))):
            an = TrackerOnly(enc)
            faults = refine(an, an.enumerate())
            total += len(faults)
            harmful = [f for f in faults if f.status == "harmful"]
            part = [f for f in harmful if f.block in ("init", "syndrome", "measure")]
            clean, dirty = Counter(), Counter()
            big = []
            for f in harmful:
                w, c = fp(an, f)
                (clean if c else dirty)[w] += 1
                if an.lay.style == "bus" and w > 2:
                    big.append((w, c, f))
            print(f"   {name:12s} n={enc.num_qubits:2d} faults={len(faults):5d} potentially-harmful undetected={len(harmful):4d} in init/syndrome/measure={len(part)} "
                  f"clean weights={dict(sorted(clean.items()))} bus-residual footprints={dict(sorted(dirty.items()))}")
            for f in part[:4]:
                print("       PART:", describe(enc, f))
            for w, c, f in big[:6]:
                print(f"       BUS >2 (footprint {w}, clean={c}):", describe(enc, f))
    print(f"total faults {total}, {time.time()-t0:.0f}s")
