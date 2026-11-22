"""Claim 1 at the sizes the package is meant for (single noiseless runs), plus postselect_probs cross-check."""
import time
import numpy as np
import iceberg, qsel
from qsel import Angles
from vlib import *
rng = np.random.default_rng(8)
for k, ef, seed, p in [(7, 0.4, 1, 2), (8, 0.4, 2, 2), (9, 0.3, 3, 1), (10, 0.3, 4, 1), (10, 0.5, 5, 2)]:
    q = qsel.random_qubo(k, edge_fraction=ef, seed=seed)
    bare = qsel.qaoa_circuit(q, Angles(tuple(rng.uniform(-1, 1, p)), tuple(rng.uniform(-1, 1, p))))
    pb = exact_dist(bare)
    for name, enc in (("direct", iceberg.encode_direct(bare)), ("bus[direct]", iceberg.encode_bus(bare)), ("bus[bus]", iceberg.encode_bus(bare, edge_mode="bus"))):
        t0 = time.time()
        pe = exact_dist(enc)
        lay = iceberg.layout_of(enc)
        lg, kept = my_postselect_dist(pe, lay)
        lg2, kept2 = iceberg.postselect_probs(pe, enc)
        print(f"k={k} p={p} E={q.num_edges} {name:12s} n={enc.num_qubits}: max|logical-bare|={np.abs(lg-pb).max():.1e} kept={kept:.12f} | postselect_probs agrees: {np.abs(lg2-lg).max():.1e}, {abs(kept2-kept):.1e} ({time.time()-t0:.1f}s)")
