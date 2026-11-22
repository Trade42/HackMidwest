"""Extra: bare circuit WITHOUT the leading H layer (logical |0..0> start) - does (a) still hold?"""
import numpy as np
from qiskit import QuantumCircuit
import iceberg
from vlib import *
from faultlib import *
from test_claim3 import run
rng = np.random.default_rng(4)
qc = QuantumCircuit(4, 4)
for i in range(4):
    qc.rx(rng.uniform(0.4, 1.2), i)
for i, j in [(0, 1), (1, 2), (2, 3), (0, 2)]:
    qc.rzz(rng.uniform(0.4, 1.2), i, j)
for i in range(4):
    qc.rz(rng.uniform(0.4, 1.2), i)
for i in range(4):
    qc.rx(rng.uniform(0.4, 1.2), i)
qc.measure(range(4), range(4))
for title, enc in (("direct, no leading H", iceberg.encode_direct(qc)), ("bus, no leading H", iceberg.encode_bus(qc)),
                   ("direct checks=x (default QAOA-style H start)", None), ("direct checks=z", None)):
    if enc is None:
        import qsel
        from qsel import Angles
        q = qsel.random_qubo(4, edge_fraction=0.5, seed=1)
        bare = qsel.qaoa_circuit(q, Angles((0.6,), (0.5,)))
        enc = iceberg.encode_direct(bare, checks="x" if "x" in title.split("=")[1][:1] else "z")
    try:
        run(title, enc, verbose_harmful=3)
    except AssertionError as e:
        print("== ", title, "-> assertion:", e)
