"""Noise channels and prior statistics of the kit's circuit versus PriorTilt's, one basis and p.

  python -I compare_noise.py <kit folder> <PriorTiltDecoder folder> X 0.004
"""
import sys
from collections import Counter

import numpy as np
import stim

KIT, PT, BASIS, P = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
sys.path.insert(0, KIT)
sys.path.insert(1, PT)
from circuit import GROSS_CIRCUITS, dem_matrices, load_circuit  # noqa: E402
from priortilt.dem import dem_to_matrices  # noqa: E402


def channels(circuit):
    out = Counter()
    for inst in circuit.flattened():
        if stim.gate_data(inst.name).is_noisy_gate:
            out[(inst.name, tuple(round(a, 9) for a in inst.gate_args_copy()))] += len(inst.targets_copy())
    return out


def gates(circuit):
    return Counter(inst.name for inst in circuit.flattened() if not stim.gate_data(inst.name).is_noisy_gate)


kc = load_circuit(GROSS_CIRCUITS[BASIS], P)
pc = stim.Circuit.from_file(f"{PT}/StimCircuit/BB[[144,12,12]],memory_{BASIS},error_rate={P},syndrome_rounds=12.stim")
print("kit noise channels (name, args): targets", dict(channels(kc)))
print("PriorTilt noise channels (name, args): targets", dict(channels(pc)))
print("kit gates", dict(gates(kc)))
print("PriorTilt gates", dict(gates(pc)))
_, _, kp = dem_matrices(kc, BASIS)
m = dem_to_matrices(pc.detector_error_model())
for name, pri in (("kit", kp), ("PriorTilt", m.priors)):
    q = np.quantile(pri, [0, 0.1, 0.5, 0.9, 1])
    vals = Counter(np.round(pri / P, 4).tolist())
    print(f"{name}: sum of priors {pri.sum():.2f} (expected faults per shot), quantiles/p {np.round(q / P, 4).tolist()}, "
          f"{len(vals)} distinct values; most common prior/p: {vals.most_common(6)}")
