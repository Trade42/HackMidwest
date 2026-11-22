"""Does a Stim seed give the same shots? The two checks behind a rule in ml_decoder/README.md.

  python ml_decoder/stim_seed_check.py data/p0.004/test_X.npz

1. Sample the saved file's circuit again, with its own basis, p, seed and shot count, as make_dataset.py does, and
   compare the detector bits with the saved ones. A file saved on another machine can differ, because Stim's sampler
   depends on its build. For scale, two independent samples with a share q of ones differ in about 2 q (1 - q) of bits.
2. Is sample(circuit, 400, seed) the first 400 shots of a larger sample with the same seed? (X memory, p = 0.004)
"""
import os
import sys

import numpy as np
import stim

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(here))
from circuit import GROSS_CIRCUITS, load_circuit, sample  # noqa: E402

saved = np.load(sys.argv[1])
basis, p, seed = str(saved["basis"]), float(saved["p"]), int(saved["seed"])
old = saved["detectors"]
new, new_obs = sample(load_circuit(GROSS_CIRCUITS[basis], p), len(old), seed)
q = old.mean()
print(f"stim {stim.__version__}. {sys.argv[1]}: {basis} memory, p={p}, seed {seed}, {len(old)} shots")
print(f"  sampled again here: {(new != old).mean():.1%} of detector bits differ, "
      f"{(new_obs != saved['observables']).any(axis=1).mean():.1%} of shots have a different logical label. "
      f"Share of ones: saved {q:.2%}, here {new.mean():.2%}; independent samples would differ in {2 * q * (1 - q):.1%}")
circuit = load_circuit(GROSS_CIRCUITS["X"], 0.004)
small, _ = sample(circuit, 400, 1)
for n in (2000, 5000, 10000):
    large, _ = sample(circuit, n, 1)
    print(f"  sample(circuit, 400, 1) equals the first 400 shots of sample(circuit, {n}, 1): {np.array_equal(small, large[:400])}")
