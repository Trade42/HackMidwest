"""Not declared: the wrong-p task against the pipeline's stored decodes of the same validation shots (server-logs
files/predecoder-tcbp/validation_X_p0.005_per_shot.npz: TCBP-JOBS with the kit's priors at 0.005 ('fresh') and the head told
0.005 ('head'), 5,000 shots). From challenge-1-ml-decoder-kit/: python -m neutral_atom_noise.tcbp_crosscheck_wrongp <wrong-p per_shot.npz> <stored per_shot.npz> <out.json>"""
import json
import sys

import numpy as np

from neutral_atom_noise.tcbp_check import sign_test, wilson

ours, stored, out = sys.argv[1:]
with np.load(ours) as f:
    a = {k: f[k] for k in f.files}
with np.load(stored) as f:
    b = {k: f[k] for k in f.files}
n = int(a["shots_scored"])
assert np.array_equal(a["truth"], b["truth"][:n]), "not the same shots"
res = {"shots": n, "what": "not declared: a check of the harness and the cost of the wrong p to the head"}
same = (a["predicted_m"] == b["predicted_fresh"][:n]).all(axis=1)
res["m_vs_stored_fresh"] = {"same_12_bits": int(same.sum()), "failures_m": int((a["failed_m"] > 0).sum()),
                            "failures_stored_fresh": int(b["failed_fresh"][:n].sum())}
for arm, stored_arm in (("h", "head"), ("g", "fresh"), ("m", "head")):
    x, y = a[f"failed_{arm}"] > 0, b[f"failed_{stored_arm}"][:n] > 0
    res[f"{arm}_vs_stored_{stored_arm}"] = {"failures": [int(x.sum()), int(y.sum())], "only_ours": int((x & ~y).sum()),
                                            "only_stored": int((y & ~x).sum()), "sign_test_p": sign_test(int((x & ~y).sum()), int((y & ~x).sum())),
                                            "wilson_ours": wilson(int(x.sum()), n), "wilson_stored": wilson(int(y.sum()), n)}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
