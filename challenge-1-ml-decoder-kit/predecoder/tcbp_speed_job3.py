"""tcbp-speed job 3: its numbers on any machine, from the files it sent back (no decoder, no torch).

  python -m predecoder.tcbp_speed_job3 <folder of files/tcbp-speed/job3/> <the validation run's per_shot.npz (ctl/130)>

Writes <folder>/analysis.json (committed as logs/tcbp_speed_job3_analysis.json).
"""
import json
import sys
from pathlib import Path

import numpy as np

from predecoder.decode import paired, wilson

J = Path(sys.argv[1])
JOB3_HEAD = Path(sys.argv[2])                  # the head arm of the validation run (predecoder-tcbp job 3, ctl/130)
out = {}
with np.load(JOB3_HEAD) as file:
    ref = {key: file[key] for key in file.files}
for name in ("summary.json", "summary_compile.json"):
    if (J / name).exists():
        summary = json.loads((J / name).read_text())
        for arm, row in summary["arms"].items():
            print(f"network {arm:8s}", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.get("ms", {}).items()},
                  row.get("notes"), row.get("against_eager"), row.get("error", "")[:300])
        out[name] = summary["arms"]
if (J / "probe_rates.json").exists():
    probe = json.loads((J / "probe_rates.json").read_text())
    print("probe: decoder names", probe["Decoder"]["names"])
    print("probe: candidates", probe["decoder_candidates"], "plugin", probe["plugin"])
    for label in ("cudaq_qec", "nv_qldpc", "Decoder", "DecoderResult"):
        print(f"  {label} matching names:", list(probe[label]["docs"]))
    out["probe"] = {"candidates": probe["decoder_candidates"], "plugin": probe["plugin"]}
for name, label, shots in (("bp_only_per_shot.npz", "bp-only (tcbp preset)", 2000), ("decode_graph_per_shot.npz", "graph logits", 1000)):
    if not (J / name).exists():
        continue
    with np.load(J / name) as file:
        ours = {key: file[key] for key in file.files}
    n = len(ours["truth"])
    assert np.array_equal(ours["truth"], ref["truth"][:n])
    failed, theirs = ours["failed_head"] > 0, ref["failed_head"][:n] > 0
    both, only_ours, only_theirs, p = paired(failed, theirs)
    low, high = wilson(int(failed.sum()), n)
    bits = int((ours["predicted_code_head"] != ref["predicted_code_head"][:n]).sum())
    faults = int((ours["n_faults_head"] != ref["n_faults_head"][:n]).sum())
    row = {"shots": n, "failures": int(failed.sum()), "wilson95": [low, high], "job3_head_failures": int(theirs.sum()),
           "both": both, "only_this": only_ours, "only_job3_head": only_theirs, "p": p, "bits_differ": bits,
           "faults_differ": faults, "not_reproduced": int((ours["reproduced_head"] == 0).sum()),
           "build_ms": float(ours["seconds_build_head"].mean() * 1e3), "decode_ms": float(ours["seconds_decode_head"].mean() * 1e3),
           "p99_ms": float(np.percentile((ours["seconds_build_head"] + ours["seconds_decode_head"]) * 1e3, 99))}
    out[label] = row
    print(label, row)
(J / "analysis.json").write_text(json.dumps(out, indent=1, default=str))
