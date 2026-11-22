"""L4: what per-shot priors cost. Run alone, one process on one physical core:

    flock /home/sal/hackMidwest/dq/run/sim-d.lock nice -n 15 taskset -c 0 \\
        python -m loss.timing --loss-data DATA_s1e-3.npz --clean-data DATA_s0.npz --out loss/results/L4_timing

Holding lane D keeps CPU 1, the sibling thread of CPU 0, idle. All times are wall-clock perf_counter.

1. Relay-BP construction, in ms (relay_bp.RelayDecoderF64 at the kit's settings), with the lossless priors and
   with per-shot flag-aware priors (the starter's map), and the time to build those priors.
2. Relay-BP has no method that changes its priors (its methods: decode, decode_batch, decode_detailed,
   decode_detailed_batch), so per-shot priors need a new decoder per shot.
3. ldpc's BpOsdDecoder: construction, and update_channel_probs, which changes the priors in place. Checked: a
   decoder built with the lossless priors and then updated to a shot's priors decodes 20 loss shots exactly as a
   decoder built with that shot's priors.
4. Relay-BP decode time per shot with and without the rebuild, on 100 lossless shots (same priors, so only the
   rebuild differs) and on 30 loss shots with flag-aware priors.
5. Clifft's sampling speed on the Gross X circuit at p = 0.004, single thread.
"""
import argparse
import json
import time

import numpy as np

from decoders import bposd_decoder, relay_decoder
from loss.lossmap import LossMap, PriorBuilder, parse_condition
from loss.lossmodel import clifft_text, kit_circuit, load_dataset, sample_chunk


def stats(ms):
    ms = np.asarray(ms)
    return {"n": int(len(ms)), "mean_ms": float(ms.mean()), "median_ms": float(np.median(ms)),
            "p95_ms": float(np.percentile(ms, 95)), "max_ms": float(ms.max())}


def timed(f, *args, **kw):
    t = time.perf_counter()
    r = f(*args, **kw)
    return r, 1e3 * (time.perf_counter() - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss-data", required=True)
    ap.add_argument("--clean-data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    loss, clean = load_dataset(a.loss_data), load_dataset(a.clean_data)
    lmap = LossMap("X", loss["meta"]["p"])
    builder = PriorBuilder(lmap, loss["meta"]["s"])
    spec = parse_condition("starter")
    H, pri = lmap.H, lmap.pri
    det_loss, det_clean = loss["detectors"][:, lmap.rows], clean["detectors"][:, lmap.rows]
    out = {"machine": "AMD Ryzen 7 7730U, one process on CPU 0 (sibling CPU 1 idle), nice 15, other lanes busy",
           "loss_data": loss["meta"], "clean_data": clean["meta"]}

    # 1. construction
    relay_decoder(H, pri, seed=1)          # warm-up
    out["relay_build_lossless"] = stats([timed(relay_decoder, H, pri, seed=1)[1] for _ in range(30)])
    priors, prior_ms = zip(*[timed(builder.priors, spec, loss["heralds"][k]) for k in range(200)])
    out["prior_build_starter"] = stats(prior_ms)
    out["relay_build_flag_aware"] = stats([timed(relay_decoder, H, priors[k], seed=1)[1] for k in range(30)])
    out["relay_methods"] = sorted(m for m in dir(relay_decoder(H, pri, seed=1)) if not m.startswith("_"))

    # 3. BP-OSD
    out["bposd_build"] = stats([timed(bposd_decoder, H, pri)[1] for _ in range(5)])
    upd = bposd_decoder(H, pri)
    out["bposd_update_channel_probs"] = stats([timed(upd.update_channel_probs, priors[k])[1] for k in range(30)])
    same, upd_ms, fresh_ms = 0, [], []
    for k in range(20):
        fresh = bposd_decoder(H, priors[k])
        e1, t1 = timed(fresh.decode, det_loss[k])
        upd.update_channel_probs(priors[k])
        e2, t2 = timed(upd.decode, det_loss[k])
        same += bool(np.array_equal(e1, e2))
        fresh_ms.append(t1); upd_ms.append(t2)
    out["bposd_update_equals_rebuild"] = {"shots": 20, "identical_outputs": same}
    out["bposd_decode_after_update"] = stats(upd_ms)
    out["bposd_decode_fresh"] = stats(fresh_ms)

    # 4. relay-BP decode with and without the rebuild
    static = relay_decoder(H, pri, seed=1)
    out["relay_decode_static_lossless_shots"] = stats([timed(static.decode, det_clean[k])[1] for k in range(100)])
    rebuilt = []
    for k in range(100):
        d, tb = timed(relay_decoder, H, pri, seed=1)
        _, td = timed(d.decode, det_clean[k])
        rebuilt.append(tb + td)
    out["relay_build_plus_decode_lossless_shots"] = stats(rebuilt)
    aware = []
    for k in range(30):
        p_k, tp = timed(builder.priors, spec, loss["heralds"][k])
        d, tb = timed(relay_decoder, H, p_k, seed=1)
        _, td = timed(d.decode, det_loss[k])
        aware.append((tp, tb, td))
    aware = np.array(aware)
    out["relay_flag_aware_loss_shots"] = {"priors": stats(aware[:, 0]), "build": stats(aware[:, 1]),
                                          "decode": stats(aware[:, 2]), "total": stats(aware.sum(1))}

    # 5. Clifft
    circuit = kit_circuit("X", 0.004)
    out["clifft"] = {}
    for s, probes in ((0.0, False), (1e-3, False), (1e-3, True)):
        text, _, _ = clifft_text(circuit, s, probes=probes)
        sample_chunk(text, 50, 1)
        _, ms = timed(sample_chunk, text, 1000, 7)
        out["clifft"][f"s={s:g}, probes={probes}"] = {"shots": 1000, "seconds": ms / 1e3, "shots_per_second": 1e6 / ms}
    with open(a.out + ".json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
