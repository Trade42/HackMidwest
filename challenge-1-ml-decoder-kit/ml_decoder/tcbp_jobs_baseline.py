"""Row 12: NVIDIA's TCBP-JOBS on the same shots as the kit's baseline.py.

  python tcbp_jobs_baseline.py --p 0.004 --shots 10000             serial decode(), timed per shot like the kit
  python tcbp_jobs_baseline.py --p 0.004 --shots 10000 --batch     decode_batch(): fast, for error rates only
  python tcbp_jobs_baseline.py --p 0.004 --shots 200 --decoder relay     no GPU: checks the plumbing with relay-BP
  python tcbp_jobs_baseline.py --p 0.004 --opt jobs_draws=32 --opt jobs_beam_width=64     tune JOBS
  python tcbp_jobs_baseline.py --p 0.004 --preset tcbp-jobs-sc     another nv-qldpc preset

It lives in challenge-1-ml-decoder-kit/ml_decoder/ and finds the kit one folder up (or pass --kit). It needs Linux, an NVIDIA GPU and `pip install cudaq-qec`,
version 0.9.0 or later. Circuits, seed and shots are the same as baseline.py, so its rows pair with the kit's rows.
Failures are listed as [X Z]. The first decode of each basis runs once untimed, to keep GPU start-up out of the latency.
"""
import argparse
import ast
import os
import sys
import time

import numpy as np


def default_kit():
    """The kit folder: this file's folder if circuit.py is in it, else the folder above."""
    here = os.path.dirname(os.path.abspath(__file__))
    return here if os.path.exists(os.path.join(here, "circuit.py")) else os.path.dirname(here)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kit", default=default_kit(), help="folder with baseline.py")
    parser.add_argument("--p", type=float, default=0.004)
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--decoder", choices=["tcbp-jobs", "relay"], default="tcbp-jobs")
    parser.add_argument("--preset", default="tcbp-jobs", help="nv-qldpc preset: tcbp-jobs, tcbp-jobs-sc, ...")
    parser.add_argument("--opt", action="append", default=[], metavar="NAME=VALUE", help="override one decoder option")
    parser.add_argument("--batch", action="store_true", help="use decode_batch(): error rates only, not latency")
    return parser.parse_args()


def tcbp_jobs(H, priors, preset, overrides):
    """(decoder, seconds to fit the TCBP model, seconds to build the decoder)."""
    import cudaq_qec as qec
    from cudaq_qec.plugins import nv_qldpc
    dense = np.ascontiguousarray(H.toarray(), dtype=np.uint8)        # row-major, as the plugin's docs ask
    start = time.perf_counter()
    model = nv_qldpc.init_tcbp_model(dense)                          # once per code family and basis
    fitted = time.perf_counter()
    decoder = qec.get_decoder("nv-qldpc-decoder", dense, error_rate_vec=np.asarray(priors, float),
                              **nv_qldpc.preset(preset, tcbp_model=model, **overrides))
    return decoder, fitted - start, time.perf_counter() - fitted


def hard(result):
    """Error vector(s) over the columns of H from a decoder result."""
    return (np.asarray(result) > 0.5).astype(np.uint8)


def batch_score(decoder, L, det, obs, chunk=4096):
    """(failed, seconds) with decode_batch. seconds is the average over the run, so it is not a latency."""
    failed, start = np.zeros(len(det), bool), time.perf_counter()
    for a in range(0, len(det), chunk):
        errors = hard(decoder.decode_batch(np.ascontiguousarray(det[a:a + chunk], dtype=np.uint8)).result)
        failed[a:a + chunk] = (((L @ errors.T) % 2).T != obs[a:a + chunk]).any(axis=1)
    return failed, np.full(len(det), (time.perf_counter() - start) / len(det))


def main():
    args = parse_args()
    sys.path.insert(0, args.kit)
    from baseline import memory_experiments
    from circuit import GROSS_CIRCUITS
    from decoders import print_score, relay_decoder, score
    overrides = {name: ast.literal_eval(value) for name, value in (item.split("=", 1) for item in args.opt)}
    batched = args.batch and args.decoder != "relay"
    failed, seconds = [], []
    experiments = memory_experiments([GROSS_CIRCUITS["X"], GROSS_CIRCUITS["Z"]], args.p, args.shots, args.seed)
    for basis, ex in zip("XZ", experiments):
        if args.decoder == "relay":
            decoder = relay_decoder(ex.H, ex.priors)
            decode = decoder.decode
        else:
            decoder, fit, build = tcbp_jobs(ex.H, ex.priors, args.preset, overrides)
            print(f"{basis} memory: H is {ex.H.shape[0]} x {ex.H.shape[1]}, TCBP model fitted in {fit:.1f} s, "
                  f"decoder built in {build * 1e3:.0f} ms", flush=True)
            decode = lambda syndrome, decoder=decoder: hard(decoder.decode(syndrome).result)
        decode(ex.det[0])                                             # warm-up, not timed
        f, s = batch_score(decoder, ex.L, ex.det, ex.obs) if batched else score(decode, ex.L, ex.det, ex.obs)
        failed.append(f)
        seconds.append(s)
    mode = "decode_batch, the time is an average" if batched else "serial decode"
    print(f"{args.shots} shots at p={args.p}, {args.cycles} cycles, failures listed as [X Z], {mode}")
    name = args.preset if args.decoder == "tcbp-jobs" else "relay-bp"
    print_score(name + ("*" if overrides else ""), np.column_stack(failed), np.column_stack(seconds), args.cycles)


if __name__ == "__main__":
    main()
