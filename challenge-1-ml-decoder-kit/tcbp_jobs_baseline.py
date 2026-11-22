"""Row 12: NVIDIA's TCBP-JOBS on the same shots as the kit's baseline.py.

  python tcbp_jobs_baseline.py --p 0.004 --shots 10000             serial decode(), timed per shot like the kit
  python tcbp_jobs_baseline.py --p 0.004 --shots 10000 --batch     decode_batch(): fast, for error rates only
  python tcbp_jobs_baseline.py --p 0.004 --shots 200 --decoder relay     no GPU: checks the plumbing with relay-BP
  python tcbp_jobs_baseline.py --p 0.004 --opt jobs_draws=32 --opt jobs_beam_width=64     tune JOBS
  python tcbp_jobs_baseline.py --p 0.004 --preset tcbp-jobs-sc     another nv-qldpc preset

Put this file in challenge-1-ml-decoder-kit/ (or pass --kit). It needs Linux, an NVIDIA GPU and `pip install cudaq-qec`,
version 0.9.0 or later. Circuits, seed and shots are the same as baseline.py, so its rows pair with the kit's rows.
Failures are listed as [X Z]. The first decode of each basis runs once untimed, to keep GPU start-up out of the latency.

Serial runs automatically save Python stdout, stderr, and exception tracebacks
to a unique .log file in results/ beside this script, while still displaying
output in the terminal. Use --log-dir to choose another directory. Batch runs
keep their original terminal-only behavior. Native-library output written
directly to operating-system file descriptors is not intercepted.
"""
import argparse
import ast
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import os
from pathlib import Path
import shlex
import socket
import sys
import time
import traceback

import numpy as np


class Tee:
    """Copy Python stream writes to the terminal and an open log file."""

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text):
        self.terminal.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kit", default=os.path.dirname(os.path.abspath(__file__)), help="folder with baseline.py")
    parser.add_argument("--p", type=float, default=0.004)
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--decoder", choices=["tcbp-jobs", "relay"], default="tcbp-jobs")
    parser.add_argument("--preset", default="tcbp-jobs", help="nv-qldpc preset: tcbp-jobs, tcbp-jobs-sc, ...")
    parser.add_argument("--opt", action="append", default=[], metavar="NAME=VALUE", help="override one decoder option")
    parser.add_argument("--batch", action="store_true", help="use decode_batch(): error rates only, not latency")
    parser.add_argument("--log-dir", default=str(Path(__file__).resolve().parent / "results"),
                        help="directory for automatic serial-run logs (default: results/ beside this script)")
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


def run(args):
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


def main():
    args = parse_args()
    if args.batch and args.decoder != "relay":
        return run(args)

    log_dir = Path(args.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f%z")
    log_path = log_dir / (
        f"{args.decoder}_p{args.p:g}_shots{args.shots}_seed{args.seed}_{timestamp}.log"
    )

    # Set up logging outside the decoder and its per-shot timers. Line buffering
    # writes completed log lines promptly, including during a long-running job.
    with log_path.open("x", encoding="utf-8", buffering=1) as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)), redirect_stderr(Tee(sys.stderr, log_file)):
            print(f"Log file: {log_path}", flush=True)
            print(f"Started: {datetime.now().astimezone().isoformat()}", flush=True)
            print(f"Host: {socket.gethostname()}", flush=True)
            print(f"Command: {shlex.join([sys.executable, *sys.argv])}", flush=True)
            started = time.perf_counter()
            try:
                run(args)
            except BaseException:
                # Python prints the terminal traceback after streams are
                # restored. Write a separate copy to the log before closing it.
                traceback.print_exc(file=log_file)
                log_file.flush()
                raise
            else:
                print("Run completed successfully.", flush=True)
            finally:
                print(f"Total wall time (includes setup and fitting): {time.perf_counter() - started:.3f} s",
                      flush=True)


if __name__ == "__main__":
    main()
