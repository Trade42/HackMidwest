"""tcbp-speed job 3 (a): the network and head on the GPU at batch one, faster, with the same weights and inputs.

Every arm computes the logits of the first --limit shots ONE SHOT AT A TIME and is timed doing it, the way
predecoder.evaluate times a shot: the shot's bits copied to the device, the network and head, the logits copied back,
torch.cuda.synchronize. Float32, no autocast, no TF32 (evaluate.py --precision fp32), 1 CPU thread.

  eager    predecoder.evaluate.Predecoder: job 4's path, the reference
  cached   the same, with the head's sparse incidence matrix (H transposed, 30,672 ones) built once instead of on every
           call (PriorHead._incidence builds and coalesces it per call)
  dense    the same, with the head's sum over each column's rows done as a dense float32 matmul (no TF32)
  graph    the batch-one forward captured once as a CUDA graph and replayed per shot; the sparse product if it
           captures, else the dense one (the summary says which)
  compile  torch.compile(mode="reduce-overhead") of the dense forward

Each arm's logits are saved in evaluate.py's format (logits.npy float16, meta.json), so tcbp.py decode reads them.
The decoders read the float16 logits: an arm whose float16 rows equal eager's gives TCBP-JOBS the same priors, so the
same answers (the decoder is repeatable: job 1). The summary counts the rows that differ, and the largest gap.

  python -m predecoder.fast_network --head <head.pt> --trunk <trunk.pt> --data data/p0.005/validation_X.npz \\
      --limit 1000 --arms eager cached dense graph --out <folder>
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from evaluation.data import load_dataset
from predecoder.decode import kit_problem
from predecoder.evaluate import Predecoder

ARMS = ("eager", "cached", "dense", "graph", "compile")


def use_cached(model):
    """The head's incidence matrix built once (the same coalesced tensor every call)."""
    incidence = model.head._incidence(model.device, torch.float32)
    model.head._incidence = lambda device, dtype: incidence
    return incidence


def use_dense(model):
    """The head's per-column sums as one dense matmul with H transposed, [columns, rows], in float32."""
    dense = use_cached(model).to_dense()
    columns = model.head.num_columns

    def per_column(incidence, per_row):
        shots, rows, width = per_row.shape
        flat = per_row.permute(1, 0, 2).reshape(rows, shots * width)
        return (dense @ flat).reshape(columns, shots, width).permute(1, 0, 2)

    model.head._per_column = per_column


def forward_of(model, p, priors):
    prior = model.prior_logit(p, priors)

    def forward(full, basis):
        trunk_input = full if model.trunk.num_inputs == full.shape[1] else basis
        return model.head(model.trunk.features(trunk_input), basis, prior) - model.shift

    return forward


def run_arm(arm, args, data, problem, shots):
    """(logits float32 [shots, columns], seconds per shot, notes) of one arm, one shot at a time."""
    model = Predecoder(args.head, args.trunk, args.device, 1, "fp32")
    notes = {"product": "sparse"}
    if arm == "cached":
        use_cached(model)
    elif arm in ("dense", "compile"):
        use_dense(model)
        notes["product"] = "dense"
    detectors, basis = data.detectors[:shots], data.basis_input[:shots]
    out = np.zeros((shots, problem.H.shape[1]), np.float32)
    seconds = []
    cuda = model.device.type == "cuda"
    if arm in ("eager", "cached", "dense"):
        def one(k):
            return model(detectors[k:k + 1], basis[k:k + 1], data.p, problem.priors)
    else:
        full_in = torch.zeros((1, detectors.shape[1]), dtype=torch.uint8, device=model.device)
        basis_in = torch.zeros((1, basis.shape[1]), dtype=torch.uint8, device=model.device)
        full_in.copy_(torch.from_numpy(np.ascontiguousarray(detectors[:1], np.uint8)))
        basis_in.copy_(torch.from_numpy(np.ascontiguousarray(basis[:1], np.uint8)))
        if arm == "graph":
            if not cuda:
                raise SystemExit("the graph arm needs a GPU")
            use_cached(model)
            graph = None
            for product in ("sparse", "dense"):
                if product == "dense":
                    use_dense(model)
                forward = forward_of(model, data.p, problem.priors)
                try:
                    with torch.no_grad():
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for _ in range(3):
                                forward(full_in, basis_in)
                        torch.cuda.current_stream().wait_stream(stream)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            static_out = forward(full_in, basis_in)
                    notes["product"] = product
                    break
                except Exception as error:      # noqa: BLE001  try the dense product
                    notes[f"capture_{product}"] = repr(error)[:500]
                    graph = None
                    torch.cuda.synchronize()
            if graph is None:
                raise SystemExit(f"no CUDA graph captured: {notes}")

            def run():
                graph.replay()
                return static_out
        else:                                   # compile
            compiled = torch.compile(forward_of(model, data.p, problem.priors), mode="reduce-overhead", dynamic=False)
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(3):
                    compiled(full_in, basis_in)
            notes["compile_and_warm_seconds"] = time.perf_counter() - start

            def run():
                return compiled(full_in, basis_in)

        def one(k):
            full_in.copy_(torch.from_numpy(np.ascontiguousarray(detectors[k:k + 1], np.uint8)))
            basis_in.copy_(torch.from_numpy(np.ascontiguousarray(basis[k:k + 1], np.uint8)))
            with torch.no_grad():
                return run().float().cpu().numpy()
    one(0)                                                              # warm-up, untimed
    for k in range(shots):
        tick = time.perf_counter()
        logits = one(k)
        if cuda:
            torch.cuda.synchronize()
        seconds.append(time.perf_counter() - tick)
        out[k] = logits[0]
    return out, np.asarray(seconds), notes, model.record


def save(folder, logits, data, problem, record, seconds, arm, notes):
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "logits.npy", logits.astype(np.float16))
    meta = {"data": data.identity(), "shots": len(logits), "columns": int(problem.H.shape[1]),
            "model": record | {"arm": arm, "precision": "float32"}, "notes": notes,
            "per_shot_seconds": {"what": "network + head, one shot at a time (inputs to the device, forward, logits back, "
                                         "synchronize)", "shots": len(seconds), "mean": float(seconds.mean()),
                                 "median": float(np.median(seconds)), "p99": float(np.percentile(seconds, 99))}}
    (folder / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", required=True)
    parser.add_argument("--trunk", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=["eager", "cached", "dense", "graph"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reference", help="an eager arm's folder of an earlier run (default: this run's eager arm)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if Path(args.data).name.startswith(("test_", "go_")):
        raise SystemExit("validation or train files only")
    torch.set_num_threads(1)
    data = load_dataset(args.data)
    problem = kit_problem(data.basis, data.p)
    shots = min(args.limit, len(data.detectors))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"data": data.identity(), "shots": shots, "torch": torch.__version__, "device": args.device,
               "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None, "arms": {}}
    reference = None
    if args.reference:
        reference = np.load(Path(args.reference) / "logits.npy").astype(np.float32)[:shots]
    for arm in args.arms:
        try:
            logits, seconds, notes, record = run_arm(arm, args, data, problem, shots)
            meta = save(out / arm, logits, data, problem, record, seconds, arm, notes)
            row = {"ms": {key: meta["per_shot_seconds"][key] * 1e3 for key in ("mean", "median", "p99")}, "notes": notes}
            if reference is None and arm == "eager":
                reference = logits
            if reference is not None:
                half, half_ref = logits.astype(np.float16), reference.astype(np.float16)
                row["against_eager"] = {"largest_gap_float32": float(np.abs(logits - reference).max()),
                                        "rows_differ_float32": int((logits != reference).any(axis=1).sum()),
                                        "rows_differ_float16": int((half != half_ref).any(axis=1).sum()),
                                        "entries_differ_float16": int((half != half_ref).sum())}
            summary["arms"][arm] = row
            print(f"{arm:8s} {row['ms']['mean']:6.2f} ms mean (median {row['ms']['median']:.2f}, p99 {row['ms']['p99']:.2f}) "
                  f"per shot; {notes}; against eager {row.get('against_eager')}", flush=True)
        except BaseException as error:          # noqa: BLE001  one arm failing must not stop the others
            summary["arms"][arm] = {"error": repr(error)[:1000], "traceback": traceback.format_exc()[-3000:]}
            print(f"{arm:8s} FAILED: {error!r}", flush=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"saved {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
