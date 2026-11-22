"""Compile evaluation circuits with Superstaq's Sqale compiler and verify them locally.

Our Superstaq role may compile (``cq_compile``) but not run jobs. This script sends
the bare and Iceberg-encoded QAOA circuits of the first ``--n`` evaluation questions to
``cq_compile`` for ``cq_sqale_qpu`` in ONE call, then checks the returned circuits on
our exact local simulator:

* gate counts before and after the Sqale compiler (qubits, two-qubit gates, depth);
* bare: the compiled circuit's output distribution matches the uncompiled one;
* Iceberg: without noise the compiled circuit still keeps 100% of shots, and its
  decoded logical distribution matches the bare circuit. If the compiler had removed
  or reordered the check gates, this is where it would show.

The compiled circuits are saved as QPY, so the analysis can be redone with
``--from-cache`` without spending credits. The balance is logged before and after.

    python scripts/sqale_compile.py --n 1                  # 2 circuits: shows per-call vs per-circuit billing
    python scripts/sqale_compile.py --n 12 --styles bare,bus,direct
    python scripts/sqale_compile.py --n 12 --from-cache    # re-analyse, no network
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qiskit import QuantumCircuit, qpy  # noqa: E402

import iceberg  # noqa: E402
from qctx.config import load_config  # noqa: E402
from qctx.studies import load_instances  # noqa: E402
from qsel.backends import IdealBackend, circuit_stats  # noqa: E402

CIRCUIT_KEY = {"bare": "bare_all", "bus": "bus", "direct": "direct"}


def simulable(circuit: QuantumCircuit, max_rounds: int = 6) -> QuantumCircuit:
    """Decompose until no gate acts on more than two qubits, then drop spectator qubits.

    Sqale's compiler places circuits on all 24 device qubits and uses global rotations
    (GR) that touch every one of them. After decomposition, qubits that never take part
    in a two-qubit gate and are never measured stay in a product state and cannot change
    the measured distribution, so they are removed (2^24 -> 2^18 amplitudes for the bus
    circuit).
    """
    out = circuit
    for _ in range(max_rounds):
        wide = [i for i in out.data if i.operation.name not in ("barrier", "measure", "delay")
                and len(i.qubits) > 2]
        if not wide:
            break
        out = out.decompose(gates_to_decompose=sorted({i.operation.name for i in wide}))
    keep = sorted({out.find_bit(q).index for i in out.data
                   if i.operation.name == "measure" or (len(i.qubits) == 2 and i.operation.name != "barrier")
                   for q in i.qubits})
    if len(keep) == out.num_qubits:
        return out
    index = {old: new for new, old in enumerate(keep)}
    small = QuantumCircuit(len(keep), out.num_clbits, name=out.name)
    for inst in out.data:
        qs = [out.find_bit(q).index for q in inst.qubits]
        if inst.operation.name == "barrier" or any(q not in index for q in qs):
            continue
        small.append(inst.operation, [small.qubits[index[q]] for q in qs],
                     [small.clbits[out.find_bit(c).index] for c in inst.clbits])
    return small


def sendable(circuit: QuantumCircuit) -> QuantumCircuit:
    """A plain copy for the server: no metadata, no instruction labels, no barriers.

    Our first cq_compile of encoded circuits returned HTTP 500 while a plain Bell circuit
    compiled fine; the encoder's metadata and per-instruction labels are the difference.
    Gates, qubits and measurements are unchanged, so post-selection still uses the
    original circuit's layout.
    """
    out = QuantumCircuit(*circuit.qregs, *circuit.cregs, name=circuit.name)
    for inst in circuit.data:
        if inst.operation.name == "barrier":
            continue
        op = inst.operation.to_mutable() if hasattr(inst.operation, "to_mutable") else inst.operation.copy()
        try:
            op.label = None
        except Exception:
            pass
        out.append(op, inst.qubits, inst.clbits)
    return out


CZ_COST = {"cz": 1, "cx": 1, "cy": 1, "ecr": 1, "rzz": 2, "rxx": 2, "ryy": 2, "swap": 3}


def cz_equivalent(circuit: QuantumCircuit) -> int:
    """Two-qubit cost in CZ units, so it compares with Sqale's output (CZ is its native gate).

    ``circuit_stats``' two-qubit count treats an RZZ like a CX, but an RZZ needs two CZs;
    comparing raw gate counts before and after compilation overstates the compiler's
    blow-up (found in review: bare looked x3.5 instead of x1.8).
    """
    ops = circuit.count_ops()
    unknown = [k for k, i in ((k, None) for k in ops) if k not in CZ_COST and k not in
               ("h", "rz", "rx", "ry", "x", "y", "z", "s", "sdg", "sx", "t", "tdg", "u", "p", "measure", "barrier")]
    if unknown:
        raise ValueError(f"no CZ cost for gates {unknown}")
    return int(sum(n * CZ_COST[k] for k, n in ops.items() if k in CZ_COST))


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def balance(provider) -> float | None:
    try:
        return float(str(provider.get_balance()).replace("$", "").replace(",", "").split()[0])
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/offline.yaml",
                    help="whose QUBOs to compile (offline = the circuits behind the reported results)")
    ap.add_argument("--n", type=int, default=1, help="number of evaluation questions")
    ap.add_argument("--styles", default="bare,bus", help="comma list from bare,bus,direct")
    ap.add_argument("--target", default="cq_sqale_qpu")
    ap.add_argument("--from-cache", action="store_true", help="use saved compiled circuits; no network")
    ap.add_argument("--out", default=None, help="output folder (default results/<run>/sqale_compile)")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    out_dir = Path(args.out) if args.out else ROOT / cfg["output"] / "sqale_compile"
    out_dir.mkdir(parents=True, exist_ok=True)
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    instances = load_instances({**cfg, "data": {**cfg["data"], "questions": str(ROOT / cfg["data"]["questions"])},
                                "scores": {**cfg["scores"], "dir": str(ROOT / cfg["scores"]["dir"])}}, args.n)
    circuits, names = [], []
    for inst in instances:
        for style in styles:
            c = inst.circuits[CIRCUIT_KEY[style]]
            circuits.append(c)
            names.append((inst.qid, style))
    print(f"{len(circuits)} circuits from {len(instances)} questions ({', '.join(styles)})")

    qpy_path = out_dir / f"compiled_n{args.n}_{'-'.join(styles)}.qpy"
    record: dict = {"config": args.config, "target": args.target, "n": args.n, "styles": styles,
                    "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if args.from_cache:
        with qpy_path.open("rb") as fh:
            compiled = qpy.load(fh)
        print(f"loaded {len(compiled)} compiled circuits from {qpy_path}")
    else:
        import qiskit_superstaq as qss

        provider = qss.SuperstaqProvider()
        before = balance(provider)
        print("balance before:", before)
        t0 = time.time()
        try:
            result = provider.cq_compile([sendable(c) for c in circuits], target=args.target)
        except Exception as exc:
            after = balance(provider)
            record["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            record["credits"] = {"before": before, "after": after, "circuits": len(circuits), "calls": 1,
                                 "cost": (before - after) if before is not None and after is not None else None}
            path = out_dir / f"compile_n{args.n}_{'-'.join(styles)}_error.json"
            path.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
            print(f"cq_compile failed: {record['error']}\nbalance after: {after}; log {path}")
            return 1
        seconds = time.time() - t0
        compiled = list(getattr(result, "circuits", None) or [result.circuit])
        after = balance(provider)
        cost = (before - after) if before is not None and after is not None else None
        print(f"cq_compile: {len(compiled)} circuits in {seconds:.1f} s; balance after {after}; cost {cost}")
        record["credits"] = {"before": before, "after": after, "cost": cost, "circuits": len(circuits), "calls": 1}
        record["compile_seconds"] = seconds
        with qpy_path.open("wb") as fh:
            qpy.dump(compiled, fh)
    if len(compiled) != len(circuits):
        print(f"WARNING: sent {len(circuits)} circuits, got {len(compiled)} back")

    ideal = IdealBackend()
    rows = []
    for (qid, style), src, comp in zip(names, circuits, compiled):
        row = {"qid": qid, "style": style, "input": circuit_stats(src), "sqale": circuit_stats(comp),
               "input_cz_equiv": cz_equivalent(src)}
        try:
            p_src = ideal.probabilities(simulable(src))
            p_comp = ideal.probabilities(simulable(comp))
            if style == "bare":
                row["tvd_vs_uncompiled"] = tvd(p_comp, p_src)
            else:
                logical_src, kept_src = iceberg.postselect_probs(p_src, src)
                logical_comp, kept_comp = iceberg.postselect_probs(p_comp, src)
                row["kept_share_noiseless"] = float(kept_comp)
                row["tvd_logical_vs_uncompiled"] = tvd(logical_comp / max(kept_comp, 1e-12),
                                                       logical_src / max(kept_src, 1e-12))
        except Exception as exc:
            row["verify_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        s_in, s_out = row["input"], row["sqale"]
        extra = (f"TVD {row['tvd_vs_uncompiled']:.2e}" if "tvd_vs_uncompiled" in row else
                 f"kept {row['kept_share_noiseless']:.4f}, logical TVD {row['tvd_logical_vs_uncompiled']:.2e}"
                 if "kept_share_noiseless" in row else row.get("verify_error", ""))
        print(f"{qid[:10]} {style:6s} CZ-equiv {row['input_cz_equiv']:4d} -> {s_out['two_qubit_gates']:4d}  "
              f"depth {s_in['depth']:4d} -> {s_out['depth']:4d}  qubits {s_in['qubits']} -> {s_out['qubits']}  {extra}")

    summary = {}
    for style in styles:
        sel = [r for r in rows if r["style"] == style]
        summary[style] = {
            "two_qubit_in": float(np.mean([r["input"]["two_qubit_gates"] for r in sel])),
            "cz_equiv_in": float(np.mean([r["input_cz_equiv"] for r in sel])),
            "two_qubit_sqale": float(np.mean([r["sqale"]["two_qubit_gates"] for r in sel])),
            "depth_in": float(np.mean([r["input"]["depth"] for r in sel])),
            "depth_sqale": float(np.mean([r["sqale"]["depth"] for r in sel])),
        }
        kept = [r["kept_share_noiseless"] for r in sel if "kept_share_noiseless" in r]
        if kept:
            summary[style]["min_kept_share_noiseless"] = float(min(kept))
    record.update({"rows": rows, "mean": summary})
    path = out_dir / f"compile_n{args.n}_{'-'.join(styles)}.json"
    path.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    print("\nmean:", json.dumps(summary, indent=1))
    bad = [r for r in rows if r.get("kept_share_noiseless", 1.0) < 0.999 or "verify_error" in r]
    print("PASS: the Sqale-compiled circuits keep every check" if not bad else
          f"CHECK: {len(bad)} circuit(s) lost shots or could not be verified; see {path}")
    print(f"written {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
