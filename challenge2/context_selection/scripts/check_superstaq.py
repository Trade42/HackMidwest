"""First-hour Superstaq check. Run this before spending credits on real experiments.

    export SUPERSTAQ_API_KEY=...          (PowerShell: $env:SUPERSTAQ_API_KEY = "...")
    python scripts/check_superstaq.py --budget        # cheapest useful check
    python scripts/check_superstaq.py                 # everything, including Bell tests and IBM

Steps:
  1. print the balance and the available Sqale (``cq_``) targets;
  2. (skipped by --budget) Bell circuit on the Sqale simulator, noiseless and noise-sim,
     and one batched call;
  3. one bare and one Iceberg-encoded QAOA circuit, as ONE job on the noiseless simulator.
     All Iceberg shots must pass post-selection and the decoded distribution must match
     the local ideal one. If this fails, bit order or the compiler broke the checks, and
     no noisy run would mean anything;
  4. the same two circuits, as ONE job with ``method="noise-sim"``, next to the local
     Pauli-noise prediction (tells you which CZ error rate the local model should use);
  5. ``cq_compile`` of both circuits in one call (IBM compile skipped by --budget).

The balance is read after every step, so the log shows what each kind of call costs:
step 3 sends one job with two circuits, which tells you whether billing is per job or per
circuit before any large run. Everything is written to ``--log`` (default
``results/superstaq_check.json``).
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

from qiskit import QuantumCircuit  # noqa: E402

import iceberg  # noqa: E402
from qsel import make_backend, qaoa_circuit, random_qubo, tune_angles  # noqa: E402
from qsel.backends import IdealBackend, LocalNoisyBackend  # noqa: E402
from qsel.superstaq_backend import SuperstaqBackend, superstaq_available  # noqa: E402

LOCAL_NOISE = {"p1": 0.0005, "p2": 0.005, "p_meas": 0.005}   # configs/offline.yaml


def tvd(counts: dict[str, int], reference: np.ndarray) -> float:
    total = sum(counts.values())
    emp = np.zeros_like(reference)
    for key, c in counts.items():
        emp[int(key, 2)] = c / total
    return 0.5 * float(np.abs(emp - reference).sum())


class Ledger:
    """Balance after each step, so the log shows what every call cost."""

    def __init__(self, provider):
        self.provider = provider
        self.rows: list[dict] = []
        self.last = self._balance()
        self.start = self.last
        print("balance:", self.last if self.last is not None else "unavailable")

    def _balance(self):
        try:
            raw = self.provider.get_balance()
        except Exception:
            return None
        try:
            return float(str(raw).replace("$", "").replace(",", "").split()[0])
        except (ValueError, IndexError):
            return None

    def step(self, name: str, circuits: int, calls: int) -> None:
        now = self._balance()
        cost = (self.last - now) if (now is not None and self.last is not None) else None
        self.rows.append({"step": name, "circuits": circuits, "calls": calls, "balance_after": now, "cost": cost})
        shown = f"{cost:.3f}" if cost is not None else "unknown"
        print(f"  [credits] {name}: cost {shown} ({calls} call(s), {circuits} circuit(s)); balance {now}")
        self.last = now


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="cq_sqale_simulator")
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--n", type=int, default=6, help="paragraphs in the test problem (even)")
    ap.add_argument("--budget", action="store_true", help="skip the Bell tests and the IBM compile")
    ap.add_argument("--skip-bell", action="store_true", help="skip the Bell and batched-call steps")
    ap.add_argument("--skip-noise", action="store_true", help="skip the noise-sim run of the QAOA circuits")
    ap.add_argument("--skip-compile", action="store_true", help="skip the cq_compile / ibmq_compile step")
    ap.add_argument("--no-ibm", action="store_true", help="skip the IBM compile")
    ap.add_argument("--targets", action="store_true",
                    help="only list the targets your account can use (free), then stop")
    ap.add_argument("--probe", action="store_true",
                    help="try one tiny job, compile and QUBO call to see what the account's role allows, then stop")
    ap.add_argument("--log", default=str(ROOT / "results" / "superstaq_check.json"))
    args = ap.parse_args()
    if args.budget:
        args.skip_bell = args.no_ibm = True

    if not superstaq_available():
        try:
            import qiskit_superstaq  # noqa: F401
        except ImportError:
            print("qiskit-superstaq is not installed in this environment: pip install -e \".[superstaq]\"")
            return 2
        print("No Superstaq API key found. This process cannot see SUPERSTAQ_API_KEY.\n"
              "Set it in the same terminal you run this from:\n"
              "  PowerShell: $env:SUPERSTAQ_API_KEY = \"<key>\"\n"
              "  bash:       export SUPERSTAQ_API_KEY=<key>\n"
              "  cmd:        set SUPERSTAQ_API_KEY=<key>")
        return 2

    noiseless = SuperstaqBackend(target=args.target, method=None)
    noisy = SuperstaqBackend(target=args.target, method="noise-sim")
    provider = noiseless.provider
    log: dict = {"target": args.target, "shots": args.shots, "n": args.n,
                 "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    print("== account")
    ledger = Ledger(provider)
    try:
        targets = [t.name for t in provider.backends() if t.name.startswith("cq_")]
        print("cq targets:", ", ".join(targets) or "none listed")
        log["cq_targets"] = targets
    except Exception as exc:
        print("target list unavailable:", exc)

    access = target_access(provider)
    log["target_access"] = access
    if args.targets:
        print("\n== targets as listed by Superstaq (accessible=True is NOT a guarantee: a role can still get 403;"
              " use --probe)")
        for row in access:
            if row["target"].startswith(("cq_", "ss_")) or row["accessible"]:
                print(f"  {row['target']:32s} accessible={row['accessible']!s:5s} submit={row['supports_submit']!s:5s} "
                      f"compile={row['supports_compile']!s:5s} qubo={row['supports_submit_qubo']!s:5s} "
                      f"available={row['available']}")
        _write_log(args.log, log)
        return 0
    if args.probe:
        print("\n== probe: one tiny call of each kind; a 403 costs nothing, a success may cost up to ~0.5")
        log["probe"] = probe_operations(provider, ledger)
        log["credits"] = {"start": ledger.start, "end": ledger.last, "steps": ledger.rows}
        _write_log(args.log, log)
        return 0
    known = {row["target"]: row for row in access}
    if args.target in known and not known[args.target]["accessible"]:
        print(f"\nSTOP: your account's role cannot use {args.target} (nothing was submitted, no credits used).\n"
              "Ask the challenge organizers or Superstaq support to enable it, or list what you can use with\n"
              "  python scripts/check_superstaq.py --targets")
        log["stopped"] = f"{args.target} not accessible"
        _write_log(args.log, log)
        return 3

    if not args.skip_bell:
        print("\n== Bell circuit")
        bell = QuantumCircuit(2, 2, name="bell")
        bell.h(0)
        bell.cx(0, 1)
        bell.measure([0, 1], [0, 1])
        t0 = time.time()
        clean = noiseless.run([bell], args.shots)[0]
        print(f"noiseless ({time.time() - t0:.1f} s): {clean}")
        ledger.step("bell noiseless", 1, 1)
        t0 = time.time()
        dirty = noisy.run([bell], args.shots)[0]
        print(f"noise-sim ({time.time() - t0:.1f} s): {dirty}")
        ledger.step("bell noise-sim", 1, 1)
        if sum(c for k, c in clean.items() if k in ("01", "10")):
            print("WARNING: the noiseless simulator returned odd-parity Bell outcomes")
        print("\n== batched call")
        t0 = time.time()
        noisy.run([_rotated_bell(a) for a in (0.3, 0.6, 0.9, 1.2)], args.shots)
        print(f"4 circuits, {args.shots} shots each: {time.time() - t0:.1f} s wall")
        ledger.step("batched noise-sim", 4, 1)

    print("\n== bare vs Iceberg on the noiseless simulator (one job, two circuits)")
    qubo = random_qubo(args.n, edge_fraction=0.3, seed=1)
    angles = tune_angles(qubo, p=1).angles
    bare = qaoa_circuit(qubo, angles)
    enc = iceberg.encode(bare)
    ideal = IdealBackend().probabilities(bare)
    t0 = time.time()
    c_bare, c_enc = noiseless.run([bare, enc], args.shots)
    print(f"ran both in {time.time() - t0:.1f} s")
    ledger.step("qaoa noiseless", 2, 1)
    logical, stats = iceberg.postselect(c_enc, enc)
    scale = float(np.sqrt(len(ideal) / args.shots))
    d_bare, d_enc = tvd(c_bare, ideal), (tvd(logical, ideal) if logical else 1.0)
    print(f"bare:    TVD to local ideal = {d_bare:.3f}")
    print(f"iceberg: kept {stats.kept_share:.1%} of shots, TVD to local ideal = {d_enc:.3f}")
    print(f"(sampling noise alone gives a TVD of roughly {0.4 * scale:.3f})")
    ok = stats.kept_share >= 0.99 and d_enc < max(0.1, 1.5 * scale) and d_bare < max(0.1, 1.5 * scale)
    log["noiseless"] = {"kept_share": stats.kept_share, "tvd_bare": d_bare, "tvd_iceberg": d_enc, "pass": ok,
                        "qubits": {"bare": bare.num_qubits, "iceberg": enc.num_qubits}}
    if stats.kept_share < 0.99:
        print("FAIL: Iceberg shots were discarded without noise. Check bit order and whether the "
              "compiler removed or reordered the check gates.")
    if not ok:
        print("FAIL: do not start noisy runs until this passes.")
    else:
        print("PASS: encoding, bit order and post-selection agree with the local simulator.")

    if ok and not args.skip_noise:
        print("\n== the same two circuits with noise-sim (one job), against the local Pauli model")
        n_bare, n_enc = noisy.run([bare, enc], args.shots)
        ledger.step("qaoa noise-sim", 2, 1)
        n_logical, n_stats = iceberg.postselect(n_enc, enc)
        local = make_backend({"kind": "local-pauli", "noise": LOCAL_NOISE, "seed": 0})
        l_bare, l_enc = local.run([bare, enc], args.shots)
        l_logical, l_stats = iceberg.postselect(l_enc, enc)
        row = {
            "sqale": {"kept_share": n_stats.kept_share, "tvd_bare": tvd(n_bare, ideal),
                      "tvd_iceberg": tvd(n_logical, ideal) if n_logical else None},
            "local_model": {"noise": LOCAL_NOISE, "kept_share": l_stats.kept_share, "tvd_bare": tvd(l_bare, ideal),
                            "tvd_iceberg": tvd(l_logical, ideal) if l_logical else None},
        }
        log["noise_sim"] = row
        for name, r in row.items():
            ti = f"{r['tvd_iceberg']:.3f}" if r["tvd_iceberg"] is not None else "n/a"
            print(f"{name:12s} kept {r['kept_share']:.1%}  TVD bare {r['tvd_bare']:.3f}  TVD iceberg {ti}")
        print("If Sqale keeps far fewer shots than the local model, its CZ error is above 0.5%; far more, below.")

    if not args.skip_compile:
        print("\n== compile statistics (cq_compile" + ("" if args.no_ibm else ", ibmq_compile") + ")")
        try:
            local_grid = LocalNoisyBackend(coupling="grid", grid_shape=(4, 6))
            rows = noiseless.compile_stats([bare, enc], ibm=not args.no_ibm)
            ledger.step("cq_compile" + ("" if args.no_ibm else " + ibmq_compile"), 2, 1 if args.no_ibm else 2)
            log["compile"] = rows
            for row, circuit in zip(rows, (bare, enc)):
                ours = local_grid.compiled_stats(circuit)["two_qubit_gates"]
                row["local_grid_two_qubit_gates"] = ours
                print(f"{row['name']}: input {circuit_stats_line(row['input'])} | Sqale {circuit_stats_line(row['sqale'])} | "
                      f"IBM {circuit_stats_line(row['ibm'])} | our local compile: {ours} two-qubit gates")
            print("The Iceberg circuit is nearest-neighbour by construction. If Sqale reports many more "
                  "two-qubit gates than our local compile, cq_compile did not find the layout and added SWAPs.")
        except Exception as exc:
            print("compile step failed:", exc)
            log["compile_error"] = repr(exc)

    log["credits"] = {"start": ledger.start, "end": ledger.last, "steps": ledger.rows}
    if ledger.start is not None and ledger.last is not None:
        print(f"\n== credits used: {ledger.start - ledger.last:.3f} (balance {ledger.last})")
    _write_log(args.log, log)
    return 0 if ok else 1


def probe_operations(provider, ledger: Ledger) -> list[dict]:
    """Try a job, a compile and a QUBO solve once each on a tiny problem; record what the role allows."""
    tiny = QuantumCircuit(1, 1, name="probe")
    tiny.x(0)
    tiny.measure(0, 0)
    bell = QuantumCircuit(2, 2, name="probe_bell")
    bell.h(0)
    bell.cx(0, 1)
    bell.measure([0, 1], [0, 1])
    qubo = random_qubo(2, edge_fraction=1.0, seed=0)
    attempts = [
        ("job on ss_unconstrained_simulator",
         lambda: SuperstaqBackend(target="ss_unconstrained_simulator", method=None, provider=provider).run([tiny], 10)),
        ("job on cq_sqale_simulator",
         lambda: SuperstaqBackend(target="cq_sqale_simulator", method=None, provider=provider).run([tiny], 10)),
        ("cq_compile for cq_sqale_qpu", lambda: provider.cq_compile(bell, target="cq_sqale_qpu")),
        ("compile for ss_unconstrained_simulator",
         lambda: provider.compile(bell, target="ss_unconstrained_simulator")),
        ("submit_qubo on ss_unconstrained_simulator",
         lambda: SuperstaqBackend(provider=provider).submit_qubo(qubo, repetitions=10, max_solutions=10)),
    ]
    rows = []
    for name, call in attempts:
        try:
            call()
            status, detail = "allowed", ""
        except Exception as exc:
            text = str(exc).replace("\n", " ")
            status = "refused (403)" if "403" in text or "permit" in text else "error"
            detail = f"{type(exc).__name__}: {text[:160]}"
        ledger.step(name, 1, 1)
        print(f"  {name:42s} {status} {detail}")
        rows.append({"operation": name, "status": status, "detail": detail})
    return rows


def target_access(provider) -> list[dict]:
    """Per-target flags from Superstaq's /get_targets (a free call); empty if unavailable."""
    try:
        rows = provider._client.get_targets()
    except Exception:
        return []
    out = []
    for t in rows:
        get = (lambda k, t=t: t.get(k) if isinstance(t, dict) else getattr(t, k, None))
        out.append({k: get(k) for k in ("target", "accessible", "supports_submit", "supports_compile",
                                         "supports_submit_qubo", "available")})
    return [r for r in out if r["target"]]


def _write_log(path: str, log: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(log, indent=1, default=str), encoding="utf-8")
    print(f"log written to {path}")


def _rotated_bell(angle: float) -> QuantumCircuit:
    qc = QuantumCircuit(2, 2, name=f"bell_ry_{angle}")
    qc.ry(angle, 0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def circuit_stats_line(stats: dict) -> str:
    if "error" in stats:
        return f"unavailable ({stats['error'][:60]})"
    return f"{stats['qubits']} qubits, {stats['two_qubit_gates']} two-qubit gates, depth {stats['depth']}"


if __name__ == "__main__":
    raise SystemExit(main())
