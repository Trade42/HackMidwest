"""Solver-side studies that need no language model.

All of them run on the QUBOs of the evaluation questions (not synthetic toys), except
the roadmap projection, which needs sizes the dataset does not have.

    noise_sweep    approximation ratio and shots kept against CZ error rate
    edge_sweep     the same against the number of redundant edges (tau)
    fault_budget   every single fault scored: what the checks catch, per block
    compile_stats  two-qubit gates per tier and connectivity   -> circuits/stats.json (if circuit_stats is set)
    roadmap        qubits, gates and brute-force time for 10 to 30 paragraphs
    equal_kept_shots  bare QAOA at R raw shots against bus-Iceberg QAOA given about R
                   accepted shots (run separately: scripts/equal_kept_shots.py)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from qiskit import QuantumCircuit

import iceberg
from iceberg.frames import FrameSimulator, PauliNoise
from qsel import Angles, QaoaSimulator, build_qubo, qaoa_circuit, random_qubo, tune_angles
from qsel.pauli_backend import route_on_grid

from .data import read_questions
from .scoring import load_scores, make_scorer, score_path, score_questions

TIERS = ("bare", "bus", "direct")
TIER_LABEL = {"bare": "Bare QAOA", "bus": "Error-detected (bus)", "direct": "Textbook Iceberg"}


def cz_count(circuit: QuantumCircuit) -> int:
    """CZ gates needed by a circuit: one per CNOT/CZ, two per rzz/rxx, three per SWAP."""
    cost = {"cx": 1, "cz": 1, "rzz": 2, "rxx": 2, "swap": 3}
    return int(sum(cost.get(inst.operation.name, 0) for inst in circuit.data))


def _write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")


def _energy_metrics(counts: Mapping[str, int], sim: QaoaSimulator) -> tuple[float, float]:
    idx = np.fromiter((int(k, 2) for k in counts), dtype=np.int64, count=len(counts))
    c = np.fromiter(counts.values(), dtype=float, count=len(counts))
    e = sim.energies[idx]
    return sim.approx_ratio(float(c @ e / c.sum())), float(c[e <= sim.e_min + 1e-9].sum() / c.sum())


class Instance:
    """One question's QUBO with its circuits in every tier."""

    def __init__(self, qid: str, qubo, angles: Angles, *, grid_shape=(4, 6), seed: int = 0):
        self.qid = qid
        self.qubo = qubo
        self.angles = angles
        self.sim = QaoaSimulator(qubo)
        bare = qaoa_circuit(qubo, angles)
        self.circuits = {
            "bare_all": bare,
            "bare": route_on_grid(bare, grid_shape, seed=seed),
            "bus": iceberg.encode(bare, style="bus", seed=seed),
            "direct": iceberg.encode(bare, style="direct"),
        }
        self._frames: dict[str, FrameSimulator] = {}

    def frames(self, tier: str) -> FrameSimulator:
        if tier not in self._frames:
            self._frames[tier] = FrameSimulator(self.circuits[tier], level="cz")
        return self._frames[tier]

    def run(self, tier: str, noise: PauliNoise, shots: int, seed: int) -> dict[str, float]:
        counts = self.frames(tier).sample(shots, noise, seed=seed)
        kept = 1.0
        if tier != "bare":
            counts, stats = iceberg.postselect(counts, self.circuits[tier])
            kept = stats.kept_share
        kept_shots = int(sum(counts.values()))
        if not counts:
            return {"kept": 0.0, "ar": float("nan"), "p_opt": float("nan"), "raw_shots": shots,
                    "kept_shots": 0, "hit": 0.0}
        ar, p_opt = _energy_metrics(counts, self.sim)
        return {"kept": kept, "ar": ar, "p_opt": p_opt, "raw_shots": shots, "kept_shots": kept_shots,
                "hit": float(p_opt > 0.0)}


def load_instances(cfg: Mapping[str, Any], n: int, *, tau: float | None = None, seed: int = 0) -> list[Instance]:
    """The first ``n`` evaluation questions as instances (angles tuned on the ideal tier)."""
    questions = read_questions(cfg["data"]["questions"])[:n]
    qp = {k: v for k, v in dict(cfg["qubo"]).items() if v is not None}
    if tau is not None:
        qp["tau"] = tau
    p = int(cfg["qaoa"]["p"])
    out = []
    for q in questions:
        r, s = load_scores(q.qid, cfg["scores"]["dir"])
        qubo = build_qubo(r, s, **qp)
        out.append(Instance(q.qid, qubo, tune_angles(qubo, p=p, seed=seed).angles, seed=seed))
    return out


# ----------------------------------------------------------------- 1. noise
def noise_sweep(cfg: Mapping[str, Any], instances: Sequence[Instance], *, rates: Sequence[float],
                shots: int, kinds: Sequence[str] = ("depolarizing", "dephasing")) -> dict[str, Any]:
    rows = []
    for kind in kinds:
        for p2 in rates:
            noise = PauliNoise(p1=p2 / 10.0, p2=p2, p_meas=p2, two_qubit=kind)
            for tier in TIERS:
                res = [inst.run(tier, noise, shots, seed=1000 * i + 7) for i, inst in enumerate(instances)]
                ar = np.array([r["ar"] for r in res])
                kept = np.array([r["kept"] for r in res])
                p_opt = np.array([r["p_opt"] for r in res])
                rows.append({
                    "noise": kind, "p2": p2, "tier": tier, "label": TIER_LABEL[tier],
                    "ar": float(np.nanmean(ar)), "ar_se": float(np.nanstd(ar) / np.sqrt(max(1, np.sum(~np.isnan(ar))))),
                    "kept": float(kept.mean()), "p_opt": float(np.nanmean(p_opt)),
                    "p_opt_x_kept": float(np.nanmean(p_opt * kept)),
                    "cz": float(np.mean([inst.frames(tier).two_qubit_gates for inst in instances])),
                    "per_instance_ar": [float(x) for x in ar],
                })
    ideal = float(np.mean([inst.sim.approx_ratio(float(inst.sim.probabilities(inst.angles) @ inst.sim.energies)) for inst in instances]))
    uniform = float(np.mean([inst.sim.approx_ratio(float(inst.sim.energies.mean())) for inst in instances]))
    return {"rows": rows, "ideal_ar": ideal, "uniform_ar": uniform, "instances": len(instances), "shots": shots,
            "note": "p1 = p2 / 10 and p_meas = p2. Assumed rates, not Sqale calibration data. "
                    "Bare is routed on a 4x6 grid; the textbook encoding gets free all-to-all wiring."}


# ----------------------------------------------------------------- 2. edges
def edge_sweep(cfg: Mapping[str, Any], n: int, *, quantiles: Sequence[float], shots: int) -> dict[str, Any]:
    """Vary tau to change the number of redundant edges; default noise."""
    noise = PauliNoise(**((cfg["backends"].get("physical") or {}).get("noise") or {}))
    calib = read_questions(cfg["data"]["calibration"])
    sdir = Path(cfg["scores"]["dir"])
    missing = [q for q in calib if not score_path(q.qid, sdir).exists()]
    if missing:   # calibrate.py scores in memory only; a fresh scorer needs them written once
        score_questions(missing, make_scorer(cfg["scores"]["scorer"]), sdir)
    pooled = np.concatenate([load_scores(q.qid, cfg["scores"]["dir"])[1][np.triu_indices(len(q.paragraphs), 1)] for q in calib])
    points = []
    for qt in quantiles:
        tau = float(np.quantile(pooled, qt))
        for i, inst in enumerate(load_instances(cfg, n, tau=tau)):
            bare = inst.run("bare", noise, shots, seed=31 * i + 1)
            bus = inst.run("bus", noise, shots, seed=31 * i + 2)
            direct = inst.run("direct", noise, shots, seed=31 * i + 3)
            ideal = inst.sim.approx_ratio(float(inst.sim.probabilities(inst.angles) @ inst.sim.energies))
            points.append({
                "qid": inst.qid, "tau_quantile": qt, "tau": tau, "edges": inst.qubo.num_edges,
                "ideal_ar": ideal, "bare_ar": bare["ar"], "bus_ar": bus["ar"], "direct_ar": direct["ar"],
                "bus_kept": bus["kept"], "direct_kept": direct["kept"],
                "bare_cz": inst.frames("bare").two_qubit_gates, "bus_cz": inst.frames("bus").two_qubit_gates,
                "direct_cz": inst.frames("direct").two_qubit_gates,
            })
    return {"points": points, "noise": noise.to_dict(), "shots": shots}


# ---------------------------------------------------------- 3. fault budget
def fault_budget(instances: Sequence[Instance]) -> dict[str, Any]:
    """Exhaustive single-fault analysis at the CZ level, averaged over instances."""
    tiers: dict[str, dict[str, list[float]]] = {t: {} for t in TIERS}
    blocks: dict[str, dict[str, list[float]]] = {}
    paulis: dict[str, dict[str, list[float]]] = {t: {} for t in TIERS}
    for inst in instances:
        e = inst.sim.energies
        span = inst.sim.e_max - inst.sim.e_min
        for tier in TIERS:
            report = inst.frames(tier).single_faults(e)
            co = report.coefficients()
            two = [f for f in report.faults if f.kind == "gate2"]
            row = {
                "cz": co["two_qubit_gates"], "discard_slope": co["discard"],
                "ar_loss_slope": co["d_energy"] / span, "tvd_slope": co["error"],
                "undetected_share": sum(f.outcome == "undetected error" for f in two) / max(1, len(two)),
                "detected_share": sum(f.outcome == "detected" for f in two) / max(1, len(two)),
                "harmless_share": sum(f.outcome == "harmless" for f in two) / max(1, len(two)),
            }
            for k, v in row.items():
                tiers[tier].setdefault(k, []).append(float(v))
            for kind, members in (("phase (Z-type) faults", [f for f in two if set(f.pauli) <= {"I", "Z"}]),
                                  ("bit-flip (X-type) faults", [f for f in two if set(f.pauli) <= {"I", "X"}]),
                                  ("all two-qubit faults", two)):
                paulis[tier].setdefault(kind, []).append(
                    sum(f.outcome == "undetected error" for f in members) / max(1, len(members)))
            if tier == "bus":
                for name, r in report.by_block().items():
                    b = blocks.setdefault(name, {})
                    for k in ("faults", "detected", "harmless", "undetected_error"):
                        b.setdefault(k, []).append(float(r[k]))
    mean = lambda d: {k: float(np.mean(v)) for k, v in d.items()}  # noqa: E731
    return {"tiers": {t: mean(v) for t, v in tiers.items()}, "bus_blocks": {b: mean(v) for b, v in blocks.items()},
            "undetected_by_fault_type": {t: mean(v) for t, v in paulis.items()}, "instances": len(instances),
            "note": "Slopes are per unit CZ error rate in the low-noise limit (one fault at most)."}


# --------------------------------------------------------- 4. compile stats
def compile_stats(cfg: Mapping[str, Any], instances: Sequence[Instance], *, sabre_trials: int = 6) -> dict[str, Any]:
    """Two-qubit gate counts per tier and connectivity; Superstaq numbers when available."""
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap

    grid = CouplingMap.from_grid(4, 6)
    basis = ["cz", "rz", "sx", "x"]
    rows = []
    for inst in instances:
        direct = inst.circuits["direct"]
        best = None
        for s in range(sabre_trials):
            out = transpile(direct, basis_gates=basis, coupling_map=grid, optimization_level=3, seed_transpiler=s)
            n2 = sum(1 for i in out.data if len(i.qubits) == 2 and i.operation.name != "barrier")
            best = n2 if best is None else min(best, n2)
        rows.append({
            "qid": inst.qid, "edges": inst.qubo.num_edges,
            "bare_all_to_all": cz_count(inst.circuits["bare_all"]), "bare_grid": cz_count(inst.circuits["bare"]),
            "bus_grid": cz_count(inst.circuits["bus"]), "direct_all_to_all": cz_count(direct), "direct_grid": int(best),
            "qubits": {"bare": inst.qubo.n, "bus": inst.circuits["bus"].num_qubits, "direct": direct.num_qubits},
        })
    keys = ("bare_all_to_all", "bare_grid", "bus_grid", "direct_all_to_all", "direct_grid", "edges")
    out: dict[str, Any] = {"rows": rows, "mean": {k: float(np.mean([r[k] for r in rows])) for k in keys},
                           "grid": [4, 6], "note": "CZ counts. Grid counts for bare use SWAP routing (best of 16 "
                                                   "SABRE seeds); the bus circuit needs no SWAPs; the textbook "
                                                   "circuit on the grid is Qiskit level 3, best of several seeds."}
    spec = cfg["backends"].get("physical") or {}
    if spec.get("kind") == "superstaq":
        try:
            from qsel import make_backend

            backend = make_backend({k: v for k, v in spec.items()})
            circuits = [c for inst in instances[:5] for c in (inst.circuits["bare_all"], inst.circuits["bus"])]
            out["superstaq"] = backend.compile_stats(circuits)
        except Exception as exc:  # network or access problems must not sink the run
            out["superstaq_error"] = repr(exc)
    return out


# ---------------------------------------------------------------- 5. roadmap
def roadmap(cfg: Mapping[str, Any], *, sizes: Sequence[int], edge_fraction: float, max_brute: int = 22) -> dict[str, Any]:
    """Compile-only projection on synthetic problems of 10 to 30 paragraphs."""
    p = int(cfg["qaoa"]["p"])
    fixed = Angles((0.4,) * p, (0.5,) * p)        # gate counts do not depend on the angles
    rows = []
    for n in sizes:
        qubo = random_qubo(n, edge_fraction=edge_fraction, seed=n)
        brute = None
        if n <= max_brute:
            t0 = time.perf_counter()
            e = qubo.energies_all()
            ok = qubo.feasible_mask_all()
            int(np.flatnonzero(ok)[np.argmin(e[ok])])
            brute = time.perf_counter() - t0
        bare = qaoa_circuit(qubo, fixed)
        k_code = n + n % 2
        cols = k_code // 2 + 1
        side = int(np.ceil(np.sqrt(n)))
        shape = (4, 6) if n <= 24 else (side, int(np.ceil(n / side)))
        routed = route_on_grid(bare, shape, trials=4, seed=0)
        bus = iceberg.encode(bare, style="bus", seed=0)
        direct = iceberg.encode(bare, style="direct")
        rows.append({
            "paragraphs": n, "edges": qubo.num_edges, "brute_force_seconds": brute,
            "bare_qubits": n, "bare_cz_all_to_all": cz_count(bare), "bare_cz_grid": cz_count(routed),
            "bare_grid": list(shape), "bus_qubits": bus.num_qubits, "bus_cz": cz_count(bus),
            "bus_patch": [3, cols], "bus_fits_6x4": bool(cols <= 6),
            "direct_qubits": direct.num_qubits, "direct_cz_all_to_all": cz_count(direct),
        })
    timed = [(r["paragraphs"], r["brute_force_seconds"]) for r in rows if r["brute_force_seconds"]]
    fit = None
    if len(timed) >= 3:
        xs = np.array([t[0] for t in timed[-4:]], dtype=float)
        ys = np.log2(np.array([t[1] for t in timed[-4:]]))
        slope, intercept = np.polyfit(xs, ys, 1)
        fit = {"log2_seconds_per_paragraph": float(slope), "intercept": float(intercept),
               "paragraphs_at_1s": float(-intercept / slope)}
        for r in rows:
            r["brute_force_seconds_fit"] = float(2 ** (slope * r["paragraphs"] + intercept))
    return {"rows": rows, "brute_force_fit": fit, "edge_fraction": edge_fraction, "p": p,
            "marks": {"sqale_grid_qubits": 24, "infleqtion_2028_logical_qubits": 100},
            "note": "Brute force is exhaustive enumeration in numpy on this machine; it is not the best "
                    "classical method (branch and bound solves these sizes instantly). Beyond the timed "
                    "sizes the curve is an exponential fit."}


# ------------------------------------------------------- 6. equal kept shots
def equal_kept_shots(cfg: Mapping[str, Any], instances: Sequence[Instance], *, shots: int = 4000,
                     tier: str = "bus", n_boot: int = 10000, seed: int = 0) -> dict[str, Any]:
    """Bare QAOA at ``shots`` raw shots against the error-detected tier at ``shots / kept_share``.

    Equal raw shots (lanes 3b and 3c) charge the error-detected tier for its discards in
    sample size; here it gets enough raw shots to end with about as many accepted shots
    as bare QAOA, and the price is reported as the raw-shot multiplier. Per question the
    kept share is estimated by a pilot run at ``shots`` raw shots (itself reported as the
    equal-raw-shot row); the equal-kept run then draws ``ceil(shots / pilot kept share)``
    raw shots with a fresh seed. Noise is the config's physical backend (local Pauli
    model). Metrics use kept shots before repair: approximation ratio of the mean sampled
    energy, probability of the optimum per kept shot, and whether any kept shot is optimal.
    """
    from .metrics import paired_bootstrap_diff

    noise = PauliNoise(**((cfg["backends"].get("physical") or {}).get("noise") or {}))
    per: list[dict[str, Any]] = []
    for i, inst in enumerate(instances):
        bare = inst.run("bare", noise, shots, seed=seed + 7919 * i + 1)
        pilot = inst.run(tier, noise, shots, seed=seed + 7919 * i + 2)
        raw = int(np.ceil(shots / pilot["kept"])) if pilot["kept"] > 0 else shots
        matched = inst.run(tier, noise, raw, seed=seed + 7919 * i + 3)
        per.append({"qid": inst.qid, "edges": inst.qubo.num_edges, "bare": bare,
                    f"{tier}_equal_raw": pilot, f"{tier}_equal_kept": matched})

    def summarise(key: str) -> dict[str, Any]:
        rows = [p[key] for p in per]
        col = lambda name: np.array([r[name] for r in rows], dtype=float)  # noqa: E731
        return {"mean_approx_ratio": float(np.nanmean(col("ar"))),
                "p_optimum_per_kept_shot": float(np.nanmean(col("p_opt"))),
                "optimum_in_sample_share": float(np.mean(col("hit"))),
                "mean_kept_share": float(np.mean(col("kept"))),
                "mean_raw_shots": float(np.mean(col("raw_shots"))),
                "mean_kept_shots": float(np.mean(col("kept_shots"))),
                "raw_shot_multiplier": float(np.mean(col("raw_shots")) / shots),
                "raw_shot_multiplier_per_question": {"mean": float(np.mean(col("raw_shots") / shots)),
                                                     "min": float(np.min(col("raw_shots") / shots)),
                                                     "max": float(np.max(col("raw_shots") / shots))}}

    keys = ("bare", f"{tier}_equal_raw", f"{tier}_equal_kept")
    labels = {"bare": f"Bare QAOA, {shots} raw shots",
              f"{tier}_equal_raw": f"{TIER_LABEL[tier]}, {shots} raw shots",
              f"{tier}_equal_kept": f"{TIER_LABEL[tier]}, raw shots for ~{shots} kept"}
    rows = {k: {"label": labels[k], **summarise(k)} for k in keys}
    diffs = {}
    for k in keys[1:]:
        for metric, name in (("ar", "approx_ratio"), ("p_opt", "p_optimum"), ("hit", "optimum_in_sample")):
            a = np.array([p[k][metric] for p in per], dtype=float)
            b = np.array([p["bare"][metric] for p in per], dtype=float)
            ok = ~(np.isnan(a) | np.isnan(b))
            diffs[f"{k}_minus_bare_{name}"] = paired_bootstrap_diff(a[ok], b[ok], n_boot=n_boot, seed=seed)
    return {"experiment": cfg.get("name"), "instances": len(instances), "shots": shots, "tier": tier,
            "noise": noise.to_dict(), "tiers": rows, "paired_differences": diffs, "per_question": per,
            "note": "Kept shots before repair. Bare QAOA keeps every shot (routed on the 4x6 grid). "
                    "The equal-kept run's raw shots come from a per-question pilot kept share, so its "
                    "kept count is about, not exactly, the bare shot count. Paired bootstrap over questions."}


# -------------------------------------------------------------------- driver
def run_studies(cfg: Mapping[str, Any], *, quick: bool = False, log=print) -> dict[str, Any]:
    out_dir = Path(cfg["output"]) / "studies"
    shots = int(cfg["qaoa"]["shots"])
    n = 8 if quick else 30
    t0 = time.perf_counter()
    instances = load_instances(cfg, n)
    log(f"studies: {len(instances)} instances, {np.mean([i.qubo.num_edges for i in instances]):.1f} edges on average")

    rates = (0.002, 0.005, 0.01) if quick else (0.001, 0.002, 0.005, 0.01, 0.02)
    sweep = noise_sweep(cfg, instances, rates=rates, shots=shots)
    _write(out_dir / "noise_sweep.json", sweep)
    log(f"studies: noise sweep done ({time.perf_counter() - t0:.0f} s)")

    budget = fault_budget(instances[: (4 if quick else 12)])
    _write(out_dir / "fault_budget.json", budget)
    log(f"studies: fault budget done ({time.perf_counter() - t0:.0f} s)")

    stats = compile_stats(cfg, instances[: (4 if quick else 12)], sabre_trials=3 if quick else 6)
    _write(out_dir / "compile_stats.json", stats)
    if cfg.get("circuit_stats"):      # the hand-over file; smoke and test runs leave it alone
        _write(Path(cfg["circuit_stats"]), {"experiment": cfg["name"], **stats})
    log(f"studies: compile statistics done ({time.perf_counter() - t0:.0f} s)")

    edges = edge_sweep(cfg, 6 if quick else 20, quantiles=(0.9, 0.6) if quick else (0.95, 0.9, 0.8, 0.7, 0.6, 0.45, 0.3),
                       shots=shots)
    _write(out_dir / "edge_sweep.json", edges)
    log(f"studies: edge sweep done ({time.perf_counter() - t0:.0f} s)")

    density = float(np.mean([i.qubo.num_edges for i in instances]) / 45.0)
    road = roadmap(cfg, sizes=(10, 14, 18) if quick else tuple(range(10, 31, 2)), edge_fraction=density,
                   max_brute=18 if quick else 24)
    _write(out_dir / "roadmap.json", road)
    log(f"studies: roadmap done ({time.perf_counter() - t0:.0f} s)")
    return {"noise_sweep": sweep, "fault_budget": budget, "compile_stats": stats, "edge_sweep": edges, "roadmap": road}
