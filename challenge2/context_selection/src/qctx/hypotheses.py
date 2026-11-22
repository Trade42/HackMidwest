"""Judge the pre-registered hypotheses exactly as written in HYPOTHESES.md."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .metrics import paired_bootstrap_diff, wilson_interval

Table = Mapping[str, Mapping[str, Sequence[float]]]     # lane -> metric -> per-question values


def _verdict(lo: float, hi: float, *, predicted: str = "positive") -> str:
    if predicted == "zero":
        return "supported" if lo <= 0.0 <= hi else "not supported"
    if lo > 0:
        return "supported"
    if hi < 0:
        return "contradicted"
    return "not supported"


def _diff(table: Table, a: tuple[str, str], b: tuple[str, str], n_boot: int, seed: int) -> dict[str, float] | None:
    try:
        xa = np.asarray(table[a[0]][a[1]], dtype=float)
        xb = np.asarray(table[b[0]][b[1]], dtype=float)
    except KeyError:
        return None
    if len(xa) == 0 or len(xa) != len(xb):
        return None
    ok = ~(np.isnan(xa) | np.isnan(xb))
    if not ok.any():
        return None
    out = paired_bootstrap_diff(xa[ok], xb[ok], n_boot=n_boot, seed=seed)
    out["n"] = int(ok.sum())
    return out


def evaluate_hypotheses(table: Table, *, lanes: Mapping[str, str] | None = None,
                        n_boot: int = 10000, seed: int = 0) -> list[dict[str, Any]]:
    """Return one record per hypothesis: quantity, interval, verdict.

    ``table[lane][metric]`` holds one value per question, in the same question order
    for every lane. ``lanes`` maps the roles used below to lane names, so the same
    code judges the hardware lanes or a noise-sweep variant.
    """
    L = {"all": "1a", "exact": "2a", "ideal_p2": "3a-p2", "ideal": "3a", "bare": "3b", "iceberg": "3c"}
    L.update(lanes or {})
    out: list[dict[str, Any]] = []

    def missing(name: str, statement: str, need: str) -> None:
        out.append({"id": name, "statement": statement, "verdict": "not evaluated", "detail": f"needs {need}"})

    # ---- H1
    d = _diff(table, (L["exact"], "em_single"), (L["all"], "em_single"), n_boot, seed)
    stmt = "Selection helps: exact selection beats all 10 paragraphs on exact match"
    if d is None:
        missing("H1", stmt, f"lanes {L['exact']} and {L['all']}")
    else:
        out.append({"id": "H1", "statement": stmt, "quantity": f"EM({L['exact']}) - EM({L['all']})",
                    **d, "verdict": _verdict(d["lo"], d["hi"])})

    # ---- H2
    stmt = "Ideal QAOA at p = 2 is close to exact"
    ideal = L["ideal_p2"] if L["ideal_p2"] in table else L["ideal"]
    d = _diff(table, (ideal, "em_single"), (L["exact"], "em_single"), n_boot, seed)
    if d is None or "hit_optimum" not in table.get(ideal, {}):
        missing("H2", stmt, f"lanes {ideal} and {L['exact']}")
    else:
        hits = np.asarray(table[ideal]["hit_optimum"], dtype=float)
        hits = hits[~np.isnan(hits)]
        k, n = int(hits.sum()), int(len(hits))
        lo, hi = wilson_interval(k, n)
        part_a = lo > 0.5
        part_b = d["lo"] <= 0.0 <= d["hi"]
        out.append({
            "id": "H2", "statement": stmt, "lane": ideal,
            "quantity": f"(a) optimum hit rate of {ideal}; (b) EM({ideal}) - EM({L['exact']})",
            "hit_rate": k / n if n else float("nan"), "hit_lo": lo, "hit_hi": hi,
            **d, "verdict": "supported" if (part_a and part_b) else "not supported",
            "detail": f"(a) {'holds' if part_a else 'fails'}, (b) {'holds' if part_b else 'fails'}",
        })

    # ---- H3
    stmt = "Error detection helps the solver: higher approximation ratio after discarding flagged shots"
    d3 = _diff(table, (L["iceberg"], "approx_ratio"), (L["bare"], "approx_ratio"), n_boot, seed)
    if d3 is None:
        missing("H3", stmt, f"lanes {L['iceberg']} and {L['bare']}")
    else:
        out.append({"id": "H3", "statement": stmt, "quantity": f"AR({L['iceberg']}) - AR({L['bare']})",
                    **d3, "verdict": _verdict(d3["lo"], d3["hi"])})

    # ---- H4
    stmt = "The benefit shrinks downstream: accuracy gain is smaller than the solver-quality gain"
    try:
        ar = np.asarray(table[L["iceberg"]]["approx_ratio"], float) - np.asarray(table[L["bare"]]["approx_ratio"], float)
        em = np.asarray(table[L["iceberg"]]["em_single"], float) - np.asarray(table[L["bare"]]["em_single"], float)
    except KeyError:
        ar = em = None
    if ar is None or d3 is None:
        missing("H4", stmt, f"lanes {L['iceberg']} and {L['bare']}")
    else:
        ok = ~(np.isnan(ar) | np.isnan(em))
        d4 = paired_bootstrap_diff(ar[ok], em[ok], n_boot=n_boot, seed=seed)
        d_em = _diff(table, (L["iceberg"], "em_single"), (L["bare"], "em_single"), n_boot, seed)
        rec = {"id": "H4", "statement": stmt,
               "quantity": f"[AR({L['iceberg']}) - AR({L['bare']})] - [EM({L['iceberg']}) - EM({L['bare']})]",
               **d4, "n": int(ok.sum()), "solver_gain": d3["mean_diff"],
               "accuracy_gain": d_em["mean_diff"] if d_em else float("nan"),
               "accuracy_gain_lo": d_em["lo"] if d_em else float("nan"),
               "accuracy_gain_hi": d_em["hi"] if d_em else float("nan")}
        if out[-1]["verdict"] != "supported":
            rec["verdict"] = "not applicable"
            rec["detail"] = "H3 is not supported, so there is no solver gain to pass on"
        else:
            rec["verdict"] = _verdict(d4["lo"], d4["hi"])
        out.append(rec)

    # ---- H5
    stmt = "The vote helps: 3-selection vote beats the single best selection"
    d = _diff(table, (L["exact"], "em_vote"), (L["exact"], "em_single"), n_boot, seed)
    if d is None:
        missing("H5", stmt, f"lane {L['exact']}")
    else:
        out.append({"id": "H5", "statement": stmt, "quantity": f"EM vote({L['exact']}) - EM single({L['exact']})",
                    **d, "verdict": _verdict(d["lo"], d["hi"])})
    return out


def hypotheses_markdown(records: Sequence[Mapping[str, Any]], title: str = "Hypothesis report") -> str:
    """Render the verdicts as a short Markdown report."""
    lines = [f"# {title}", "", "Judged as pre-registered in `HYPOTHESES.md`. Differences are in points on a 0 to 1 scale.", "",
             "| | Prediction | Result | 95% interval | Verdict |", "|---|---|---|---|---|"]
    for r in records:
        if r["verdict"] == "not evaluated":
            lines.append(f"| {r['id']} | {r['statement']} | | | not evaluated ({r['detail']}) |")
            continue
        result = f"{r['mean_diff']:+.3f}"
        if r["id"] == "H2":
            result = f"hit rate {r['hit_rate']:.3f} [{r['hit_lo']:.3f}, {r['hit_hi']:.3f}]; EM diff {r['mean_diff']:+.3f}"
        if r["id"] == "H4":
            result = f"solver gain {r['solver_gain']:+.4f}, accuracy gain {r['accuracy_gain']:+.3f}"
        verdict = r["verdict"] + (f" ({r['detail']})" if r.get("detail") else "")
        lines.append(f"| {r['id']} | {r['statement']} | {result} | [{r['lo']:+.3f}, {r['hi']:+.3f}] | **{verdict}** |")
    lines.append("")
    return "\n".join(lines)
