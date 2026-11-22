"""Experiment configuration: one YAML file per experiment, merged over defaults."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULTS: dict[str, Any] = {
    "name": "experiment",
    "seed": 0,
    "data": {
        "questions": "data/questions.jsonl",
        "calibration": "data/calibration.jsonl",
        "demo": "data/demo.jsonl",
        "limit": None,                 # evaluate only the first N questions
    },
    "scores": {"dir": "scores", "scorer": {"kind": "tfidf"}},
    "qubo": {"tau": 0.08, "theta": 0.0, "penalty_scale": 1.5, "normalize": "zscore", "max_edges": None},
    "qaoa": {
        "p": 1,                        # layers used by every quantum tier
        "p_ideal_extra": 2,            # extra ideal-only lane for hypothesis H2 (null to skip)
        "shots": 4000,
        "top": 3,                      # selections fed to the vote
        "min_select": 1,
        "shot_budgets": [64],          # equal-raw-shot comparisons, subsampled from the full run
    },
    "iceberg": {"style": "bus"},
    "top_k": 3,
    "lm": {"kind": "stub"},
    "backends": {
        "physical": {"kind": "local-pauli", "noise": {"p1": 0.0005, "p2": 0.005, "p_meas": 0.005}},
        "annealer": {"kind": "local"},   # "superstaq" uses submit_qubo
        "hardware": None,                # e.g. {"kind": "superstaq", "target": "cq_sqale_qpu", "limit": 30}
    },
    "lanes": ["1a", "1b", "2a", "2b", "2c", "rand", "3a", "3b", "3c"],
    "bootstrap": {"n_boot": 10000, "seed": 0},
    "output": "results/experiment",
    "figures": "figures/experiment",
}


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML experiment file and fill in defaults."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = deep_merge(DEFAULTS, raw)
    cfg["config_path"] = str(path)
    if "name" not in raw:
        cfg["name"] = path.stem
    if "output" not in raw:
        cfg["output"] = f"results/{cfg['name']}"
    if "figures" not in raw:
        cfg["figures"] = f"figures/{cfg['name']}"
    return cfg
