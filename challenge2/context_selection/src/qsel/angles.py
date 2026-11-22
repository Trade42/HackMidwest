"""Angle tuning on the ideal tier.

Angles are tuned once per question on the exact simulator and then reused by the
physical and error-detected tiers, so those tiers differ only in noise and
encoding. Only the final sampling circuits ever go to Superstaq.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .qaoa import Angles, QaoaSimulator
from .qubo import Qubo


@dataclass(frozen=True)
class TunedAngles:
    angles: Angles
    expectation: float
    approx_ratio: float
    p_optimum: float
    evaluations: int

    def to_dict(self) -> dict:
        return {
            **self.angles.to_dict(),
            "expectation": self.expectation,
            "approx_ratio": self.approx_ratio,
            "p_optimum": self.p_optimum,
            "evaluations": self.evaluations,
        }


def tune_angles(
    qubo: Qubo,
    p: int = 1,
    *,
    grid_gamma: int = 49,
    grid_beta: int = 25,
    gamma_max: float = 2.5,
    restarts: int = 4,
    seed: int = 0,
    simulator: QaoaSimulator | None = None,
) -> TunedAngles:
    """Minimise the expected QUBO energy over QAOA angles.

    ``p = 1`` is a dense grid search followed by a local polish. Deeper circuits
    start from the ``p - 1`` solution (repeated last layer) plus a few random
    restarts, which is enough at these sizes and keeps tuning deterministic.
    """
    if p < 1:
        raise ValueError("p must be at least 1")
    sim = simulator or QaoaSimulator(qubo)
    evals = 0

    def objective(theta: np.ndarray) -> float:
        nonlocal evals
        evals += 1
        k = len(theta) // 2
        return sim.expectation(theta[:k] / sim.scale, theta[k:])

    # ---- p = 1: grid in natural units (gamma * scale), then polish
    g_grid = np.linspace(-gamma_max, gamma_max, grid_gamma)
    b_grid = np.linspace(0.0, np.pi / 2, grid_beta)
    surface = sim.grid_expectation(g_grid / sim.scale, b_grid)
    evals += surface.size
    gi, bi = np.unravel_index(int(np.argmin(surface)), surface.shape)
    best = minimize(objective, np.array([g_grid[gi], b_grid[bi]]), method="Nelder-Mead",
                    options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 400})
    theta, value = best.x, float(best.fun)

    # ---- p > 1: grow layer by layer
    rng = np.random.default_rng(seed)
    for depth in range(2, p + 1):
        k = depth - 1
        gam, bet = theta[:k], theta[k:]
        starts = [np.concatenate([np.append(gam, gam[-1]), np.append(bet, bet[-1] * 0.5)])]
        starts.append(np.concatenate([np.append(gam * 0.7, gam[-1]), np.append(bet, bet[-1] * 0.7)]))
        for _ in range(restarts):
            starts.append(np.concatenate([
                rng.uniform(-gamma_max, gamma_max, size=depth),
                rng.uniform(0.0, np.pi / 2, size=depth),
            ]))
        cand_theta, cand_value = None, np.inf
        for x0 in starts:
            res = minimize(objective, x0, method="Nelder-Mead",
                           options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 1200})
            if res.fun < cand_value:
                cand_theta, cand_value = res.x, float(res.fun)
        # never accept a deeper circuit that is worse than padding with identity
        padded = np.concatenate([np.append(gam, 0.0), np.append(bet, 0.0)])
        if cand_value > value:
            cand_theta, cand_value = padded, value
        theta, value = cand_theta, cand_value

    k = len(theta) // 2
    angles = Angles(tuple(theta[:k] / sim.scale), tuple(theta[k:]))
    probs = sim.probabilities(angles)
    return TunedAngles(
        angles=angles,
        expectation=value,
        approx_ratio=sim.approx_ratio(value),
        p_optimum=float(probs[sim.optimum_mask()].sum()),
        evaluations=evals,
    )
