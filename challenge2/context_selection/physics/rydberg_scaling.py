"""How caesium nS Rydberg properties scale with the principal quantum number n.

Why it matters for the project: choosing n trades a stronger blockade (C6) against
more sensitivity to stray electric fields (polarizability) and a different decay
rate (lifetime). That trade-off sets the gate error that error detection has to catch.

Computed with ARC for n = 30 .. 100:

* lifetime            ``getStateLifetime``  (0 K radiative, plus 300 K with black-body radiation)
* polarizability      ``StarkMap(...).getPolarizability``  (fit of the DC Stark shift)
* C6 (Cs nS + Cs nS)  ``PairStateInteractions(...).getC6perturbatively``
* level spacing       ``getEnergy(n + 1) - getEnergy(n)``

Each series gets a power-law fit on log-log axes. The fitted exponents are printed
next to the textbook ones and reported as they come out, including when they differ.
The textbook exponents hold for the effective quantum number n* = n - delta
(delta is the quantum defect, about 4.05 for caesium nS), so both fits are reported.

No particular n is implied for Sqale: the whole range is illustrative.

Run:  python physics/rydberg_scaling.py
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as C
from arc import Caesium, PairStateInteractions, StarkMap

from _style import ACCENT, INK, MUTED, apply_style, save

N_VALUES = np.arange(30, 101, 5)
TEMPERATURE_K = 300.0            # for the black-body series only
BBR_LEVELS_ABOVE = 20            # levels above n included in black-body transfer
STARK_BASIS_DN = 4               # basis n - 4 .. n + 4 for the polarizability
STARK_BASIS_MAXL = 6
STARK_FIT_FRACTION = 0.05        # fit up to 5% of the Inglis-Teller field 1 / (3 n^5) a.u.
C6_N_RANGE = 5
C6_ENERGY_DELTA_HZ = 25e9
ATOMIC_FIELD_V_PER_M = 5.14220674763e11


def power_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least squares on log-log axes. Returns (exponent, prefactor)."""
    slope, intercept = np.polyfit(np.log(x), np.log(np.abs(y)), 1)
    return float(slope), float(np.exp(intercept))


def compute() -> dict:
    atom = Caesium()
    n = N_VALUES
    n_star = np.array([nn - atom.getQuantumDefect(int(nn), 0, 0.5) for nn in n])

    lifetime_0k = np.array([atom.getStateLifetime(int(nn), 0, 0.5) for nn in n]) * 1e6          # microseconds
    lifetime_300k = np.array([
        atom.getStateLifetime(int(nn), 0, 0.5, temperature=TEMPERATURE_K,
                              includeLevelsUpTo=int(nn) + BBR_LEVELS_ABOVE) for nn in n
    ]) * 1e6

    polarizability = []
    for nn in n:
        nn = int(nn)
        fit_max = STARK_FIT_FRACTION * ATOMIC_FIELD_V_PER_M / (3 * nn**5)                       # V/m
        sm = StarkMap(atom)
        sm.defineBasis(nn, 0, 0.5, 0.5, nn - STARK_BASIS_DN, nn + STARK_BASIS_DN, STARK_BASIS_MAXL)
        sm.diagonalise(np.linspace(0.0, fit_max, 30))
        polarizability.append(sm.getPolarizability(maxField=fit_max))                            # MHz cm^2 / V^2
    polarizability = np.array(polarizability)

    c6 = np.array([
        PairStateInteractions(atom, int(nn), 0, 0.5, int(nn), 0, 0.5, 0.5, 0.5)
        .getC6perturbatively(0, 0, C6_N_RANGE, C6_ENERGY_DELTA_HZ) for nn in n
    ])                                                                                           # GHz um^6

    spacing = np.array([
        (atom.getEnergy(int(nn) + 1, 0, 0.5) - atom.getEnergy(int(nn), 0, 0.5)) * C.e / C.h * 1e-9 for nn in n
    ])                                                                                           # GHz

    return {
        "n": n, "n_star": n_star, "lifetime_0k_us": lifetime_0k, "lifetime_300k_us": lifetime_300k,
        "polarizability_mhz_cm2_v2": polarizability, "c6_ghz_um6": c6, "spacing_ghz": spacing,
    }


PANELS = [
    # key, title, y label, textbook exponent
    ("lifetime_0k_us", "Lifetime grows", "Lifetime (microseconds)", 3),
    ("polarizability_mhz_cm2_v2", "Stray-field sensitivity grows much faster",
     "Scalar polarizability (MHz cm$^2$ / V$^2$)", 7),
    ("c6_ghz_um6", "Blockade strength grows fastest", "|C$_6$| (GHz $\\mu$m$^6$)", 11),
    ("spacing_ghz", "Neighbouring levels crowd together", "Spacing to the next nS level (GHz)", -3),
]


def main() -> dict:
    t0 = time.time()
    apply_style()
    d = compute()
    n, n_star = d["n"], d["n_star"]

    fits: dict[str, dict] = {}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))
    for ax, (key, title, ylabel, textbook) in zip(axes.ravel(), PANELS):
        y = np.abs(d[key])
        exp_n, pre_n = power_fit(n, y)
        exp_ns, _ = power_fit(n_star, y)
        fits[key] = {"fit_vs_n": round(exp_n, 3), "fit_vs_n_star": round(exp_ns, 3), "textbook": textbook}

        grid = np.linspace(n.min(), n.max(), 200)
        ax.plot(grid, pre_n * grid**exp_n, color=MUTED, lw=1.2, zorder=1)
        main_pts, = ax.plot(n, y, "o", color=ACCENT, ms=5.5, zorder=2)
        if key == "lifetime_0k_us":
            y300 = d["lifetime_300k_us"]
            exp300, pre300 = power_fit(n, y300)
            exp300_ns, _ = power_fit(n_star, y300)
            fits["lifetime_300k_us"] = {"fit_vs_n": round(exp300, 3), "fit_vs_n_star": round(exp300_ns, 3),
                                        "textbook": None}
            ax.plot(grid, pre300 * grid**exp300, color=MUTED, lw=1.0, ls=(0, (3, 2)), zorder=1)
            bbr_pts, = ax.plot(n, y300, "s", mfc="white", mec=INK, ms=5, zorder=2)
            ax.legend([main_pts, bbr_pts],
                      ["radiative decay only (0 K)",
                       f"with {TEMPERATURE_K:.0f} K black-body radiation, fit $n^{{{exp300:.2f}}}$"],
                      loc="lower right", fontsize=9.5, handletextpad=0.4, borderaxespad=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([30, 40, 50, 60, 80, 100])
        ax.set_xticklabels(["30", "40", "50", "60", "80", "100"])
        ax.set_xlabel("Principal quantum number n")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.text(0.04, 0.95 if textbook > 0 else 0.12,
                f"fit: $n^{{{exp_n:.2f}}}$\ntextbook: $n^{{{textbook}}}$",
                transform=ax.transAxes, va="top" if textbook > 0 else "bottom", ha="left", fontsize=10.5, color=INK)
        ax.grid(True, which="major", color="#e6e9ed", lw=0.6)
        ax.set_axisbelow(True)

    fig.suptitle("Caesium nS Rydberg states: what a higher n buys and what it costs (illustrative range)",
                 x=0.012, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.955))

    c6_sign = "repulsive" if np.all(d["c6_ghz_um6"] < 0) else "mixed sign"
    high = n >= 50
    c6_high_exp, _ = power_fit(n_star[high], d["c6_ghz_um6"][high])
    fits["c6_ghz_um6"]["fit_vs_n_star_n_ge_50"] = round(c6_high_exp, 3)
    f = fits
    caption = (
        f"Caesium nS1/2 properties for n = {n.min()} to {n.max()}, computed with ARC. Points are computed values, "
        f"grey lines are power-law fits against n. Fitted exponents against n (and against the effective quantum "
        f"number n* = n minus the quantum defect, where the textbook scalings apply): radiative lifetime at 0 K "
        f"{f['lifetime_0k_us']['fit_vs_n']:.2f} ({f['lifetime_0k_us']['fit_vs_n_star']:.2f}), textbook 3; "
        f"scalar polarizability {f['polarizability_mhz_cm2_v2']['fit_vs_n']:.2f} "
        f"({f['polarizability_mhz_cm2_v2']['fit_vs_n_star']:.2f}), textbook 7; "
        f"C6 {f['c6_ghz_um6']['fit_vs_n']:.2f} ({f['c6_ghz_um6']['fit_vs_n_star']:.2f}), textbook 11; "
        f"level spacing {f['spacing_ghz']['fit_vs_n']:.2f} ({f['spacing_ghz']['fit_vs_n_star']:.2f}), textbook -3. "
        f"With {TEMPERATURE_K:.0f} K black-body radiation included the lifetime grows more slowly, as n to the "
        f"{f['lifetime_300k_us']['fit_vs_n']:.2f}. C6 is for two caesium atoms in the same nS state "
        f"(V = -C6/R^6, {c6_sign} over this range). Its low end falls below a single power law; fitting only "
        f"n of 50 and above gives {c6_high_exp:.2f} against n*. No specific n is implied for Sqale."
    )
    data = {
        "n": n.tolist(),
        "n_star": np.round(n_star, 4).tolist(),
        "lifetime_0k_us": np.round(d["lifetime_0k_us"], 3).tolist(),
        "lifetime_300k_us": np.round(d["lifetime_300k_us"], 3).tolist(),
        "polarizability_mhz_cm2_v2": np.round(d["polarizability_mhz_cm2_v2"], 4).tolist(),
        "c6_ghz_um6": [float(f"{v:.5g}") for v in d["c6_ghz_um6"]],
        "spacing_ghz": np.round(d["spacing_ghz"], 4).tolist(),
        "fits": fits,
        "assumptions": {
            "temperature_K_for_blackbody_series": TEMPERATURE_K,
            "stark_fit_up_to_fraction_of_inglis_teller_field": STARK_FIT_FRACTION,
            "stark_basis": f"n-{STARK_BASIS_DN}..n+{STARK_BASIS_DN}, l <= {STARK_BASIS_MAXL}",
            "c6": f"theta=0, nRange={C6_N_RANGE}, energyDelta={C6_ENERGY_DELTA_HZ:.0e} Hz",
        },
    }
    save(fig, "rydberg_scaling", caption, data)

    print("Power-law exponents for caesium nS, n = 30..100 (fit vs n | fit vs n* | textbook):")
    for key, label in (("lifetime_0k_us", "lifetime, 0 K"), ("lifetime_300k_us", "lifetime, 300 K"),
                       ("polarizability_mhz_cm2_v2", "polarizability"), ("c6_ghz_um6", "C6"),
                       ("spacing_ghz", "level spacing")):
        r = fits[key]
        print(f"  {label:<16} {r['fit_vs_n']:>7.2f} | {r['fit_vs_n_star']:>7.2f} | {r['textbook']}")
    i60 = int(np.where(n == 60)[0][0])
    print(f"  at n = 60: lifetime {d['lifetime_0k_us'][i60]:.1f} us (0 K), {d['lifetime_300k_us'][i60]:.1f} us "
          f"(300 K); polarizability {d['polarizability_mhz_cm2_v2'][i60]:.1f} MHz cm^2/V^2; "
          f"C6 {d['c6_ghz_um6'][i60]:.1f} GHz um^6; spacing {d['spacing_ghz'][i60]:.2f} GHz")
    runtime = time.time() - t0
    print(f"rydberg_scaling.py done in {runtime:.1f} s")
    return {"figure": "rydberg_scaling.png", "runtime_s": round(runtime, 1), "fits": fits}


if __name__ == "__main__":
    main()
