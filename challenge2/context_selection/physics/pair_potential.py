"""Caesium-rubidium Rydberg pair potential and the blockade radius.

Why it matters for the project: the blockade is what makes each CZ, and it is
short-range. That is why the Iceberg code's two hub qubits (t and b), which must
interact with every data qubit, need routing on a grid, unlike on trapped ions.

Computed with ARC: ``PairStateInteractions(Caesium(), n, 0, 1/2, n, 0, 1/2, m1, m2,
atom2=Rubidium87())``, then ``defineBasis`` and ``diagonalise`` for the full level
diagram, ``getC6perturbatively`` for the van der Waals curve V = -C6 / R^6, and
``getLeRoyRadius`` for where the calculation stops being valid.

Assumptions, all stated on the figure or in the caption:
* the pair state Cs 60S + Rb 60S is ILLUSTRATIVE (we do not know Sqale's levels);
* the laser drive (Rabi frequency) is ASSUMED, it is not a Sqale number;
* 5.5 um is the Rb-Cs spacing quoted in the plan's sources (Infleqtion, "A Rb-Cs
  Quantum Computing Testbed", APS March Meeting 2026).

Run:  python physics/pair_potential.py
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
from arc import Caesium, PairStateInteractions, Rubidium87

from _style import ACCENT, ACCENT_2, FAINT, INK, MUTED, apply_style, save

N = 60                         # illustrative principal quantum number for both atoms
RABI_MHZ = 1.0                 # ASSUMED drive, Omega / 2 pi
RABI_ALT_MHZ = 5.0             # second assumed drive, quoted in the caption only
SPACING_UM = 5.5               # cited Rb-Cs spacing
THETA = 0.0                    # interatomic axis along the quantisation axis
BASIS_N_RANGE = 4
BASIS_L_RANGE = 4
BASIS_ENERGY_DELTA_HZ = 15e9
EIGENVECTORS = 100
R_MIN_UM, R_MAX_UM, R_POINTS = 1.2, 14.0, 160
C6_N_RANGE = 5
C6_ENERGY_DELTA_HZ = 25e9
FULL_DIAGONALISATION_BUDGET_S = 300.0   # the plan's fallback: perturbative curve only if slower


def blockade_radius(c6_abs_ghz_um6: float, rabi_mhz: float) -> float:
    """Separation where |C6| / R^6 equals the drive (both as frequencies)."""
    return (c6_abs_ghz_um6 * 1e3 / rabi_mhz) ** (1.0 / 6.0)


def main() -> dict:
    t0 = time.time()
    apply_style()
    cs, rb = Caesium(), Rubidium87()
    calc = PairStateInteractions(cs, N, 0, 0.5, N, 0, 0.5, 0.5, 0.5, atom2=rb)
    c6 = float(calc.getC6perturbatively(THETA, 0.0, C6_N_RANGE, C6_ENERGY_DELTA_HZ))   # GHz um^6, V = -C6/R^6
    leroy = float(calc.getLeRoyRadius())                                                # um
    c6_abs = abs(c6)
    character = "repulsive" if c6 < 0 else "attractive"
    r_block = blockade_radius(c6_abs, RABI_MHZ)
    r_block_alt = blockade_radius(c6_abs, RABI_ALT_MHZ)
    v_spacing = c6_abs * 1e3 / SPACING_UM**6                                            # MHz
    v_double = c6_abs * 1e3 / (2 * SPACING_UM) ** 6

    # ---- full diagonalisation (falls back to the perturbative curve if too slow)
    r = np.linspace(R_MIN_UM, R_MAX_UM, R_POINTS)                       # ascending, micrometres
    t_diag = time.time()
    used_fallback = False
    try:
        calc.defineBasis(THETA, 0.0, BASIS_N_RANGE, BASIS_L_RANGE, BASIS_ENERGY_DELTA_HZ, progressOutput=False)
        n_basis = int(calc.matDiagonal.shape[0])
        calc.diagonalise(r, EIGENVECTORS, progressOutput=False)
        # ARC stores its results for separations sorted in DESCENDING order (calc.r),
        # so put them back in ascending order before pairing them with ``r``.
        order = np.argsort(np.asarray(calc.r))
        if not np.allclose(np.asarray(calc.r)[order], r):
            raise RuntimeError("separation grid returned by ARC does not match the requested one")
        energies = [np.array(calc.y[i]) * 1e3 for i in order]         # MHz, relative to the pair state
        overlaps = [np.array(calc.highlight[i]) for i in order]       # share of the original pair state
        diag_seconds = time.time() - t_diag
        if diag_seconds > FULL_DIAGONALISATION_BUDGET_S:
            print(f"note: full diagonalisation took {diag_seconds:.0f} s, above the budget")
    except Exception as exc:  # pragma: no cover - documented fallback path
        print(f"full diagonalisation failed ({exc!r}); using the perturbative curve only")
        used_fallback = True
        n_basis, energies, overlaps, diag_seconds = 0, [], [], time.time() - t_diag

    tracked = tracked_overlap = None
    r_block_diag = v_spacing_diag = None
    if not used_fallback:
        pick = [int(np.argmax(o)) for o in overlaps]
        tracked = np.array([energies[i][pick[i]] for i in range(len(r))])
        tracked_overlap = np.array([overlaps[i][pick[i]] for i in range(len(r))])
        clean = tracked_overlap > 0.5
        v_spacing_diag = float(np.interp(SPACING_UM, r[clean], np.abs(tracked[clean])))
        # outermost separation where the diagonalised shift still exceeds the drive
        above = np.flatnonzero(clean & (np.abs(tracked) >= RABI_MHZ))
        if len(above) and above[-1] + 1 < len(r):
            i = above[-1]
            r_block_diag = float(np.interp(RABI_MHZ, [abs(tracked[i + 1]), abs(tracked[i])], [r[i + 1], r[i]]))

    fig, (ax, az) = plt.subplots(1, 2, figsize=(13.0, 6.0), gridspec_kw={"width_ratios": [1.15, 1]})
    grid = np.linspace(R_MIN_UM, R_MAX_UM, 400)
    vdw = -c6 * 1e3 / grid**6                                         # MHz, signed

    # ---- left: the level diagram
    y_lim = 420.0
    if not used_fallback:
        for i, ri in enumerate(r):
            e, o = energies[i], overlaps[i]
            inside = np.abs(e) < y_lim
            ax.scatter(np.full(inside.sum(), ri), e[inside], s=2.2, color=MUTED, alpha=0.55, lw=0, zorder=1)
            strong = inside & (o > 0.05)
            if strong.any():
                ax.scatter(np.full(strong.sum(), ri), e[strong], s=4 + 26 * o[strong], color=ACCENT,
                           alpha=np.clip(0.25 + 0.75 * o[strong], 0, 1), lw=0, zorder=3)
    ax.plot(grid, vdw, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=2, label=r"van der Waals curve, $-C_6/R^6$")
    ax.axvspan(0, leroy, color=FAINT, alpha=0.7, lw=0)
    ax.text((0.6 + leroy) / 2, 0.0, "inside the Le Roy radius", fontsize=8.5, color=INK, ha="center",
            va="center", rotation=90)
    ax.axvline(SPACING_UM, color=ACCENT_2, lw=1.3)
    ax.text(SPACING_UM + 0.15, -y_lim * 0.58, f"{SPACING_UM} $\\mu$m\nRb-Cs spacing (cited)", color=ACCENT_2,
            fontsize=9.5, va="center")
    ax.set_xlim(0.6, R_MAX_UM)
    ax.set_ylim(-y_lim, y_lim)
    ax.set_xlabel("Distance between the two atoms ($\\mu$m)")
    ax.set_ylabel(f"Pair energy relative to Cs {N}S + Rb {N}S (MHz)")
    ax.set_title("Close together the pair state is pushed far off resonance")
    ax.legend(loc="upper right", fontsize=9.5)
    if not used_fallback:
        ax.text(0.985, 0.66, "blue: share of the\noriginal pair state", transform=ax.transAxes, ha="right",
                va="center", fontsize=9.5, color=ACCENT)

    # ---- right: where the interaction equals the drive
    az.plot(grid, np.abs(vdw), color=INK, lw=1.0, ls=(0, (4, 3)), label=r"van der Waals, $|C_6|/R^6$")
    if not used_fallback:
        ok = tracked_overlap > 0.5
        az.plot(r[ok], np.abs(tracked[ok]), "o", color=ACCENT, ms=3.6, label="full diagonalisation")
    az.axhline(RABI_MHZ, color=MUTED, lw=1.2)
    az.text(R_MAX_UM - 0.15, RABI_MHZ * 1.25, f"assumed drive, {RABI_MHZ:g} MHz", ha="right", va="bottom",
            fontsize=9.5, color=INK)
    az.axvline(SPACING_UM, color=ACCENT_2, lw=1.3)
    az.axvline(r_block, color=INK, lw=1.0, ls=":")
    az.set_yscale("log")
    az.set_xlim(2.0, R_MAX_UM)
    az.set_ylim(0.02, 3e3)
    az.text(SPACING_UM + 0.2, 1400, f"{SPACING_UM} $\\mu$m, cited Rb-Cs spacing", color=ACCENT_2, fontsize=9.5,
            ha="left", va="center")
    az.text(r_block + 0.2, 220, f"blockade radius\n{r_block:.1f} $\\mu$m for the assumed drive", color=INK,
            fontsize=9.5, ha="left", va="center")
    az.plot([SPACING_UM], [v_spacing], "o", color=ACCENT_2, ms=7, zorder=5)
    az.annotate(f"{v_spacing:.1f} MHz at {SPACING_UM} $\\mu$m", xy=(SPACING_UM, v_spacing),
                xytext=(2.25, 2.3), fontsize=9.5, color=ACCENT_2, ha="left", va="center",
                arrowprops={"arrowstyle": "-", "color": ACCENT_2, "lw": 0.7})
    az.plot([2 * SPACING_UM], [v_double], "o", mfc="white", mec=ACCENT_2, ms=7, zorder=5)
    az.annotate(f"{v_double:.2f} MHz at {2 * SPACING_UM:g} $\\mu$m", xy=(2 * SPACING_UM, v_double),
                xytext=(13.85, 0.42), fontsize=9.5, color=ACCENT_2, ha="right", va="center",
                arrowprops={"arrowstyle": "-", "color": ACCENT_2, "lw": 0.7})
    az.set_xlabel("Distance between the two atoms ($\\mu$m)")
    az.set_ylabel("Interaction strength (MHz)")
    az.set_title("The blockade is short-range")
    az.legend(loc="lower left", fontsize=9.5)

    fig.suptitle(f"Cs-Rb Rydberg pair potential, Cs {N}S + Rb {N}S (illustrative levels)", x=0.012, ha="left",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    method = ("perturbative van der Waals curve only (full diagonalisation did not run)" if used_fallback else
              f"full diagonalisation of {n_basis} pair states (blue shows the share of the original pair state) "
              f"with the perturbative van der Waals curve on top")
    diag_note = "" if used_fallback or v_spacing_diag is None else (
        f" The diagonalised shift at {SPACING_UM} um is {v_spacing_diag:.1f} MHz"
        + (f" and it crosses the assumed drive at {r_block_diag:.1f} um." if r_block_diag else "."))
    caption = (
        f"Interaction between a caesium atom and a rubidium-87 atom, both in {N}S1/2 (illustrative levels, "
        f"m = 1/2 each, interatomic axis along the quantisation axis), computed with ARC: {method}. "
        f"C6 = {c6:.1f} GHz um^6 with V = -C6/R^6, so the interaction is {character}. "
        f"At the cited {SPACING_UM} um Rb-Cs spacing the van der Waals shift is {v_spacing:.1f} MHz; at twice "
        f"that distance it is {v_double:.2f} MHz, 64 times weaker.{diag_note} "
        f"For an ASSUMED laser drive of {RABI_MHZ:g} MHz (Rabi frequency over 2 pi) the blockade radius, where the "
        f"interaction equals the drive, is {r_block:.1f} um; for an assumed {RABI_ALT_MHZ:g} MHz it is "
        f"{r_block_alt:.1f} um. The Le Roy radius, below which the calculation is not valid, is {leroy:.2f} um. "
        f"The drive is an assumption and the levels are illustrative: neither is a Sqale number. Only close "
        f"neighbours blockade each other, so a qubit that must interact with every data qubit needs routing on a grid."
    )
    data = {
        "pair_state_illustrative": f"Cs {N}S1/2 + Rb87 {N}S1/2, m1 = m2 = 1/2, theta = {THETA}",
        "c6_ghz_um6": round(c6, 3),
        "convention": "V = -C6 / R^6",
        "character": character,
        "le_roy_radius_um": round(leroy, 3),
        "assumed_rabi_mhz": RABI_MHZ,
        "blockade_radius_um": round(r_block, 3),
        "assumed_rabi_alt_mhz": RABI_ALT_MHZ,
        "blockade_radius_alt_um": round(r_block_alt, 3),
        "cited_spacing_um": SPACING_UM,
        "vdw_shift_mhz_at_spacing": round(v_spacing, 3),
        "vdw_shift_mhz_at_twice_spacing": round(v_double, 4),
        "full_diagonalisation": not used_fallback,
        "basis_pair_states": n_basis,
        "basis": f"nRange={BASIS_N_RANGE}, lrange={BASIS_L_RANGE}, energyDelta={BASIS_ENERGY_DELTA_HZ:.0e} Hz",
        "diagonalisation_seconds": round(diag_seconds, 1),
        "diagonalised_shift_mhz_at_spacing": None if v_spacing_diag is None else round(v_spacing_diag, 3),
        "diagonalised_blockade_radius_um": None if r_block_diag is None else round(r_block_diag, 3),
    }
    save(fig, "pair_potential", caption, data)

    print(f"Cs {N}S + Rb {N}S pair potential (illustrative levels):")
    print(f"  C6 = {c6:.2f} GHz um^6 (V = -C6/R^6, {character}); Le Roy radius {leroy:.2f} um")
    print(f"  van der Waals shift at {SPACING_UM} um: {v_spacing:.2f} MHz; at {2 * SPACING_UM} um: {v_double:.3f} MHz")
    print(f"  blockade radius for an ASSUMED {RABI_MHZ:g} MHz drive: {r_block:.2f} um "
          f"(for {RABI_ALT_MHZ:g} MHz: {r_block_alt:.2f} um)")
    if not used_fallback:
        print(f"  full diagonalisation: {n_basis} pair states, {diag_seconds:.1f} s; shift at {SPACING_UM} um "
              f"{v_spacing_diag:.2f} MHz; crosses the drive at {r_block_diag} um")
    else:
        print("  FALLBACK used: perturbative curve only")
    runtime = time.time() - t0
    print(f"pair_potential.py done in {runtime:.1f} s")
    return {"figure": "pair_potential.png", "runtime_s": round(runtime, 1), **data}


if __name__ == "__main__":
    main()
