"""Stark map of caesium near n = 60: how a DC electric field moves the levels.

Why it matters for the project: a small stray field shifts the Rydberg level and
adds a phase error to every CZ. Phase errors on data qubits are what the all-X
stabilizer check of the Iceberg code detects.

Computed with ARC: ``StarkMap(Caesium())``, ``defineBasis``, ``diagonalise``,
``getPolarizability``. Left panel: low-l states (S, P, D) shift quadratically while
the high-l manifold fans out linearly. Right panel: the shift of the nS level itself
at small fields with the quadratic law  -alpha E^2 / 2  fitted by ARC.

The level n = 60 is ILLUSTRATIVE. We do not know which levels Sqale uses, and no
stray-field value for Sqale is assumed anywhere.

Run:  python physics/stark_map.py          (about 2 minutes the first time)
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as C
from arc import Caesium, StarkMap

from _style import ACCENT, FAINT, INK, MUTED, apply_style, save

N = 60                      # illustrative
N_MIN, N_MAX = 54, 61       # basis: every l for these n, |mj| = 1/2
FIELD_MAX_V_PER_CM = 3.0
FIELD_POINTS = 151
ZOOM_MAX_V_PER_CM = 0.3
ZOOM_POINTS = 31
FIT_MAX_V_PER_CM = 0.1      # range of the quadratic fit
WINDOW_GHZ = (-25.0, 23.0)  # energy window of the left panel, relative to nS at zero field
QUADRATIC_TOLERANCE = 0.05  # "quadratic region": within 5% of the fitted law
L_NAMES = "SPD"


def follow_state(sm: StarkMap, e0_ghz: float) -> tuple[np.ndarray, np.ndarray]:
    """Energy shift (GHz) and overlap of the eigenstate that is most nS-like at each field."""
    overlap = np.array(sm.highlight)
    idx = overlap.argmax(axis=1)
    shift = np.array([sm.y[i][idx[i]] for i in range(len(sm.y))]) - e0_ghz
    return shift, overlap[np.arange(len(idx)), idx]


def main() -> dict:
    t0 = time.time()
    apply_style()
    atom = Caesium()
    to_ghz = C.e / C.h * 1e-9
    e0 = atom.getEnergy(N, 0, 0.5) * to_ghz

    sm = StarkMap(atom)
    sm.defineBasis(N, 0, 0.5, 0.5, N_MIN, N_MAX, N_MAX)
    n_basis = len(sm.basisStates)

    # ---- low-field run: polarizability and the zoom panel
    zoom_fields = np.linspace(0.0, ZOOM_MAX_V_PER_CM, ZOOM_POINTS)          # V/cm
    sm.diagonalise(zoom_fields * 100.0)                                     # ARC wants V/m
    alpha = float(sm.getPolarizability(maxField=FIT_MAX_V_PER_CM * 100.0))  # MHz cm^2 / V^2
    zoom_shift_mhz, _ = follow_state(sm, e0)
    zoom_shift_mhz = zoom_shift_mhz * 1e3

    # ---- wide run: the map itself
    fields = np.linspace(0.0, FIELD_MAX_V_PER_CM, FIELD_POINTS)             # V/cm
    sm.diagonalise(fields * 100.0)
    levels = np.array(sm.y) - e0                                            # (fields, states), GHz
    shift_ghz, overlap = follow_state(sm, e0)
    quad_ghz = -0.5 * alpha * fields**2 * 1e-3
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_dev = np.abs(shift_ghz - quad_ghz) / np.abs(quad_ghz)
    bad = np.flatnonzero((rel_dev[1:] > QUADRATIC_TOLERANCE) | (overlap[1:] < 0.5)) + 1   # grid indices
    quad_holds_everywhere = len(bad) == 0
    # last grid point before the nS level first leaves the tolerance band
    quad_limit = float(fields[-1]) if quad_holds_everywhere else float(fields[bad[0] - 1])

    # field at which the nS level first meets the fan of the manifold just above it
    idx = np.array(sm.highlight).argmax(axis=1)
    gap_above = np.array([sm.y[i][idx[i] + 1] - sm.y[i][idx[i]] for i in range(len(fields))])
    near = np.flatnonzero(gap_above < 0.05 * gap_above[0])
    first_meet = float(fields[near[0]]) if len(near) else None

    fig, (ax, az) = plt.subplots(1, 2, figsize=(13.0, 6.2), gridspec_kw={"width_ratios": [1.55, 1]})

    # ---- left: the Stark map
    lo, hi = WINDOW_GHZ
    keep = np.flatnonzero((levels.min(axis=0) < hi) & (levels.max(axis=0) > lo))
    for k in keep:
        ax.plot(fields, levels[:, k], color=MUTED, lw=0.45, alpha=0.85)
    ax.plot(fields, shift_ghz, color=ACCENT, lw=2.2)
    ax.axvspan(0, quad_limit, color=FAINT, alpha=0.4, lw=0, zorder=0)
    ax.text(0.04, 8.2, f"{N}S stays quadratic within\n{QUADRATIC_TOLERANCE:.0%} up to {quad_limit:.2f} V/cm",
            fontsize=9.5, color=ACCENT, va="bottom", ha="left")
    ax.set_xlim(0, FIELD_MAX_V_PER_CM)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("DC electric field (V/cm)")
    ax.set_ylabel(f"Energy relative to {N}S at zero field (GHz)")
    ax.set_title("Low-l states bend, the high-l manifold fans out")

    def label(y, text, colour=INK, dy=0.0, x=0.04):
        ax.text(x, y + dy, text, fontsize=9.5, color=colour, va="bottom", ha="left",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85})

    label(0.0, f"{N}S (illustrative)", ACCENT, dy=-2.6)
    for n_l, l in ((N - 1, 1), (N, 1), (N - 2, 2), (N - 1, 2)):
        y = (atom.getEnergy(n_l, l, l + 0.5) * to_ghz) - e0
        if lo < y < hi:
            label(y, f"{n_l}{L_NAMES[l]}", dy=0.5)
    n_manifold = int(round(N - atom.getQuantumDefect(N, 0, 0.5)))
    y_manifold = (atom.getEnergy(n_manifold, n_manifold - 1, n_manifold - 0.5) * to_ghz) - e0
    ax.annotate(f"n = {n_manifold} manifold, l of 3 and above", xy=(1.2, y_manifold + 4.5),
                xytext=(1.25, 13.0), fontsize=9.5, color=INK,
                arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.7})

    # ---- right: the nS level at small fields
    grid = np.linspace(0, ZOOM_MAX_V_PER_CM, 200)
    az.axvspan(0, FIT_MAX_V_PER_CM, color=FAINT, alpha=0.45, lw=0)
    az.plot(grid, -0.5 * alpha * grid**2, color=MUTED, lw=1.4, label=r"quadratic law, $-\alpha E^2/2$")
    az.plot(zoom_fields, zoom_shift_mhz, "o", color=ACCENT, ms=4.5, label="full diagonalisation")
    az.set_xlim(0, ZOOM_MAX_V_PER_CM)
    az.set_xlabel("DC electric field (V/cm)")
    az.set_ylabel(f"Shift of the {N}S level (MHz)")
    az.set_title("A stray field shifts the Rydberg level")
    az.legend(loc="lower left", fontsize=9.5)
    az.text(FIT_MAX_V_PER_CM / 2, 0.62 * zoom_shift_mhz.min(), "range of\nthe fit", ha="center", va="center",
            fontsize=9.5, color=INK)
    shift_at_01 = -0.5 * alpha * 0.1**2
    az.annotate(f"{shift_at_01:.2f} MHz at 0.1 V/cm", xy=(0.1, shift_at_01),
                xytext=(0.012, 0.30 * zoom_shift_mhz.min()), fontsize=9.5, color=INK, ha="left", va="top",
                arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.7})
    az.text(0.97, 0.95, f"fitted polarizability\n{alpha:.0f} MHz cm$^2$ / V$^2$", transform=az.transAxes,
            ha="right", va="top", fontsize=10, color=INK)

    fig.suptitle(f"Stark map of caesium near n = {N} (illustrative level)", x=0.012, ha="left",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if quad_holds_everywhere:
        quad_sentence = (f"The {N}S shift stays within {QUADRATIC_TOLERANCE:.0%} of the quadratic law over the "
                         f"whole computed range, up to {FIELD_MAX_V_PER_CM:.1f} V/cm")
    else:
        quad_sentence = (f"The {N}S shift stays within {QUADRATIC_TOLERANCE:.0%} of the quadratic law up to "
                         f"{quad_limit:.2f} V/cm (shaded), where avoided crossings with the manifold start to "
                         f"mix the state")
    meet_sentence = (f". It first meets the fan of the n = {n_manifold} manifold near {first_meet:.2f} V/cm, "
                     f"in crossings too narrow to matter at that point." if first_meet is not None else ".")
    caption = (
        f"Caesium energy levels near {N}S (blue) against a DC electric field, computed with ARC by diagonalising "
        f"{n_basis} states (n = {N_MIN} to {N_MAX}, all l, |mj| = 1/2). Left: the low-l states ({N}S, "
        f"{N - 1}P, {N}P, {N - 2}D, {N - 1}D) shift quadratically, the high-l manifold fans out linearly. "
        f"Right: the {N}S shift at small fields with the quadratic law fitted below {FIT_MAX_V_PER_CM} V/cm, "
        f"giving a scalar polarizability of {alpha:.1f} MHz cm^2/V^2, so a field of 0.1 V/cm shifts the level by "
        f"{shift_at_01:.2f} MHz. {quad_sentence}{meet_sentence} The level n = {N} is illustrative: we do not know "
        f"which levels Sqale uses, and the field values are not Sqale numbers. A shift of the Rydberg level "
        f"during a gate is a phase error on the CZ."
    )
    data = {
        "n_illustrative": N,
        "basis": {"n_min": N_MIN, "n_max": N_MAX, "max_l": N_MAX, "mj": 0.5, "states": n_basis},
        "polarizability_mhz_cm2_v2": round(alpha, 3),
        "fit_max_v_per_cm": FIT_MAX_V_PER_CM,
        "shift_mhz_at_0p1_v_per_cm": round(shift_at_01, 4),
        "quadratic_within_tolerance_up_to_v_per_cm": round(quad_limit, 3),
        "quadratic_holds_over_whole_range": quad_holds_everywhere,
        "quadratic_tolerance": QUADRATIC_TOLERANCE,
        "first_meets_manifold_v_per_cm": None if first_meet is None else round(first_meet, 3),
        "manifold_n": n_manifold,
        "zoom_fields_v_per_cm": np.round(zoom_fields, 4).tolist(),
        "zoom_shift_mhz": np.round(zoom_shift_mhz, 4).tolist(),
    }
    save(fig, "stark_map", caption, data)

    print(f"Stark map, caesium {N}S (illustrative), basis of {n_basis} states:")
    print(f"  fitted scalar polarizability: {alpha:.2f} MHz cm^2/V^2 (fit below {FIT_MAX_V_PER_CM} V/cm)")
    print(f"  shift at 0.1 V/cm: {shift_at_01:.3f} MHz")
    print(f"  within {QUADRATIC_TOLERANCE:.0%} of the quadratic law up to {quad_limit:.2f} V/cm"
          f"{' (whole computed range)' if quad_holds_everywhere else ''}")
    print(f"  first meets the n = {n_manifold} manifold near {first_meet} V/cm")
    runtime = time.time() - t0
    print(f"stark_map.py done in {runtime:.1f} s")
    return {"figure": "stark_map.png", "runtime_s": round(runtime, 1), **data}


if __name__ == "__main__":
    main()
