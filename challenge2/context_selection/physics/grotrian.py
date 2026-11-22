"""Grotrian diagrams for caesium and rubidium-87 with the two-photon Rydberg path.

Why it matters for the project: during a CZ gate only the qubit's |1> state is
excited along this path, so it is the physical step behind every two-qubit gate
in our circuits.

Everything drawn or printed is computed with ARC (``getEnergy``,
``getTransitionWavelength``). The plan's nominal wavelengths are only used to
pick which fine-structure level of the intermediate P state they refer to, and
are printed next to the computed values for comparison.

The Rydberg level (n = 60) is ILLUSTRATIVE. We do not know which levels Sqale uses.

Run:  python physics/grotrian.py
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
from arc import Caesium, LevelPlot, Rubidium87
from matplotlib.patches import FancyArrowPatch

from _style import ACCENT, ACCENT_2, FAINT, INK, MUTED, apply_style, save

ILLUSTRATIVE_N = 60     # assumption, labelled on the figure
N_LABELLED = 3          # lowest levels per column that get a text label
N_MAX_DRAWN = 40        # the series is drawn up to here, then it merges with the limit
L_NAMES = "SPDF"

SPECIES = [
    {
        "atom": Caesium, "title": "Caesium (data qubits)", "symbol": "Cs",
        "ground": (6, 0, 0.5), "mid_n": 7, "nominal_nm": (459, 1040),
        "n_start": {0: 6, 1: 6, 2: 5, 3: 4},
    },
    {
        "atom": Rubidium87, "title": "Rubidium-87 (ancilla qubits)", "symbol": "Rb",
        "ground": (5, 0, 0.5), "mid_n": 6, "nominal_nm": (421, 1005),
        "n_start": {0: 5, 1: 5, 2: 4, 3: 4},
    },
]


def path_wavelengths(atom, ground, mid_n, n_ryd):
    """Both candidate two-photon paths (via P1/2 and via P3/2), in nm."""
    out = {}
    for j_mid in (0.5, 1.5):
        first = abs(atom.getTransitionWavelength(*ground, mid_n, 1, j_mid)) * 1e9
        second = abs(atom.getTransitionWavelength(mid_n, 1, j_mid, n_ryd, 0, 0.5)) * 1e9
        out[j_mid] = (first, second)
    return out


def draw_species(ax, spec) -> dict:
    atom = spec["atom"]()
    ground = spec["ground"]
    e_ground = atom.getEnergy(*ground)          # eV, negative, relative to the ionisation limit
    e_ion = -e_ground                           # ionisation energy above the ground state

    def above_ground(n, l, j):
        return atom.getEnergy(n, l, j) - e_ground

    # ---- which intermediate level do the plan's nominal wavelengths refer to?
    paths = path_wavelengths(atom, ground, spec["mid_n"], ILLUSTRATIVE_N)
    j_mid = min(paths, key=lambda j: abs(paths[j][0] - spec["nominal_nm"][0]))
    lam1, lam2 = paths[j_mid]

    # ---- levels: one column per orbital angular momentum, fine structure side by side
    half = 0.36
    for l in range(4):
        js = (0.5,) if l == 0 else (l - 0.5, l + 0.5)
        for n in range(spec["n_start"][l], N_MAX_DRAWN + 1):
            for k, j in enumerate(js):
                y = above_ground(n, l, j)
                if len(js) == 1:
                    x0, x1 = l - half, l + half
                else:
                    x0, x1 = (l - half, l - 0.03) if k == 0 else (l + 0.03, l + half)
                rank = n - spec["n_start"][l]
                ax.plot([x0, x1], [y, y], color=INK if rank < 8 else MUTED,
                        lw=1.4 if rank < 8 else 0.6, solid_capstyle="butt")
            if n - spec["n_start"][l] < N_LABELLED:
                y_lab = above_ground(n, l, js[-1])
                ax.text(l + half + 0.05, y_lab, f"{n}{L_NAMES[l]}", va="center", ha="left",
                        fontsize=8.5, color=INK)
        if l > 0:
            ax.text(l - half / 2 - 0.01, -0.33, f"j={int(2 * js[0])}/2", ha="center", fontsize=7.5, color=MUTED)
            ax.text(l + half / 2 + 0.01, -0.33, f"j={int(2 * js[1])}/2", ha="center", fontsize=7.5, color=MUTED)
        else:
            ax.text(0, -0.33, "j=1/2", ha="center", fontsize=7.5, color=MUTED)

    # ---- ionisation limit and the illustrative Rydberg level
    ax.axhline(e_ion, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.text(3.45, e_ion + 0.05, f"ionisation limit, {e_ion:.3f} eV", ha="right", va="bottom",
            fontsize=8.5, color=MUTED)
    y_ryd = above_ground(ILLUSTRATIVE_N, 0, 0.5)
    ax.plot([-half, half], [y_ryd, y_ryd], color=ACCENT, lw=2.4, solid_capstyle="butt")
    binding_mev = -atom.getEnergy(ILLUSTRATIVE_N, 0, 0.5) * 1e3
    ax.annotate(
        f"{ILLUSTRATIVE_N}S (illustrative), {binding_mev:.1f} meV below the limit",
        xy=(-half + 0.04, y_ryd), xytext=(-0.7, e_ion + 0.2), fontsize=8.5, color=ACCENT, ha="left", va="bottom",
        arrowprops={"arrowstyle": "-", "color": ACCENT, "lw": 0.7},
    )

    # ---- two-photon path: ground S -> n P_j -> n S Rydberg
    x_mid = 1 - half / 2 - 0.015 if j_mid == 0.5 else 1 + half / 2 + 0.015
    y_mid = above_ground(spec["mid_n"], 1, j_mid)
    for (xa, ya), (xb, yb), colour in (((0.12, 0.0), (x_mid, y_mid), ACCENT),
                                       ((x_mid, y_mid), (0.16, y_ryd), ACCENT_2)):
        ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle="-|>", mutation_scale=13,
                                     lw=1.9, color=colour, shrinkA=1.5, shrinkB=1.5, zorder=5))
    j_txt = f"{int(2 * j_mid)}/2"
    # Both labels sit in the empty lower-right corner so they never cover a level.
    ax.text(1.62, 1.02, f"First photon: {lam1:.1f} nm\n{ground[0]}S(1/2) to {spec['mid_n']}P({j_txt})",
            color=ACCENT, fontsize=9.5, ha="left", va="center", fontweight="bold")
    ax.text(1.62, 0.42, f"Second photon: {lam2:.1f} nm\n{spec['mid_n']}P({j_txt}) to {ILLUSTRATIVE_N}S(1/2)",
            color=ACCENT_2, fontsize=9.5, ha="left", va="center", fontweight="bold")

    ax.set_xlim(-0.75, 3.75)
    ax.set_ylim(-0.42, e_ion + 0.42)
    ax.set_xticks(range(4))
    ax.set_xticklabels(list(L_NAMES))
    ax.tick_params(axis="x", length=0, pad=6)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel("Orbital angular momentum of the valence electron")
    ax.set_ylabel("Energy above the ground state (eV)")
    ax.set_title(spec["title"])
    for l in range(4):
        ax.axvspan(l - 0.46, l + 0.46, color=FAINT, alpha=0.18, lw=0, zorder=0)

    return {
        "species": spec["symbol"],
        "ground": f"{ground[0]}S1/2",
        "intermediate": f"{spec['mid_n']}P{j_txt}",
        "rydberg_illustrative": f"{ILLUSTRATIVE_N}S1/2",
        "first_photon_nm": round(lam1, 2),
        "second_photon_nm": round(lam2, 2),
        "nominal_nm_from_plan": list(spec["nominal_nm"]),
        "other_fine_structure_path_nm": {
            f"{spec['mid_n']}P{int(2 * j)}/2": [round(v, 2) for v in paths[j]] for j in paths if j != j_mid
        },
        "ionisation_energy_eV": round(e_ion, 4),
        "rydberg_binding_meV": round(binding_mev, 3),
    }


def levelplot_headless_ok() -> bool:
    """ARC's own LevelPlot also runs headless; kept as a cross-check of the level data."""
    try:
        lp = LevelPlot(Caesium())
        lp.makeLevels(6, 12, 0, 3)
        lp.drawLevels()
        plt.close(lp.fig)
        return True
    except Exception:  # pragma: no cover - only reports capability
        return False


def main() -> dict:
    t0 = time.time()
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.6))
    results = [draw_species(ax, spec) for ax, spec in zip(axes, SPECIES)]
    fig.suptitle("Two-photon path to a Rydberg state: the physical step behind every CZ gate",
                 x=0.012, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    cs, rb = results
    caption = (
        f"Energy levels of caesium and rubidium-87 computed with ARC {_arc_version()}. Arrows mark the two-photon "
        f"path to a Rydberg state. Caesium: {cs['ground']} to {cs['intermediate']} at {cs['first_photon_nm']:.1f} nm, "
        f"then to {cs['rydberg_illustrative']} at {cs['second_photon_nm']:.1f} nm. Rubidium-87: {rb['ground']} to "
        f"{rb['intermediate']} at {rb['first_photon_nm']:.1f} nm, then to {rb['rydberg_illustrative']} at "
        f"{rb['second_photon_nm']:.1f} nm. The Rydberg level n = {ILLUSTRATIVE_N} is illustrative: we do not know "
        f"which levels Sqale uses, and the second wavelength shifts by a few nm with n. During a CZ only the "
        f"qubit's |1> state is excited along this path."
    )
    save(fig, "grotrian", caption, {"species": results, "levelplot_headless_ok": levelplot_headless_ok()})

    print("Two-photon paths (computed with ARC | nominal in the plan):")
    for r in results:
        print(f"  {r['species']}: {r['ground']} -> {r['intermediate']} {r['first_photon_nm']:.2f} nm | "
              f"{r['nominal_nm_from_plan'][0]} nm ;  -> {r['rydberg_illustrative']} "
              f"{r['second_photon_nm']:.2f} nm | {r['nominal_nm_from_plan'][1]} nm")
        for name, (a, b) in r["other_fine_structure_path_nm"].items():
            print(f"      other fine-structure path via {name}: {a:.2f} nm + {b:.2f} nm")
    runtime = time.time() - t0
    print(f"grotrian.py done in {runtime:.1f} s")
    return {"figure": "grotrian.png", "runtime_s": round(runtime, 1), "species": results}


def _arc_version() -> str:
    import arc
    return getattr(arc, "__version__", "")


if __name__ == "__main__":
    main()
