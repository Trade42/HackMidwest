# Physics backing

Four figures that explain where the errors in our circuits come from, which is what
the error-detected tier exists to catch. Every figure regenerates from a script in
this folder using ARC (Alkali Rydberg Calculator, version 3.10.2 here).

## Run

```bash
pip install ARC-Alkali-Rydberg-Calculator      # or: pip install -e ".[physics]"
python physics/make_all.py                     # all four, one after the other
```

Or one at a time:

```bash
python physics/grotrian.py
python physics/rydberg_scaling.py
python physics/stark_map.py
python physics/pair_potential.py
```

Output goes to `figures/physics/`. Each figure `X.png` (200 dpi) comes with `X.txt`
(the caption, plain words, ready to paste under the slide) and `X.json` (every number
the figure carries). `summary.json` collects the headline values and runtimes. Set
`PHYSICS_OUT=/some/dir` to write somewhere else. ARC needs no network: its data ships
with the package and is cached in `~/.arc-data`.

## Runtimes measured here

Measured on 2 shared CPU cores. "First run" is with an empty ARC cache.

| Script | First run | Later runs |
| --- | --- | --- |
| `grotrian.py` | 1.2 s | 0.9 s |
| `rydberg_scaling.py` | 117 s | 8 s |
| `stark_map.py` | 65 s | 25 s |
| `pair_potential.py` | 13 s | 12 s |
| `make_all.py` | 195 s | about 50 s |

The pair potential uses the full diagonalisation (561 pair states, 11 s), so the
perturbative fallback from the plan was not needed. It is still drawn as the dashed
van der Waals curve, and the two agree: 7.5 MHz against 7.4 MHz at 5.5 um.

## What each figure shows, with the computed numbers

| Figure | Computed | Ties to the project |
| --- | --- | --- |
| `grotrian.png` | Cs: 6S1/2 to 7P1/2 at 459.4 nm, then to 60S1/2 at 1041.0 nm. Rb-87: 5S1/2 to 6P1/2 at 421.7 nm, then to 60S1/2 at 1005.9 nm. | During a CZ only the qubit's \|1> state is excited along this path. It is the physical step behind every two-qubit gate. |
| `rydberg_scaling.png` | Fitted exponents against n (against n* in brackets): lifetime 3.24 (3.00), polarizability 7.18 (6.64), C6 12.49 (11.55), level spacing -3.21 (-2.97). | Choosing n trades a stronger blockade against more sensitivity to stray fields and decay. That trade-off sets the gate error that error detection has to catch. |
| `stark_map.png` | 60S polarizability 197 MHz cm^2/V^2, so 0.1 V/cm shifts the level by -0.99 MHz. Quadratic within 5% up to 0.68 V/cm. | A stray field shifts the Rydberg level and adds a phase error to every CZ. Phase errors on data qubits are what the all-X check detects. |
| `pair_potential.png` | Cs 60S + Rb 60S: C6 = -205.4 GHz um^6 (repulsive). 7.4 MHz at 5.5 um, 0.12 MHz at 11 um. Blockade radius 7.7 um for an assumed 1 MHz drive, 5.9 um for 5 MHz. Le Roy radius 1.07 um. | The blockade makes each CZ and is short-range. That is why the Iceberg code's hub qubits need routing on a grid, unlike on trapped ions. |

## The physics chain on one slide

1. Grotrian diagram: the laser path that drives each gate.
2. Scaling plot: why a given Rydberg level is chosen, and what it costs.
3. Stark map: stray fields cause phase errors on every CZ.
4. Pair potential: the blockade is short-range, so connectivity is local. The textbook Iceberg mapping needs SWAP chains for its two hub qubits; our bus mapping uses nearest-neighbour gates only.
5. Our result: which of those errors the Iceberg checks catch, how many shots survive, and what that does to the final answer. See `figures/offline/fig6_fault_budget.png` and `fig7b_noise_sweep_phase_errors.png` (from the evaluation harness, not from this folder).

## Assumptions, all of them

Nothing here is a Sqale number unless it says "cited".

| Assumption | Value | Where it is used |
| --- | --- | --- |
| Rydberg level | n = 60, S1/2, illustrative | Grotrian, Stark map, pair potential. We do not know which levels Sqale uses. |
| Range of n | 30 to 100 in steps of 5, illustrative | Scaling plot |
| Laser drive (Rabi frequency over 2 pi) | 1 MHz, assumed. 5 MHz quoted in the caption as a second assumption | Blockade radius only. It scales as drive^(-1/6), so a 5 times stronger drive moves it from 7.7 to 5.9 um. |
| Temperature | 0 K for the fitted lifetime, 300 K for the black-body series | Scaling plot |
| Pair geometry | m = 1/2 for both atoms, interatomic axis along the quantisation axis | Pair potential |
| Pair basis | nRange 4, lrange 4, energy window 15 GHz, 100 eigenvectors | Pair potential, full diagonalisation |
| C6 | second-order perturbation theory, nRange 5, energy window 25 GHz | Pair potential (dashed curve), scaling plot |
| Stark basis | n = 54 to 61, every l, \|mj\| = 1/2 (912 states) | Stark map |
| Polarizability fit | below 0.1 V/cm for the Stark map; below 5% of the Inglis-Teller field with a small basis (n-4 to n+4, l up to 6) for the scaling plot | Both give 197 to 198 MHz cm^2/V^2 at n = 60 |
| "Quadratic region" | the 60S shift within 5% of -alpha E^2 / 2 and the state still more than half 60S | Stark map, shaded band |
| Rb-Cs spacing | 5.5 um, cited (Infleqtion, "A Rb-Cs Quantum Computing Testbed", APS March Meeting 2026, from the plan's sources) | Pair potential. The 11 um marker is just twice that, to show the 1/R^6 falloff. The actual array geometry is not assumed. |

## Things worth knowing before presenting

- **The plan's rubidium wavelengths match 6P1/2, not 6P3/2.** ARC gives 421.67 nm + 1005.85 nm
  through 6P1/2 and 420.30 nm + 1013.76 nm through 6P3/2. The script picks the fine-structure
  level whose first photon is closest to the plan's nominal value and prints both paths.
  Caesium is unambiguous: 459.45 nm is 7P1/2 (7P3/2 would be 455.66 nm).
- **The second wavelength depends on n.** For caesium it runs from 1046 nm at n = 40 to
  1039 nm at n = 80, so "1040 nm" fits n near 60 to 70. Say "about 1040 nm".
- **The textbook exponents hold against n\*, not n.** n\* = n minus the quantum defect
  (about 4.05 for caesium nS). Against n\* the lifetime and level spacing land on 3.00 and
  -2.97. Polarizability (6.64) and C6 (11.55, or 11.28 for n of 50 and above) do not land
  exactly on 7 and 11 over this range. The figure reports what was fitted.
- **With 300 K black-body radiation the lifetime grows more slowly**, as n^2.63, and is
  94 us instead of 210 us at n = 60.
- **ARC returns pair-potential results for separations sorted in descending order**
  (`calc.r`), whatever order was passed in. `pair_potential.py` re-sorts them and checks the
  grid. Without that the curve comes out mirrored.
- **ARC's `LevelPlot` runs headless** (checked, recorded in `grotrian.json`). The figure
  is drawn from `getEnergy` directly so both species and the excitation path fit in one panel pair.

## Ground rules these scripts follow

- Caesium for data qubits, rubidium-87 for ancilla qubits.
- Every Rydberg level is labelled illustrative.
- No number appears on a figure unless a script computed it or it comes from a cited source.
- Scripts live here so every figure regenerates.
