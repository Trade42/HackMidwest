# Results: the Gross code under noise shaped like neutral-atom hardware

Task `infleqtion-noise`, branch `infleqtion-noise`. Plain-language summary: [README.md](README.md).
**A model built from Infleqtion's published figures, not Infleqtion's noise model.** Simulated, not measured.

**Status (2026-11-22, 08:00 UTC): the model, its tests and the pilot are done; the runs start when a lane frees
(about 09:00 UTC). Every table below marked "to come" is unmeasured.**

## The model

Every rate is multiplied by one **factor**; factor 1 is the published figures. The published figures
(`model.PUBLISHED`; "M" measured on hardware, "F" a parameter of the authors' simulator, fitted to their hardware).
**Every figure below was checked by us** against the paper's own LaTeX source (arxiv.org/e-print) on 2026-11-22
(`checked=True` in `model.py`); none is taken from the research notes unchecked:

| Figure | Value | Kind | Source |
|---|---|---|---|
| phase error on each atom per CZ | 0.73% | F | arXiv:2509.13247v2, Appendix "System error model", table "Simulation Noise Model", row CZ |
| transitions per CZ between the five atom levels (scattering, leakage, loss) | matrix, ×10⁻⁶ | F | same row; the 4×4 block is Eq. 3 of arXiv:2408.08288v3 |
| loss per CZ of an atom in state 1 | 0.5% (none from state 0) | F | same row (matrix entry 5000×10⁻⁶) |
| CZ fidelity after discarding lost atoms, array median | 98.7% | M | arXiv:2509.13247v2, Appendix "Gate Fidelities" |
| readout: 0 read as 1, 1 read as 0 | 0.2%, 2.3% | F | arXiv:2509.13247v2, same table, row Measure |
| state preparation fidelity | 97.9% | F | same table, row State Prep |
| one-qubit gates: local, global | 99.80(5)%, 99.96(1)% | M | arXiv:2509.13247v2, Appendix "Gate Fidelities" |
| coherence with decoupling, T2 | 2.8 s (97% lower bound 128 ms) | M | arXiv:2408.08288v3, Sec. "Quantum processor implementation and characterization" |
| coherence without decoupling, T2* | 12.7(5) ms | M | same |
| readout time | 6 ms | M | arXiv:2408.08288v3, Sec. "Non-destructive State Selective Readout" |
| loss per CZ, one benchmarked pair | about 0.38% (CZ fidelity 99.35(4)%, 99.73(3)% after discarding lost atoms) | M | arXiv:2408.08288v3, Sec. "Quantum processor implementation and characterization" |

Our rates at factor 1 (`model.MODEL`):

| Where in the kit's circuit | Our channel | Rate | Status | How |
|---|---|---|---|---|
| after each CX | Z on the control, X on the target | 0.73% each | published | the CZ phase error per atom, carried to CX through the Hadamards on the target (README.md) |
| after each CX | one-atom depolarizing on both atoms | 0.0157% | derived | 3/4 of the state-averaged chance (2.09×10⁻⁴) that scattering leaves the atom in a qubit state |
| idle data atoms (the kit's three idle slots) | Z | 0.107% | our choice | (1 − e^(−t/T2))/2 with t = 6 ms, T2 = 2.8 s. No idle error per step is published |
| before each check readout | flip | 1.25% | derived | mean of 0.2% and 2.3%; mid-circuit use is our choice (their machine reads out only at the end) |
| after each check reset | flip | 2.1% | our choice | 1 − preparation fidelity; their machine has no mid-circuit reset |
| after each CX (question 2 only) | atom loss, each atom | 0.25% | derived | half of 0.5%: an atom is in state 1 half the time (our assumption) |

Total two-qubit Pauli error at factor 1: 1.486% per CX (`Noise.two_qubit_error`). Left out though published:
leakage (5.4×10⁻⁴ per atom per CZ), one-qubit gate errors (the kit's circuit has no one-qubit gate), loss during
readout (0.9%), state-dependent readout, and atom moves (2.6% to 4.1% phase error per move). Each would push
today's hardware further above factor 1.

## Checks

| Check | Result | Where |
|---|---|---|
| With every rate at the kit's uniform p = 0.004, our circuit's detector error model equals the kit's term for term (72,864 error terms for X memory, 73,296 for Z), and the circuits are equal | pass, X and Z | `test_uniform_rates_give_the_kits_detector_error_model` |
| At the model's rates: gates, detectors (1,728), observables (12), detector coordinates equal the kit's; noiseless shots give zero detectors and observables | pass, X and Z | `test_only_the_noise_differs_from_the_kit` |
| Matched priors are the model circuit's error model on the kit's columns; every model mechanism is a kit column (all 8,784 fire); generic priors come from the kit's uniform circuit; H and L do not depend on p | pass | `test_matched_priors_*`, `test_generic_priors_*` |
| No fitted quantity is scored on the shots it was fitted on: the generic p is fitted on train seed 5100, scored on test seed 4100 | enforced | `job.py` refuses equal seeds |
| Rows of one point decode the same shot file | enforced | `summarize.py` refuses to pair rows whose shot files differ (sha256) |
| A shot fails when any of the 12 observables is wrong; unconverged shots are scored on the decoder's output | by construction | `run._decode` |
| The truth reaches no decoder: a decoder gets the 936 X-check detector bits (and, with loss, the heralds); the observables are compared after decoding | by construction | `run._decode`; the loss shot files hold no loss times (`clifft_text(..., probes=False)`) |
| Our loss loop equals `python -m loss.job` (`starter`) on the kit's noise: shots bit-identical, decoding identical shot for shot | pass on 12 shots in the suite; 100-shot run to come | `test_our_loss_loop_equals_*`, `loss_check.py` |
| Clifft (used for loss) samples our circuit like Stim at loss 0 | pass | `test_clifft_samples_our_circuit_like_stim_*` |
| Our loop on the kit's own noise gives the kit's failure rates | pass, both decoders (two seeds for relay-BP; table below) | `runs/lane_b.sh reference`, `checks`; `logs/reference.log`, `logs/checks.log` |
| A lost atom's partner keeps its own error in Clifft (our one-atom channels), unlike under a two-atom channel | pass | `test_the_partner_of_a_lost_atom_keeps_*` |

**The reference.** The kit's own uniform noise at p = 0.004, X memory, sampled by `make_shots.py` and decoded by
this folder's loop with the kit's priors; luminaty, 4 processes, 2026-11-22 (lane B 08:50-08:58, lane D 10:53-11:05
UTC). Results: `results/reference_*`.

| Decoder | Shots (seed) | Failures | Fraction (95% Wilson) | Kit README, X memory | Unconverged | Mean iterations |
|---|---|---|---|---|---|---|
| BP-OSD | 2,000 (4100) | 42 | 2.10% (1.56–2.83%) | 253 / 10,000 = 2.53% | 733 | 52 |
| relay-BP | 2,000 (4100) | 3 | 0.15% (0.05–0.44%) | | 1 | 524 |
| relay-BP | 2,000 (4101) | 12 | 0.60% (0.34–1.05%) | | 4 | 547 |
| relay-BP, both seeds | 4,000 | 15 | 0.38% (0.23–0.62%) | 45 / 10,000 = 0.45% | 5 | 535 |

Both decoders agree with the kit. The first relay-BP seed alone was low (3 against about 9 expected), so we ran a
second seed before using any relay-BP row; pooled, the two agree with the kit's 0.45%. Our relay-BP gets a fresh
decoder with seed 1 for every shot, as `loss/` does; the kit's `baseline.py` reuses one decoder per process.

**A control for the fitted generic p.** Under our model, BP-OSD decodes best with a uniform p about 2 to 4 times the
model's two-qubit error (question 1). Is that BP-OSD preferring inflated priors on any noise? No: on the kit's own
noise at p = 0.004 (seed 4100), BP-OSD with p = 0.011 in its priors fails 52 of the same 2,000 shots against 42 with
the true p (30 shots only the inflated priors get wrong, 20 only the true ones; exact sign test p = 0.20). So the
shift of the best uniform p is a property of the neutral-atom shape, not of BP-OSD.

## Pilot (not results)

The factors 0.6, 0.7, 0.8, 0.9 and the marked point 1 were chosen from a pilot of 40 to 100 shots per point (X
memory, seed 900, run inline under a one-minute cap; **no committed log**, so none of its numbers is quoted here).
Every number in the tables below comes from the logged runs.

## Question 1: what knowing the shape buys

To come.

## Question 3: where the code starts to work

Every rate of the model times one factor; matched priors; X memory, 12 cycles. A shot fails when any of the 12
logical bits is wrong. Test shots seed 4100 (luminaty, `logs/shots.log`); 2,000 shots per point except relay-BP at
0.9 (500) and 1 (250), where its slow shots are costly.

| Factor | Two-qubit error per CX | relay-BP: failed shots (95% Wilson) | per cycle | BP-OSD: failed shots (95% Wilson) | per cycle |
|---|---|---|---|---|---|
| 0.6 | 0.89% | 2 / 2,000 = 0.10% (0.03–0.36%) | 8.3e-5 | 5 / 2,000 = 0.25% (0.11–0.58%) | 2.1e-4 |
| 0.7 | 1.04% | 13 / 2,000 = 0.65% (0.38–1.11%) | 5.4e-4 | 31 / 2,000 = 1.55% (1.09–2.19%) | 1.3e-3 |
| 0.8 | 1.19% | 39 / 2,000 = 1.95% (1.43–2.65%) | 1.6e-3 | 91 / 2,000 = 4.55% (3.72–5.55%) | 3.9e-3 |
| 0.9 | 1.34% | 31 / 500 = 6.2% (4.4–8.7%) | 5.3e-3 | 238 / 2,000 = 11.9% (10.6–13.4%) | 1.1e-2 |
| 1 (the published figures) | 1.49% | 42 / 250 = 16.8% (12.7–21.9%) | 1.5e-2 | 599 / 2,000 = 30.0% (28.0–32.0%) | 2.9e-2 |

**Relay-BP fails between 0.1% and 10% of shots from factor 0.6 to just above 0.9.** BP-OSD reaches 10% a little
below 0.9.

**Where today's hardware sits.** The Pauli part of the published figures (factor 1) is just above that range:
relay-BP fails about one shot in six. That is **not** today's machine running this experiment, which sits further
above it, for four reasons this model leaves out:

- **atom loss**: about 0.4% of a pair per gate on their best pair, 0.5% from state 1 in their simulator. Question
  2 below adds it;
- **leakage** (5×10⁻⁴ per atom per gate) and **atom moves**: the Gross code needs gates between distant atoms,
  and on today's machine a move costs 2.6% to 4.1% phase error in their table;
- **mid-circuit readout and reset**: their machine has neither today; the rates here are our assumptions;
- the **figures are the authors' simulator parameters** fitted to small circuits, not a benchmark of this code.

And one choice of ours moves factor 1 itself: the idle error. We charge each idle slot 6 ms at the decoupled
T2 = 2.8 s (0.107% phase error). The paper's 97% lower bound on T2, 128 ms, would make it 2.3%, about 20 times
larger, and then factor 1 would sit further above the range.

**Why the factor reaches so far.** At factor 0.7 the two-qubit error is 1.04% per CX, 2.6 times the kit's
p = 0.004; yet relay-BP fails 0.65% of shots, close to the kit's 0.45% at p = 0.004 (the kit's `README.md`; our
reference above: 0.15%). A phase error on a CZ arrives on the kit's CX as Z on the control and X on the target;
neither spreads through the rest of the check's gates, so this noise makes almost no "hook" faults (one check atom
spreading its fault onto several data atoms). In the matched picture 0.2% of the expected faults flip 5 or 6
detectors, against 7.9% in the kit's uniform picture (`logs/explain.log`). Noise of this shape is easier for this
code than its strength suggests, **provided the decoder knows the shape** (question 1).

## Question 2: what the loss flags add

To come, after question 1.

## Machine, commands, seeds

To come with the tables. Shot files and their sha256: `logs/shots.log`.
