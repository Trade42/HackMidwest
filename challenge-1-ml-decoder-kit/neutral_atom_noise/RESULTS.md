# Results: the Gross code under noise shaped like neutral-atom hardware

Task `infleqtion-noise`, branch `infleqtion-noise`. Plain-language summary: [README.md](README.md).
**A model built from Infleqtion's published figures, not Infleqtion's noise model.** Simulated, not measured.

**Status (2026-11-22, 08:00 UTC): the model, its tests and the pilot are done; the runs start when a lane frees
(about 09:00 UTC). Every table below marked "to come" is unmeasured.**

## The model

Every rate is multiplied by one **factor**; factor 1 is the published figures. The published figures, as read in
the papers' own LaTeX sources on 2026-11-22 (`model.PUBLISHED`; "M" measured on hardware, "F" a parameter of the
authors' simulator, fitted to their hardware):

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
| Our loop on the kit's own noise gives the kit's failure rates | to come | `runs/lane_b.sh reference` |

## Pilot (not results)

X memory, seed 900, 40 to 100 shots per point, this PC under load. It fixed the factors 0.6, 0.7, 0.8, 0.9 and the
marked point 1. Relay-BP with matched priors: 0 of 100 failures at factors 0.4 to 0.7, 1 of 100 at 0.8, 5 of 40 at
0.9. With generic priors at the same two-qubit error: no failure at 0.5 to 0.7 (82, 58, 23 shots), but 891, 1,357
and 1,861 mean iterations against 76, 109 and 243 matched.

## Question 1: what knowing the shape buys

To come.

## Question 3: where the code starts to work

To come.

## Question 2: what the loss flags add

To come, after question 1.

## Machine, commands, seeds

To come with the tables. Shot files and their sha256: `logs/shots.log`.
