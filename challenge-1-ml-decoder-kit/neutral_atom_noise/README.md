# Neutral-atom-shaped noise: what a decoder gains from knowing the shape of the noise

A demonstration beside the required sweep. The results are in [RESULTS.md](RESULTS.md); the plot of failures
against the noise factor is `results/failure_vs_factor_X.png`.

**This is not Infleqtion's noise model.** It is our own model, built from figures Infleqtion has published, with
our own choices wherever nothing is published. Everything here is simulated; nothing was measured on hardware.

## The question

The kit's required experiment uses one noise strength p for every operation, and hands every decoder that exact
description. Real neutral-atom hardware does not look like that. Its errors have a **shape**:

- a two-qubit gate mostly makes **phase errors** (one kind of error, not all kinds equally);
- **reading out** an atom is worse than a gate, and so is preparing it;
- an atom that **waits** barely degrades;
- atoms are sometimes **lost** altogether, and the machine can tell which ones.

A decoder works from a list of everything that can go wrong and how likely each thing is (its **priors**). We ask
three things:

1. **What does knowing the shape buy?** We decode the same shots twice with each decoder: once with **generic**
   priors (the kit's one-strength model: the decoder is told nothing about the hardware) and once with **matched**
   priors (the true description of the noise that made the shots).
2. **What do loss flags add?** The same, with atoms being lost, and the decoder told which atoms were found
   missing when they were read out.
3. **Where does the code start to work?** We scale every error rate by one factor. Factor 1 is the published
   figures; factor 0.5 is a machine twice as good in every rate.

## What "the shape" is here

`model.py` holds the table: every published figure with its paper and section, and every rate of our model with
how we got it. In short, at factor 1:

| Operation | The kit's model | Our model | Where it comes from |
|---|---|---|---|
| after a two-qubit gate | every two-qubit error, equally likely, p in total | a phase error on each atom, 0.73% each; rare scattering, 0.016% | arXiv:2509.13247, the authors' simulator table (CZ) |
| check-atom readout | flip with probability p | flip with probability 1.25% | same table: 0.2% and 2.3% for the two states; we take the mean |
| check-atom reset | flip with probability p | flip with probability 2.1% | same table: preparation fidelity 97.9%. **Our choice**: their machine has no mid-circuit reset |
| a waiting atom | every one-qubit error, p in total | a phase error, 0.11% | **our choice**: 6 ms of waiting at a coherence time of 2.8 s (arXiv:2408.08288) |
| atom loss (question 2 only) | none | 0.25% per atom per gate | same table: 0.5% for an atom in state 1; we halve it |

**From their gate to the kit's gate.** Their two-qubit gate is CZ; the kit's circuit uses CX. A CX is a CZ with a
Hadamard gate on the target atom before and after. A phase error on each atom of the CZ therefore arrives as a
phase error (Z) on the control and a bit error (X) on the target. One consequence carries much of the result: in
the kit's circuit a check atom then only ever gets the error that flips its own readout. The errors that a
check atom spreads onto several data atoms ("hook errors"), which the kit's model is full of, almost never happen.

**The circuit is the kit's, untouched.** The same gates, schedule, detectors and observables; only the noise
channels are replaced. The test `test_uniform_rates_give_the_kits_detector_error_model` proves the rewrite: with
every rate set to the kit's uniform p, our circuit and its error model equal the kit's, term for term.

**The decoders are the kit's, untouched.** Relay-BP and BP-OSD at the kit's settings, on the kit's decoding
matrix. Between two rows of a table the only thing that differs is the list of probabilities the decoder is
given.

## What the tables show

A "failure" is a run of 12 rounds after which at least one of the 12 stored logical bits comes out wrong. All
numbers are for X memory; every comparison is on the same shots.

1. **Knowing the shape cuts failures several times over.** At factor 0.8 (0.8 of the published rates), on the
   same 2,000 shots BP-OSD fails 91 times with matched priors and 442 times with the best generic picture we could
   fit; relay-BP, on 500 shots, 9 times against 65. Shots that only the matched decoder gets wrong are rare (21 and
   1). The wrong picture also makes relay-BP work about 4 times harder per shot. The generic picture's mistake is
   specific: it expects faults that spread from one check atom onto several data atoms ("hook" faults), and this
   hardware hardly makes them.
2. **What loss flags add** (question 2). With atoms being lost (about 10 data atoms per run at 0.4 of the
   published rates), relay-BP that ignores the loss flags fails 18 of 300 runs at factor 0.4 and 62 of 100 at
   0.6; told which atoms were found missing, it fails 0 and 9 of the same runs, and it never lost a run by using
   the flags. Knowing the shape still helps on top of the flags: at 0.6, 47 failures with generic priors against 10
   with matched ones, on the same 150 runs.
3. **Where the code starts to work.** With matched priors relay-BP fails 0.1% of runs at factor 0.6, 2% at 0.8,
   6% at 0.9 and 17% at factor 1: without atom loss the code works from about 0.9 of the published rates down.
   **With atom loss** (and its flags) it fails 0.8% at 0.5, 6% at 0.6 and 57% at 0.8: it works only up to somewhere
   between 0.6 and 0.8 of the published rates. **So today's published figures sit far above the range where this code works**, and
   the model still leaves out leakage, atom moves and the mid-circuit readout their machine does not have yet.

4. **Exploration, after the main results: the team's TCBP pipeline on this noise** (TCBP-JOBS, RESULTS.md). On the
   same 2,000 shots at factor 0.8, TCBP-JOBS fails 25 times with matched priors, 209 with the network's priors and
   957 with the kit's generic priors at BP-OSD's fitted p. That generic p was weak for TCBP-JOBS (BP-OSD itself fails 442 at
   it), so the network's gain over it is overstated. Knowing the shape is still worth about 8 times more than the network.
   Phase 2 (4,000 shots each): the network helps on the kit's own noise even when told the wrong strength (24
   failures against 75 and 68). On SI1000 at p = 0.004 it fails 2, against 13 with generic priors (the only significant pair, p = 0.007)
   and 7 with matched ones (not significant, p = 0.18). It **hurts on Z-biased noise** (605 against 457 and 421).
   Phase 2b, SI1000 at p = 0.006 and 0.008 (a bracket chosen after p = 0.004 failed nothing in its first 600 shots): the network beats the
   generic priors at both (273 against 546; 2,060 against 2,560 of 3,000), whose p is not fitted for TCBP-JOBS. Against the matched ones
   it is not significantly different at 0.006 (273 against 303, p = 0.07) and better at 0.008 (2,060 against 2,172), where about 69% to 85% of shots fail in every arm.

5. **Exploration, after the main results: the TCBP pipeline with atom loss** (TCBP-JOBS, RESULTS.md, 1,000 shots of each loss file).
   The flags help TCBP-JOBS as much as relay-BP (matched: 49 to 0 failures at f0.4, 562 to 37 at f0.6), and the network on a generic base with the flags (5 and 242) does not make up for knowing the noise's shape (matched + flags); the head's own effect is unmeasured.

## What the tables do not show

- **A model, not a machine.** The figures come from Infleqtion's papers; the model is ours. Where a paper gives no
  number (mid-circuit readout and reset, waiting errors per step) the value is our choice, and `model.py` says so.
- **Left out, though published:** leakage (an atom leaving the two qubit states without being lost, 0.05% per atom
  per gate), one-qubit gate errors (the kit's circuit has no one-qubit gates, so the Hadamards above are free),
  atoms lost during readout (0.9%), and **atom moves**. The Gross code needs gates between distant atoms; on
  today's machine that means moving atoms, at 2.6% to 4.1% phase error per move in the same table. The kit's
  circuit treats every CX as one gate. Each of these would make today's hardware look worse than factor 1 here.
- **Their machine cannot run this experiment yet.** It reads atoms out only at the end of a circuit and detects
  loss only then. Twelve rounds of mid-circuit readout are an assumption in every row.
- **Simulated, not measured.** Shots come from Stim (and from Clifft when atoms are lost).

## Files

| File | What it does |
|---|---|
| `model.py` | the published figures with sources, our rates, the factor, and the rewrite of the kit's circuit |
| `priors.py` | the kit's decoding matrix and the two pictures of the noise (generic, matched) |
| `make_shots.py` | sample shots and save them (`--loss` adds atom loss) |
| `run.py` | decode one shot file with one decoder and one picture of the noise |
| `job.py` | everything in one command: shots, all rows, the fitted generic p, tables, plot |
| `summarize.py` | tables with intervals and paired tests, and the plot |
| `loss_check.py` | checks our loss loop against `loss/`'s own job |
| `explain.py` | how the matched and the generic picture of the noise differ, kind of fault by kind of fault |
| `test_neutral_atom_noise.py` | tests |

## Run it

From `challenge-1-ml-decoder-kit/`, with Python 3.12 and `neutral_atom_noise/requirements.txt`:

```sh
python -m pytest -p no:cacheprovider test_kit.py neutral_atom_noise
python -m neutral_atom_noise.model                      # print the table of rates
python -m neutral_atom_noise.job --basis X --factors 0.6,0.7,0.8,0.9 --shots 2000 --seed 4100 \
    --decoders relay,bposd --rows matched,generic-2q,generic-fit --workers 8 --out OUT
```

The job keeps finished parts, so it can be stopped and started again. Seeds give the same shots only on the same
machine and library versions; RESULTS.md gives the sha256 of every shot file we used.

## What we rely on inside `loss/`

For the runs with atom loss we import, unchanged, from `loss/` as it is on `integration` at `6723626` (the last commit that touched it is `2e5c810`):
`lossmodel.clifft_text` and `lossmodel.sample_chunk` (the Clifft circuit and its readout model),
`lossmap.LossMap`, `lossmap.PriorBuilder` and `lossmap.parse_condition` (which columns a flagged atom raises, and
by how much). The checks also use `lossmodel.clifft_model` (the readout model, in one test),
`lossmodel.load_dataset` (to read `loss.job`'s shots in `loss_check.py`) and `python -m loss.job` itself, run as a
program by `loss_check.py`. These are not a public interface. We set one attribute from outside: `LossMap.pri`, the base priors
that the flag bumps are added to. `loss_check.py` and its test confirm that, on the kit's own noise, our loop
gives the same shots and the same decoding, shot for shot, as `python -m loss.job`.
