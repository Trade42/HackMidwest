# TCBP pipeline, phase 2: three more pictures of the noise, declared before any number

**Label: exploration, after the main test results; a separate branch (`infleqtion-noise`).** Written 2026-11-22
about 16:31 UTC by the implementer of task `tcbp-noise`, at the orchestrator's request on Sal's word (16:24 UTC), before the server job
ran and before any TCBP-JOBS decode of these shots existed. Every number is reported whatever it shows.

## What carries over from phase 1 (`tcbp_check_declared.md`)

TCBP-JOBS (cudaq-qec preset `tcbp-jobs`), one decoder built per shot, one CPU thread, the server's A100. Every task
decodes the kit's X-memory H and L (936 x 8,784); only the prior vector differs between arms. The three arms:

- **(g) generic:** the kit's uniform model at a stated p (below, per task). **Nothing is fitted for TCBP-JOBS** in
  phase 2.
- **(m) matched:** the error model of the circuit that made the shots, on the kit's columns.
- **(h) head:** the network and head (trunk `d8d1c70f...`, head `38ab667e...`, fp32 on the A100), the frozen TCBP
  knobs of `predecoder/logs/fit_tcbp_X_p0.005.json` (log-odds, alpha 0.4, eps 1e-4, cap 0.5). No refit. The
  network's p input and the mixing base are the generic arm's p, so h at alpha 0 is exactly g.

The same pairs, h vs g, h vs m and m vs g, each get an exact two-sided sign test on the shots where only one arm
fails, and every failure count a 95% Wilson interval.

**A lesson from phase 1, stated before these numbers.** In phase 1 the generic arm used BP-OSD's fitted p. TCBP-JOBS
with it failed 957 of 2,000 shots, where BP-OSD itself failed 442 at the same p. That generic arm was weak for TCBP-JOBS, so
the head's gain over it (209 against 957) overstates what the head adds over a good generic picture. In phase 2 the
generic p is not fitted at all, and the text will compare the head with the generic and the matched arms only, not with
a "best generic" claim.

## The tasks (`tcbp_models.py prepare`, all X memory, 4,000 shots each)

1. **wrong-p, the wrong-noise test.** The kit's own shots at p = 0.005: the first 4,000 shots of
   `data/p0.005/validation_X.npz` (seed 201, sha256 `25e9ce8e2871...`; **validation** shots: the TCBP knobs were
   fitted on train shots, and the validation file was used for the pipeline's check, so these are exploration shots,
   not a test). (g): the kit's priors at **p = 0.004** (the wrong p). (m): the kit's priors at **p = 0.005** (the
   truth). (h): the network told **p = 0.004** (its p input and the mixing base). The question: does the head
   recover what the wrong p loses?
2. **si1000.** SI1000 from Gidney, Newman, Fowler and Broughton, "A Fault-Tolerant Honeycomb Memory", Quantum 5,
   605 (2021), arXiv:2108.10457, Table 2 (column SI1000, superconducting-inspired). It gives CZ(p) (two-qubit
   depolarizing p after the gate), AnyClifford1(p/10), InitZ(2p) (bit flip 2p after reset), MZ(5p) (bit flip 5p
   before readout), Idle(p/10) and ResonatorIdle(2p) (depolarizing 2p on a qubit idle while others are read out or
   reset). On the kit's circuit, at **p = 0.004**: two-qubit depolarizing p after each CX, readout flip 5p = 2%,
   reset flip 2p = 0.8%, every idle slot p/10. **Left out:** one-qubit-gate noise (the kit's circuit has no
   one-qubit gates), and ResonatorIdle, because the rewrite (`model.with_noise`) gives every idle slot of the kit one
   rate. So this is SI1000 without its resonator idle, milder than the paper's. Shots: Stim, seed 7100.
   (g): the kit's priors at the model's two-qubit error, p = 0.004.
3. **biased.** Z-biased Pauli noise, **bias eta = 100**, at **p = 0.003**. After each CX, on each of its two qubits,
   and on each idle qubit: one-qubit Pauli noise of total p, with pZ = p eta / (eta + 1) and pX = pY = p / (2 (eta + 1)).
   Readout and reset flips are p. Shots: Stim, seed 7200. (g): the kit's priors at the model's two-qubit error
   1 - (1 - p)^2 = 0.005991 (the branch's `generic-2q` rule). **Why only eta = 100, not 10:** X memory reads only
   the X checks, which see Z and Y errors. At equal total p, eta = 10 and eta = 100 differ little in what X memory
   sees (pZ + pY = 0.955p against 0.995p), so the shots are spent on one bias.

The strengths were chosen before any decode of these models, by reasoning only. They were meant to sit near the
kit's p = 0.005 in what X memory sees, where TCBP-JOBS fails about 1% to 2% of shots. No pilot was run. If a task fails
almost never or almost always, that is reported as it is.

## The stopping rule

One server job, one TCBP model (fitted once: every task has the same H) shared by all tasks. Blocks of 100 shots,
taken in turn: block 0 of wrong-p, of si1000, of biased, then block 1 of each, and so on. Inside each block the arms
run m, g, h. No new block starts after **17:35 UTC**. **A task's scored set is the longest prefix of whole blocks
that all three of its arms finished.** If all 40 blocks of a task finish, its scored set is all 4,000 shots.
