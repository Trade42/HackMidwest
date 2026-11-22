# Atom loss on the Gross code: what it costs relay-BP, what the heralds give back, and the ceiling for a loss-time tracker

Task `loss-gates` (gate G5 of `orchestration/PLAN.md`). These are exploration numbers on shots that Clifft
simulates. They are not final results, and no held-out file of the team is used.

**Read this first: how L2 differs from the brief.** The brief asks to add, one at a time, four pieces of the
loss-to-prior recipe that "the starter lacks": the final-readout flip at 0.5, the hook columns of flagged check
atoms of the other basis, lossless priors wherever no flag fired, and the location weights. The starter's code
(`starter_scripts/loss_starter.py`) already has the first three. Its docstring says that Z-check heralds are not
used, but its flag-aware decoder does use them, through the hook columns. So L2 is run as leave-one-out from the
starter's map, plus the one missing piece, the exact location weights. That piece includes a loss inside a run of
same-role CX, which the starter leaves out. The orchestrator agreed to this by message, 2026-11-22.

<!-- TABLES BELOW ARE FILLED IN AS THE RUNS FINISH -->

## The loss model (our assumption)

| What | Our choice |
|---|---|
| Simulator | Clifft 0.13.0, `noncomp` module, online execution. The circuit is the kit's memory circuit (`circuits/gross_X_r12_p0.001.stim`, noise set to p), flattened. |
| Pauli noise | The kit's, unchanged: strength p on every operation (`circuits/README.md`). p = 0.004 unless stated. |
| When an atom is lost | After every CX, each of its two atoms is lost with probability s: Clifft's `LOSS(s)` on both operands. `LOSS(p)` moves any occupied level to "lost", g and e alike. So s is **per atom per CX**: this is also what the starter's `p_loss` means. No loss at resets, measurements or idles. Losses are independent: no correlated pair loss. |
| How often | Every atom has 6 CX per cycle, 72 in the experiment. A data atom is lost by the end with probability 1 − (1 − s)^72; a check atom is lost in a given cycle with probability 1 − (1 − s)^6. The mean numbers of lost data atoms and of check heralds per shot are measured beside every result. |
| What a lost atom does | Nothing. Clifft skips every later gate and every noise channel that touches a lost atom ("DEPOLARIZE2/3 and PAULI_CHANNEL_2/3 touching a leaked or lost site are skipped completely"), and by default leaves the gate partner unchanged. No partner effect is configured. |
| What is flagged, and when | A measurement of a lost atom uses the readout classifier, whose third symbol is the herald. The classifier is perfect: a lost atom is always heralded, a present atom never. The measurement record then holds a uniformly random bit. |
| What is restored | `reset_restores_lost=True`. Check atoms are reset every cycle (X checks by RX in layer 0, Z checks by R in layer 7). So a check atom lost in cycle c is flagged at its measurement in cycle c and is back from cycle c + 1. Data atoms are never reset. A lost data atom stays lost and is flagged only at the final readout. |
| Leakage | None. |
| Loss fraction | A CX pair loses an atom with probability about 2s; its Pauli error is p. At p = 0.004: 2s/(2s + p) = 0.09, 0.20, 0.33, 0.50 for s = 2e-4, 5e-4, 1e-3, 2e-3. |
| Why this range | Infleqtion's hardware paper (arXiv:2408.08288) measures about 0.38% loss per pair per CZ, about 2e-3 per atom: s = 2e-3 is about today's hardware, the lower values are targets (the brief, from `reference/infleqtion-noise-models.md`). |

**The true loss times** come from status probes: a `HERALD_LOSS_EVENT` on both atoms after every CX. Clifft's
documentation says these checks "report the status at a circuit position without changing the site's quantum
state or occupation", are noiseless by default, and append one ordinary record bit per target. The probes shift
the measurement record, so `lossmodel.clifft_text` remaps every `DETECTOR` and `OBSERVABLE_INCLUDE` lookback.
Checked:

- noiseless circuits give zero detectors and zero observables, X and Z memory, with and without probes;
- at the same seed, the detectors, observables, heralds and measurement records are bit-identical with and
  without probes (300 shots, X and Z, p = 0.004, s = 1e-3). So the probes change neither the physics nor Clifft's
  random stream;
- Clifft's thread count does not change the shots;
- on every dataset (`lossmodel.loss_times` asserts it): every herald equals the probe after the atom's last CX,
  and a lost data atom never comes back.

The decoders never see the probes. Only the oracle conditions (L3) and the flicker table (L5) read them.

## How a loss enters the decoder: the map, checked on Clifft shots

X memory is decoded from the 936 X-check detectors, with the kit's H (936 × 8,784), L and priors
(`circuit.dem_matrices`). There the decoder sees Z errors on data atoms. `lossmap.py` reads the circuit and
finds the columns of H that each loss can flip:

| Lost atom | Columns that fire with probability 1/2 |
|---|---|
| X-check atom (own basis), lost in cycle c | its measurement flip in cycle c |
| Z-check atom (other basis), lost after its k-th CX of cycle c | Z on that atom right after its k-th CX (a "hook": the data atoms it already met are dephased by the partial check), k = 1..5; nothing for k = 6 |
| data atom, lost after its j-th CX | Z on it right before the first CX after j in which it is the target; Z right before every later CX that starts a run of target CXs (one per cycle in this schedule); its final readout flip |

The data-atom rule is the Pauli envelope of Liu et al. (arXiv:2603.04156): an absent CX control acts like a
control in |0⟩, an absent target like a target in |+⟩. Z memory is the mirror image (X errors, control role).

`check_map.py` tests this on Clifft shots with loss and no Pauli noise. It stacks each shot's 936 detector bits
and 12 observable bits, and solves over GF(2):

| Test | X, s = 1e-3 | X, s = 2e-3 | Z, s = 1e-3 | Z, s = 2e-3 |
|---|---|---|---|---|
| shots | 1,000 | 1,000 | 1,000 | 1,000 |
| lost data atoms per shot | 10.2 | 19.4 | 10.0 | 19.4 |
| in the span of the columns its flags allow (all loss times) | 1,000 | 1,000 | 1,000 | 1,000 |
| in the span of the generators of its TRUE loss times | 1,000 | 1,000 | 1,000 | 1,000 |
| true generators independent, so the fired set is fixed | 932 | 748 | 884 | 603 |
| two generators of a shot both fired (independent: 0.25) | 0.252 | 0.254 | 0.267 | 0.234 |

Fire rate of each kind of generator, over the shots where it is fixed (expected 0.5):

| Generator | X, 1e-3 | X, 2e-3 | Z, 1e-3 | Z, 2e-3 |
|---|---|---|---|---|
| own check: measurement flip | 0.503 (4,764) | 0.497 (7,642) | 0.510 (4,448) | 0.507 (6,223) |
| other check: hook, k = 1 | 0.497 (784) | 0.512 (1,232) | 0.503 (690) | 0.504 (968) |
| other check: hook, k = 2 | 0.504 (776) | 0.487 (1,262) | 0.460 (754) | 0.486 (1,046) |
| other check: hook, k = 3 | 0.488 (810) | 0.482 (1,267) | 0.481 (805) | 0.486 (1,043) |
| other check: hook, k = 4 | 0.507 (836) | 0.506 (1,280) | 0.507 (768) | 0.497 (1,008) |
| other check: hook, k = 5 | 0.488 (775) | 0.514 (1,186) | 0.503 (730) | 0.498 (946) |
| data: first column after the loss, inside a run | 0.497 (3,092) | 0.483 (4,630) | 0.492 (2,828) | 0.494 (3,705) |
| data: first column after the loss, a run start | 0.513 (6,147) | 0.504 (9,378) | 0.503 (5,644) | 0.494 (7,431) |
| data: later run starts | 0.494 (50,473) | 0.499 (78,053) | 0.499 (47,165) | 0.501 (62,260) |
| data: final readout flip | 0.500 (9,433) | 0.495 (14,271) | 0.494 (8,725) | 0.491 (11,459) |

Forced single losses (`--forced`: one atom lost for certain after one chosen CX, 400 shots per location; own and
other check atom after each of their 6 CX in cycle 5, a left and a right data atom after CX 1-6, 31-36, 71, 72)
pass the span tests at every location in both bases: 400 of 400 shots in both spans. Every fire rate is 0.44 to
0.56 except one: in Z memory, a right data atom lost after its first CX (a control CX in layer 0 of cycle 0)
is still in its prepared |0⟩ state, untouched by any X check, so the first column after the loss never fires
(0 of 400). The envelope then over-counts one generator. This concerns 1 of the 72 loss slots of half the data
atoms, in Z memory only: about 0.7% of data losses there. In the rate-tested random shots it is inside the
"inside a run" row. The other check atom lost after its 6th CX leaves no trace in the decoded basis, as the rule
says.

So the hook rule and the envelope, both marked "our inference" in the notes, hold in the simulator, observables
included. A third of the data-atom losses (3,092 of 9,433 at X, 1e-3) happen inside a run of target CX. Their
first column is one the starter never raises, and the uniform guess gives it only (1/2)(1/72).

Commands:

```sh
flock /home/sal/hackMidwest/dq/run/sim-d.lock nice -n 15 taskset -c 0-3 loss/runs/check_map.sh
```

Results: `results/check_map_*.json`, `results/check_forced_*.json`; logs: `logs/check_map_*.log`, `logs/check_forced_*.log`.

## L0: no loss (s = 0): Clifft's shots against Stim's and the kit's

Dataset: X memory, p = 0.004, s = 0, 10,000 shots, seed 1000 (chunks of 2,000, seeds 1000 to 1004), with probes.
Decoder: relay-BP at the kit's settings, a fresh decoder with seed 1 for every shot.

| Check | Clifft (s = 0) | Reference | Verdict |
|---|---|---|---|
| mean detector weight per shot (1,728 detectors) | 232.78 | Stim, same circuit, 10,000 shots, seed 1000: 233.01 | z = −0.60 |
| largest per-detector difference in firing rate | |z| = 3.64 over 1,728 detectors, none above 4 | (expected maximum of 1,728 normal z: about 3.4) | consistent |
| observable flip rates (undecoded) | 0.493 to 0.507 | Stim: 0.491 to 0.501 | both fair coins, as the kit says |
| relay-BP failures, X | **64 / 10,000 = 0.64% (95% Wilson 0.50–0.82%)**; 27 unconverged, mean 542 iterations | kit README 45 / 10,000; `baselines.log` 51 / 10,000; relay-BP authors 53 / 10,000; the three pooled 149 / 30,000 = 0.50% (0.42–0.58%) | z = 1.82, 1.22, 1.02; pooled z = 1.71 (two-sided p ≈ 0.09) |

**L0 passes.** The per-shot detector statistics, which are far more precise than a failure count, match Stim's.
The relay-BP failure count is 1.7σ above the pooled reference, which is within the statistical error; the
brief's "about 0.5%" lies inside our interval.

Clifft's sampling speed on this circuit (four threads, with probes, in `sample.py`): 400 shots/s at s = 0,
301 shots/s at s = 1e-3. Single-thread speeds on one physical core are in L4.

Commands:

```sh
loss/runs/L0_L1_s1e-3.sh          # samples the dataset, decodes it (parts of 2,500 shots, lanes A+D or D)
python -m loss.l0_check --data $S/data/X_p0.004_s0_seed1000.npz --parts "$S/parts/L0_X_p0.004_s0/blind/*.npz" --out loss/results/L0_X_p0.004
```

`S=/home/sal/hackMidwest/orchestration/scratch/loss-gates`. Results: `results/L0_X_p0.004.json`,
`results/L0_X_p0.004_s0_table.{json,md}`. Logs: `logs/L0_L1_s1e-3.log`, `logs/L0_check.log`.
