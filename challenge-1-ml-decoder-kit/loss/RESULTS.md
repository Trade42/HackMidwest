# Atom loss in the Gross-code memory experiment

This report studies how simulated atom loss affects Relay-BP decoding of the Gross code and how much information from loss flags helps. The experiments use Clifft and the loss model specified below. They are **exploratory X-memory shots**, separate from the held-out X/Z software comparison in the [main decoder report](../../fpga_board/MAIN_ROUTER.md). They are not measurements of Infleqtion hardware.

At physical error rate `p = 0.004` and per-atom, per-CX loss probability `s = 0.001`, the flag-aware decoder failed on 176 of 4,000 shots (4.4%). An oracle given the true round of each loss failed on 107 of those same 4,000 shots (2.67%). The oracle shows the possible value of knowing loss timing; it is not an implemented loss-time tracker. A loss-blind comparison used only 300 shots from this experiment and failed on 173 of them (57.67%). The detailed tables below distinguish these sample sizes and paired comparisons.

The starter script already accounts for final-readout flips, hooks from flagged check atoms, and lossless priors where no flag appears. The location-weight experiment tests an additional part of the loss-to-prior mapping; its conditions and measured effect are recorded below.

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

The first three were run inline during development, not by a committed script (the commands are in this
task's session log); the fourth is asserted by committed code on every dataset:

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

## The decoders compared

All use relay-BP at the kit's settings (`decoders.RELAY_SETTINGS`: a first leg of 80 iterations, up to 300
legs of 60, **stop after 5 solutions**), H and L of the lossless circuit, and a fresh decoder with seed 1 for
every shot and condition (`decode.py`). Only the priors differ. "Raise column c by v" means c's prior becomes
p + v − 2pv (independent flips combined), and every prior is capped at 0.5 at the end.

| Condition | Priors |
|---|---|
| `blind` | the lossless priors of the kit (`dem_matrices`). It sees no flag. |
| `starter` (**flag-aware**, the brief's L1 decoder) | exactly the starter's `aware_priors` (checked inline against a copy of the starter's code: identical columns and priors, to 6e-17, on 50 shots). A flagged X-check atom: its measurement flip to 0.5. A flagged Z-check atom: its 5 hook columns raised by 0.5/6. A flagged data atom: one column per cycle (the run start of cycle c) raised by 0.5 (c + 0.5)/12 (the uniform guess on the loss time), and its final readout flip by 0.5. |
| `starter@0.5/stop1` | `starter` with relay-BP stopping at its first solution (the starter's setting) |
| `exact` | the location weights done exactly: the geometric weights (1 − s)^(k−1) over the loss slot, P(lost before the column) for every run start, (1/2) P(lost right before it) for every column inside a run, geometric hook weights |
| ORACLE `oracle-round` | `exact`, with the uniform guess replaced by the TRUE loss round of each lost data atom (uniform over the 6 CX of that round). Uses the probes. |
| ORACLE `oracle-slot` | `exact`, with the TRUE loss CX of each lost data atom |
| ORACLE `starter-oracle-round` | `starter`'s map (one column per cycle, uniform hooks), with the TRUE loss round in place of the uniform guess |
| `starter-nofinal`, `starter-nohook` | `starter` without the final-readout flip, without the hook columns (L2) |
| `starter-ratebase` | `starter`, but every unflagged atom also gets the loss priors it would have if flags were not trusted (the rate-averaged loss); this removes "lossless priors where no flag fired" (L2) |
| `X@cap` | condition X with the cap at `cap` instead of 0.5 |

The decoders see the 936 X-check detectors and the heralds. The true loss times reach only the ORACLE rows (the
decode passes them to the prior builder for those rows only; for every other row it passes nothing, so any use
would fail). A shot fails when any of the 12 observables is wrong; an unconverged shot is scored on the
decoder's output like any other, and the tables count how many unconverged shots were still right.

## L1: blind against flag-aware, X memory, p = 0.004

Each loss rate has its own dataset of 20,000 sampled shots (seeds and sha256 under "Datasets"). Blind decodes a
first prefix, sized for about 100 failures; flag-aware decodes a longer prefix of the same shots. The paired
columns compare the two on the shots both decoded.

| s per atom per CX | lost data atoms per shot | check heralds per shot | blind: failures / shots (95% Wilson) | flag-aware (`starter`): failures / shots (95% Wilson) | flag-aware, stop at first solution | paired shots: blind-only / aware-only | blind ÷ flag-aware (95% bootstrap) | unconverged: blind, flag-aware |
|---|---|---|---|---|---|---|---|---|
| 2e-4 | 2.07 | 2.08 | unmeasured here: server job B | unmeasured here: server job B | unmeasured here: server job B | | | |
| 5e-4 | 5.11 | 5.17 | 57 / 400 = 14.25% (11.16–18.02) | 29 / 2,000 = 1.45% (1.01–2.07) | 36 / 2,000 = 1.80% (1.30–2.48) | 400: 53 / 1 | 11.4 (5.9–53.0) | 45 of 400, 15 of 2,000 |
| 1e-3 | 9.99 | 10.36 | 173 / 300 = 57.67% (52.01–63.13) | 176 / 4,000 = 4.40% (3.81–5.08) | 200 / 4,000 = 5.00% (4.37–5.72) | 300: 158 / 0 | 11.5 (7.6–22.0) | 161 of 300, 108 of 4,000 |
| 2e-3 | 19.29 | 20.64 | 148 / 150 = 98.67% (95.27–99.63) | 173 / 750 = 23.07% (20.19–26.21) | 186 / 750 = 24.80% (21.84–28.01) | 150: 115 / 0 | 4.5 (3.4–6.2) | 147 of 150, 151 of 750 |

Rates per cycle (1 − (1 − P)^(1/12)) for the flag-aware decoder: 1.22e-3 (5e-4), 3.74e-3 (1e-3), 2.16e-2 (2e-3);
for blind: 1.27e-2, 6.91e-2, 0.30. Lossless at the same p (L0): 5.35e-4.

**How far it got.** On this PC, s = 5e-4 stopped at 400 blind and 2,000 flag-aware shots (29 flag-aware
failures: fewer than the brief's 100). s = 2e-4 has its dataset but no decodes. The orchestrator stopped both
at 2026-11-22 06:00 UTC because server jobs B (s = 2e-4: 3,000 blind, 20,000 flag-aware shots, seeds
5100–5109) and C (s = 5e-4: 800 blind, 10,000 flag-aware, seeds 5200–5204) measure them on far more shots. Their
tables come from a second machine and are not yet in this file (see HANDOFF.md).

Shots only the flag-aware decoder fails: 1, 0 and 0, against 53, 158 and 115 that only blind fails. The heralds
never cost a shot in a way that matters.

Stopping at the first solution costs failures at every rate (36 against 29, 200 against 176, 186 against 173).
On the 4,000 shots at 1e-3 the paired count is 34 against 10 (sign test p = 0.0004). It does not change the
unconverged count, so the extra failures are converged shots that settle on the wrong logical class. It saves
56% of the iterations at 1e-3.

Per-rate tables: `results/L1_X_p0.004_s{5e-4,1e-3,2e-3}.{json,md}`; the combined table:
`results/L1_combined.md`. Commands: `loss/runs/L0_L1_s1e-3.sh` (1e-3), `loss/runs/L1_rates.sh` (the other
rates; it ran as a lane-A stream (`LANE=A`), then on lane D), `loss/runs/summarize.sh`. Logs:
`logs/L0_L1_s1e-3.log`, `logs/L1_rates_laneA.log`, `logs/L1_rates_laneD.log`.

Server jobs B and C (to come; run from `challenge-1-ml-decoder-kit/` at `6ea830c`, 10 workers, on the team's server):

```sh
python -m loss.job --basis X --p 0.004 --s 2e-4 --shots 20000 --seed 5100 --workers 10 --out ~/mlBranchMidwest/runs/loss/L1_s2e-4_seed5100 --group blind:3000 --group starter,starter@0.5/stop1:20000 --table blind:blind,starter,starter@0.5/stop1 --table starter:starter,starter@0.5/stop1
python -m loss.job --basis X --p 0.004 --s 5e-4 --shots 10000 --seed 5200 --workers 10 --out ~/mlBranchMidwest/runs/loss/L1_s5e-4_seed5200 --group blind:800 --group starter,starter@0.5/stop1:10000 --table blind:blind,starter,starter@0.5/stop1 --table starter:starter,starter@0.5/stop1
```

## L3: the oracle on the loss time (s = 1e-3)

Dataset: X memory, p = 0.004, s = 1e-3, 20,000 shots sampled, seed 2300 (chunks 2300 to 2309), sha256
`378ee22c9cdf1f25ac44a72f68931f925c67e529ecb65485326afe44b4c3d532`; 9.99 lost data atoms and 10.36 check
heralds per shot. The first 4,000 shots, every condition on the same shots.

| Condition | Failures / 4,000 | Fail fraction (95% Wilson) | Per cycle | Unconverged (of them right) | Mean iterations |
|---|---|---|---|---|---|
| `starter` (flag-aware, uniform guess) | 176 | 4.40% (3.81–5.08) | 3.74e-3 | 108 (3) | 2,339 |
| `exact` (uniform guess, exact weights) | 189 | 4.73% (4.11–5.43) | 4.03e-3 | 140 (4) | 2,609 |
| **ORACLE `oracle-round`** | **107** | **2.67% (2.22–3.22)** | 2.26e-3 | 63 (1) | 1,655 |
| `starter@0.5/stop1` | 200 | 5.00% (4.37–5.72) | 4.27e-3 | 108 (3) | 1,026 |

Paired, on the same 4,000 shots:

| Comparison | Failures | Shots only A fails / only B fails | Sign test | A ÷ B (95% bootstrap) |
|---|---|---|---|---|
| A = `starter`, B = ORACLE `oracle-round` | 176, 107 | 94 / 25 | p = 1.3e-10 | **1.64 (1.42–1.94)** |
| A = `exact`, B = ORACLE `oracle-round` (same map, only the loss time differs) | 189, 107 | 105 / 23 | p = 1e-13 | **1.77 (1.52–2.07)** |
| A = `exact`, B = `starter` | 189, 176 | 59 / 46 | p = 0.24 | 1.07 (0.96–1.20) |
| A = `starter@0.5/stop1`, B = `starter` | 200, 176 | 34 / 10 | p = 0.0004 | 1.14 (1.05–1.22) |

The oracle built on the starter's own map (ORACLE `starter-oracle-round`: one column per cycle, uniform hooks,
the true loss round in place of the uniform guess), on the first 2,000 of the same shots:

| Comparison, first 2,000 shots | Failures | Shots only A fails / only B fails | Sign test | A ÷ B (95% bootstrap) |
|---|---|---|---|---|
| A = `starter`, B = ORACLE `starter-oracle-round` (same map, only the loss time differs) | 81, 53 | 42 / 14 | p = 0.0002 | **1.53 (1.22–1.97)** |
| A = ORACLE `oracle-round`, B = ORACLE `starter-oracle-round` | 51, 53 | 14 / 16 | p = 0.86 | 0.96 (0.78–1.19) |

So the two round-level oracles are equally good, and on these 2,000 shots the starter goes from 81 failures
to 51 or 53 with the true loss round: 1.5 to 1.6.

**The gate.** The brief's pass mark is a cut by 1.5 or more; below 1.2, the tracker idea stops. The round-level
ORACLE cuts the failures by 1.64 against the starter's flag-aware decoder (4,000 shots) and by 1.77 against the
decoder with the same map. On the starter's own map it cuts them by 1.53 (2,000 shots). Every point estimate
passes 1.5. The intervals reach down to 1.42 and 1.22, so 1.5 is likely but not certain; 1.2 is excluded on the
4,000 shots. **L3 passes on the point estimate: the tracker idea goes on.** These are oracle numbers, a ceiling
for any tracker of the loss round, not decoder results. Server job A (seed 5300, below) replicates this on 4,000 more shots and adds the slot-level oracle.

Two more observations. The oracle also converges faster: 1,655 mean iterations against 2,339, and 63
unconverged shots against 108. And the exact location weights do not beat the starter's rounder ones
(189 against 176 failures, p = 0.24).

Commands: `loss/runs/L0_L1_s1e-3.sh` and `loss/runs/L3b_s1e-3.sh` (decodes), `loss/runs/summarize.sh`
(tables). Results: `results/L3_X_p0.004_s1e-3.{json,md}` (reference `exact`),
`results/L3_starter_X_p0.004_s1e-3.{json,md}` (reference `starter`), `results/L3_oracles_X_p0.004_s1e-3.{json,md}`.
Logs: `logs/L0_L1_s1e-3.log`, `logs/L3b_part1_laneD.log`.

## Second machine: server job A (L3 replication, seed 5300)

**A second machine's numbers.** The team's Linux server (ICRN), run by Sal: 10 worker processes, checkout of
`loss-gates` at `6ea830ce35d1b6ab912e58f673e3fc542cd7b902`. Failure counts do not depend on the machine; times
do, and none is quoted from it. Command, from `challenge-1-ml-decoder-kit/`:

```sh
python -m loss.job --basis X --p 0.004 --s 1e-3 --shots 4000 --seed 5300 --workers 10 --out ~/mlBranchMidwest/runs/loss/L3_s1e-3_seed5300 \
  --group starter,exact,oracle-round,starter-oracle-round,starter@0.5/stop1:4000 --group oracle-slot:4000 \
  --table starter:starter,starter-oracle-round,oracle-round,oracle-slot,exact,starter@0.5/stop1 --table exact:exact,oracle-round,oracle-slot
```

Dataset: 4,000 shots, seeds 5300 and 5301, sha256 `60c09da58402097fd70cca4e6044fc4affadcfbfbcbc9fb2227694e4642a9310`,
10.08 lost data atoms and 10.33 check heralds per shot. Its log and tables are on the branch `server-logs`
(`loss_jobs.out`, `files/loss/L3_s1e-3_seed5300/tables.md`; "job done in 1682s", no error). Every row on
the same 4,000 shots:

| Condition | Failures / 4,000 | Fail fraction (95% Wilson) | Unconverged (right) | Mean iterations | vs `starter`: only starter / only cond | `starter` ÷ cond (95% bootstrap) |
|---|---|---|---|---|---|---|
| `starter` | 162 | 4.05% (3.48–4.71) | 112 (1) | 2,325 | reference | |
| ORACLE `starter-oracle-round` | 107 | 2.67% (2.22–3.22) | 70 (0) | 1,698 | 86 / 31 (p = 4e-7) | 1.51 (1.28–1.79) |
| ORACLE `oracle-round` | 96 | 2.40% (1.97–2.92) | 54 (0) | 1,634 | 90 / 24 (p = 4e-10) | 1.69 (1.44–2.01) |
| ORACLE `oracle-slot` | 87 | 2.17% (1.77–2.68) | 70 (9) | 1,545 | 111 / 36 (p = 4e-10) | 1.86 (1.54–2.30) |
| `exact` | 181 | 4.52% (3.92–5.21) | 131 (3) | 2,554 | 44 / 63 (p = 0.08) | 0.90 (0.80–1.01) |
| `starter@0.5/stop1` | 178 | 4.45% (3.85–5.13) | 112 (1) | 1,020 | 9 / 25 (p = 0.009) | 0.91 (0.85–0.97) |

Against `exact` (the same map): ORACLE `oracle-round` 1.89 (1.63–2.25), ORACLE `oracle-slot` 2.08 (1.73–2.58).

It replicates every L3 finding of this PC on independent shots. The round-level oracle cuts failures 1.69×
against the starter (this PC: 1.64×). Pooled over the 8,000 shots of both machines, it is 338 → 203, or 1.67×.
The starter-map oracle cuts them 1.51× (this PC, 2,000 shots: 1.53×). The exact weights do not help, and
stopping at the first solution costs failures. The new row is the slot-level oracle: knowing the exact loss CX
instead of the round adds a little more (87 against 96 failures), for 1.86× in all.

## L2: the flag-to-prior map, piece by piece (s = 1e-3)

How far it got: one of the five comparisons is measured on this PC; the rest is server job D, on job A's
dataset (seed 5300), 1,000 shots per condition, its table to come (HANDOFF.md says where):

```sh
python -m loss.job --basis X --p 0.004 --s 1e-3 --shots 4000 --seed 5300 --workers 10 --out ~/mlBranchMidwest/runs/loss/L3_s1e-3_seed5300 --group starter-nofinal,starter-nohook,starter-ratebase:1000 --group exact@0.4,oracle-round@0.4,exact@0.25,oracle-round@0.25:1000 --table starter:starter,starter-nofinal,starter-nohook,starter-ratebase,exact --table exact@0.4:exact@0.4,oracle-round@0.4,exact --table exact@0.25:exact@0.25,oracle-round@0.25,exact --table oracle-round:oracle-round,oracle-round@0.4,oracle-round@0.25
```

| Piece | Status in the starter | Measured here (4,000 shots of the L3 dataset) | Keep (20% fewer failures)? |
|---|---|---|---|
| final-readout flip at 0.5 | in the starter | leave-one-out (`starter-nofinal`): server job D | to come |
| hook columns of flagged check atoms of the other basis | in the starter (its docstring says otherwise) | leave-one-out (`starter-nohook`): server job D | to come |
| lossless priors wherever no flag fired | in the starter | leave-one-out (`starter-ratebase`): server job D | to come |
| location weights (geometric, exact P(lost before), the columns inside a run) | **missing from the starter** | `exact` 189 against `starter` 176 failures; 59 / 46 discordant, p = 0.24; ratio 0.93 (0.83–1.04). Server job A, 4,000 other shots: 181 against 162, 63 / 44, p = 0.08, ratio 0.90 (0.80–1.01) | **no**: it does not lower the failures. Pooled over both machines the discordant shots are 122 (only `exact` fails) against 90 (only `starter`): if anything it hurts slightly |
| cap on a raised prior: 0.5, 0.4, 0.25 | 0.5 | `exact@0.4`, `oracle-round@0.4`, `exact@0.25`, `oracle-round@0.25`: server job D | to come |

The local L2 runs (`loss/runs/L2_s1e-3.sh`, 500 shots) were not reached: the orchestrator moved L2 to server
job D, on the 4,000 shots of job A (seed 5300), 1,000 shots per condition.

## L4: what per-shot priors cost

Machine: this PC (AMD Ryzen 7 7730U), one process on CPU 0 with its sibling thread idle (lane D held), nice 15,
the other lanes busy with other workers. Wall-clock `perf_counter`. Command: `loss/runs/L4_timing.sh`
(`python -m loss.timing`). Data: the L0 dataset (lossless shots) and the L3 dataset (s = 1e-3).

| What | Mean | Median | p95 | n |
|---|---|---|---|---|
| relay-BP decoder construction, lossless priors (H 936 × 8,784) | 7.8 ms | 7.4 ms | 9.9 ms | 30 |
| relay-BP decoder construction, a shot's flag-aware priors | 8.1 ms | 7.4 ms | 10.9 ms | 30 |
| building a shot's flag-aware priors (`starter`) | 0.26 ms | 0.25 ms | 0.36 ms | 200 |
| BP-OSD (`ldpc` 2.4.1) construction | 113 ms | 113 ms | 118 ms | 5 |
| BP-OSD `update_channel_probs` (new priors in place) | 1.7 ms | 1.7 ms | 1.8 ms | 30 |
| relay-BP decode, lossless shots, one decoder reused | 571 ms | 76 ms | 1,458 ms | 100 |
| relay-BP build + decode, lossless shots, a new decoder per shot | 625 ms | 96 ms | 1,206 ms | 100 |
| flag-aware at s = 1e-3, per shot: priors + build + decode | 1,785 ms (0.5 + 7.9 + 1,776) | 248 ms | 10,658 ms | 30 |

- **Relay-BP takes its priors only at construction.** Its Python methods are `decode`, `decode_batch`,
  `decode_detailed` and `decode_detailed_batch`. So per-shot priors mean a new decoder per shot: about 8 ms,
  against a median decode of 76 ms (lossless) or 239 ms (flag-aware, s = 1e-3). That is 10% and 3% of the
  median, and 1.4% and 0.4% of the mean, which the shots that run all 300 legs dominate.
- The reused and the rebuilt decoder are not the same experiment: a reused decoder carries its random state
  from shot to shot, so the two decode-time rows differ by more than the 8 ms of construction. Their medians
  differ by 20 ms; their means are dominated by a few long shots.
- **BP-OSD can change its priors without a rebuild.** `update_channel_probs` takes 1.7 ms, against 113 ms for
  a new decoder. Checked: on 20 loss shots, a decoder built with the lossless priors and then updated to each
  shot's priors decoded every shot exactly as a decoder built with that shot's priors (20 of 20 identical).
- Clifft, single thread, on this core: 220 shots/s without loss, 177 shots/s at s = 1e-3, 162 shots/s with the
  probes (1,000 shots each, compilation included).

Results: `results/L4_timing.json`. Log: `logs/L4_timing.log`.

## L5: the flicker table

Moved to the next task, `loss-tracker`, by the orchestrator (2026-11-22, 05:40 UTC). `loss/flicker.py` is
written and tested, and `loss/runs/L5_flicker.sh` names the command, but this task did not run it. One pilot run on
200 shots (s = 1e-3, seed 999, not a result) showed what to expect. Neighbouring detectors fired at 0.12 to 0.19
before the loss round and at about 0.5 from the round after it, in X and Z checks alike.

## Datasets (in /home/sal/hackMidwest/orchestration/scratch/loss-gates/data/, not committed)

All: X memory, p = 0.004, 20,000 shots (10,000 for L0), Clifft 0.13.0, probes on, chunks of 2,000 shots with
seeds seed, seed + 1, ... (`loss/sample.py`).

| File | s | Seed | Lost data atoms per shot | Check heralds per shot | sha256 |
|---|---|---|---|---|---|
| `X_p0.004_s0_seed1000.npz` | 0 | 1000 | 0 | 0 | `34a7fd8c43ba1aa69210253c6691b6ead34167f7facd83d5cb18778208c620b8` |
| `X_p0.004_s2e-4_seed2100.npz` | 2e-4 | 2100 | 2.07 | 2.08 | `797d4d06f9e12f9f5a0c70b3a823827f1fd1cccf9f141ff37fdc2f0eb186f2ce` |
| `X_p0.004_s5e-4_seed2200.npz` | 5e-4 | 2200 | 5.11 | 5.17 | `318cc463218072218a0dba4f86d9de26d8ca8cb6a2ee3b3c0b7742692284056a` |
| `X_p0.004_s1e-3_seed2300.npz` | 1e-3 | 2300 | 9.99 | 10.36 | `378ee22c9cdf1f25ac44a72f68931f925c67e529ecb65485326afe44b4c3d532` |
| `X_p0.004_s2e-3_seed2400.npz` | 2e-3 | 2400 | 19.29 | 20.64 | `e9eed3c8fad2719a120358c0884cc5937b284cc0eb1a8f6b63b8f4eeb8d86474` |

The expected numbers per shot, 144 (1 − (1 − s)^72) lost data atoms and 1,728 (1 − (1 − s)^6) check heralds,
are 2.06/2.07, 5.09/5.18, 10.01/10.34 and 19.33/20.63: the measured ones agree, which confirms that s is per atom
per CX. The per-shot decode results are in `scratch/loss-gates/parts/` (one file per part; `summarize.py`
reads them).

**Reproducibility.** Every shot and condition gets a fresh relay-BP decoder with seed 1, so a result depends only
on the shot and the condition. Checked inline (not a committed script): 12 shots rerun with 1 worker give the same failures,
convergence, iterations and wrong-observable masks as the run with 4 workers. Clifft's shots do not depend on its thread
count, and the probes do not change them. The bootstrap uses seed 0. Rerunning a command gives the same table.

## What this says

- **Loss without its flags is ruinous for relay-BP.** On lossless priors, relay-BP fails 14% of the shots at
  s = 5e-4 (5 lost data atoms per shot), 58% at 1e-3 and 99% at 2e-3, mostly by not converging. Without loss
  it fails 0.64%.
- **The heralds give most of it back.** With the starter's flag-aware priors the failures fall to 1.45%, 4.4% and
  23%: 11.4×, 11.5× and 4.5× fewer on paired shots. A loss-aware decoder never lost a shot that blind got right
  (1, 0 and 0 such shots). Loss still costs: 4.4% at 1e-3 is 7 times the lossless 0.64%.
- **Knowing when each data atom was lost is worth about 1.6×** at s = 1e-3. ORACLE: 176 → 107 failures with the
  true loss round, interval 1.42–1.94. This is the ceiling for a tracker of the loss round, and it clears the
  brief's 1.5 on the point estimate.
- **Not worth it:** the exact location weights (no gain over the starter's rounder ones), and stopping relay-BP at
  its first solution (14% more failures under loss).
- **The map from a loss to the decoder's columns is exact in the simulator.** It held for every shot tested,
  observables included, in both bases. The one exception to the 1/2 rate is in Z memory (0.7% of data losses).
- **Not measured here:** Z memory decoding, p = 0.003 and 0.005, L2's leave-one-out and cap sweep (server job
  D), L1 at 2e-4 and the rest of 5e-4 (server jobs B, C), and the flicker table (task `loss-tracker`).
