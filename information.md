# Information: what we are building, why, and where everything is

Last updated: 2026-11-22 05:50 UTC. This file is the overview of our part of the team's work on
Infleqtion's Challenge 1 (ML-assisted decoding of the Gross code). It lives on the branch
`integration`, our one working branch: everything we have built is merged here.

Read this first, then the README of the folder you care about. Every number below names where
it was measured. **Unless a line says otherwise, numbers are on exploration shots (the train
and validation files), not on the held-out test files, and the code behind them is not reviewed
yet.**

## 1. In one paragraph

We are building a hierarchical decoder for the Gross code `[[144, 12, 12]]`. A fast decoder runs
twice with different random seeds. When its two answers agree, that answer is final. When they
disagree, which happens on about 1 shot in 250, a third decoder joins and a neural network
that follows the code's structure picks the most likely answer. On 50,000 practice shots at
p = 0.004 the version without the network fails 34% fewer shots than relay-BP in about a
quarter less time, and with the network on X memory it fails 47% fewer. NVIDIA's GPU decoder
TCBP-JOBS, which teammates ran as an extra reference, is stronger still at p = 0.004 and 0.005,
so the next stage adds stronger members and lets the network choose among them. Beside it we
study atom loss, the part the challenge stresses: a decoder that ignores loss fails on most
shots, one that uses the loss flags on about 4%, and we measured that knowing when an atom was
lost would cut that 1.6 times more, so we are building a tracker that estimates that time from
the syndrome history.

## 2. The challenge

From the organisers' statement (`README.md` at the top of the repository):

- **Task:** build an ML-assisted decoder that improves on BP-OSD and relay-BP for the Gross
  code, in accuracy, speed, or the handling of atom loss. Hierarchical decoding is named as one
  approach. The stated goal is to beat the baseline "particularly when loss information is
  available".
- **Required experiment:** a sweep of our decoder over the physical error rates 0.003, 0.004
  and 0.005, against both baselines, on the same test data and noise.
- **Required metrics:** logical error rate, decoding latency and resource requirements, with a
  discussion of where the approach works and where it falls short.
- **Deliverables:** a working decoder, the code that makes the data, a script that reproduces
  the evaluation, at least one plot, and a short presentation.
- **Stretch goals:** a trained router, a second code, sliding windows, scaling with distance,
  hardware cost.

## 3. The experiment and the baselines

Defined by the kit (`challenge-1-ml-decoder-kit/README.md`):

- **Memory experiment.** 144 data qubits hold 12 logical qubits. X memory prepares |+>, runs 12
  noisy rounds of checks, and measures. Z memory is the same with |0>.
- **What a decoder sees.** One shot gives 1,728 detector bits. The baselines decode X memory
  from the 936 bits on X checks, a problem with 8,784 possible error mechanisms. Using all
  1,728 bits ("joint decoding") gives a problem with about 68,000 mechanisms.
- **Failure.** A shot fails when any of the 12 logical bits is predicted wrong. Shot k of X is
  paired with shot k of Z, and the pair fails when either fails.
- **Rate.** With failure fraction P over the pairs, the rate per cycle is `1 - (1 - P)^(1/12)`.
- **Noise.** One strength p on every operation. Each decoder gets the exact error model.

Baselines, 10,000 shots per basis (the kit's README, the organisers' reference run):

| p | Decoder | Failures X, Z | Rate per cycle | Mean time per pair |
|---|---|---|---|---|
| 0.003 | BP-OSD | 26, 30 | 4.7e-4 | 0.14 s |
| 0.003 | relay-BP | 3, 2 (43, 54 of 100,000) | 4.2e-5 (8.0e-5) | 0.10 s |
| 0.004 | BP-OSD | 253, 269 | 4.4e-3 | 0.33 s |
| 0.004 | relay-BP | 45, 60 | 8.7e-4 | 0.38 s |
| 0.005 | BP-OSD | 1302, 1286 | 2.3e-2 | 0.59 s |
| 0.005 | relay-BP | 411, 414 | 6.9e-3 | 1.7 s |

Relay-BP is the stronger of the two baselines the challenge names. A third reference, run by
teammates on a GPU, is NVIDIA's TCBP-JOBS (`tcbp_jobs_baseline.py`, logs in `results/` on
`main`; its own sampled shots, 10,000 pairs each):

| p | TCBP-JOBS failures X, Z | Rate per cycle | Against relay-BP |
|---|---|---|---|
| 0.003 | 2, 3 | 4.2e-5 | equal |
| 0.004 | 18, 23 | 3.4e-4 | 2.5 times lower |
| 0.005 | 192, 188 | 3.2e-3 | 2.2 times lower |

- **TCBP-JOBS is therefore the real bar at p = 0.004 and 0.005.**
- Its logged times (4.3 ms, 42.9 ms and 81.7 ms per pair) are not comparable yet: the runs at
  p = 0.004 and 0.005 shared the GPU with one of our trainings, so they are inflated. They need
  repeating on an idle GPU before any latency is quoted.
- Its rows come from its own sampled shots, not from the team's saved test files.

## 4. Who does what

- **Teammates** own the shared evaluation: the saved shot files for all three noise levels
  (`data/p0.003`, `data/p0.004`, `data/p0.005`: train 20,000, validation 5,000 and test 10,000
  shots per basis), the evaluator that scores any decoder on them (`evaluation/`), the baseline
  rows, the TCBP-JOBS baseline, and a trained router on relay-BP's first answer
  (`evaluation/router*.py`, `results/router_p0.00x/`). They push to `main`.
  - Their router on the p = 0.003 test shots (`results/icrn/`): 6 failed pairs against
    relay-BP's 5, at 47 ms against 86 ms per pair. That is relay-BP's accuracy in 45% less time.
- **This document covers our line:** the hierarchical decoder, the neural network, the loss
  study, and the measurements that decided the design. Our decoder plugs into the team's
  evaluator, so every row of the final table comes from the same saved test shots.

## 5. The pipeline

```mermaid
flowchart TD
    S["One shot: detector bits"] --> P1["PriorTilt, seed A"]
    S --> P2["PriorTilt, seed B"]
    P1 --> Q{"Same 12 logical bits?"}
    P2 --> Q
    Q -- "yes: about 99.6% of shots at p = 0.004" --> A["Answer"]
    Q -- "no: about 0.4%" --> R["relay-BP gives a third candidate"]
    R --> N["Neural network reads all 1,728 bits and picks the most likely candidate"]
    N --> A
    N -. "planned" .-> J["Joint decoding of both check types for the shots that stay doubtful"]
    J -.-> A
```

1. **Stage 1, fast: PriorTilt twice.** PriorTilt is IonQ's belief-propagation decoder with 14
   fitted parameters, published for this code and noise in November 2026. We feed it the kit's
   own check matrix and priors. Alone it fails 15% fewer shots than relay-BP at p = 0.004 at
   about a third of the time. We run it twice with different seeds.
2. **The agreement test.** If both runs predict the same 12 logical bits, the answer is final.
   The two runs differ on 0.40% of X shots, and those shots hold most of the failures.
3. **Stage 2, on disagreement only: relay-BP and a learned tie-breaker.** The kit's relay-BP
   gives a third candidate. The neural network of section 6 reads the shot's full syndrome and
   scores each candidate; the most likely one wins. Without a network the rule is "lowest
   weight".
4. **Stage 3, planned: joint decoding.** Decoding both check types together fixes about 9 in 10
   of relay-BP's failures, but costs about 70 to 100 times the time, and run on every shot its
   own mistakes cancel most of the gain at p = 0.004. Behind a filter that sends it only the
   doubtful shots it should pay. The network's confidence is the natural filter.

**What is learned:** PriorTilt's parameters (fitted by its authors), the neural tie-breaker
(trained by us), and later the filter for stage 3 and the loss tracker. **What is not:** the
agreement test, relay-BP, and the scoring.

Measured on the team's p = 0.004 train and validation files (laptop CPU; times were taken
beside a GPU training, so compare them with each other only):

| Rule | Shots | Failures | Mean time per shot |
|---|---|---|---|
| relay-BP, the kit's settings | 50,000 (X and Z) | 245 | 169 ms |
| PriorTilt once | 50,000 | 208 | 54 ms |
| PriorTilt twice, relay-BP on disagreement, lowest weight | 50,000 | **162 (34% fewer)** | 124 ms |
| relay-BP | 25,000 (X only) | 122 | 167 ms |
| the same rule with lowest weight | 25,000 (X only) | 77 | 118 ms |
| **the same rule with the network as tie-breaker** (step 40,000 of its training) | 25,000 (X only) | **65 (47% fewer)** | 118 ms |
| a perfect tie-breaker on the same candidates | 25,000 (X only) | 50 | |

- The paired comparison of PriorTilt against relay-BP: 127 shots only relay-BP fails, 90 only
  PriorTilt fails, p = 0.014.
- 29 X shots have two agreeing wrong answers. No tie-breaker fixes those; only a better filter
  or stage 3 can.
- The network gained little as a tie-breaker between steps 30,000 and 40,000 (66 to 65) while
  improving a lot on its own. The hard shots are hard for it too.
- **Not measured yet:** this decoder at p = 0.003 and 0.005 (running), on Z with a network
  (the Z network started training on 2026-11-22), through the team's evaluator, and on the test
  shots. The one run on the test shots happens once, with every setting frozen.
- First number at p = 0.005 (20,000 X shots made on the laptop): PriorTilt alone fails 612
  against relay-BP's 883, 31% fewer, at 201 ms against 937 ms per shot.

**Where this stands against TCBP-JOBS.** At p = 0.004 the hierarchy is at about 5e-4 per cycle
by estimate from the practice shots: below relay-BP (8.7e-4) and above TCBP-JOBS (3.4e-4). Even
a perfect tie-breaker over its three candidates would not pass TCBP-JOBS. The next stage is
therefore an ensemble: stronger members where the fast ones disagree or look doubtful (joint
decoding, more relay-BP re-rolls, TCBP-JOBS itself), and the network choosing among all of
them. The evidence that this can work: with 16 relay-BP candidates per shot the right answer is
among them in 79% of relay-BP's converged-wrong failures, while "lowest weight" picks it in
only 49% (section 8). A set of decoders whose mistakes differ can beat each of its members.

Code: `challenge-1-ml-decoder-kit/hierarchy/` (`candidates.py`, `select.py`, `rules.py`,
`stages.py`; the evaluator plugin is being written). Its README has the install and the
commands.

## 6. The neural network

Folder `challenge-1-ml-decoder-kit/neural/`. Its README explains the design in plain words and
lists every measured number.

- **Recipe.** A convolutional network after the "Cascade" decoder (arXiv:2604.08358), the one
  published neural recipe that works on this code, scaled down to one laptop GPU. No code was
  copied; the authors released none.
- **Three ingredients.** Layers that pass information along the code's own connections (each
  check touches 6 data qubits, each data qubit 6 checks, the same weights at all 72 positions
  of the torus). Pooling on the logical operators, so the output is the 12 logical bits. A
  curriculum that starts training at low noise and climbs.
- **Input and output.** All 1,728 detector bits in, 12 logical bits out. One network per
  memory basis. The logits give a confidence.
- **Training data.** Shots drawn fresh from the kit's error model at every step, so no shot is
  seen twice and there is no dataset to memorise. Scores come from the kit's Stim shots.
- **Size and speed.** Width 128, depth 8, 0.89 M parameters, about 725 shots per second on an
  RTX 4060. A width-256 version (3.6 M parameters) trains on an A100 slice.
- **Z memory needed a fix.** Five of the kit's Z logical operators are heavy (24 to 38 qubits)
  and learn badly. The network is trained on 12 lighter operators from which the kit's 12 bits
  follow exactly (checked on Stim records). The X network's weights, mirrored, give Z a head
  start: before any Z training it already gets 97.8% of shots right at p = 0.001.

X network, width 128, scored on the team's Stim shot files (train plus validation, 25,000 shots
per noise level):

| Training step | p = 0.003 | p = 0.004 | p = 0.005 |
|---|---|---|---|
| 30,000 of 230,000 | 8.3% of shots wrong (other files) | 26.9% | 55.9% (other files) |
| 40,000 | 6.1% | 21.7% | 48.4% |

- Alone it is far from relay-BP (0.5% at p = 0.004). Its use is the tie-breaker of section 5.
- Its most confident quarter of shots is about as clean as the classical decoders (3 failures
  in 6,250 shots at p = 0.004, where relay-BP has 3).
- The run ends at step 230,000. The width-256 run and the Z run are behind it.

An earlier plain CNN (`ml_decoder/`, pull request #1) is the stepping stone: a first stage with
a confidence threshold in front of relay-BP. Its default size learned nothing in 50 minutes;
a small one learned the low noise levels only. See section 9.

## 7. Atom loss

Folder `challenge-1-ml-decoder-kit/loss/`; `RESULTS.md` there is the full record.

- **Model (our assumption, stated as such).** Simulated with Clifft's `noncomp` module on the
  kit's circuit. After every CX each of its two atoms is lost with probability s. A lost check
  atom is flagged at its measurement and replaced. A lost data atom stays lost and is flagged
  only at the end. Flags are perfect. s = 2e-3 is about today's hardware by Infleqtion's own
  paper (arXiv:2408.08288); lower values are targets. This is not "Infleqtion's noise model":
  it is a model built from their published figures.
- **How a loss reaches the decoder.** A map from each possible loss to the columns of the
  decoding problem it can flip, checked against the simulator. A third of the data-atom losses
  land on a column the starter script never raises.
- **Flag-aware decoding** raises the priors of those columns for flagged atoms, spreading a
  data atom's loss evenly over all possible times (the published method).

X memory, p = 0.004, s = 1e-3 per atom per CX, relay-BP at the kit's settings, Clifft shots
(about 10 lost data atoms per shot):

| Decoder | Shots | Failures |
|---|---|---|
| loss-blind (ignores the flags) | 300 to 400 (a first, small run) | 57.7% of shots |
| flag-aware, uniform guess over the loss time | 4,000 | 176 (4.40%) |
| flag-aware with exact location weights | 4,000 | 189 (no gain) |
| flag-aware, stopping relay-BP at its first solution | 4,000 | 200 (14% more) |
| **ORACLE: told the true loss round** (a ceiling, not a decoder) | 4,000 | **107 (2.67%)** |

- The oracle fails 1.64 times less often than the flag-aware decoder (paired 94 against 25
  shots, interval 1.42 to 1.94). So the timing of a loss is worth knowing.
- At s = 2e-3, about today's hardware: the loss-blind decoder fails 148 of 150 shots, the
  flag-aware one 173 of 750 (23.1%).
- **The loss-time tracker (next task).** A lost data atom makes its six neighbouring checks
  fire at random from the moment it is lost. A hidden Markov model over that history, then a
  small network, estimates the loss round and feeds it to relay-BP as per-shot priors. No paper
  reports this for a BP decoder. Target: recover at least half of the oracle's gain.

## 8. Measurements that decided the design

Folder `challenge-1-ml-decoder-kit/analysis/`; `RESULTS.md` there has the tables and commands.

| Question | Answer | Where |
|---|---|---|
| How does relay-BP fail at p = 0.004? | 245 failures in 50,000 shots: 100 did not converge, 145 converged to a wrong answer. 224 of them are among the 14% of shots whose first leg does not converge. | `analysis/`, G1 |
| What fixes those failures? | A re-run with other random draws: 108 of 245. BP-OSD: 79. Tesseract: 80. Joint decoding: about 9 in 10. | G2 |
| Is the right answer among relay-BP's candidates? | With 16 candidates per shot it is there in 114 of the 145 converged-wrong failures (79%). The lowest weight picks it in 71 (49%), a vote in 69. Weight is the wrong score; a learned one has room. | G2 |
| Joint decoding at p = 0.003 | Fixes 43 of relay-BP's 44 failures, breaks 0 of 300 control shots, about 9 s against 0.13 s per shot. | G2 |
| Joint decoding at p = 0.004 | Fixes 89 of 100 failures, breaks 1 of 200 controls, 135 s per failed shot and 31 s per control shot on our slow PC. | G2 |
| Can relay-BP's schedule be faster for free? | Stopping after 3 solutions instead of 5 costs nothing measurable (42 against 51 failed pairs) at 1.44 times fewer iterations. Legs of 30 iterations cost accuracy badly (93 to 108). | G3, 16 schedules |
| Does a Stim seed give the same shots on every machine? | No. Paired comparisons need saved shot files. | `ml_decoder/stim_seed_check.py` |
| Do two machines decode the same shots the same way? | Yes: our PC, a laptop (Windows) and a Linux server agree shot for shot on relay-BP and PriorTilt. | G3 machine check; `hierarchy/` |

The relay-BP authors' own data agree on joint decoding: about 2.5 times fewer failures at
p = 0.003, 1.3 times at 0.004, worse at 0.005, at about 100 times the time
(https://github.com/trmue/relay, `examples/data`).

## 9. What we tried that did not work

Each is kept with its numbers, as the challenge asks where an approach falls short.

- **The default plain CNN** (5.6 M parameters, batch 2,048): after 50 minutes it equalled the
  always-0 guess at every noise level.
- **A small plain CNN** learned low noise only: after 117 M shots, 90.6% of shots still wrong
  at p = 0.003. Plain CNNs stall without the code's structure.
- **A faster relay-BP schedule as the fallback** (40 / 30 / stop at 1): 49 failures against 20
  on 5,000 shots.
- **Classic predecoders** (commit to errors early, pass the rest on): no gain in our tests.
  NVIDIA's released Ising predecoders are for surface and colour codes and cannot read this
  code's syndrome.
- **Exact location weights for loss:** no gain over the simple map (189 against 176).
- **Stopping relay-BP at the first solution under loss:** 14% more failures.
- **Longer training as a tie-breaker:** 66 to 65 failures from step 30,000 to 40,000.

## 10. Result sets planned for the submission

| Set | Noise | Compared | State |
|---|---|---|---|
| A (required) | the kit's circuits at p = 0.003, 0.004, 0.005 | our hierarchy against BP-OSD and relay-BP on the team's test shots; PriorTilt and TCBP-JOBS as extra rows | decoder being finished; test run not done |
| B | A plus atom loss at several rates | loss-blind, flag-aware, and ours with the tracker | flag-aware and oracle measured at one rate |
| C | a neutral-atom-style model from Infleqtion's published figures | each decoder with generic and with matching priors | not started |
| D | decode at one p with priors for another | every decoder | not started |
| E | what did not work | section 9 | being collected |

## 11. Rules we hold ourselves to

- Final numbers come from the kit's Stim shots in saved files, the same files for every decoder.
- Three splits with different seeds. Everything is chosen on train and validation shots. The
  test shots are decoded once, at the end. Our tools refuse `test_*` and `go_*` files by name.
- The truth (the observables, the true loss times) reaches training and scoring only, never a
  decoder at decode time. Oracle rows are labelled ORACLE.
- "Better" means paired counts on the same shots, with a test or an interval.
- Every latency names its machine and what the timer wraps; every stage of a hierarchy counts.
- A result that looks too good is treated as a bug until shown otherwise.
- All of our work is on one branch, `integration`. Each task owns its own folder there, and
  an independent review re-runs its numbers before the branch merges into `main`.

## 12. Where everything is

Branches of this repository:

| Branch | What it holds | State |
|---|---|---|
| `main` | the kit, and the teammates' evaluator, shot files, router and baselines | shared by the team |
| **`integration`** | **all of our work, and this file** | our one working branch; not reviewed as a whole yet |
| `server-logs` | training logs and small result files pushed from the GPU server every 5 minutes | data only, never merged |
| `laptop-artifacts` | the laptop's per-shot decode files, network checkpoints and scores | data only, never merged |

Five older task branches are being retired. Their work is already merged into `integration`,
and each is deleted once its last commits are in:

| Old branch | Its folder on `integration` | State |
|---|---|---|
| `MLDecoder` | `ml_decoder/` (plain CNN first stage), `starter_scripts/` (loss and sizing starters), `fpga_router/` | pull request #1, in its second review |
| `gates` | `analysis/`: the measurements of section 8 | finishing |
| `loss-gates` | `loss/`: section 7 | finishing |
| `cascade` | `neural/`: section 6 | finishing |
| `hierarchy` | `hierarchy/`: section 5 | in progress |

Folders (inside `challenge-1-ml-decoder-kit/` unless noted):

| Folder | Read first |
|---|---|
| `hierarchy/` | `README.md` |
| `neural/` | `README.md` |
| `loss/` | `RESULTS.md` |
| `analysis/` | `RESULTS.md` |
| `ml_decoder/` | `README.md` |
| `evaluation/`, `data/`, `results/` | `evaluation/README.md` (the teammates') |
| `fpga_router/` (top level) | `README.txt`: Verilog for a router's decision, simulated with stand-in weights; a stretch goal |

Main commands, all from `challenge-1-ml-decoder-kit/` with Python 3.12 and the kit's
`requirements.txt` (each folder's README has the full list):

```sh
# the three candidates of every shot of a noise level, and the rules that need no network
python -m hierarchy.candidates --data data/p0.004 --out runs/cand_p0.004 --processes 12
# score the rules again, with the network's per-shot logits
python -m hierarchy.select --candidates runs/cand_p0.004/validation_X.chunks.npz
# train the X network, then the Z network from it
python -m neural.train --out runs/cascade_X_s1 --seed 1 --steps 230000
python -m neural.train --basis Z --logicals light --init-from runs/cascade_X_s1/checkpoint.pt --out runs/cascade_Z_s1 --seed 1 --steps-start 1000 --steps-anneal 6000 --steps 230000
# score a network on saved Stim shots, and save its per-shot logits
python -m neural.evaluate_stim --checkpoint runs/cascade_X_s1/checkpoint.pt --data data/p0.004/validation_X.npz --save runs/eval/val_X
```

Machines: a 16-thread PC for the CPU studies, a laptop with an RTX 4060 (8 GB) for the X
network and CPU decoding, and a university notebook server (an A100 slice of 20 GB, 12 CPUs)
for the wider X network, the Z network and large decoding runs. Much of the code was written
with AI coding assistants under our direction, which the challenge allows; this file and the
READMEs are how we keep being able to explain it.

## 13. What is next

1. Finish the `hierarchy` evaluator plugin; decode the practice shots at p = 0.003 and 0.005.
2. Train the Z network; keep training X.
3. One run of the frozen decoder on the team's test shots at the three noise levels, through
   the team's evaluator, beside the baseline rows.
4. The ensemble: joint decoding, more relay-BP candidates and TCBP-JOBS as further members,
   with the network as the selector and as the filter that decides when the slow members run.
   This is what can pass TCBP-JOBS.
5. The loss-time tracker, then loss at other rates and in Z memory.
6. The plots, the reproduce script, the README for the judges and the slides.
7. Stretch: the realistic-noise set, the mismatch test, the FPGA router, a second code.

## 14. Plain-words glossary

- **Gross code `[[144, 12, 12]]`:** 144 physical qubits protect 12 logical qubits; any 5
  errors can be corrected (distance 12).
- **Check, syndrome, detector:** a check measures a parity of 6 data qubits each round. A
  detector is the comparison of a check with itself one round earlier; it fires when something
  changed. The pattern of fired detectors is the syndrome.
- **Decoder:** the program that reads the syndrome and says which logical bits flipped.
- **Logical error rate:** how often the decoder's answer is wrong, here per cycle.
- **Belief propagation (BP):** a fast message-passing decoder. It often fails to settle on
  these codes.
- **BP-OSD:** BP followed by a slower algebraic clean-up. The older baseline.
- **Relay-BP:** many BP runs in a chain with changed memory strengths, stopping after a few
  solutions and keeping the lightest. The strong baseline.
- **PriorTilt:** BP with a few fitted parameters and a random stage, from IonQ.
- **Joint decoding:** decoding X and Z checks together. More accurate at low noise, far slower.
- **Hierarchical decoding:** a fast decoder for the easy shots, a stronger one for the hard
  ones, and a rule (a router) that tells them apart.
- **Atom loss, herald:** in neutral-atom machines an atom can leave its trap. A herald, or
  flag, is the measurement telling us an atom is gone.
- **Oracle:** a test that is given the truth on purpose, to measure the best any method could do.

## 15. Sources

What we build on directly. The figures we quote from papers were read by a research pass and
are not all checked against the papers by us: check the source before quoting one.

| Source | What we use it for |
|---|---|
| Bravyi et al., "High-threshold and low-overhead fault-tolerant quantum memory", Nature 627, 778 (2024) | the Gross code and its syndrome circuit (through the kit) |
| The relay-BP paper, arXiv:2506.01779; code and data at https://github.com/trmue/relay; PyPI `relay-bp` 0.2.2 | the strong baseline, its settings, and the authors' published failure counts for split and joint decoding |
| Ye and Delfosse (IonQ), PriorTilt, arXiv:2611.13373; https://github.com/ionq-publications/PriorTiltDecoder at commit `8eefd37` | stage 1 of the hierarchy. Licence CC BY-NC-SA 4.0: used as a dependency, none of its code is in this repository |
| Gu, Bonilla Ataides, Lukin, Yelin, "Scalable Neural Decoders for Practical Fault-Tolerant Quantum Computation" (the "Cascade" decoder), arXiv:2604.08358 | the recipe of the neural network (no code released; ours is written from the paper's description) |
| NVIDIA TCBP-JOBS (run by teammates, `tcbp_jobs_baseline.py`) | the GPU reference row |
| arXiv:2609.37629 and arXiv:2601.17757 | evidence that disagreement between decoders predicts failure far better than syndrome weight |
| Murri, "Maximally-Matched Mixture Decoding for Correlated Atom Loss in Quantum LDPC Codes", arXiv:2610.38283 | loss on the Gross code with relay-BP: the uniform mixture over loss times, and the tracker named as future work |
| Baranes et al., "Leveraging Qubit Loss Detection in Fault Tolerant Quantum Algorithms"; Liu et al., "Low-Overhead Surface Code with Delayed Atom-Loss Detection via Pauli Envelope", arXiv:2603.04156; Suchara, Cross, Gambetta, "Leakage Suppression in the Toric Code" | the recipe for loss-aware priors |
| Varbanov et al., "Leakage detection for a transmon-based surface code", arXiv:2002.07119 | the hidden Markov model that the loss tracker adapts |
| Wang et al., "AI-Enabled Qubit Loss Detection and Replenishment Strategy for Atom Array Quantum Computing", arXiv:2604.14269 | a network that locates lost qubits from the syndrome |
| Radnaev et al., arXiv:2408.08288; arXiv:2412.07670; Rines et al., arXiv:2509.13247 | Infleqtion's measured loss and error rates, and their five-level circuit model |
| Stim (https://github.com/quantumlib/stim); Clifft v0.13.0 (https://github.com/unitaryfoundation/clifft, arXiv:2604.27058); `ldpc` (BP-OSD); qLDPC (https://github.com/qLDPCOrg/qLDPC) | simulators and decoder libraries |
| NVIDIA Ising decoding (https://github.com/NVIDIA/ising-decoding) | read for the predecoder idea; its models do not apply to this code |
