# Material for a five-minute talk

Eight slides. Each has a title that states its point, the figure or table to show, and three lines to say. Every
number is in a generated table (`report/tables/`) or a file under `report/data/`; the source is named beside it.
"Practice shots" below means the exploration shots (the team's train and validation files): numbers on them are not
results on the test shots, and the slides say so each time.

> **Status:** a draft. Slides 4, 5 and 7 are finished when the rows on the test shots have landed.

Words used, each with its meaning:

- **Decoder:** the program that reads which parity checks fired and says which of the 12 stored bits flipped.
- **Shot:** one run of the memory experiment. It **fails** when the decoder gets any of the 12 bits wrong.
- **Pair:** one X-memory shot plus one Z-memory shot. The kit scores pairs; a pair fails when either shot fails.
- **p:** the strength of the noise. Higher p, more errors.
- **BP-OSD, relay-BP:** the two baseline decoders of the kit. Relay-BP is the stronger one.
- **PriorTilt:** a fast decoder published by IonQ for this code. It is not ours; we build on it.
- **TCBP-JOBS:** NVIDIA's decoder, which runs on a GPU. The teammates ran it as an extra reference.

---

## Slide 1. Two fast decodes that agree are almost always right; a neural network settles the rest

**Show:** the diagram of `REPORT.md`, section 2.

- A decoder reads the syndrome, the list of checks that fired, and says which of the 12 stored bits flipped.
- We run a fast decoder twice with different random seeds. If the two answers agree, that is the answer.
- If they differ, a third decoder joins, and a neural network picks the answer it finds most likely.

## Slide 2. The failures hide in the <!--f:expl_disagree_share-->0.4%<!--/f--> of shots where the two decodes disagree

**Show:** `figures/where_failures.png` (practice shots, p = 0.004).

- On <!--f:expl_shots-->50,000<!--/f--> practice shots the two runs disagree on only <!--f:expl_disagree-->214<!--/f--> shots.
- Those <!--f:expl_disagree-->214<!--/f--> shots hold <!--f:expl_one_run_wrong_disagree-->145<!--/f--> of the <!--f:expl_one_run_wrong-->208<!--/f--> failures of a single run. So "do they agree?" is a cheap alarm for a hard shot.
- <!--f:expl_left_agree-->63<!--/f--> failures are shots where both runs agree on a wrong answer. No tie-break can fix those: that is our floor.

Source: `report/tables/exploration_p0.004.md`, the last table.

## Slide 3. The network is a weak decoder but a useful judge

**Show:** `figures/learning_curves.png`.

- The network reads all 1,728 detector bits of a shot. It is built along the code's own wiring, with the same weights
  at all 72 positions, and has <!--f:net_parameters-->893,697<!--/f--> parameters.
- Alone it is far worse than relay-BP, so we never use it alone. We only ask it which of two or three given answers
  is more likely.
- On 25,000 practice shots of X memory, the hierarchy fails <!--f:expl_X_C-->67<!--/f--> times with its best classical tie-break and <!--f:expl_X_N-->59<!--/f-->
  times with the network (at training step <!--f:expl_X_N_step-->70,000<!--/f-->); relay-BP fails <!--f:expl_X_R-->122<!--/f--> times.

Source: `report/tables/exploration_p0.004.md` and `report/tables/networks.md`.

## Slide 4. On the test shots: *title pending the measured rows*

**Show:** the table below and `figures/ler_vs_p.png`.

<!-- BEGIN table:test_sweep -->
<!-- END table:test_sweep -->

- <!--f:framing-->not measured<!--/f-->
- *Pending.* Our decoder against relay-BP at each p, as a paired count on the same 10,000 pairs.
- *Pending.* Where TCBP-JOBS stands on the test shots. On the practice shots it is ahead of us (see the questions below).

## Slide 5. Time: *title pending the measured rows*

**Show:** `figures/time_vs_p.png`, then `figures/ler_vs_time.png`.

- Every time is per pair, with every stage counted, one process per row on the same server.
- *Pending.* Our mean time against relay-BP's, and the slowest shots (the p99 column of the table).
- A GPU row (TCBP-JOBS) is marked as such: it is a different kind of machine.

## Slide 6. With atom loss, the flags matter most, and knowing when an atom was lost is worth more still

**Show:** `figures/loss.png`.

- In a neutral-atom machine an atom can leave its trap. A flag is the measurement that tells us it is gone.
- At a loss rate of 0.001 per atom per gate, relay-BP that ignores loss fails <!--f:loss_blind_share-->57.7%<!--/f--> of shots
  (<!--f:loss_blind-->173 of 300<!--/f-->); with the flags it fails <!--f:loss_aware_share-->4.4%<!--/f--> (<!--f:loss_aware-->176 of 4,000<!--/f-->), <!--f:loss_aware_ratio-->11.5<!--/f--> times fewer on the shots both decoded. Loss without its flags is ruinous; the flags give most of it back.
- An ORACLE that is told the true round of each loss fails <!--f:loss_oracle_share-->2.7%<!--/f--> (<!--f:loss_oracle-->107 of 4,000<!--/f-->, the same shots):
  <!--f:loss_oracle_ratio-->1.64<!--/f--> times fewer than with the flags alone (<!--f:loss_oracle_ratio_interval-->1.42 to 1.94<!--/f-->).
  That is a ceiling, not a decoder. Our tracker tries to estimate that round from the syndrome. *Pending: its result.*

Source: `report/tables/loss.md`. The loss model is our own simplified one, not Infleqtion's hardware model.

## Slide 7. What did not work, and where we fall short

**Show:** four lines of `report/tables/background.md`.

- A plain neural network learned only at noise far below the challenge's; the network that follows the code's wiring
  learns at the challenge's noise, but only as a judge, not as a decoder.
- Predecoders and faster relay-BP schedules cost accuracy or gained nothing (the table has each count).
- *Pending the test rows:* where TCBP-JOBS is ahead of us, and what the slowest shots cost.

## Slide 8. Whose work is whose, and how to check us

**Show:** the table of `REPORT.md`, section 3, and the two commands of section 9.

- PriorTilt is IonQ's decoder; relay-BP and BP-OSD are the kit's; the router and the evaluator are our teammates'.
- Ours: the hierarchy (two seeds, the agreement test, the third candidate), the learned tie-break, and the loss study.
- Every table and figure is regenerated by two commands from committed files; much of the code was written with AI
  assistants under our direction, and the READMEs are how we keep being able to explain it.

---

## Questions a judge will ask, with honest answers

**Whose gain is whose?**
On the test shots: <!--f:framing-->not measured<!--/f--> The table `report/tables/test_ladder.md` has every rung. On the practice shots:
PriorTilt alone already beats relay-BP on the practice shots, and that part is IonQ's. Our hierarchy adds to it: on
<!--f:expl_pairs-->25,000<!--/f--> practice pairs at p = 0.004, relay-BP fails <!--f:expl_pairs_R-->245<!--/f--> pairs, one PriorTilt run <!--f:expl_pairs_P1-->207<!--/f--> or <!--f:expl_pairs_P2-->188<!--/f--> depending on its seed, and
the hierarchy with its best classical tie-break <!--f:expl_pairs_C-->140<!--/f--> (only relay-BP fails <!--f:expl_pairs_C_only_relay-->128<!--/f--> pairs, only the hierarchy <!--f:expl_pairs_C_only_ours-->23<!--/f-->). The neural tie-break adds a smaller step on top, measured on X memory
(<!--f:expl_X_C-->67<!--/f--> to <!--f:expl_X_N-->59<!--/f--> failures of 25,000 shots; only the network rule fails <!--f:expl_X_N_vs_C_only_network-->7<!--/f-->, only the classical rule <!--f:expl_X_N_vs_C_only_classical-->15<!--/f-->, exact p = <!--f:expl_X_N_vs_C_p-->0.13<!--/f-->). *Pending: the same split on the test shots.* Source:
`report/tables/exploration_p0.004.md`.

**What is on the test shots and what is not?**
On the test shots: the rows of the comparison table and the three plots made from it, decoded once with everything
frozen. Not on the test shots: every number used to design the decoder (the practice shots), the loss study (its own
simulated shots), the learning curves (shots drawn for training reports), and the measurements of slide 7. Each
table says which it is.

**Why are the latency rows comparable?**
Each CPU row ran through the same evaluator, in one process with one thread, on the same server, and the timer wraps
the whole decode of a pair, with every stage and rerun. The table says which other rows ran beside each row. Times
from other machines (a teammate's Mac, our laptop) are in separate tables and are not compared. TCBP-JOBS runs on a
GPU: its time is shown and marked, and it is a different kind of machine. *Pending: the measured overlaps.*

**What does the loss model assume?**
After every two-qubit gate each of its two atoms is lost with a fixed probability. A lost check atom is flagged at
its measurement and replaced; a lost data atom stays lost and is flagged only at the end; flags are never wrong.
Losses are independent. It is our simplified model; it is not Infleqtion's hardware model, and the results are for
X memory at p = 0.004 only. For scale, Infleqtion's paper arXiv:2408.08288 reports about 0.38% atom loss per CZ gate
for a pair of atoms; the top of our range, 2e-3 per atom, is half of that, by our own reading.

**Where does TCBP-JOBS stand against you?**
Ahead, at p = 0.004 and 0.005, as measured so far. Its logged rate per cycle at p = 0.004, on its own sampled shots, is
<!--f:tcbp_own4_rate-->3.42e-04<!--/f-->; on the practice pairs relay-BP is at <!--f:expl_rate_R-->8.20e-04<!--/f--> and our hierarchy with its best classical tie-break at
<!--f:expl_rate_C-->4.68e-04<!--/f-->. We also ran TCBP-JOBS on those same practice pairs: it fails <!--f:expl_pairs_T-->113<!--/f--> of <!--f:expl_pairs-->25,000<!--/f-->,
our hierarchy <!--f:expl_pairs_C-->140<!--/f-->, relay-BP <!--f:expl_pairs_R-->245<!--/f-->. *Pending: the same comparison on the saved test shots, pair by pair.* It needs a GPU;
our hierarchy runs on one CPU core.

**Did anything tune on the test shots?**
No. Rules, seeds and checkpoints were chosen on the train and validation files; our tools refuse files named `test_`
unless told that the run is final; the test files were decoded once.

**What does machine learning add, exactly?**
Three things are learned: PriorTilt's 14 parameters (by its authors), the neural tie-break (ours), and the
teammates' router. The measurement that motivated a learned tie-break: among 16 relay-BP candidates the right answer
is present on <!--f:cand_present-->114<!--/f--> of <!--f:cand_shots-->145<!--/f--> hard shots, but the classical score picks it on only <!--f:cand_lowest-->71<!--/f--> (`report/tables/background.md`).
