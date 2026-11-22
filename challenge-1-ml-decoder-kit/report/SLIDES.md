# Material for a five-minute talk

Eight slides. Each has a title that states its point, the figure or table to show, and three lines to say. Every
number is in a generated table (`report/tables/`) or a file under `report/data/`; the source is named beside it.
"Practice shots" below means the exploration shots (the team's train and validation files): numbers on them are not
results on the test shots, and the slides say so each time.

Words used, each with its meaning:

- **Decoder:** the program that reads which parity checks fired and says which of the 12 stored bits flipped.
- **Shot:** one run of the memory experiment. It **fails** when the decoder gets any of the 12 bits wrong.
- **Pair:** one X-memory shot plus one Z-memory shot. The kit scores pairs; a pair fails when either shot fails.
- **p:** the strength of the noise. Higher p, more errors.
- **BP-OSD, relay-BP:** the two baseline decoders of the kit. Relay-BP is the stronger one.
- **PriorTilt:** a fast decoder published by IonQ for this code. It is not ours; we build on it.
- **Tie-break:** the rule that picks one answer when the fast decodes disagree.
- **Paired count:** two decoders on the same pairs; "only A fails 15, only B fails 54" counts the pairs where they differ.
- **exact p:** the chance of a difference at least that large between two decoders if they were really equally good (an
  exact binomial test on the pairs only one of them fails); below 0.05 we call it significant. Plain "p" on these
  slides is always the noise strength.
- **ORACLE:** a test given the truth on purpose, to show the best any method could do. It is a ceiling, not a decoder.
- **TCBP-JOBS:** NVIDIA's decoder, which runs on a GPU. The teammates ran it as an extra reference.

---

## Slide 1. Two fast decodes that agree are almost always right; only the hard shots pay for more

**Show:** the diagram of `REPORT.md`, section 2.

- A decoder reads the syndrome, the list of checks that fired, and says which of the 12 stored bits flipped.
- We run a fast decoder twice with different random seeds. If the two answers agree, that is the answer.
- If they differ, relay-BP joins as a third opinion and a tie-break picks one of the answers: a hand-written rule, or
  a neural network that reads the whole syndrome.

## Slide 2. The failures hide in the <!--f:tsplit_share-->0.45%<!--/f--> of shots where the two fast decodes disagree

**Show:** `figures/where_failures.png` (the test shots at p = 0.004, our decoder as it ran).

- On the <!--f:tsplit_shots-->20,000<!--/f--> test shots at p = 0.004 (X and Z memory), the two PriorTilt runs disagree on only <!--f:tsplit_disagree-->90<!--/f-->.
- Those few shots hold <!--f:tsplit_fail_disagree-->27<!--/f--> of our <!--f:tsplit_fail-->44<!--/f--> failed shots. So "do they agree?" is a cheap alarm for a hard shot, and
  only those shots pay for relay-BP and the tie-break.
- The other <!--f:tsplit_fail_agree-->17<!--/f--> failures are shots where both runs agree on a wrong answer. No tie-break can fix those: that is our floor.

Source: `report/tables/where_failures.md`. The design was chosen on practice shots before, where the same pattern held.

## Slide 3. <!--f:slide3_title-->The network is a poor decoder alone, and as a tie-break it is no better than the best hand-written rule<!--/f-->

**Show:** `figures/learning_curves.png`, then the ablation table of `REPORT.md`, section 4.

- The network reads all 1,728 detector bits of a shot, is built along the code's own wiring, and has
  <!--f:net_parameters-->893,697<!--/f--> parameters. Alone, on the test shots at p = 0.004, it gets <!--f:netalone4_X-->4.2%<!--/f--> of X-memory shots wrong; relay-BP gets
  <!--f:t4_relay_bp_xz-->36, 47<!--/f--> (X, Z) of 10,000 wrong.
- As a tie-break it clearly beats the plain lowest-weight rule (on <!--f:a80_shots-->80,000<!--/f--> fresh shots: <!--f:a80_a-->237<!--/f--> against <!--f:a80_aw-->298<!--/f--> failures,
  exact p = <!--f:a80_a_vs_aw_p-->7e-09<!--/f-->). Against the best hand-written rule there is no significant difference: <!--f:a80_a-->237<!--/f--> against <!--f:a80_ac-->253<!--/f-->
  there (exact p = <!--f:a80_a_vs_ac_p-->0.12<!--/f-->), and <!--f:abl_pooled_a-->544<!--/f--> against <!--f:abl_pooled_aw-->525<!--/f--> failed pairs on the test shots (exact p = <!--f:abl_pooled_p-->0.15<!--/f-->; one offline decode of all three
  p, which is why it is not the sum of slide 4's timed counts). <!--f:abl_timed_slide-->In the timed rows at p = 0.003 and 0.004 the network is ahead (47 against 56 failed pairs, exact p = 0.11): not significant either way.<!--/f-->
- <!--f:slide3_close-->So the gain over relay-BP comes from PriorTilt and the hierarchy; on the test shots the network does not add to it.<!--/f-->

Source: `report/tables/networks.md`, `ablation_80k.md`, `test_ablation.md`.

## Slide 4. <!--f:slide4_title-->On the test shots our decoder fails fewer pairs than relay-BP at p = 0.004 and 0.005, and as many within the error at p = 0.003; it is level with TCBP-JOBS at p = 0.003 and 0.004, and behind it at p = 0.005<!--/f-->

**Show:** the table below and `figures/ler_vs_p.png`.

<!-- BEGIN table:test_sweep_slide -->
| Decoder | p = 0.003 | p = 0.004 | p = 0.005 |
|---|---|---|---|
| BP-OSD | 60 · 131 ms | 472 · 302 ms | 2,433 · 581 ms |
| relay-BP | 5 · 85.3 ms | 83 · 319 ms | 845 · 1,471 ms |
| router (teammates) | 6 · 41.0 ms | 87 · 315 ms | 807 · 1,928 ms |
| PriorTilt (IonQ) | 4 · 22.7 ms | 74 · 87.3 ms | 601 · 264 ms |
| ours, 1 core | 3 · 45.7 ms | 44 · 216 ms | 479 · 825 ms |
| ours, 2 cores | 3 · 28.9 ms | 44 · 150 ms | 479 · 608 ms |
| TCBP-JOBS (GPU) | 3 · 3.9 ms | 41 · 7.8 ms | 360 · 16.9 ms |
| ours, ensemble | 2 · offline | 16 · offline | 360 · offline |

Failed pairs of 10,000 test pairs · mean time per pair. "Offline": combined from saved answers, no time. TCBP-JOBS runs on a GPU.
<!-- END table:test_sweep_slide -->

- <!--f:slide4_relay-->Failed pairs of 10,000, ours against relay-BP on the same pairs: p = 0.003: 3 against 5 (exact p = 0.50); p = 0.004: 44 against 83 (exact p = 3e-06); p = 0.005: 479 against 845 (exact p = 4e-59).<!--/f-->
- <!--f:tcbp_statement-->Against TCBP-JOBS, NVIDIA's GPU decoder, on the same test pairs: at p = 0.003 TCBP-JOBS and our decoder fail as many pairs (3 against 3; only TCBP-JOBS fails 1, only ours 1, exact p = 1.00, not a significant difference); at p = 0.004 TCBP-JOBS fails fewer pairs than our decoder (41 against 44; only TCBP-JOBS fails 17, only ours 20, exact p = 0.74, not a significant difference); at p = 0.005 TCBP-JOBS fails fewer pairs than our decoder (360 against 479; only TCBP-JOBS fails 95, only ours 214, exact p = 1e-11). TCBP-JOBS against relay-BP's timed row: 3 against 5 at p = 0.003 (exact p = 0.62), 41 against 83 at p = 0.004 (exact p = 4e-07), 360 against 845 at p = 0.005 (exact p = 1e-101).<!--/f-->
- At p = 0.003 every good decoder fails only a handful of pairs in 10,000, so nothing separates there.

## Slide 5. <!--f:slide5_title-->Our decoder takes 0.54 to 0.68 of relay-BP's mean time per pair on one core and 0.34 to 0.47 with its two fast decodes on two cores<!--/f-->

**Show:** `figures/time_vs_p.png`, then `figures/ler_vs_time.png`.

- Every time is per pair (an X shot and a Z shot), with every stage counted, one process per row on the same server.
- <!--f:slide5_lines-->p = 0.003: ours 45.7 ms mean and 395 ms at the 99th percentile, relay-BP 85.3 and 557; p = 0.004: ours 216 ms mean and 2,064 ms at the 99th percentile, relay-BP 319 and 5,514; p = 0.005: ours 825 ms mean and 9,885 ms at the 99th percentile, relay-BP 1,471 and 8,270.<!--/f-->
- TCBP-JOBS runs on a GPU and its timer covers the decode call alone: its time is shown, not compared as a ratio.

## Slide 6. With atom loss, the flags matter most, and knowing when an atom was lost is worth more still

**Show:** `figures/loss.png`.

- In a neutral-atom machine an atom can leave its trap. A flag is the measurement that tells us it is gone.
- At a loss rate of 0.001 per atom per gate, relay-BP that ignores loss fails <!--f:loss_blind_share-->57.7%<!--/f--> of shots
  (<!--f:loss_blind-->173 of 300<!--/f-->); with the flags it fails <!--f:loss_aware_share-->4.4%<!--/f--> (<!--f:loss_aware-->176 of 4,000<!--/f-->), <!--f:loss_aware_ratio-->11.5<!--/f--> times fewer on the shots both decoded. Loss without its flags is ruinous; the flags give most of it back.
- An ORACLE that is told the true round of each loss fails <!--f:loss_oracle_share-->2.7%<!--/f--> (<!--f:loss_oracle-->107 of 4,000<!--/f-->, the same shots):
  <!--f:loss_oracle_ratio-->1.64<!--/f--> times fewer than with the flags alone (<!--f:loss_oracle_ratio_interval-->1.42 to 1.94<!--/f-->).
  That is a ceiling, not a decoder. Our tracker estimates that round from the syndrome history: it recovers
  <!--f:trk_gain-->58% (38% to 78%)<!--/f--> of the oracle's gain on its own test shots.

Source: `report/tables/loss.md`. The loss model is our own simplified one, not Infleqtion's hardware model.

## Slide 7. What did not work, and where we fall short

**Show:** four lines of `report/tables/background.md`.

- A plain neural network learned only at noise far below the challenge's; the network that follows the code's wiring
  learns at the challenge's noise, but only as a judge, not as a decoder.
- Predecoders and faster relay-BP schedules cost accuracy or gained nothing (the table has each count).
- Where we fall short: TCBP-JOBS fails fewer pairs than our decoder at p = 0.005 (<!--f:t5_tcbp_pairs-->360<!--/f--> against <!--f:t5_hierarchy_pairs-->479<!--/f-->),
  and our slowest shots are slower than relay-BP's there (<!--f:t5_hierarchy_p99_ms-->9,885<!--/f--> against <!--f:t5_relay_bp_p99_ms-->8,270<!--/f--> ms at the 99th percentile):
  a hard shot pays for three decoders one after the other.

## Slide 8. Whose work is whose, and how to check us

**Show:** the table of `REPORT.md`, section 3, and the two commands of section 10.

- PriorTilt is IonQ's decoder; relay-BP and BP-OSD are the kit's; the router and the evaluator are our teammates'.
- Ours: the hierarchy (two seeds, the agreement test, the third candidate), the learned tie-break (<!--f:abl_verdict-->no better than<!--/f--> the best
  hand-written rule), the ensemble's rule, and the loss study with its tracker.
- Every table and figure is regenerated by two commands from committed files; much of the code was written with AI
  assistants under our direction, and the READMEs are how we keep being able to explain it.

---

## Backup slide. <!--f:net_timed_slide_title-->Timed against timed, the network is not significantly different from the best hand-written tie-break (47 against 56 failed pairs at p = 0.003, 0.004, exact p = 0.11)<!--/f-->

Shows: `report/tables/test_network_timed.md`. The rows marked as run after the test results were seen are an ablation the
team asked for, not our decoder. The best hand-written rule's one-core rows at p = 0.003 and 0.004 ran in the test run
itself, among its pre-registered arms.

- The same decoder ran on the same test pairs, through the same evaluator, with the network swapped for a hand-written tie-break.
- <!--f:net_timed_per_p-->With the network against without it, timed rows on one core: at p = 0.003, against the best hand-written rule 3 against 4 failed pairs (only with the network 0, only without 1, exact p = 1.00; 45.7 against 46.3 ms per pair) and against plain lowest weight [run after the test results were seen (an ablation)] 3 against 4 failed pairs (only with the network 0, only without 1, exact p = 1.00; 45.7 against 46.9 ms per pair); at p = 0.004, against the best hand-written rule 44 against 52 failed pairs (only with the network 8, only without 16, exact p = 0.15; 216 against 218 ms per pair) and against plain lowest weight [run after the test results were seen (an ablation)] 44 against 60 failed pairs (only with the network 5, only without 21, exact p = 0.002; 216 against 215 ms per pair).<!--/f-->
- The network changes the answer only on the few shots where the two fast decodes disagree. <!--f:net_timed_plain-->In plain words: on the timed rows the network does not do measurably more than a good hand-written rule, and on one core its time is within 4% of the rule's at every p, the size of the run-to-run spread of these rows, so it costs no time the timer can see. The gain over relay-BP is PriorTilt's and the hierarchy's, as in the offline ablation above.<!--/f-->

## Questions a judge will ask, with honest answers

**Whose gain is whose?**
<!--f:framing_slide-->On the test shots (one offline decode of each shot, the same candidates for every rung; p = 0.003, 0.004, 0.005 together), relay-BP fails 889 pairs, PriorTilt run once (IonQ's decoder) 672, and our decoder as frozen (two PriorTilt runs, relay-BP where they disagree, the neural tie-break) 544; a perfect tie-break would fail 429. Of the drop from relay-BP to our decoder, 63% is already there with PriorTilt run once; the hierarchy around it gives the rest. As an ablation, with the best hand-written tie-break (lowest weight among the converged candidates) in place of the network, the hierarchy fails 525 (only the network rule fails 87 pairs, only the hand-written rule 68, exact p = 0.15): the network is no better than the hand-written rule.<!--/f--> The ensemble's lead over TCBP-JOBS at p = 0.004 comes from joint decoding of the
disputed shots: its rule is ours, its members are NVIDIA's, IonQ's and the kit's. Source: `report/tables/test_ladder.md`,
`test_ablation.md`, `test_p0.004.md`.

**What is on the test shots and what is not?**
On the test shots: the rows of the comparison table and the three plots made from it, decoded with everything frozen
beforehand. Not on the test shots: every number used to design the decoder (the practice shots), the loss study (its own
simulated shots), the learning curves (shots drawn for training reports), and the measurements of slide 7. Each
table says which it is.

**Why are the latency rows comparable?**
Each CPU row ran through the same evaluator, in one process with one thread, on the same server, and the timer wraps
the whole decode of a pair, with every stage and rerun. The table says which other rows ran beside each row. Times
from other machines (a teammate's Mac, our laptop) are in separate tables and are not compared. TCBP-JOBS runs on a
GPU: its time is shown and marked, and it is a different kind of machine. Up to <!--f:max_beside-->9<!--/f--> evaluator rows ran side by side
on the server, and the CPU quota throttled each row for at most <!--f:max_throttle-->0.4%<!--/f--> of its time (the table gives every row's figure).

**What does the loss model assume?**
After every two-qubit gate each of its two atoms is lost with a fixed probability. A lost check atom is flagged at
its measurement and replaced; a lost data atom stays lost and is flagged only at the end; flags are never wrong.
Losses are independent. It is our simplified model; it is not Infleqtion's hardware model, and the results are for
X memory at p = 0.004 only. For scale, Infleqtion's paper arXiv:2408.08288 reports about 0.38% atom loss per CZ gate
for a pair of atoms; the top of our range, 2e-3 per atom, is half of that, by our own reading.

**Where does TCBP-JOBS stand against you?**
<!--f:tcbp_statement-->Against TCBP-JOBS, NVIDIA's GPU decoder, on the same test pairs: at p = 0.003 TCBP-JOBS and our decoder fail as many pairs (3 against 3; only TCBP-JOBS fails 1, only ours 1, exact p = 1.00, not a significant difference); at p = 0.004 TCBP-JOBS fails fewer pairs than our decoder (41 against 44; only TCBP-JOBS fails 17, only ours 20, exact p = 0.74, not a significant difference); at p = 0.005 TCBP-JOBS fails fewer pairs than our decoder (360 against 479; only TCBP-JOBS fails 95, only ours 214, exact p = 1e-11). TCBP-JOBS against relay-BP's timed row: 3 against 5 at p = 0.003 (exact p = 0.62), 41 against 83 at p = 0.004 (exact p = 4e-07), 360 against 845 at p = 0.005 (exact p = 1e-101).<!--/f--> It needs a GPU; our decoder runs on one CPU core. <!--f:ensemble_statement-->Our ensemble, combined offline from saved answers with its rule frozen beforehand (where two PriorTilt runs and TCBP-JOBS agree, that answer; otherwise joint decoding of both check types, the slow decoder of the measurements before building): at p = 0.003 2 against TCBP-JOBS's 3 failed pairs (only the ensemble fails 0, only TCBP-JOBS 1, exact p = 1.00; joint decoding ran on 5 disputed shots, about 5 s each); at p = 0.004 16 against TCBP-JOBS's 41 failed pairs (only the ensemble fails 1, only TCBP-JOBS 26, exact p = 4e-07; joint decoding ran on 93 disputed shots, about 24 s each); at p = 0.005 joint decoding was not run, so the ensemble fails exactly TCBP-JOBS's 360 pairs. Its members are NVIDIA's, IonQ's and the kit's; the rule is ours, and it has no time measured as one decoder.<!--/f-->

**"Another team is at 0.3 times the baseline's time." How do you compare?**
A time ratio means nothing without the accuracy beside it: stopping relay-BP at its first answer is several times
faster and fails more (`report/tables/background.md`). Our mean time divided by relay-BP's, on the same server and
the same test pairs: <!--f:t3_hierarchy_time_ratio-->0.54<!--/f--> at p = 0.003, <!--f:t4_hierarchy_time_ratio-->0.68<!--/f--> at p = 0.004,
<!--f:t5_hierarchy_time_ratio-->0.56<!--/f--> at p = 0.005, with <!--f:t3_hierarchy_pairs-->3<!--/f-->, <!--f:t4_hierarchy_pairs-->44<!--/f--> and <!--f:t5_hierarchy_pairs-->479<!--/f--> failed pairs
against relay-BP's <!--f:t3_relay_bp_pairs-->5<!--/f-->, <!--f:t4_relay_bp_pairs-->83<!--/f--> and <!--f:t5_relay_bp_pairs-->845<!--/f-->. The table `report/tables/test_time.md`
puts every ratio beside its failures.

**Does the network do more than break ties?**
In one secondary result, yes. A second output of the network gives relay-BP its starting beliefs for each shot (`predecoder/`).
The gate fixed in advance, on practice shots at p = 0.004, was not passed (<!--f:pregate_H-->14<!--/f--> against <!--f:pregate_fresh-->20<!--/f--> failures, exact
p = <!--f:pregate_p-->0.24<!--/f-->). One run on the test shots at p = 0.005, chosen after that and on X memory only, gave <!--f:pre5X_H-->151<!--/f--> failures against
<!--f:pre5X_fresh-->433<!--/f--> for relay-BP with the kit's beliefs (exact p = <!--f:pre5X_vs_fresh_p-->2e-62<!--/f-->) and against TCBP-JOBS's <!--f:pre5X_T-->186<!--/f--> (exact p = <!--f:pre5X_vs_T_p-->0.006<!--/f-->). It counts
single shots, not the kit's pairs, and its times are not clean. Source: `report/tables/test_predecoder.md`.

**Did anything tune on the test shots?**
No. Rules, seeds and checkpoints were chosen on the train and validation files; our tools refuse files named `test_`
unless told that the run is final. Every setting was frozen before the test files were opened; the frozen decoder and
its ablation arms (the hand-written tie-breaks) then decoded them, and nothing was chosen from those runs.

**What does machine learning add, exactly?**
Four things are learned: PriorTilt's 14 parameters (by its authors, IonQ), our neural tie-break, our loss-time tracker
(a statistical model fitted on training shots), and the teammates' router. The honest account of ours:
- The neural tie-break was motivated by a measurement: among 16 relay-BP candidates the right answer is present on
  <!--f:cand_present-->114<!--/f--> of <!--f:cand_shots-->145<!--/f--> hard shots, but the classical score picks it on only <!--f:cand_lowest-->71<!--/f--> (`report/tables/background.md`).
  It clearly beats that classical score (on 80,000 fresh shots, <!--f:a80_a-->237<!--/f--> against <!--f:a80_aw-->298<!--/f--> failures); on the test shots it
  is <!--f:abl_verdict-->no better than<!--/f--> the best hand-written rule (<!--f:abl_pooled_a-->544<!--/f--> against <!--f:abl_pooled_aw-->525<!--/f--> failed pairs over the three p in one offline decode, exact p = <!--f:abl_pooled_p-->0.15<!--/f-->). <!--f:slide3_close-->So the gain over relay-BP comes from PriorTilt and the hierarchy; on the test shots the network does not add to it.<!--/f-->
- The loss-time tracker is the learned part with a clear gain: it recovers <!--f:trk_gain-->58% (38% to 78%)<!--/f--> of what knowing the true loss time
  would give, on its own test shots.
