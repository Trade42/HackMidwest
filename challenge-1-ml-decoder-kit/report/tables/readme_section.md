<!-- BEGIN our front page: written by challenge-1-ml-decoder-kit/report/make_tables.py from committed result files; edit it there -->
# ML-assisted decoding of the Gross code [[144, 12, 12]]

Our entry to Infleqtion's Challenge 1 at HackMidwest 2026, the main entry of this repository; the challenge statement is in [CHALLENGE.md](CHALLENGE.md). Every number on this page is written by `challenge-1-ml-decoder-kit/report/make_tables.py` from committed result files.

**In short.** A small neural network reads a shot's syndrome and tells NVIDIA's GPU decoder TCBP-JOBS which of the 8,784 possible faults to suspect in that shot. With these per-shot priors, TCBP-JOBS fails **132 of 10,000 test pairs at p = 0.005** (1.11e-03 ± 9.6e-05 per cycle), against 360 with its fixed priors (3.05e-03 ± 1.6e-04) and 845 for the kit's relay-BP (7.33e-03 ± 2.5e-04); at p = 0.004 10 against 41 and 83; at p = 0.003 1 against 3 and 5 (too few to claim anything). **This version was designed after the test results were seen** (no validation at p = 0.004 and 0.003); on 5,000 fresh X shots at p = 0.005, simulated after it was designed (exploration on new Stim shots), it fails 48 against TCBP-JOBS's 112. The decoder we froze before the test run is the first pipeline below (the table's row "ours, frozen before the test run").

![Logical error rate per cycle against p, with the predecoder and the TCBP version](challenge-1-ml-decoder-kit/report/figures/front_ler_vs_p.png)

## Results on the test shots

The kit's score on the team's saved test files (Stim circuit shots, seeds 301 and 302): failed pairs of 10,000 (a pair is shot k of X memory and shot k of Z memory, and fails when either fails) · logical error rate per cycle ± the kit's error. Every row decodes the same shots. The label says what kind of number it is; rows of different labels are not equals.

| Method | Label | p = 0.003 | p = 0.004 | p = 0.005 |
|---|---|---|---|---|
| BP-OSD (kit baseline) | the kit's baseline | 60 · 5.01e-04 ± 6.5e-05 | 472 · 4.02e-03 ± 1.8e-04 | 2,433 · 2.30e-02 ± 4.6e-04 |
| relay-BP (kit baseline) | the kit's baseline | 5 · 4.17e-05 ± 1.9e-05 | 83 · 6.94e-04 ± 7.6e-05 | 845 · 7.33e-03 ± 2.5e-04 |
| relay-BP with a learned router | the teammates' (pre-registered) | 6 · 5.00e-05 ± 2.0e-05 | 87 · 7.28e-04 ± 7.8e-05 | 807 · 6.99e-03 ± 2.5e-04 |
| PriorTilt, one run | IonQ's decoder (pre-registered) | 4 · 3.33e-05 ± 1.7e-05 | 74 · 6.19e-04 ± 7.2e-05 | 601 · 5.15e-03 ± 2.1e-04 |
| TCBP-JOBS (NVIDIA, GPU) | NVIDIA's decoder (pre-registered) | 3 · 2.50e-05 ± 1.4e-05 | 41 · 3.42e-04 ± 5.3e-05 | 360 · 3.05e-03 ± 1.6e-04 |
| our hierarchy, neural tie-break | **ours, frozen before the test run** (pre-registered) | 3 · 2.50e-05 ± 1.4e-05 | 44 · 3.67e-04 ± 5.5e-05 | 479 · 4.08e-03 ± 1.9e-04 |
| the relay-BP predecoder: relay-BP with the network's per-shot priors | ours; secondary at p = 0.005 (chosen after its own gate failed), the rest a registered extension run after it | X only: 2 of 10,000 X shots | X only: 15 of 10,000 X shots | 281 · 2.37e-03 ± 1.4e-04 |
| the predecoder with a fallback to TCBP-JOBS where its relay-BP does not converge | ours; post hoc (designed after the first test result) | X only: 1 of 10,000 X shots | X only: 10 of 10,000 X shots | 206 · 1.73e-03 ± 1.2e-04 |
| **the TCBP version: TCBP-JOBS with the network's per-shot priors** | **ours; designed after the test results were seen; formerly a separate branch (predecoder-tcbp)**; no validation at p = 0.004 or 0.003 | 1 · 8.33e-06 ± 8.3e-06 (no claim: exact p = 0.50 against TCBP-JOBS) | **10 · 8.34e-05 ± 2.6e-05** | **132 · 1.11e-03 ± 9.6e-05** |

Paired on the same pairs, the TCBP version against TCBP-JOBS alone: only the TCBP version fails 41 and only TCBP-JOBS 269 at p = 0.005 (exact p = 3e-42); 4 and 35 at p = 0.004 (exact p = 3e-07). TCBP-JOBS fits its model when it starts, and three fits on the same validation shots fail 91, 96 and 100, so its counts carry about ±5%. Every method with its label, its X failures and its time per pair: [report/tables/master_comparison.md](challenge-1-ml-decoder-kit/report/tables/master_comparison.md).

## How we got here

1. **The first pipeline, frozen before the test run** (`hierarchy/`, `neural/`). Two runs of a fast decoder (IonQ's PriorTilt) settle the shots they agree on; where they disagree, relay-BP joins and a neural network (893,697 parameters) picks the most likely of the answers. At p = 0.005 our decoder fails 479 of 10,000 pairs, relay-BP 845 (only ours fails 93, only relay-BP 459, exact p = 4e-59), in 825 ms per pair against relay-BP's 1,471 ms per pair, or 608 ms per pair with its two fast decodes side by side on two cores; PriorTilt run once fails 601 in 264 ms per pair. In plain words: on the timed rows the network does not do measurably more than a good hand-written rule, and on one core its time is within 4% of the rule's at every p, the size of the run-to-run spread of these rows, so it costs no time the timer can see. The gain over relay-BP is PriorTilt's and the hierarchy's, as in the offline ablation above. (The offline ablation: [report/REPORT.md](challenge-1-ml-decoder-kit/report/REPORT.md), section 4, "The ablation: what the learned part adds".)
2. **The relay-BP predecoder** (`predecoder/`). A second output of the same network gives every one of the 8,784 fault columns a probability for this shot, and relay-BP decodes with those priors. Scored as the kit scores (a pair of an X and a Z shot fails when either fails), at p = 0.005: relay-BP with the head's priors fails 281 pairs, a logical error rate of 2.37e-03 ± 1.4e-04 per cycle, against TCBP-JOBS's 360 (3.05e-03 ± 1.6e-04) (only H 114, only TCBP-JOBS 193, exact p = 8e-06) and relay-BP's 845 (exact p = 6e-131); the fallback rule, post hoc on its X half, fails 206, 1.73e-03 ± 1.2e-04 per cycle (against TCBP-JOBS: only the rule 39, only TCBP-JOBS 193, exact p = 1e-25). This pair score is part of the registered extension, not a held-out result, and it is not in the pair-scored tables of our decoder; accuracy only (its times come from a shared pool on the team's PC); TCBP-JOBS's counts come from its stored test file and carry its fit's spread (above).
3. **The TCBP version** (`predecoder/README_tcbp.md`). The same per-shot priors given to TCBP-JOBS instead of relay-BP. On the test shots (designed after the test results were seen; formerly a separate branch (predecoder-tcbp); decoded one basis at a time; its knobs were fitted on X at p = 0.005 and applied unchanged to Z and at p = 0.004 and 0.003, no validation at these p): X at p = 0.005, TH fails 72 against TCBP-JOBS's 186 (only TH 22, only TCBP-JOBS 136, exact p = 3e-21; the same run's kit-priors arm fails 186); X at p = 0.004, TH fails 4 against TCBP-JOBS's 20 (only TH 2, only TCBP-JOBS 18, exact p = 4e-04); X at p = 0.003, TH fails 0 against TCBP-JOBS's 1 (only TH 0, only TCBP-JOBS 1, exact p = 1.00: no claim); Z at p = 0.005, TH fails 61 against TCBP-JOBS's 177 (only TH 20, only TCBP-JOBS 136, exact p = 2e-22); Z at p = 0.004, TH fails 6 against TCBP-JOBS's 21 (only TH 2, only TCBP-JOBS 17, exact p = 7e-04); Z at p = 0.003, TH fails 1 against TCBP-JOBS's 2 (only TH 0, only TCBP-JOBS 1, exact p = 1.00: no claim). Scored as the kit scores (a pair fails when its X or its Z shot fails), TH's failed pairs of 10,000 against TCBP-JOBS's: at p = 0.005, 132 against TCBP-JOBS's 360 (1.11e-03 ± 9.6e-05 against 3.05e-03 ± 1.6e-04 per cycle; only TH 41, only TCBP-JOBS 269, exact p = 3e-42; relay-BP 845, 7.33e-03 ± 2.5e-04); at p = 0.004 (no validation at these p), 10 against TCBP-JOBS's 41 (8.34e-05 ± 2.6e-05 against 3.42e-04 ± 5.3e-05 per cycle; only TH 4, only TCBP-JOBS 35, exact p = 3e-07); at p = 0.003 (no validation at these p), 1 against TCBP-JOBS's 3 (8.33e-06 ± 8.3e-06 against 2.50e-05 ± 1.4e-05 per cycle; only TH 0, only TCBP-JOBS 2, exact p = 0.50: no claim). TCBP-JOBS's stored counts carry its fit's spread (above).
4. Also tried, none of them replacing the above: Our ensemble, combined offline from saved answers with its rule frozen beforehand (where two PriorTilt runs and TCBP-JOBS agree, that answer; otherwise joint decoding of both check types, the slow decoder of the measurements before building): at p = 0.003 2 against TCBP-JOBS's 3 failed pairs (only the ensemble fails 0, only TCBP-JOBS 1, exact p = 1.00; joint decoding ran on 5 disputed shots, about 5 s each); at p = 0.004 16 against TCBP-JOBS's 41 failed pairs (only the ensemble fails 1, only TCBP-JOBS 26, exact p = 4e-07; joint decoding ran on 93 disputed shots, about 24 s each); at p = 0.005 joint decoding was not run, so the ensemble fails exactly TCBP-JOBS's 360 pairs. Its members are NVIDIA's, IonQ's and the kit's; the rule is ours, and it has no time measured as one decoder. A learned selector among these decoders' answers (`selector/`) fails 353 pairs at p = 0.005 against TCBP-JOBS's 360 (exact p = 0.41): its declared line was not met.

## Speed

Mean time per X shot in ms (one X-memory decode; a pair is an X shot and a Z shot). CPU and GPU times are on different machines and timers and do not compare as equals.

| Method | Where and how timed | p = 0.003 | p = 0.004 | p = 0.005 |
|---|---|---|---|---|
| BP-OSD (kit baseline) | CPU, one process, the team's server; the whole decode | 62.7 | 143 | 286 |
| relay-BP (kit baseline) | CPU, one process, the team's server; the whole decode | 44.3 | 156 | 760 |
| relay-BP with a learned router | CPU, one process, the team's server; the whole decode | 20.8 | 153 | 996 |
| our hierarchy, neural tie-break | CPU, one process, the team's server; the whole decode | 23.7 | 104 | 420 |
| TCBP-JOBS (NVIDIA, GPU) | an A100 GPU slice; the decode call alone | 2.0 | 3.9 | 8.7 |
| the relay-BP predecoder | the team's server, one process, one thread; network and decode; 1,000 validation shots | 52.9 | 94.0 | 212 |
| the TCBP version, as registered | an A100 slice shared with another job (not a clean timing); network on validation shots | not measured | not measured | 55.8 |
| the TCBP version after the speed work | the A100 slice; each part timed alone on 1,000 validation shots | not measured | not measured | about 38 |
| the TCBP version with the fast path | the A100 slice; 5,000 fresh X shots; its own time model, in which the TCBP version alone takes 46.2 | not measured | not measured | 33.5 |

After the speed work (branch `tcbp-speed`; exploration shots: the first 1,000 validation X shots at p = 0.005 on the A100, each part timed alone, not one end-to-end timer), the same decoder takes about 38 ms per X shot instead of about 52 measured the same way on the same shots: the network as a CUDA graph 1.7 ms instead of 8.3, TCBP-JOBS built with H as a scipy CSR matrix 34.0 instead of 41.2, its decode 2.7; freeing the decoder adds 4.6 to 4.8 ms before the next shot. The answers were the same on all 1,000 shots checked. The registered time (55.8 ms, jobs 4 and 5) is another measurement, on other shots; the test numbers were made with the registered path and were not rerun.

**A fast path for easy shots** (exploration after the test results): when the network is confident, the shot goes to TCBP-JOBS with fixed priors, built once; the rest to the TCBP version. On 5,000 fresh X shots at p = 0.005 (seed 2026112251), simulated after the rule was frozen, it sends 35.3% of the shots down the fast path and fails 48 against the TCBP version's 48, in 33.5 ms mean per X shot against 46.2. Details: `predecoder/fastpath/README.md`.

![X test failures against time per X shot at p = 0.005, the hardware in the legend](challenge-1-ml-decoder-kit/report/figures/front_x_vs_time.png)

## Robustness: other noise, and atom loss (exploration, on our own simulated shots)

- **Atom loss** (our own simplified loss model, X memory, p = 0.004): relay-BP that uses the loss flags fails 4.4% of shots where a loss-blind one fails 57.7%; our loss-time tracker, which estimates from the syndrome when each atom was lost, recovers 58% (38% to 78%) of what knowing the true loss time would add (`loss/`, `loss_tracker/`, [report/tables/loss.md](challenge-1-ml-decoder-kit/report/tables/loss.md)).
- **Noise shaped like neutral-atom hardware** (a model built from Infleqtion's published figures, not Infleqtion's noise model; `neutral_atom_noise/RESULTS.md`): decoders that know the shape of the noise fail several times less than with the kit's uniform priors. The TCBP version's network helps on some noise pictures and hurts on others (Z-biased noise), and under atom loss it does not make up for not knowing the shape.
- **The head retrained on that noise** makes TCBP-JOBS worse on its own base: 27 failures against 8 for the matched priors alone (run 1; exploration).

## Resources

- The network: 893,697 parameters, trained on a laptop GPU (RTX 4060, 8 GB) for X memory and on an A100 slice for Z memory ([report/tables/networks.md](challenge-1-ml-decoder-kit/report/tables/networks.md)).
- The first pipeline runs on one CPU core; the TCBP version needs an NVIDIA GPU (cudaq-qec 0.9.0) and builds one decoder per shot, which is most of its time.
- Peak memory and start-up time of each row: [report/tables/resources.md](challenge-1-ml-decoder-kit/report/tables/resources.md).

## Reproduce

From `challenge-1-ml-decoder-kit/`, with Python 3.12 and its `requirements.txt` (the full list and the order: [report/REPRODUCE.md](challenge-1-ml-decoder-kit/report/REPRODUCE.md)):

```sh
python -m pytest -p no:cacheprovider report     # the report's tests
sh report/reproduce_ours.sh smoke                # our decoder and relay-BP on a few shots, through the team's evaluator
sh report/reproduce_ours.sh report               # every table and figure, and this page, from report/data/
```

The TCBP-JOBS steps need Linux, an NVIDIA GPU and `cudaq-qec` 0.9.0 (`predecoder/README_tcbp.md`); training the network needs a GPU.

## Where it falls short

- The TCBP version was designed after the test results were seen, its two knobs were fitted on X memory at p = 0.005 only and applied unchanged to Z and to the lower p (no validation at these p); its only fresh check is the 5,000 new X shots above.
- TCBP-JOBS's answers vary from one model fit to the next (see the results), so every count of it, and of the TCBP version, carries that spread.
- Building TCBP-JOBS for every shot costs 34.0 ms of the TCBP version's time even after the speed work; cudaq-qec 0.9.0 cannot change a built decoder's priors.
- In the first pipeline the network is no better than the best hand-written tie-break on the test shots.
- Our pipeline under atom loss was not tested on the kit's shots; the loss and neutral-atom studies use our own simulated shots.

## Where things are

| Folder | What |
|---|---|
| [`report/`](challenge-1-ml-decoder-kit/report/) | the write-up ([REPORT.md](challenge-1-ml-decoder-kit/report/REPORT.md)), every table and figure, made by scripts from `report/data/` |
| [`predecoder/`](challenge-1-ml-decoder-kit/predecoder/) | the network's per-shot priors, the relay-BP predecoder, the TCBP version (`README_tcbp.md`) and its fast path (`fastpath/`) |
| [`hierarchy/`](challenge-1-ml-decoder-kit/hierarchy/), [`neural/`](challenge-1-ml-decoder-kit/neural/) | the first pipeline and its network |
| [`neutral_atom_noise/`](challenge-1-ml-decoder-kit/neutral_atom_noise/), [`loss/`](challenge-1-ml-decoder-kit/loss/), [`loss_tracker/`](challenge-1-ml-decoder-kit/loss_tracker/) | noise shaped like neutral-atom hardware, and atom loss |
| [`selector/`](challenge-1-ml-decoder-kit/selector/), [`ml_decoder/`](challenge-1-ml-decoder-kit/ml_decoder/) | a learned selector; an ML-first decoder |
| [`evaluation/`](challenge-1-ml-decoder-kit/evaluation/) | the team's evaluator and the teammates' learned router |
| [information.md](information.md) | everything we built and why |
| [`challenge2/`](challenge2/) | the teammates' Challenge 2 work (Give Your LLM a Quantum Backend); see its README |

The section below is the teammates' earlier technical review; current results are above.
<!-- END our front page -->
