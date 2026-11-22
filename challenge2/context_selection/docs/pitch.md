# Pitch: Error-detected quantum context selection

MidQWest 2026, Infleqtion Challenge 2, "Give your LLM a quantum backend". Demo and judging
Sunday 2 PM; content freeze Sunday 11:00.

Every number below is taken from a committed file, named next to it. Main run =
`configs/main.yaml` (Qwen2.5-1.5B-Instruct reader, bge-small scorer). Offline run =
`configs/offline.yaml` (t5-base, TF-IDF). Both use 150 HotpotQA questions and a **local
Pauli noise model**, not Sqale's noise. Say that sentence out loud at least once.

Judging criteria (plan PDF, p. 16-17): Novelty, Significance, Infleqtion fit, Conventions,
25 points each. The deck covers each one: novelty on slides 3-4, significance on 6-8,
Infleqtion fit on 4-5, conventions on 8 (pre-registered hypotheses, CI, one reproduce command).

---

## 1. Deck outline (9 slides, 3 minutes; slide 4 is the headline)

### Slide 1: "We put error-detected QAOA inside an LLM pipeline and measured what it changes"

On-slide text:
- Small LLMs get distracted: HotpotQA gives 10 paragraphs, only 2 matter
- QAOA picks the paragraphs; the Iceberg code catches errors in the QAOA circuit
- 150 questions, 5 pre-registered hypotheses, and we report every verdict

Figure: none (title slide). Team name, repo link.

Speaker notes: Open with the question, not the technology. Say up front that we claim no
speedup, before a judge asks. Main-run setup: Qwen2.5-1.5B-Instruct at temperature 0,
bge-small scores (`README.md`, "What ran, and where").

### Slide 2: "Choosing context is a QUBO, so a quantum sampler fits in the pipeline"

On-slide text:
- Question + 10 paragraphs -> relevance and overlap scores -> weighted independent set
- Solvers: brute force, greedy, annealing, random control, QAOA ideal / bare noisy / error-detected
- Repair and rank -> LLM answers from the best selection, plus a 3-selection vote

Figure: no PNG exists for the pipeline. Draw the 5-box flow from `README.md` "How it works"
(question -> scores -> QUBO -> solver tier -> repair/rank -> LLM + vote).

Speaker notes: All solvers see the same QUBO, so any difference in the answer traces back to
the solver tier. The quantum output shapes the model twice: what it reads, and the vote.
Angles are tuned once on the ideal tier and reused, so tiers differ only in noise and encoding.

### Slide 3: "The textbook Iceberg mapping fails on Sqale's grid. Our bus mapping fixes it"

On-slide text:
- Textbook: 2 hub qubits need SWAP chains, and one fault can flip 9 of 10 answer bits
- Bus mapping: nearest-neighbour gates only, fits a 3x6 patch of the 6x4 grid
- Two-qubit faults that slip through: bare 92%, textbook 12.5%, **bus 3.4%**
- Every single fault checked exhaustively, then confirmed by an independent reviewer

Figure: `figures/main/fig6_fault_budget.png`

Speaker notes: This is our main technical contribution. He et al. ran Iceberg QAOA on MaxCut,
which has only pair terms. Our QUBO has a weight per paragraph, and each becomes a gate on the
hub qubit b. A single fault there leaves X_i X_b, which passes both checks and flips every
logical bit except i. The bus mapping fans b and t out along an ancilla row, so no data qubit
shares a gate with a hub outside preparation and the final check. Worst harmful undetected
single fault flips at most 2 bits. Approximation ratio lost per unit CZ error: bare 0.79,
textbook 1.68, bus 0.26 (main fault budget, `results/main/studies/fault_budget.json`). The
independent check ran about 1,000,000 single faults over 300 circuits and found two gaps,
both fixed and both covered by tests (`README.md`, "What we checked about the encoder").

### Slide 4: "On Sqale's own compiler, our bus mapping is the cheaper protected circuit"

On-slide text:
- 36 circuits from 12 questions through `cq_compile` for `cq_sqale_qpu` (real Superstaq, one call)
- Mean CZ before -> after (RZZ counted as 2 CZ): textbook Iceberg 113 -> **239** (x2.1); our bus Iceberg 183 -> **182** (unchanged)
- All 24 compiled encoded circuits keep **100% of shots** without noise (exact simulation)
- Sqale's compiler preserves the error checks. It does not break them

Figure: **needs to be made** (no PNG exists). Suggested `fig8_sqale_compile.png`: grouped bars,
three encodings (bare, textbook Iceberg, bus Iceberg) x two bars each (CZ before, CZ after
Sqale `cq_compile`), values 45.7/81.0, 112.7/239.1, 182.8/181.8 (CZ-equivalent before: RZZ = 2 CZ; key `cz_equiv_in`), with "x1.8", "x2.1", "x0.99"
above each pair and a footer: "Mean of 12 evaluation questions (offline-config QUBOs). Real
Superstaq `cq_compile` for `cq_sqale_qpu`; compiled circuits simulated exactly here, noiseless."
Data: `results/offline/sqale_compile/compile_n12_bare-bus-direct.json`, key `mean`. Fallback if
there is no time: put the README "Sqale compiler check" table on the slide as a 3-row table.

Speaker notes: This is the strongest slide, so let it land. Before compilation the bus circuit
is the more expensive one (183 vs 113 CZ, counting each RZZ as 2 CZ), because it pays for its
routing up front with bus ancillas. On Sqale's real compiler the textbook circuit needs routing
for its hub qubits and grows 2.1x, while the bus circuit is native: unchanged. So the protected
circuit we designed for the grid ends up cheaper, 182 vs 239 CZ. Bare QAOA grows 1.8x
(45.7 -> 81.0). Every compiled encoded circuit, simulated exactly with no noise, keeps 100% of
shots and matches the original logical distribution to TVD <= 8e-14, so no check was optimised
away. These agree with our local grid estimates (bare 69, bus 182, textbook 224, figure 4).
Cost: one call, 36 circuits, 0.5 credits, 84 s; 2 of our 20 credits spent in total. The
12 questions use the offline-config QUBOs (TF-IDF scores, denser graphs). Honest limit: our
account role may compile only; execution returned 403 (slide 9 / Q&A).

### Slide 5: "Rydberg physics makes phase errors, and the checks catch all of them"

On-slide text:
- Every CZ is a Rydberg blockade gate; a stray field shifts the level -> phase error
- Cs 60S (illustrative): 0.1 V/cm shifts the level by about 1 MHz
- Phase (Z-type) CZ faults that slip through: bare 88%, **bus Iceberg 0%**
- Blockade is short-range -> local connectivity -> why the bus mapping exists

Figure: `figures/physics/stark_map.png` (left) and `figures/offline/fig7b_noise_sweep_phase_errors.png`
(right). Backup: `figures/physics/pair_potential.png`.

Speaker notes: Numbers from `physics/README.md` (ARC 3.10.2): 60S polarizability 197 MHz
cm^2/V^2, so 0.1 V/cm gives -0.99 MHz. Cs-Rb pair potential is 7.4 MHz at the cited 5.5 um
spacing and 0.12 MHz at 11 um. Rydberg levels are illustrative; we do not know which levels
Sqale uses. Phase-fault slip-through: main run bare 87.9%, bus 0%
(`results/main/studies/fault_budget.json`); offline bare 95.2%, bus 0% (`README.md`).

### Slide 6: "Error detection recovers about half the solver quality noise removes"

On-slide text:
- H3 supported in both runs: approximation ratio up after discarding flagged shots
- Main: bare 0.863 -> bus 0.865 (ideal 0.869): 42% of the loss recovered, 51% of shots kept
- Offline: 0.934 -> 0.941 (ideal 0.947): 58% recovered, 38% of shots kept
- Noise is a local Pauli model (0.5% per CZ), not Sqale

Figure: `figures/main/fig2_noise_to_accuracy.png`

Speaker notes: H3 intervals: main +0.0026 [+0.002, +0.003], offline +0.0074 [+0.0070,
+0.0079] (`results/*/hypotheses.md`). Point at the figure left to right: valid samples and
approximation ratio move; gold recall and exact match are flat. That flat right half is
slide 8. The gain is smaller in the main run because bge gives sparser redundancy graphs (5.0
edges per question vs 22.5), so circuits are shallower. Gain grows with depth: +0.001 under 6
edges, +0.010 above 30 (offline, `figures/offline/fig3_redundant_edges.png`).

### Slide 7: "Error detection pays only in a window, and it costs shots"

On-slide text:
- Bus Iceberg beats bare from 0.1% to about 1% CZ error; loses at 2%
- At 2% CZ error only 5% of shots survive (offline)
- Per raw shot, the bare circuit finds the optimum more often at every noise level
- Our router uses this: a floor on the predicted share of shots kept

Figure: `figures/offline/fig7_noise_sweep.png` (three panels: AR of kept shots, share kept,
optimum per raw shot). `figures/main/fig7_noise_sweep.png` is the main-run version.

Speaker notes: Offline sweep (`results/offline/studies/noise_sweep.json`, 30 questions):
bus wins at 0.1, 0.2, 0.5 and 1%, loses at 2% (0.867 vs 0.893). Main sweep
(`results/main/studies/noise_sweep.json`): bus wins at 0.1-0.5%, ties at 1% (0.8614 vs
0.8612), loses at 2%. Error detection buys cleaner samples, not more of them. This answers
"isn't discarding shots cherry-picking?": we show the right panel on purpose.

### Slide 8: "Final answers don't move, and a random-sampler control shows why"

On-slide text:
- H4 supported: solver gain +0.0026, accuracy gain 0.000
- Every sampling lane, including uniform random, gives EM 0.367: 4,000 shots over 1,024 selections + repair always find the optimum
- H1 not supported: all 10 paragraphs 0.400 vs exact selection 0.367; H5 (vote) not supported
- Selection does cut the prompt from 1,455 to 428 tokens, as every classical selector does

Figure: `figures/main/fig1_accuracy_by_lane.png`

Speaker notes: This is the result we would want a judge to remember for its honesty. All
five hypotheses were written down before any result (`HYPOTHESES.md`, 18:25 CST Saturday)
and judged as written. H1: -0.033 [-0.107, +0.040]. H5: +0.007 [-0.013, +0.033]. Top-3 by
similarity (0.393) is within noise of reading everything. Because downstream accuracy cannot
separate samplers at 10 paragraphs, H3 is judged on solver quality before repair, and the
`rand` lane exists to prove the point.

### Slide 9: "No speedup at 10 paragraphs. Here is what scaling would take"

On-slide text:
- Brute force: 0.2 ms at 10 paragraphs, about 1 s at 20-22 (plain enumeration)
- Bus patch fits Sqale's 6x4 grid up to 10 paragraphs; beyond that needs a folded layout
- Not run: Sqale execution (account role 403). All noise in this talk is a local Pauli model
- Reproducible: CI tests, pre-registered hypotheses, one command rebuilds every figure

Figure: `figures/offline/fig5_roadmap.png`

Speaker notes: Brute force passes 1 s at about 20 paragraphs offline and 22 in the main run
(`results/*/studies/roadmap.json`). That is exhaustive enumeration, not the best classical
method; branch and bound solves these sizes instantly. 100 paragraphs need 153 physical
qubits with the bus mapping, against Infleqtion's 100-logical-qubit 2028 target. Close with
the one sentence: "We built and verified a grid-native error-detection mapping that is the
cheaper protected circuit on Sqale's own compiler, and measured honestly what it is and
isn't worth to a language model."

---

## 2. Three-minute script (word for word)

Target pace about 145 words per minute. Times are cumulative.

**Slide 1 (0:00-0:18)**
Small language models get distracted. In HotpotQA every question comes with ten
paragraphs, and only two matter. Can a quantum sampler pick the right ones, and does error
detection on a neutral-atom machine change the answer? We claim no speedup. We measured.

**Slide 2 (0:18-0:37)**
Choosing paragraphs is a QUBO: take relevant ones, never two that repeat each other. The
same problem goes to brute force, greedy, annealing, a random control, and QAOA in three
tiers: ideal, noisy, and error-detected. A 1.5-billion-parameter model then answers
from the best selection.

**Slide 3 (0:37-1:06)**
The error-detected tier uses the Iceberg code, built for trapped ions. On Sqale's grid it
breaks: two hub qubits need swaps, and on our problem one fault can flip nine of ten answer
bits. So we designed a bus mapping with nearest-neighbour gates only. Faults that slip through
drop from twelve and a half percent to three point four. We checked every
single fault, and an independent reviewer checked again.

**Slide 4 (1:06-1:37)**
Then we sent thirty-six circuits through Infleqtion's real Sqale compiler. Before
compiling, our bus circuit was the more expensive one: 183 CZ gates against 113. After
compiling, the textbook circuit more than doubled, to 239. Ours didn't change: 182. On
Sqale's own compiler, our protected circuit is the
cheaper one. And every compiled circuit, simulated exactly, keeps a hundred percent of
shots: the compiler preserves the checks.

**Slide 5 (1:37-1:52)**
Physically, every CZ is a Rydberg blockade gate, and stray fields add phase errors. Our
checks catch every single phase fault. And the blockade is short-range, which is exactly
why the grid needed a new mapping.

**Slide 6 (1:52-2:07)**
Our account could compile but not execute, so the noise from here on is a local model.
Under it, error detection recovers forty-two to fifty-eight percent of the solver quality
noise removes. Hypothesis three: supported.

**Slide 7 (2:07-2:19)**
But only in a window, roughly a tenth of a percent to one percent CZ error. Per raw shot,
the bare circuit still finds the optimum more often.

**Slide 8 (2:19-2:41)**
And the final answers don't move. Every sampler, even uniform random, scores the same,
because four thousand shots over a thousand selections always find the optimum. Selecting
cuts the prompt by seventy percent, but reading all ten paragraphs is still at least as
accurate. We pre-registered that risk, and we report it.

**Slide 9 (2:41-3:00)**
At ten paragraphs brute force takes a fifth of a millisecond, so this is not about speed.
It's a verified, grid-native error-detection mapping that is cheaper on Sqale's own
compiler, and an honest measurement of what it's worth to a language model. Thank you.

Word count of the script body (slide headers excluded): 423 words, about 141 per minute.

---

## 3. Sixty-second live demo

With `configs/main.yaml` the demo reads `data/demo_main.jsonl`, whose questions were each
checked live with Qwen. Each run takes 20-30 s even when cached (mostly loading the models),
and minutes if another process holds the GPU: close everything else on the GPU first. Keep a
screenshot and a backup recording of each run.

Type (PowerShell; the first three lines hide mangled characters and loader warnings):

```
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
python -m qctx.demo --config configs/main.yaml --index 0 2>$null
```

Index 0 is "Which theme park hosts both the Back to the Future Ride and The Simpsons Ride?"
(gold: Universal Studios Florida). Reading all 10 paragraphs answers "Hollywood"; top-3 is
also wrong because it reads the "Springfield (Universal Studios Hollywood)" distractor; the
QUBO and every QAOA tier read both gold paragraphs (the redundancy penalty drops the
distractor) and answer correctly at 565 tokens instead of 1,753. It is the clearest answer to
"why a QUBO and not top-k?".

What to point at, in order (about 10 s each):

1. **Top two lines.** The question, the gold answer and the two gold paragraph titles.
2. **Lane 1a row.** Reads all 10 paragraphs: note its token count (around 1,450 on average).
3. **Lanes 2a and 3a/3b/3c rows.** Same selection, so the same answer, at roughly a third of
   the tokens. The "reads:" line under each row lists the paragraph titles chosen. Check
   whether both gold titles are there.
4. **"kept" and "CZ" columns on 3c.** The error-detected tier discards about half its shots
   (51% kept on average in the main run) and runs a much larger circuit than bare 3b. That
   is the price of detection, visible live.
5. **"select s" vs "LM s".** Selection takes a fraction of a second; the language model
   dominates the latency.
6. Close: "Every quantum tier lands on the same selection as brute force. That is
   hypothesis 2, and it is why the final answers don't move."

If you have time: `--index 1` (Jack Elam) gives the same right answer on every lane at 236
tokens instead of 1,786; `--index 2` (5:15) is the failure mode: selection drops
"Quadrophenia (film)" and answers "Pete Townshend" while all-10 gets "Franc Roddam".
Indexes 3 and 4 are backups.

---

## 4. Judge Q&A prep

**1. Brute force solves this in milliseconds. Why quantum, and where's the speedup?**
There is no speedup, and we say so first. Brute force takes about 0.2 ms at 10 paragraphs
(`results/main/studies/roadmap.json`), and even per raw shot the bare circuit beats error
detection at finding the optimum (`figures/offline/fig7_noise_sweep.png`, right panel). What
we measure is what error detection on a neutral-atom grid is worth to a real application;
plain enumeration passes 1 s only at about 20-22 paragraphs (`figures/offline/fig5_roadmap.png`),
and a speedup claim would need hardware beating a strong classical solver such as branch and
bound at equal quality.

**2. Isn't a random sampler just as good?**
Downstream, yes, at this size, and we built the `rand` lane to show it: with 4,000 shots
over 1,024 selections plus repair, uniform random finds the optimum on 150 of 150
questions and gets the same EM, 0.367 (`results/main/results_table.md`). Before repair it
is far worse: approximation ratio 0.616 vs 0.869 for ideal QAOA, and at 64 shots in the main
run random finds the optimum on 27% of questions vs 85% for ideal QAOA
(`results/main/summary.json`). Even then EM does not move, which is why we judge solvers on
quality before repair.

**3. Why not just use top-k?**
In the main run top-3 by similarity scores 0.393 EM vs 0.367 for the exact QUBO selection,
and both intervals overlap reading all 10 (0.400) (`results/main/results_table.md`). So
top-k is a strong baseline and we say so. The QUBO's value is a structured problem with
two-qubit terms for QAOA, not better answers here.

**4. Did you run on hardware? Did you actually use Infleqtion's stack?**
We used Superstaq and Sqale's real compiler, but we did not execute: our account role
returned HTTP 403 for `cq_sqale_simulator`, `cq_sqale_qpu` and even
`ss_unconstrained_simulator`, and only `cq_compile` was allowed (`results/probe_output.txt`,
README "What ran, and where"). With that we compiled 36 circuits for `cq_sqale_qpu` and
simulated the output exactly (`results/offline/sqale_compile/compile_n12_bare-bus-direct.json`).
The execution path is written and tested against a fake provider, and all noise results come
from a local Pauli model.

**5. Does Sqale's compiler break your error-detection code?**
No. All 24 compiled encoded circuits (12 bus, 12 textbook) keep 100% of shots without noise,
and the logical distribution matches the original to TVD <= 8e-14
(`compile_n12_bare-bus-direct.json`). The bus circuit's CZ count does not move, 182.8 -> 181.8
(RZZ counted as 2 CZ), so the compiler neither re-routed it nor removed a check; the textbook
circuit grows 112.7 -> 239.1 because its hub qubits need routing, so ours is the cheaper
protected circuit on Sqale.

**6. Why trust a local noise model?**
Trust it for the comparison, not for absolute numbers. It is uniform Pauli noise at assumed
rates (0.5% CZ, 0.05% one-qubit, 0.5% readout), and its sampler is checked against exact
density-matrix simulation in Aer (`tests/test_iceberg_faults.py`, marked slow). We sweep the
rate from 0.1% to 2% so the conclusion doesn't hang on one guess
(`results/*/studies/noise_sweep.json`), and the fault-budget results count every single fault
exactly, with no sampling.

**7. Isn't discarding flagged shots just cherry-picking?**
We report the cost every time: 51% of shots kept in the main run, 38% offline
(`results/*/results_table.md`). The right panel of `fig7_noise_sweep.png` shows the
equal-raw-shot view, where bare wins on optimum-per-shot at every noise level. Error
detection gives cleaner samples, not more of them.

**8. Why the Iceberg code instead of Infleqtion's [[4,2,2]]?**
For 10 paragraphs, [[4,2,2]] needs 5 blocks and 20 qubits, and a ZZ term between blocks
becomes a 4-body operator. Iceberg keeps every logical ZZ two-body at 12 code qubits (18 with
our bus ancillas, which still fits a 3x6 patch of the 6x4 grid) (plan PDF p. 10; `README.md`).

**9. Does error detection change the final answer?**
No, and that was pre-registered as H4: solver gain +0.0026, accuracy gain 0.000, supported
(`results/main/hypotheses.md`; offline +0.0074 vs 0.000). The gold-recall and EM panels of
`figures/main/fig2_noise_to_accuracy.png` are flat across tiers.

**10. Hasn't this been done?**
Iceberg QAOA was done on trapped ions (He et al. 2024) on MaxCut, which has no single-variable
terms. QUBO evidence selection for HotpotQA was done without a quantum solver (Singh and
Vadlamani 2026). We found no Iceberg QAOA mapped to a neutral-atom grid, and no
error-detected quantum subroutine inside an LLM pipeline; the bus mapping and the 9-bit-flip
failure it removes are new to us (`README.md`, "Four things the plan did not anticipate").

**11. Where does Rydberg physics come in?**
Every two-qubit gate is a blockade CZ. A stray field shifts the Rydberg level and adds phase
error: 0.1 V/cm shifts Cs 60S by about 1 MHz (`physics/README.md`, illustrative level). Those
phase faults are exactly what the S_X check catches, 0% slip through
(`results/main/studies/fault_budget.json`), and the blockade's short range is why the hub
qubits needed the bus mapping.

**12. Does this scale?**
Not as built. The bus patch is 3 rows by k/2+1 columns, so it fits Sqale's 6x4 grid only up
to 10 paragraphs; larger problems need a folded layout we have not built
(`README.md`, "Limits"). Error detection alone also won't scale: He et al. saw it hurt on
larger problems, and our own sweep shows only 5% of shots kept at 2% CZ error. Going further
needs error correction at Infleqtion's 2028 scale.

Backup questions:
- *Why did selection lose to reading everything?* Selection misses a gold paragraph often:
  gold recall 0.773 in the main run (`results/main/results_table.md`). Also, paragraph
  similarity is the wrong redundancy notion for multi-hop QA, since the two gold paragraphs
  share the bridge entity, so we switched to question-coverage overlap (README, "Four things").
- *Did you tune on the test set?* No: tau, theta and hop were chosen on 50 separate
  calibration questions (`HYPOTHESES.md`, `scripts/calibrate.py`).

---

## 5. What we'd do with three more months

The roadmap (`figures/offline/fig5_roadmap.png`) says where this problem stops being free:
plain enumeration passes one second at about 20-22 paragraphs, while the bus Iceberg circuit
needs 33 physical qubits and about 300-580 CZ at 20 paragraphs (main vs offline edge
density, `results/*/studies/roadmap.json`), well past today's 24-qubit Sqale grid. Our first
job is the layout: the bus patch is three rows by k/2+1 columns, so it caps out at 10
paragraphs on the 6x4 grid. A folded layout that snakes the code and bus rows would make the
same nearest-neighbour construction work for 20-30 paragraphs, and we would re-run the
exhaustive single-fault proof on it. Second, we would get execution access and run lanes
3d-b/3d-c on `cq_sqale_simulator` noise-sim and then `cq_sqale_qpu`, and calibrate our local
model's CZ rate against the measured share of shots kept. That tells us whether Sqale sits
inside the 0.1-1% window where detection pays. Third, we would make the downstream test able
to see a difference: harder bridge questions, more paragraphs per question, and small shot
budgets where samplers actually separate. Last, we would compare against a strong classical
solver (branch and bound), not brute force, before saying anything about advantage.
