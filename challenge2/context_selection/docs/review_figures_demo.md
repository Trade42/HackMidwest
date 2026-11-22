# Review: figures, live demo, README (2026-11-21)

Reviewer pass over `figures/{main,offline,physics}/*.png`, `python -m qctx.demo --config configs/main.yaml`,
`README.md` and `Wavedysfunction_Quantum_Accelerator_Benchmark_Plan.md` at commit `ab6179f`
(plus the uncommitted offline re-run in the working tree; its summary means are unchanged from HEAD,
only timings differ). Read-only review: nothing else in the repo was changed. The demo was run from a
scratch copy of `configs/main.yaml` + `data/demo.jsonl`, so no `results/demo/` was written here.

## Ranked issues (most important first)

1. **README demo notes are wrong for the main (Qwen) config.** README says index 2 shows selection
   helping, index 1 the same answer at a fifth of the tokens, index 0 the failure mode. Under
   `configs/main.yaml` (run live, GPU, fp16):
   - index 2 (Albert Heijn): *every* lane answers "Albert Heijn" (gold "Koninklijke Ahold N.V."), all wrong.
   - index 0 (Jesus Daniel Garcia): every lane wrong, *including 1a* ("Primera Division de Mexico" vs "Liga MX"); 1b says "Cruz Azul". It does not show selection hurting.
   - index 1 (SS501 / KBS2): the only good one. 1a right at 891 tokens, 2a/3a/3b/3c right at 137 tokens (6.5x fewer), top-3 (1b) wrong ("SBS"). But the selection reads only one paragraph (Kim Hyun-joong) and misses gold "Inspiring Generation", so a judge reading the "reads:" line may ask how it got it right.
   `docs/pitch.md` section 3 still scripts `--index 2` as the "selection helps" moment. Replace (see Demo).
2. **Sqale compile table counts RZZ/RXX as one "CZ" before compilation.** In
   `results/offline/sqale_compile/compile_n12_bare-bus-direct.json`, the "input" two-qubit count of the
   bare circuit is `rzz: 19` (one per edge), the textbook circuit is `rzz 29 + cx 27 + rxx 10 = 66`. Each
   RZZ/RXX costs 2 CZ. So README / benchmark-doc "CZ before" (bare 22.8, textbook 69.8) are gate counts,
   not CZ counts, and the "x3.5 (routing)" / "x3.4 (routing)" blow-ups are overstated: in CZ terms
   bare is about 46 -> 81 (x1.8) and textbook about 112 -> 239 (x2.1); bus is ~170 CZ -> 182 (~x1.05,
   it already had 8 RZZ). The headline conclusion survives (bus is the cheapest protected circuit after
   Sqale compile, 182 vs 239 CZ, and both keep 100% noiseless), but "Before compilation it looked like
   the more expensive one (176 against 70)" is an apples-to-oranges comparison (fig4 offline itself says
   textbook all-to-all = 113 CZ). Fix the column header to "2-qubit gates in (RZZ = 1)" or convert to CZ.
3. **Main-config demo latency depends on an idle GPU.** Alone on the GPU: 18-29 s wall per run
   (cold or cached; almost all of it is Python/torch import and loading bge + Qwen, LM time per lane
   0.1-0.5 s). With another process holding the GPU (7.7 GB / 8 GB used, 100% util during this review)
   runs took 58-186 s (lane 1a LM time 108.7 s). On stage: close everything else on the GPU,
   pre-run the exact questions once, and keep the screenshot / recording as a fallback.
4. **Judge-visible noise on Windows console.** (a) Non-ASCII titles print as mojibake
   ("Jes?s Daniel Garc?a", "Primera Divisi?n de M?xic") unless `PYTHONIOENCODING=utf-8` (or
   `chcp 65001` / Windows Terminal UTF-8) is set; (b) stderr shows two `Loading weights` progress bars,
   `[transformers] torch_dtype is deprecated! Use dtype instead!`, and on fresh runs
   `Warning: You are sending unauthenticated requests to the HF Hub`. Set `HF_HUB_OFFLINE=1`,
   `PYTHONIOENCODING=utf-8`, and redirect `2>$null` (PowerShell) for the live run. No crash in 13 runs.
5. **Status contradictions in the benchmark doc.** Line 27 "Compilation benefit: ... `cq_compile`
   written, **not run**" and line 138 "Sqale compile check ... Ready" contradict line 10 and §9, which
   report two real `cq_compile` calls (36 circuits). Line 24 "Sqale untested" is now partly stale. The
   headline box (line 14) and §"Application benefit" still lead with the offline numbers only.
6. **Sqale compile results sit inside "Results with the plan's models" but used offline-config
   circuits** (`compile_n12...json`: `"config": "configs/offline.yaml"`, TF-IDF graphs, ~23 edges).
   The README says so only implicitly via the path. Say it in one sentence, or move the block.
7. **Same quantity, three values.** Offline bare-grid CZ appears as 68 (fig7 legend), 69 (README tier
   table, Sqale paragraph), 70 (fig4, fig6, fault table); bus as 181 / 182 / 183; textbook as 111 / 112 / 113.
   They come from different question subsets (30-question sweep, 150 questions, 12-question fault
   budget). Harmless but a sharp judge will notice; state the subset in each caption or use one number.
8. **"Multiple platforms" stretch-goal row lists "Sqale simulator, Sqale QPU"** although neither ever
   executed a job (403). Reword as "Sqale compiler (cq_compile); Sqale simulator/QPU code paths tested
   with a fake provider".

## Figures

Legend for each verdict: OK = readable, numbers match the results files; Note = caveat worth fixing.

### figures/main (Qwen + bge, `results/main/`)

| Figure | Agreement with results | Issues |
|---|---|---|
| fig1_accuracy_by_lane | Matches `results_table.md` (1a 0.400, 1b 0.393/0.387, 2a..4 0.367/0.373, 3a-p2 vote 0.380). 95% CIs shown, x axis from 0. | OK. Honest. Title says 150 questions; does not show 3c@64 vote 0.393, fine. |
| fig2_noise_to_accuracy | Matches summary: valid 0.859/0.835/0.849, AR 0.8690/0.8628/0.8654, recall 0.773, EM 0.367; 64-shot row matches 3a@64/3b@64/3c@64. | Note: each panel's y axis is truncated (AR panel spans 0.855-0.875, ~0.02). Caption says "Each panel has its own scale", good, but the unpaired per-lane CIs overlap heavily while H3's *paired* CI is clearly positive (+0.0026 [+0.0020, +0.0031]); add the paired difference as text, or a judge will read "no difference". |
| fig3_redundant_edges | Edge sweep (main/studies/edge_sweep.json): gain about +0.002 at ~3 edges rising to +0.011 at ~31. | Note: 3rd-panel y label ("Approx. ratio: error-detected minus bare") is long and crowds the middle panel. README only quotes the offline numbers (+0.001 / +0.010). |
| fig4_two_qubit_gates | Title "6 redundant edges on average" vs README "mean 5.0 redundant edges" (150 questions) and config comment "4.5" (calibration set). Figure is the mean over the compile_stats subset. | Note: say "(12 evaluation questions)" in the title. Bars 12/14/79/161/120 are not quoted anywhere in the README main section. Does not include the real Sqale compile numbers (would be the best compile figure; pitch.md already asks for a fig8). |
| fig5_roadmap | Brute-force fit "1 s at about 22" (offline figure says "about 20" for the same algorithm: timing noise, not a property of the scorer). | Note: "100 logical qubits (Infleqtion 2028 target)" needs a source in README Sources. CZ panel compares bus (nearest neighbour) against textbook *all-to-all* (free wiring) - flattering to textbook, stated in legend, OK. |
| fig6_fault_budget | Matches main/studies/fault_budget.json: undetected 92.0% / 12.5% / 3.4%, AR-loss slope 0.79 / 1.68 / 0.26, discard slope 0 / 66 / 110, CZ 14 / 79 / 120. | OK, cleanest "error detection works" figure. Note: in the main run textbook Iceberg loses *more* AR per unit CZ error than bare (1.68 vs 0.79), unlike offline (1.82 vs 1.94). README never states the main fault budget; the README table is offline. "Discarded share per unit CZ error rate" = 110 is a slope; a judge may read it as a percentage. |
| fig7_noise_sweep | main/studies/noise_sweep.json. | Note: in the main run bus Iceberg beats bare by only +0.001 / +0.0015 / +0.003 AR at 0.1 / 0.2 / 0.5% CZ error, ties at 1% (+0.0002), loses at 2% (-0.022); README's "wins up to 1%" is the offline result. No error bars, although the json has `ar_se` (~0.005 per point, larger than the gaps; the paired gap is what matters - say so). |
| fig7b_noise_sweep_phase_errors | as above, phase-only noise. | Note: even with phase-only noise the AR gain is small (+0.0024 at 0.5%), bus is slightly *below* bare at 1% (0.8629 vs 0.8645) and loses at 2%. Do not pair it with the "phase errors are caught every time" line without saying "single faults". |

### figures/offline (TF-IDF + t5, `results/offline/`)

| Figure | Agreement | Issues |
|---|---|---|
| fig1 | Matches README offline table (0.373; 0.313; 0.320/0.340; greedy 0.327; SA 0.333). | OK. |
| fig2 | Valid 0.759/0.619/0.718, AR 0.9468/0.9339/0.9414 match README tier table (76%/62%/72%, 0.947/0.934/0.941). | Same truncated-axis/unpaired-CI note as main fig2. |
| fig3 | +0.001 below 6 edges, +0.010 above 30: matches README. | OK. |
| fig4 | 46/70/113/224/183 match README text. | README tier table says 69 and 182 (different subset). |
| fig5 | | Same "2028 target" sourcing note. |
| fig6 | 97.9/14.8/3.9%, 1.94/1.82/0.29, CZ 70/113/183 match README fault table. | OK. Note: README also quotes "Phase (Z-type) faults that slip through 95.2% / 0% / 0%" which is not shown in the figure. |
| fig7 | AR 0.9457..0.8934 bare, 0.9475..0.8670 bus, kept 82..5% match README. | Legend CZ 68/111/181 vs 69/112/182 and 70/113/183 elsewhere. No CIs. |
| fig7b | | Phase-only: bus and bare almost tie at 1%; same caution as main. |

### figures/physics

| Figure | Check | Issues |
|---|---|---|
| grotrian | Cs 459.4/1041.0 nm, Rb 421.7/1005.9 nm match `summary.json`; 60S marked "illustrative". | Note: "Caesium (data qubits)" / "Rubidium-87 (ancilla qubits)" is an architectural claim; cite the Rb-Cs dual-species source in README Sources (only Bedalov et al. is listed). |
| pair_potential | 7.4 MHz at 5.5 um, blockade radius 7.7 um at 1 MHz drive, 0.12 MHz at 11 um. | Note: "5.5 um, cited Rb-Cs spacing" - the citation is only in `physics/pair_potential.py`; add to README Sources. |
| rydberg_scaling | Fits n^3.24, n^7.18, n^12.49, n^-3.21 vs textbook. | OK. C6 fit n^12.5 vs textbook n^11 is visibly off at n=30 (point below fit); fine for an "illustrative" figure. |
| stark_map | 197 MHz cm^2/V^2; -alpha E^2/2 at 0.1 V/cm = -0.985 MHz, matches "-0.99 MHz". | Note: left panel's 60S trace has small kinks near 1.55 and 2.05 V/cm (state tracking across avoided crossings). Cosmetic. |

Nothing in any figure is labelled "hardware"; every noisy figure footer says "local stand-in model, not
Sqale" or "Assumed rates, not Sqale calibration data". Good.

### Best figures for a 3-minute deck

1. `figures/main/fig6_fault_budget.png` - one glance: 92% of CZ faults slip through bare, 3.4% through the bus code. Exact, not sampled; no CI needed. Strongest "the quantum part works" slide.
2. `figures/main/fig1_accuracy_by_lane.png` - the honest result in one picture: every selector, quantum or classical or random, lands at the same accuracy; reading everything is best. Axis from 0, CIs shown.
3. `figures/offline/fig7_noise_sweep.png` (or main) - "error detection pays in a window": left panel wins below ~1%, right panel shows bare wins per raw shot. Good for the Q&A on shot cost.
4. A new Sqale-compile bar chart (pitch.md's proposed fig8) once the CZ-count unit issue (issue 2) is fixed; until then fig4 offline.
5. `figures/physics/stark_map.png` only if there is a physics slide; it ties "phase errors from stray fields" to fig6's "phase faults: 0% slip through".

Avoid fig2 in a 3-minute deck (eight panels, truncated axes) and fig5 (needs explanation).

## Demo

### Current demo questions under `configs/main.yaml` (run live, RTX 4070, fp16)

| Index | Question | 1a (all 10) | 1b (top-3) | 2a = 2b = 3a = 3b = 3c | Story? |
|---|---|---|---|---|---|
| 0 | Jesus Daniel Garcia Esparza / division | wrong "Primera Division de Mexico", 631 tok | wrong "Cruz Azul", 214 | wrong, 264 tok, drops gold "Cruz Azul" | No: 1a is wrong too, so it does not show selection hurting |
| 1 | SS501 rapper / station | right "KBS2", 891 tok | wrong "SBS", 237 | right, **137 tok (6.5x fewer)**, reads 1 paragraph | Yes, "same answer, far fewer tokens, top-3 fails" - but reads only one of the two gold paragraphs |
| 2 | Albert Heijn / Delhaize | wrong "Albert Heijn", 876 | wrong | wrong, 237 | No: all lanes wrong (README's "selection helps" story is offline-only) |

The quantum lanes always choose the same paragraphs as brute force (expected: H2, and the rand control in
the batch run). 3c shows 46-54% kept and 105-135 CZ vs 2-26 CZ bare: the price of detection is visible live.

### Better questions from the 150 (searched `results/main/answers_*.jsonl`, all re-run live with the demo; live output matched the batch files on every one)

Run with `--question q.json`, where `q.json` is the line from `data/questions.jsonl` (or append them to `data/demo.jsonl`).

| Role | qid | Question (gold) | Evidence (live demo, main config) |
|---|---|---|---|
| **Selection helps (lead)** | `5a89fc665542993b751ca9de` | "Which theme park is host to both the Back to the Future Ride and The Simpsons Ride?" (Universal Studios Florida) | 1a wrong "Universal Studios Hollywood" at 1753 tok; top-3 wrong "Hollywood", because it reads the distractor "Springfield (Universal Studios Hollywood)"; 2a/3a/3b/3c right at 565 tok and read both gold paragraphs (the redundancy penalty drops the Hollywood paragraph). Best story for *why a QUBO and not top-k*. Small blemish: 2b's vote column says Hollywood. |
| Selection helps (backup) | `5a83aaeb5542996488c2e483` | Kanye West album with Roc-A-Fella and Dwele (My Beautiful Dark Twisted Fantasy) | 1a "Graduation" 1749 tok, 1b "Late Registration" 511 tok, both wrong; 2a..3c right (single and vote) at 562 tok. Reads only one gold paragraph. |
| Selection helps (avoid) | `5ac289ff5542996366519a02` | Marsilea vs Brabejum species | single answer right at 367 tok vs 1a wrong at 1808, but the *vote* column shows the wrong "Brabejum" on every selected lane - confusing on stage. |
| **Same answer, far fewer tokens** | `5a90bdea55429916514e754b` | "What man with a lazy left eye starred in Sacred Ground?" (Jack Elam) | every lane right; 1a 1786 tok, top-3 569, QUBO/QAOA **236 tok (7.6x fewer)** reading exactly the two gold paragraphs. Cleanest. |
| Same answer (backup) | `5ac54c435542993e66e822d1` | "Where did Cale Gundy's brother play football in college?" (Oklahoma State University) | all right; 1936 vs 300 tok (6.5x), reads exactly Mike Gundy + Cale Gundy. |
| **Failure mode** | `5a89d36e554299515336132a` | "Who directed the film based on the rock opera 5:15 appeared in?" (Franc Roddam) | 1a right (1414 tok) and top-3 right (431 tok, both gold); QUBO and every QAOA tier read only "5.15", drop gold "Quadrophenia (film)", answer "Pete Townshend" (201 tok). Shows the main-run result (QUBO slightly below top-3 and all-context) honestly. |
| Failure (alt) | `5a72a51355429901807daf21` | Suncoast Hotel and Casino chairman (Bill Boyd) | 1a right; everything else "Donald Trump"; but QUBO = top-3 selection here, so it does not isolate the QUBO. |

Suggested 60-second order: Universal Studios (helps) -> Jack Elam (7.6x fewer tokens) -> 5:15 (failure, if time).

### Runtime

| Run | Wall time |
|---|---|
| Cold, first run of a question (GPU otherwise idle) | 18-29 s (index 0: 21 s, index 1: 29 s; candidates 18-27 s) |
| Same question again, cached in `results/demo/` | 19.5 s |
| Same, while another process held the GPU (7.7 of 8 GB, 100% util) | 58-186 s; lane 1a LM time rose from ~0.6 s to 108.7 s |

Caching saves only the LM/solver seconds; the ~18 s floor is Python + torch import and loading bge and
Qwen. Per lane on an idle GPU: selection 0.00-0.3 s, LM 0.1-1.1 s. Usable live if started while you talk,
not if the judges must watch a blank terminal for 20 s; consider showing a pre-run output first and running
live only once. A warm in-process mode (load models once, loop over questions) would make it instant.

### What a judge would see

- No crash or traceback in 13 runs.
- stderr: two `Loading weights: ...` progress bars, `[transformers] torch_dtype is deprecated! Use dtype instead!`, and on some runs `Warning: You are sending unauthenticated requests to the HF Hub` (it still calls the Hub; set `HF_HUB_OFFLINE=1`).
- stdout in the default Windows console mangles non-ASCII ("Jes?s", "Garc?a", "M?xico"); fixed by `PYTHONIOENCODING=utf-8`.
- The table is 121 characters wide; the answer column truncates at 25 characters ("My Beautiful Dark Twisted", "Universal Studios Hollywo"). Widen the terminal or shrink the font.
- The demo omits lanes `2c`, `rand` and `4`; showing `rand` would make the "all samplers tie" point live.

## README issues (judge's skim, top 5)

1. **Demo paragraph is stale** (issue 1): the index 2 / 1 / 0 story is from the offline run, but the
   command right above it uses `configs/main.yaml`, where index 2 and index 0 are wrong on every lane.
2. **Two result sets, offline still dominant.** The "plan's models" run is the real one, yet the tier
   table, fault table, noise sweep, edge-sweep numbers and the benchmark-doc headline are all offline
   (t5 / TF-IDF). A judge sees 0.320 and 0.367, 58% and 42%, 38% and 51% kept, 70 and 14 CZ for the same
   "bare" circuit, without one line saying which to quote. Suggest: lead with main, move offline to
   "stand-in run (denser graphs, larger error-detection effect)".
3. **Sqale compile table** (issue 2): "CZ before" is RZZ count, so x3.5 / x3.4 overstate routing cost;
   and the 12 questions are offline-config circuits sitting inside the main section (issue 6).
4. **Quick start implies Superstaq is needed for the main run** ("Run the plan's stack (your machine,
   Superstaq key, Hugging Face access)") while `configs/main.yaml` has `hardware: null` and runs fully
   locally; the long Superstaq-credits paragraph and "Three things to look at on the first Sqale run"
   (with the old "129 to 132 for the 9-edge test instance") read as if Sqale execution is still pending
   and are now partly answered by the compile results. Collapse into one short "Superstaq status" box.
5. **Inconsistent or unexplained counts**: 5.0 vs 6 vs 4.5 mean edges (README / fig4 title / config
   comment), CZ 68/69/70 and 181/182/183, "Sqale simulator, Sqale QPU" under multi-platform although
   no job ran, "Limits" still says only "the reader in the offline run is t5-base" without mentioning the
   main run's GPU fp16 nondeterminism noted elsewhere.

## Recommendations

- Rewrite the README demo sentence and `docs/pitch.md` section 3 with the main-config questions below; add those questions to `data/demo.jsonl` (keep the old three for the offline config) or pass them with `--question`.
- Show lane `rand` in the demo (`DEMO_LANES` lacks it): "a random sampler picks the same paragraphs" is the most honest live moment and pre-empts the judge's question.
- Live-run hygiene: `$env:HF_HUB_OFFLINE="1"; $env:PYTHONIOENCODING="utf-8"; python -m qctx.demo --config configs/main.yaml --question q.json 2>$null`, nothing else on the GPU, pre-run once so `results/demo/` is cached (it is git-ignored).
- Fix the Sqale compile "CZ before" unit (count RZZ/RXX as 2 CZ, or label it "2-qubit gates"), and make the fig8 bar chart from `compile_n12_bare-bus-direct.json`.
- Update benchmark-doc lines 24, 27, 138 to "cq_compile run: 36 circuits, all checks preserved".
- Add "(12 questions)" / "(30 questions)" to fig4/fig6/fig7 titles, and add paired-difference text to fig2.
- Add sources for "Infleqtion 2028 target", "5.5 um Rb-Cs spacing" and the Cs-data/Rb-ancilla architecture.
