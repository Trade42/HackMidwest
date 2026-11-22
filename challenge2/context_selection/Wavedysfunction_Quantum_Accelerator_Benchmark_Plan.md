# Wavedysfunction | Infleqtion
## Quantum Context-Selection Accelerator: Validation & Benchmarking Plan

**Project:** MidQWest 2026 — Infleqtion Challenge 2  
**Approach:** QAOA-based context selection for language-model question answering, using Superstaq/Sqale and an Iceberg-code error-detection layer (grid-native "bus" mapping)  
**Status (2026-11-21, 21:50):** Protocol plus two full runs on 150 questions, both with local Pauli noise.

- **Offline stand-ins:** TF-IDF scores, t5-base reader.
- **The plan's models:** Qwen 2.5 1.5B Instruct reader, bge-small scorer (`results/main/`). Results in §9, "Main run".
- **Sqale compiler:** used through `cq_compile`, and the compiled Iceberg circuit keeps 100% of shots without noise.
- **Not run:** Sqale execution, simulator or QPU. Our Superstaq role returns 403 on every job.  
**Source architecture:** *Error-Detected Quantum Context Selection — MidQWest Challenge 2 Plan* (November 21, 2026), `docs/challenge2-plan.pdf`. Implementation and full results: `README.md`, `HYPOTHESES.md`, `results/offline/`.

> **Headline (offline stand-in, 150 questions):** Selection cuts the prompt from 1,425 to 335 tokens, but loses exact match (0.320 vs 0.373 for all 10 paragraphs) because it drops a gold paragraph on about half the questions. Bus-mapped Iceberg error detection recovers 58% of the approximation ratio that noise took away (0.934 → 0.941, ideal 0.947) while keeping 38% of shots. Final-answer accuracy is identical across all sampling lanes, including a uniform random control. **No quantum speedup is claimed.**

> **Research distinction:** A working quantum backend, an application-level benefit, and quantum speedup are **three separate claims**. Test them independently. With just 10 candidate paragraphs (2^10 = 1,024 selections), exact classical enumeration takes milliseconds. **Do not claim quantum speedup based on this experiment alone.**

---

## 1. What we need to prove

| Claim | Core question | Minimum evidence | Status |
|---|---|---|---|
| **Correctness** | Does the accelerator return valid and meaningful context selections? | Unit/integration tests; agreement with exact classical references on small instances | **Done** (local); Sqale untested |
| **Application benefit** | Do quantum-selected paragraphs improve an LM's evidence use, answer accuracy, or resource efficiency? | Paired evaluation against strong classical selection and no-selection baselines | **Done (offline): no accuracy benefit**; 4× fewer prompt tokens, same as classical selectors |
| **Error-detection benefit** | Does Iceberg-protected QAOA yield better *useful* solutions after accounting for discarded shots and overhead? | Bare vs. protected QAOA at matched raw-shot and matched accepted-shot budgets | **Done (local noise model)**: better kept-shot quality; per raw shot the bare circuit finds the optimum more often. Equal-accepted-shot comparison open |
| **Compilation benefit** | Does Superstaq improve execution-relevant circuits on the Sqale target? | Pre/post compilation gate counts, depth, scheduling/resources; execution measurements where available | Local Qiskit compile done; `cq_compile` written, **not run** |
| **Quantum speedup** | Is a quantum implementation faster at **equal target solution quality** than a strong classical implementation of the same task? | End-to-end time-to-solution on actual hardware, matched objectives/quality, strong classical optimizers, scaling evidence | **Not claimed** |

A positive result in one row does **not** imply the others.

## 2. Pipeline under test

```text
HotpotQA question + 10 candidate paragraphs
                |
                v
   Classical scoring  (src/qctx/scoring.py)
 relevance r_i, overlap s_ij
 offline run: TF-IDF; HF embeddings wired, not run
                |
                v
      Shared QUBO formulation  (src/qsel/qubo.py)
   weighted independent set, w_i = r_i - theta
                |
     +----------+----------+
     |                     |
     v                     v
Classical / control     Quantum solvers
- all context (1a)      - ideal QAOA, p=1 / p=2 (3a, 3a-p2)
- similarity top-k (1b) - bare QAOA, noisy, grid-routed (3b)
- exact brute force (2a)- Iceberg QAOA, bus mapping (3c)
- greedy (2b)           - Iceberg QAOA, textbook mapping (3c-direct)
- simulated annealing   - router: classical vs quantum (4)
  (2c)
- uniform random (rand)
     |                     |
     +----------+----------+
                |
                v
  Feasibility repair, rank, keep 3 best distinct  (src/qsel/sampling.py)
                |
                v
   Selected paragraphs -> same LM  (src/qctx/lm.py)
  (single best or top-3 voting, src/qctx/vote.py)
                |
                v
      Answer + benchmark logs  (results/<run>/)
```

**Change from the original plan:** overlap is computed on question-coverage profiles (`overlap: coverage`), not raw paragraph similarity. In HotpotQA the two gold paragraphs share the bridge entity, so under paragraph similarity they sit at the 90th percentile of all pairs, and calibration picks "no edges" (no entangling gates, nothing for QAOA to do). Under coverage overlap, gold pairs sit at the 11th percentile. An optional second-hop relevance term (`hop`) was also added. `overlap: paragraph`, `hop: 0` restores the literal plan.

**Model choice.** The plan specifies "a 0.5–2B instruct model from Hugging Face, run locally at temperature 0" but names no model. `configs/main.yaml` uses **`Qwen/Qwen2.5-1.5B-Instruct`** as the reader, for these reasons:

- It sits inside the plan's 0.5–2B range and runs on a laptop CPU with no GPU.
- It is openly downloadable with no license sign-up (Llama models are gated on Hugging Face), so anyone can reproduce the run.
- It is instruction-tuned, so it answers in a short, normalizable form, which exact match needs.
- It is preferred over `Qwen2.5-0.5B-Instruct` (the code default in `src/qctx/lm.py`) because 0.5B is weak at multi-hop questions. A weak reader adds noise that buries differences between selectors, and H1 (does selection help?) depends on the reader actually using the evidence it is given.
- The same family offers 0.5B as a cheaper comparison point and has an Ollama tag (`qwen2.5:1.5b`) for the local fallback, with no code changes.

The paragraph scorer is `BAAI/bge-small-en-v1.5`, a small embedding model consistent with the plan. Calibration was redone for it on the 50 calibration questions (`results/main/calibration.json`). The result was hop 0, tau 0.6075 (90th percentile of overlaps), theta 0.25: both gold paragraphs selected on 74% of calibration questions, 2.72 paragraphs on average. Two consequences matter for the main run:

- **The QUBO selector does not beat plain top-3 under bge.** Top-3 selects both gold paragraphs on 76% of calibration questions, against 74% for the QUBO. Under TF-IDF it was the reverse (66% vs 78%), so expect H1-type comparisons to look worse for selection.
- **The redundancy graphs are sparse: 4.5 edges on average, against 22.5 offline.** That means fewer entangling gates and shallower circuits. By the offline edge sweep (fig3), the error-detection gain is smallest below 6 edges (+0.001 AR), so expect a smaller H3 effect than offline.

We kept the pre-registered calibration rule (most "both gold" within the 3-paragraph budget) rather than forcing denser graphs, because choosing settings to favour the quantum lanes would be tuning toward the result. A denser-graph run (`--min-edges`) is a possible labelled sensitivity check, not the main result.

**Control the variables:** Every solver sees the same question, candidate paragraphs, frozen relevance/overlap scores, QUBO, downstream language model, prompting template, decoding settings, and split. `tau`, `theta` and `hop` were tuned on 50 separate calibration questions (`data/calibration.jsonl`, `scripts/calibrate.py`); the 150 evaluation questions were never used for tuning. QAOA angles are tuned once per question on the exact simulator and reused across tiers, so the tiers differ only in noise and encoding.

## 3. Correctness tests — run before any performance benchmark

| Test | Procedure | Pass criterion | Status / where |
|---|---|---|---|
| QUBO formulation | Enumerate all solutions for small `n`, independently compute the intended objective and constraints | Encoded QUBO energies and feasible sets agree with the reference | Done: `tests/test_qsel_core.py` |
| QAOA circuit | Compare the circuit's statevector/probabilities against an independent Hamiltonian implementation for tiny problems | Agreement within numerical tolerance | Done: `tests/test_qsel_core.py`, `test_qsel_backends.py` |
| Iceberg encoding | Compare ideal bare and logically encoded/decoded probability distributions | Same logical output distribution, 100% of shots kept, in the zero-noise limit | Done: `tests/test_iceberg_encode.py` (sizes 2–10, p = 1 and 2, scrambled measurement maps, Qiskit level-3 compile) |
| Error detection | Inject every Pauli after every gate and every readout flip, propagate exactly | No single fault in state prep, S_X check or readout goes undetected; each flag is needed | Done: `tests/test_iceberg_faults.py` |
| Fast noise simulator | Pauli-frame sampler vs brute force (single faults) and Aer density matrix (many faults) | Agreement to 1e-9 / within sampling error | Done: `tests/test_iceberg_faults.py` (Aer test marked `slow`) |
| Independent review | Separate agent, own simulator, ~1,000,000 single faults over 300 circuits | No undetected single fault | Done. Found two gaps in the bus mapping (two-layer Y fault on a bus qubit; too-short bus row for 1–2 paragraphs), both fixed and covered by regression tests. Two-fault events not enumerated |
| Selection validity | Inspect sampled bitstrings against redundancy constraints before and after repair | All returned selections feasible after repair; pre-repair rate logged | Done: `tests/test_qsel_sampling.py`; feasibility reported per lane |
| Measurement ordering | Confirm classical-bit and logical-bit ordering using known basis states | Correct bitstring interpretation | Done: `tests/test_iceberg_postselect.py` |
| Superstaq/Sqale integration | Run small known circuits on local and Sqale simulators | Valid job/result handling; 100% kept when noise is disabled | **Written, tested with a fake provider only** (`tests/test_qsel_superstaq.py`). Live check: `scripts/check_superstaq.py` |
| Full application | Process a known question end-to-end | Valid selection, answer, timings, correctly formatted logs | Done: `tests/test_app_harness.py`; CI runs the smoke pipeline on every push |

**Important:** The Iceberg code is not a drop-in wrapper for an arbitrary circuit. The textbook ("direct") mapping, built for MaxCut, has a failure mode on this QUBO: each single-paragraph weight becomes a gate on the `b` qubit, and one undetected fault there (`X_i X_b`) flips 9 of 10 logical bits. The bus mapping keeps data qubits off `t`/`b` outside state preparation and the final check; its worst harmful undetected single fault flips at most 2 bits.

### Commands

```bash
pip install -e ".[dev]"
python -m pytest -q -m "not slow"                          # about a minute
python reproduce.py --config configs/smoke.yaml --quick    # full pipeline, 3 questions, stub reader
python reproduce.py --config configs/offline.yaml          # every offline number and figure
python scripts/check_superstaq.py                          # Bell test, timing, noiseless Iceberg check on Sqale
python scripts/calibrate.py --config configs/main.yaml     # hop, tau, theta on calibration questions
python reproduce.py --config configs/main.yaml             # the plan's stack (Superstaq key, HF access)
```

## 4. Experiment matrix

The **same 150 HotpotQA distractor questions** (hard, dev set; `data/questions.jsonl`, sampled via the HippoRAG repository), 10 paragraphs each, two gold. Seed 0. Config as run: `results/offline/config_used.json`.

| Lane | Selection method | Backend | Primary role | Status |
|---|---|---|---|---|
| 1a | All 10 paragraphs | Local | No-selection baseline | Run |
| 1b | Top-3 by relevance | Local | Simple relevance baseline | Run |
| oracle | Gold supporting paragraphs (oracle) | Local | Upper bound for evidence selection | Run: EM 0.553 main, 0.460 offline (all context 0.400 / 0.373) |
| 2a | Brute-force exact QUBO optimum | Local | Optimization reference | Run |
| 2b | Greedy | Local | Fast classical baseline | Run |
| 2c | Simulated annealing | Local | Stochastic classical comparator | Run |
| rand | Uniform random sampler, same repair and ranking | Local | Control: what repair alone achieves | Run (added; not in original plan) |
| 3a, 3a-p2 | Ideal QAOA, p = 1 and p = 2 | Local statevector | Algorithm without device noise | Run |
| 3b | Bare QAOA, grid-routed | Local Pauli noise (stand-in for `cq_sqale_simulator`) | Noisy comparator | Run locally; Sqale not run |
| 3c | Iceberg QAOA, bus mapping | Local Pauli noise | Protected circuit and postselection cost | Run locally; Sqale not run |
| 3c-direct | Iceberg QAOA, textbook mapping (free all-to-all wiring) | Local Pauli noise | Encoding comparison | Run (added) |
| 4 | Router: classical vs quantum tier | Local | Stretch goal | Run (picked 3c on all questions at default noise) |
| 3d-b, 3d-c | Bare and bus-Iceberg QAOA executed through Superstaq | `cq_sqale_simulator` noise-sim | Sqale noise vs the local model | **Blocked**: our account's role may only compile. `check_superstaq.py --probe` (2026-11-21): every job and QUBO submission returned 403 on every target, including `ss_unconstrained_simulator`. Access requested |
| — | Sqale compile check | `cq_compile` for `cq_sqale_qpu`, then exact local simulation of the compiled circuits (`scripts/sqale_compile.py`) | Real Sqale gate counts; does Sqale's compiler preserve the Iceberg checks (100% kept without noise, same logical distribution)? | Ready. 0.5 credits per call |
| — | 3d lanes on actual Sqale QPU | `cq_sqale_qpu` | Hardware evidence | **Not run**: no QPU access |

Every sampling lane was also run at 64 raw shots (`@64` variants) for an equal-raw-shot view. Every non-oracle selector reports both **single best selection** and **top-three distinct selections + voting**; the classical 2a/2b/2c lanes also vote over three distinct selections, so the vote comparison is fair.

### Pre-registered hypotheses and verdicts

Fixed in `HYPOTHESES.md` before any evaluation result existed. Verdict rule: supported when the 95% paired-bootstrap interval lies entirely on the predicted side of zero.

| | Prediction | Quantity | Result [95% CI] | Verdict |
|---|---|---|---|---|
| H1 | Selection helps | EM(2a, single) − EM(1a) | −0.053 [−0.133, +0.020] | **not supported** |
| H2 | Ideal QAOA at p = 2 is close to exact | (a) share with exact optimum on top; (b) EM(3a-p2) − EM(2a) | (a) 150/150, Wilson [0.975, 1.000]; (b) 0.000 | **supported** |
| H3 | Error detection helps the solver | AR(3c, kept shots) − AR(3b) | +0.0074 [+0.0070, +0.0079] | **supported** |
| H4 | The benefit shrinks downstream | [AR(3c) − AR(3b)] − [EM(3c) − EM(3b)] | solver +0.0074, accuracy +0.000 | **supported** |
| H5 | The vote helps | EM(2a, vote) − EM(2a, single) | +0.020 [−0.007, +0.053] | **not supported** |

A null or negative finding is valid; two of five came out that way.

## 5. Metrics: definitions and calculations

### A. Language-model outcome

| Metric | Definition | Direction |
|---|---|---|
| **Exact Match (EM)** | Fraction of final normalized answers exactly matching the gold answer | Higher |
| **Answer F1** | Token-level precision/recall F1 versus gold answer | Higher |
| **Both-evidence hit rate** | Fraction of questions for which selected paragraphs include both gold supporting paragraphs | Higher |
| **Evidence recall** | Fraction of required gold paragraphs selected (0, 0.5, or 1) | Higher |
| **Prompt tokens** | Tokens per question, including every LM call in voting mode | Lower at equal quality |
| **LM calls / question** | Number of downstream model invocations | Lower at equal quality |

Identical model weights, prompts, temperature `0`, and normalization rules across lanes (`src/qctx/metrics.py`).

### B. Optimization outcome

Let `E(x)` be the QUBO energy, `E_min` and `E_max` its exact minimum and maximum over all `2^n` bitstrings.

- **Approximation ratio (as used):** mean-sample, normalized form `AR = (E_max − mean E) / (E_max − E_min)` over the kept shots **before repair**. Because it is normalized by the full energy range, it is a normalized additive gap: well defined with negative energies, 1 at the optimum, about 0.73 for uniform random sampling here. The mean-sample form is deliberate: a best-sample metric cannot tell tiers apart at this size (see §12).
- **Optimum hit rate:** fraction of questions whose top selection is an exact optimum; and **P(optimum) per kept shot**.
- **Feasibility rate:** fraction of raw samples satisfying all redundancy constraints **before repair**.
- **Sample diversity:** distinct feasible selections per 1,000 shots.

### C. Error-detection outcome

```text
shot_acceptance_rate = shots_accepted / shots_raw
useful_shot_rate     = P(optimum | kept) * shot_acceptance_rate      # equal-raw-shot view
```

Report both. A higher-quality postselected distribution can still be less useful if most shots are discarded, and here it is: per raw shot, the bare circuit finds the optimum more often at every noise level tested (`figures/offline/fig7_noise_sweep.png`, right panel).

Fairness comparisons:

1. **Equal raw shots:** done (4,000-shot lanes and the `@64` lanes; useful-shot rate in the noise sweep).
2. **Equal accepted shots:** **done** (`scripts/equal_kept_shots.py`, 30 questions, about 4,000 kept shots each). Bus Iceberg minus bare, approximation ratio: +0.0025 [+0.0016, +0.0033] (main), +0.0069 [+0.0061, +0.0077] (offline). P(optimum) per kept shot: +0.0010 [−0.0005, +0.0026] (main, not significant), +0.0016 [+0.0006, +0.0025] (offline). Cost: 2.0× (main) and 2.7× (offline) the raw shots. The gain per kept shot does not depend on sample size; the price is in raw shots.

### D. Timing and hardware resources

| Stage | Captured | Offline value (mean per question) |
|---|---|---|
| Preprocessing | `t_embedding`, `t_build` | TF-IDF precomputed; QUBO build ≈ 0.2 ms |
| Quantum preparation | `t_tune` (angle tuning, encoding) | ≈ 77 ms |
| Selection / execution | `t_execute` | exact 0.2 ms; ideal QAOA 1.8 ms; bare noisy 183 ms; Iceberg 35 ms (local simulators, not device time) |
| Superstaq | compile time, queue, device time | **not measured** |
| Postprocessing | postselection, repair, ranking, `t_vote` | sub-millisecond |
| LM | `t_lm_single`, `t_lm_vote` | 2.4 s all-context; 0.7 s with selection (t5-base) |

Resources per tier (mean over evaluation questions):

| Tier | Qubits | CZ gates | Shots kept |
|---|---|---|---|
| Bare, grid-routed | 10 | 69 | 100% |
| Textbook Iceberg (free all-to-all wiring) | 14 | 112 | 59% |
| Bus Iceberg, nearest neighbour, no SWAPs | 18 (12 code + 6 bus) | 182 | 38% |

Before routing: bare 46 CZ all-to-all; textbook 113, rising to 224 after grid routing (`figures/offline/fig4_two_qubit_gates.png`, `circuits/stats.json`). Simulator timings are not device timings, and compilation improvements alone are not algorithmic speedups.

## 6. Quantifying quantum speedup

For instance size `n` and required solution quality `q`, define

```text
speedup(n, q) = T_classical(n, q) / T_quantum(n, q)
```

A fair comparison uses **the same objective and quality threshold**, including the complete cost of producing a usable selection. At `n = 10`, brute force takes about 0.2 ms; nothing quantum in this project is expected to beat that.

### Time-to-solution (TTS)

If an **independent raw quantum shot** has probability `p_success` of satisfying a pre-registered target, the idealized number of repetitions needed for 99% success is

```text
R_99 = ceil(log(0.01) / log(1 - p_success))
```

`TTS_99 ~= R_99 * t_per_shot` is useful for intuition. Real QPU TTS should include batching, circuit setup, queueing (reported separately), readout, repeated jobs, compilation, and postselection. **Use the raw-shot success probability**, not the rate after discarding flagged shots: for the bus Iceberg tier that means `P(optimum | kept) × 0.38`.

To claim practical quantum speedup, show **actual quantum hardware** completing the task faster than a **well-optimized classical solver** at comparable quality, ideally with favorable scaling. A local simulator beating another software implementation is **not** hardware speedup.

## 7. Scaling sweep

| Paragraph count (`n`) | Candidate selections (`2^n`) | Recommended evaluation | Status |
|---:|---:|---|---|
| 6 | 64 | Exhaustive reference and all solvers | Open |
| 8 | 256 | Exhaustive reference and all solvers | Open |
| 10 | 1,024 | Full HotpotQA application benchmark | **Done (offline)** |
| 12 | 4,096 | Synthetic graphs; classical + compile-only quantum | Open |
| 16 | 65,536 | Strong classical solvers + compile/resource projection | Open |
| 20 | 1,048,576 | Classical optimization and compile/resource projection | Open |
| 24 | 16,777,216 | Classical optimization and **compile-only projection** | Open |

**Hardware ceiling:** the bus patch is 3 rows × (k/2 + 1) columns. On Sqale's 6 × 4 grid that caps executable error-detected runs at **10 paragraphs**. Beyond that, error-detected QAOA is compile-only until a folded layout is built. Bare QAOA could go further but would need routing on the grid.

Sweeps already run at `n = 10` (`results/offline/studies/`):

- **Noise sweep** (30 questions, ideal AR 0.948):

  | CZ error | 0.1% | 0.2% | 0.5% | 1% | 2% |
  |---|---|---|---|---|---|
  | Bare AR | 0.9457 | 0.9433 | 0.9356 | 0.9214 | 0.8934 |
  | Bus Iceberg AR | 0.9475 | 0.9471 | 0.9423 | 0.9279 | 0.8670 |
  | Bus Iceberg shots kept | 82% | 68% | 38% | 17% | 5% |

- **Redundant-edge sweep:** the error-detection gain grows with circuit depth, +0.001 AR below 6 redundant edges and +0.010 above 30 (`figures/offline/fig3_redundant_edges.png`).

For sizes beyond easy enumeration, use branch-and-bound or mixed-integer solvers, not brute force alone. **Label projected quantum costs as projections.** A circuit that compiles has not been shown to run successfully or faster.

## 8. Statistical design

- Held-out evaluation set of 150 questions with stored IDs: **done**.
- Paired comparisons over the same questions, scores and LM configuration: **done**.
- 95% percentile bootstrap over questions, 10,000 resamples, seed 0; Wilson interval for proportions: **done**.
- Multiple independent seeds for stochastic methods (annealing, QAOA sampling, noise): **open**. The offline run uses one seed.
- Timing: report medians and p95, number of runs, hardware, and whether warmup/queue is included. Offline timings are means on one machine: **open** for Sqale.
- Do not read small differences as real without intervals.

**Sample size:** with 150 questions and accuracy near 30–40%, single-lane EM intervals came out at roughly **±7–8 percentage points** (e.g. 0.320 [0.247, 0.393]), as predicted. Paired differences were computed directly and are narrower.

## 9. Results (offline stand-in run)

TF-IDF scores, t5-base reader, local Pauli noise (0.5% per CZ, 0.05% per one-qubit gate, 0.5% readout), 4,000 shots. Full table with F1 and every `@64` variant: `results/offline/results_table.md`.

| Lane | EM single | EM vote | F1 | Both-gold hit | Approx. ratio | Feasibility (pre-repair) | P(opt) / kept shot | Shots kept | Prompt tokens | CZ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1a All context | 0.373 [0.293, 0.453] | 0.373 | 0.478 | 1.000 | — | — | — | — | 1425 | — |
| 1b Top-3 | 0.313 [0.240, 0.387] | 0.313 | 0.434 | — | — | — | — | — | 363 | — |
| 2a Exact QUBO | 0.320 [0.247, 0.393] | 0.340 | 0.465 | 0.473 | 1* | — | — | — | 335 | — |
| 2b Greedy | 0.320 | 0.327 | 0.458 | 0.473 | — | — | — | — | 330 | — |
| 2c Annealing | 0.320 | 0.333 | 0.465 | — | 0.999 | — | — | 100% | 335 | — |
| rand Uniform random | 0.320 | 0.340 | 0.465 | 0.473 | 0.727 | 6% | 0.09% | 100% | 335 | — |
| 3a Ideal QAOA, p=1 | 0.320 | 0.340 | 0.465 | 0.473 | 0.947 | 76% | 1.8% | 100% | 335 | — |
| 3b Bare, noisy | 0.320 | 0.340 | 0.465 | 0.473 | 0.934 | 62% | 1.5% | 100% | 335 | 69 |
| 3c-direct Textbook Iceberg | 0.320 | 0.340 | 0.465 | — | 0.931 | 69% | 1.6% | 59% | 335 | 112 |
| 3c Bus Iceberg | 0.320 | 0.340 | 0.465 | 0.473 | 0.941 | 72% | 1.7% | 38% | 335 | 182 |

`*` Exact QUBO is optimal for the **formulated objective**; that does not mean it selects the true supporting paragraphs (both-gold hit 47%) or gives the best final answer.

Fault budget, every single CZ fault scored exactly, mean of 12 evaluation questions (`figures/offline/fig6_fault_budget.png`):

| | CZ gates | Faults that slip through | AR lost per unit CZ error | Phase (Z) faults that slip through |
|---|---:|---:|---:|---:|
| Bare | 70 | 97.9% | 1.94 | 95.2% |
| Textbook Iceberg | 113 | 14.8% | 1.82 | 0% |
| Bus Iceberg | 183 | 3.9% | 0.29 | 0% |

### Main run: the plan's models (`configs/main.yaml`, `results/main/`)

Qwen 2.5 1.5B Instruct (RTX 4070, float16, temperature 0), bge-small scores, calibration
hop 0 / tau 0.6075 / theta 0.25, local Pauli noise as above. The evaluation stage took
567 s.

| Lane | EM single | EM vote | Gold recall | Prompt tokens | AR | Shots kept |
|---|---:|---:|---:|---:|---:|---:|
| 1a All context | 0.400 [0.320, 0.480] | 0.400 | 1.000 | 1455 | — | — |
| 1b Top-3 | 0.393 [0.320, 0.473] | 0.387 | 0.823 | 442 | — | — |
| 2a Exact QUBO (and 2b, 2c, rand, 3a, 3b, 3c, 4: same EM single) | 0.367 [0.293, 0.447] | 0.373 | 0.773 | 428 | — | — |
| rand Uniform random | | | | | 0.616 | 100% |
| 3a Ideal QAOA, p=1 / p=2 | | | | | 0.869 / 0.936 | 100% |
| 3b Bare, noisy | | | | | 0.863 | 100% |
| 3c Bus Iceberg, noisy | | | | | 0.865 | 51% |

Verdicts are the same as offline: H1 not supported (−0.033 [−0.107, +0.040]), H2
supported, H3 supported (+0.0026 [+0.002, +0.003]), H4 supported, H5 not supported
(+0.007 [−0.013, +0.033]). Error detection recovers 42% of the approximation ratio lost
to noise (offline 58%), with 51% of shots kept (offline 38%). Both changes follow from
shallower circuits: 5.0 redundant edges per question on average, against 22.5 offline.

### Sqale compiler check (`scripts/sqale_compile.py`, `results/offline/sqale_compile/`)

`cq_compile` for `cq_sqale_qpu`. The first call sent 2 circuits and cost 0.5 credits; a
second call sent 36 circuits (12 questions × 3 encodings) and also cost 0.5. Billing is
per call, not per circuit. Credits: 20 → 18.0 in total, including the 0.5 probe and one
failed call that returned HTTP 500 and was still charged.

| Mean of 12 questions (offline-config QUBOs) | CZ in (RZZ = 2) → Sqale | Depth in → Sqale | Qubits | Exact check of the compiled circuit |
|---|---|---|---|---|
| Bare QAOA | 45.7 → 81.0 (×1.8) | 15 → 202 | 10 → 24 | output TVD to original ≤ 8e-14 |
| Bus Iceberg | 182.8 → 181.8 (×0.99) | 142 → 427 | 18 → 24 | 100% of shots kept without noise, 12 of 12; logical TVD ≤ 8e-14 |
| Textbook Iceberg | 112.7 → 239.1 (×2.1) | 56 → 507 | 14 → 24 | 100% kept, 12 of 12 |

So the bus mapping is native on Sqale's grid: the compiler leaves its CZ count unchanged.
After real compilation it is the cheapest protected circuit, 182 vs 239 CZ, although it
was the more expensive one before (183 vs 113). That confirms the local grid estimate in
figure 4 with the vendor's own compiler. An earlier version of this table counted each
RZZ as one gate; the review caught it.

Sqale's native gate set is CZ, RZ, S and global rotations (GR). The checker decomposes
GR and drops the 6 spectator qubits before simulating.

### Figures

1. EM by lane with paired CIs, single and vote: `figures/offline/fig1_accuracy_by_lane.png` — done.
2. Noise → solver quality → accuracy: `fig2_noise_to_accuracy.png` — done. A selection-quality vs latency chart on real device timings is **open**.
3. Error-detection trade-off (AR and kept share vs CZ error): `fig7_noise_sweep.png`, `fig7b_noise_sweep_phase_errors.png` — done.
4. Compiled resources, bare vs protected, before and after routing: `fig4_two_qubit_gates.png` — done locally; pre/post Superstaq **open**.
5. Scaling plot of classical vs quantum TTS: **open**. `fig5_roadmap.png` is a projection, labelled as such.
6. Physics figures (Rydberg scaling, Stark map, pair potential, Grotrian): `figures/physics/` — done.

## 10. Machine-readable logging

The offline run writes:

- `results/<run>/answers_<lane>.jsonl`: one record per question with selections, answers, tokens and timings.
- `results/<run>/per_question.json`: per-question metrics for every lane.
- `results/<run>/summary.json`: lane means with bootstrap intervals, plus hypothesis verdicts.
- `results/<run>/raw/<lane>/<qid>.json.gz`: raw counts, so analyses replay without re-sampling.
- `results/<run>/config_used.json`, `angles.json`, `routes.json`, `studies/*.json`.

Target schema for Sqale and QPU runs, one record per question × lane × seed × backend mode. The fields not logged offline (queue time, device time, compile time) are the additions:

```json
{
  "question_id": "example_001",
  "seed": 0,
  "selector": "qaoa_iceberg",
  "backend": "cq_sqale_simulator",
  "backend_mode": "noise_sim",
  "n_paragraphs": 10,
  "qaoa_depth_p": 1,
  "shots_raw": 4000,
  "shots_accepted": null,
  "physical_qubits": 18,
  "logical_qubits": 10,
  "ancilla_qubits": 6,
  "compiled_depth": null,
  "two_qubit_gate_count": null,
  "selected_paragraph_ids": [],
  "is_feasible": null,
  "approx_ratio": null,
  "optimum_hit": null,
  "both_gold_selected": null,
  "answer_exact_match": null,
  "answer_f1": null,
  "prompt_tokens_total": null,
  "compile_time_ms": null,
  "queue_time_ms": null,
  "execution_time_ms": null,
  "postprocessing_time_ms": null,
  "llm_time_ms": null,
  "total_time_ms": null
}
```

Null means **not yet measured**; do not substitute plausible-looking numbers. Qubit fields must reflect the actual compiled circuit, including ancillas (18 for the bus mapping at 10 paragraphs, 14 for textbook).

## 11. Execution checklist

### Phase A — Baseline and sanity checks

- [x] Freeze 150 HotpotQA distractor questions, normalization, and prompt templates.
- [x] Export relevance and redundancy scores once for all lanes (`scores/`).
- [x] Implement and validate the QUBO against enumeration.
- [x] Run all-context, top-k, greedy, exact, and simulated-annealing selectors.
- [x] Add a uniform-random control lane.
- [x] Add gold-evidence oracle baseline (lane `oracle`: 0.553 main, 0.460 offline).
- [x] Verify benchmark metrics and logging schema.

### Phase B — Quantum correctness

- [x] Tune ideal QAOA angles per question (calibration parameters on separate questions).
- [x] Confirm small QAOA statevectors against a reference calculation.
- [x] Validate Iceberg encode/execute/check/decode in the zero-noise limit.
- [x] Enumerate every single fault; confirm detection and that each flag is needed.
- [x] Independent review of the encoder; fix and regression-test the gaps found.
- [x] Compare raw samples, repaired samples, and logical outputs.

### Phase C — Superstaq/Sqale

- [ ] Run `scripts/check_superstaq.py --budget`: noiseless Sqale kept share must be 100%. Read the per-step credit cost in `results/superstaq_check.json` and set `limit` for lanes 3d-b/3d-c so the run fits the 20-credit budget.
- [x] `cq_compile` the bus circuit. One evaluation question: 155 → 162 CZ, no re-routing. The compiled circuit keeps 100% of shots without noise (exact local simulation). More questions: `scripts/sqale_compile.py --n 12 --styles bare,bus,direct`, 0.5 credits per call.
- [ ] Run 3b and 3c on `cq_sqale_simulator` noise-sim; compare kept share to `results/offline/studies/noise_sweep.json` to fit the local model's CZ error.
- [ ] Record accepted and discarded shots, compile, queue and device time.
- [ ] Use `cq_sqale_qpu` only if access is granted; label hardware results unmistakably.

### Phase D — Application evaluation

- [x] Run all lanes against the same frozen question set (offline).
- [ ] Repeat stochastic runs at several logged seeds.
- [x] Evaluate single-selection and voting modes fairly.
- [x] Calculate EM/F1, evidence recall, approximation ratio, feasibility, kept share, tokens and latency.
- [x] Equal-accepted-shot comparison of 3b vs 3c (§5C).
- [x] Repeat with the instruct model (`Qwen/Qwen2.5-1.5B-Instruct`) and dense embeddings (`configs/main.yaml`), after recalibrating. Results in §9, "Main run". Optional: 0.5B as a cost comparison.
- [x] Generate bootstrap intervals and figures.

### Phase E — Scaling and reporting

- [x] Noise sweep and redundant-edge sweep at `n = 10`.
- [ ] Size sweep `n = 6, 8, 12, 16` on synthetic graphs.
- [ ] Compare against strong classical solvers (branch-and-bound / MIP) at larger `n`.
- [ ] Folded bus layout for more than 10 paragraphs.
- [x] Report simulator results separately from projections.
- [x] Publish configs, environment lock, logs, scripts, and one-command reproduction (`reproduce.py`, CI).
- [x] State which hypotheses are supported, not supported, or contradicted.

## 12. Interpretation guide

| Observed result | Appropriate claim | Claim **not** justified |
|---|---|---|
| QAOA returns valid selections | Working quantum optimization backend | Quantum speedup |
| Quantum beats greedy but not exact | Advantage over that particular heuristic | Advantage over classical computing generally |
| Error detection raises quality among accepted shots | Improved postselected distribution | More useful solutions per second |
| Error detection raises useful-shot rate at fixed raw budget | Better effective sampling under tested conditions | General fault tolerance or quantum speedup |
| LM F1 improves with quantum-selected context | Application-level gain in tested setting | Unique quantum mechanism as the cause |
| Superstaq reduces 2-qubit gates or depth | Compilation-resource improvement | Faster end-to-end inference without timing it |
| Actual QPU beats strong classical solver at equal quality | Practical speed advantage on those instances | Asymptotic speedup without scaling analysis |

### What this run actually supports

- **Working backend:** yes, locally. Ideal QAOA at p = 2 puts the exact optimum on top for 150 of 150 questions.
- **Better postselected distribution under local Pauli noise:** yes. +0.0074 AR, 58% of the noise-induced loss recovered, phase faults on CZ always caught.
- **More useful solutions per raw shot:** **no**. The bare circuit finds the optimum more often per raw shot at every noise level tested. Error detection buys cleaner samples, not more of them.
- **Application gain:** **no** accuracy gain over reading everything; a 4× token reduction that every classical selector also gets.

### Lessons the plan did not anticipate

1. **Downstream accuracy cannot separate samplers at this size.** 4,000 shots over 1,024 selections, followed by repair and ranking, find the optimum regardless of sampler: the uniform random control does so on 150 of 150 questions (91% at 64 shots, vs 84% for ideal QAOA, whose samples are less diverse). Solver quality must be judged **before repair** (approximation ratio, feasibility, P(optimum)), which is why H3 uses AR and why the `rand` lane exists.
2. **Error detection pays only in a window**, roughly 0.1–1% CZ error for 10 paragraphs at p = 1. Below it there is little to fix; above it most shots are discarded and two-fault events dominate (at 2% CZ error the bus Iceberg tier keeps 5% of shots and falls below bare). `src/qsel/router.py` encodes this as a floor on predicted kept share.
3. **The textbook Iceberg mapping is the wrong one for QUBOs with linear terms** on a grid: it needs SWAP chains to the hub qubits and has the 9-bit-flip failure mode. Even with free all-to-all wiring it ends below the bare circuit (0.931 vs 0.934).
4. **Paragraph similarity is the wrong redundancy measure for multi-hop QA** (see §2).

## 13. Final project conclusion structure

1. **Does it work?** Yes, locally: correctness suite, exhaustive single-fault analysis, independent review, end-to-end demo (`python -m qctx.demo --config configs/main.yaml`; `data/demo_main.jsonl` index 0 shows selection helping, index 2 shows it dropping a gold paragraph). Sqale: compiler only (`cq_compile` preserves the checks); job execution refused for our account.
2. **Does it help NLP?** Not on accuracy: −0.053 EM vs all context, interval includes zero. It cuts prompt tokens 4× and reader time from 2.4 s to 0.7 s, but so does every classical selector.
3. **Does it speed things up?** No. Brute force solves the 10-paragraph QUBO in about 0.2 ms. **No quantum speedup was demonstrated or claimed.**

### One-sentence pitch

> We built a grid-native Iceberg error-detection mapping for QAOA on Sqale's nearest-neighbour layout, verified it against every single fault, and showed under a local noise model that it recovers most of the solver quality noise destroys, while measuring honestly that, in a 10-paragraph LM context-selection task, neither QAOA nor error detection changes the final answers.

---

## References and scope

- **Primary design source:** *Error-Detected Quantum Context Selection — MidQWest Challenge 2 Plan* (November 21, 2026), `docs/challenge2-plan.pdf`.
- **Challenge requirements:** *Infleqtion Challenge: Qubits & Gradients*, Challenge 2; [Infleqtion MidQWest 2026 repository](https://github.com/Infleqtion/MidQWest-2026).
- Self, Benedetti, Amaro, "Protecting expressive circuits with a quantum error detection code", *Nature Physics* (2024): the Iceberg code.
- He, Amaro, Shaydulin, Pistoia, "Performance of Quantum Approximate Optimization with Quantum Error Detection" (2024): Iceberg QAOA on trapped ions.
- Jin et al., "Iceberg Beyond the Tip: Co-Compilation of a Quantum Error Detection Code and a Quantum Algorithm" (2025).
- Bedalov et al. (Infleqtion), "Fault-Tolerant Operation and Materials Science with Neutral Atom Logical Qubits" (2024).
- Yang et al., "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering" (2018), CC BY-SA 4.0; [hotpotqa.github.io](https://hotpotqa.github.io/).
- Sibalic et al., "ARC: An open-source library for calculating properties of alkali Rydberg atoms".
- [Superstaq documentation](https://superstaq.readthedocs.io/); [Sqale optimizations](https://superstaq.readthedocs.io/en/latest/optimizations/sqale/sqale.html).

**Scope note:** Every number here comes from the offline stand-in stack (TF-IDF, t5-base, uniform local Pauli noise with assumed rates). The local noise model is not Sqale, and it gives the textbook encoding free all-to-all wiring, which flatters it. Nothing here establishes Sqale or QPU behavior, hardware access, or any speed advantage.
