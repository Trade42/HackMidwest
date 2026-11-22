<img width="2172" height="724" alt="qubits-and-gradients" src="assets/qubits-and-gradients.png" />

# Infleqtion Challenge: Qubits & Gradients

Quantum computation and machine learning are two rapidly emerging fields that have largely evolved in parallel, but their respective capabilities also create opportunities for them to advance one another. Quantum computers have the potential to offer more efficient modeling of contextual relationships. Meanwhile, new machine learning techniques can help address current challenges in quantum computing by bolstering quantum error correction (QEC) decoders.

For this hackathon, choose one of two challenges exploring the intersection of these two fields. We anticipate the first challenge to require some prerequisite knowledge of QEC, whereas the second challenge is more beginner-friendly.

---

# Challenge 1: ML-Assisted QEC Decoding

## Challenge

QEC decoders must identify and correct errors quickly. They must also stay accurate. Powerful decoders can be computationally expensive. Simpler decoders may be faster but less accurate.

For this challenge, teams will work with the Gross code, a quantum low-density parity-check (qLDPC) code in the bivariate bicycle family. Its parameters are `[[144, 12, 12]]` — 12 logical qubits in 144 data qubits, with a code distance of 12.

This challenge focuses on using machine learning to improve decoding, especially in the presence of qubit loss in addition to conventional Pauli errors. These error mechanisms are relevant to neutral-atom hardware where information about loss may be available to the decoder.

## Task

Build an ML-assisted decoder that improves an existing QEC decoder. Start with the BP-OSD and relay-BP decoders as baselines and examine whether a machine-learning approach can improve its accuracy, speed, or ability to handle loss.

**We have provided a possible scaffolding for this decoder comparison [here](challenge-1-ml-decoder-kit.zip). Download the zip file and follow its setup instructions before implementing your decoder.**

One approach is hierarchical decoding: use a fast first-stage decoder for simpler error patterns and pass harder cases to a stronger decoder. Machine learning could help identify likely errors, make use of loss information, or decide when a result should be passed to the next stage.

The goal is to see whether your decoder can outperform the baseline, particularly when loss information is available.

## Requirements

- Set up the code and baseline decoders. Download the [decoder kit](challenge-1-ml-decoder-kit.zip) and run the BP-OSD and relay-BP decoders, then reproduce their baseline performance before making changes.
- Simulate memory experiments with noise. Use heralding information, such as loss-detection flags, where available.
- Build your ML-assisted decoder. Use syndrome information and, where applicable, loss information. You can build a standalone decoder or add a learned component to an existing decoding pipeline.
- Compare your decoder against the baseline approaches under the same noise conditions.

## Possible Approaches

Some possible directions:

- **[Learned predecoders](https://github.com/NVIDIA/ising-decoding):** Train a model to recognize common error patterns and resolve them before passing the remaining syndrome to a stronger decoder.
- **Confidence-based routing:** Predict whether a decoding result is reliable enough to accept or should be sent to a stronger decoder.
- **Belief propagation:** Use BP as a partial decoder, resolving confident corrections before passing the remaining syndrome onward.
- **CNNs or graph neural networks:** Use neural networks to identify patterns in syndrome data, potentially using the structure of the code or its Tanner graph.
- **Leakage tracking:** Use temporal models, such as hidden Markov models, to identify leakage events and inform decoding.

> **Note:** You do not need to implement a hierarchical decoder. These are possible ways to approach the problem, not a required set of methods.

## Experiments and Metrics

Compare your decoder with the BP-OSD and relay-BP baselines using the same test data and noise conditions. While more experiments are encouraged for a more thorough analysis, we *require* one for this challenge:

- A sweep of your decoder across the physical error rates `[0.003, 0.004, 0.005]`.

Required metrics to report include logical error rate, decoding latency*, and decoder resource requirements. In your evaluation, discuss where your approach works well and where it falls short.

\*This is where hierarchical decoding approaches really shine!

## Deliverables

- A working decoder, the code used to generate training and test data, and the script for reproducing evaluation results on GitHub.
- A comparison of your decoder to the baselines, including at least one plot of logical error rate or decoding time.
- A short presentation or demo explaining the approach and results.

## Stretch Goals

- Train a router to decide when to accept a first-stage result or invoke a stronger decoder.
- Test the approach on a second QEC code or code family.
- Run the decoder on sliding syndrome windows.
- Study how performance changes with code distance.
- Estimate the hardware cost of the decoder, such as GPU resource usage or latency.

## Recommended Tools

A few open-source tools may be useful for this Challenge:

- **[qLDPC](https://github.com/qLDPCOrg/qLDPC):** A library for working with quantum error-correcting codes. It can be used for code construction and related QEC workflows.
- **[clifft](https://github.com/unitaryfoundation/clifft/releases/tag/v0.13.0):** A non-Clifford simulation tool extended from [Stim](https://github.com/quantumlib/stim) for quantum error correction and decoding workflows that support leakage, loss, etc.

---

# Challenge 2: Give Your LLM a Quantum Backend

## Challenge

Language models use classical machine-learning methods for tasks like processing language, representing context, and handling memory and reasoning. This challenge asks whether quantum computing could play a useful role in any part of this process.

## Task

Develop a pipeline that integrates quantum computing or information processing into a sequence or language modeling task. This requires you to identify stages within a classical machine learning workflow that could benefit from quantum information processing, and develop a quantum backend that can execute those tasks.

For example, you could explore how quantum contextuality could be used to represent relationships between tokens, encode word sequences with quantum states and quantum circuits, or formulate an ML subproblem as a QUBO or related optimization problem to generate candidate solutions. However, don't feel required to tie your project to any of these.

We're keeping this challenge deliberately open-ended, so feel free to think outside the box, get creative, and surprise us!

## Requirements

- Use an existing or custom language model for a language-modeling or sequence-processing task.
- Identify a part of the workflow where quantum computing or quantum information processing could be useful.
- Implement a backend that sends this part of the computation to a quantum simulator or hardware and returns results to the application.
- Use at least one quantum algorithm or information-processing technique.
- Demonstrate the full workflow, from input to final output.

## Experiments and Metrics

Evaluate your application using metrics that make sense for the task. These must include accuracy, execution time, and resource requirements. Additional metrics can include solution quality/sampling diversity, circuit depth, and/or qubit count.

## Deliverables

- Your working application that connects a language or sequence model to a quantum backend, with the workflow uploaded to GitHub.
- An evaluation using appropriate metrics.
- A short presentation or demo highlighting your approach and results.

## Stretch Goals

- Use Superstaq to optimize circuits for execution on available hardware.
- Support multiple quantum hardware platforms or simulators.
- Route tasks between classical and quantum backends based on problem characteristics or estimated execution costs.
- Build a workflow where quantum outputs influence later language-model processing.
- Compare the quantum approach with a classical baseline.

---

# Shared Expectations

- Be prepared to explain why you chose your approach and what role quantum computing or machine learning plays in it.
- Results should be clear and reproducible, and the corresponding GitHub codebase should follow best coding practices.
- You can use existing libraries and tools rather than implementing everything from scratch.
- AI use is encouraged — but be prepared to explain your implementation!

---

# This repository: Challenge 1 implementation and technical review

The challenge text above is preserved from [Infleqtion's MidQWest 2026 challenge repository](https://github.com/Infleqtion/MidQWest-2026). The work below addresses **Challenge 1**. Challenge 2 remains part of the original problem statement; this repository does not contain a quantum backend for a language model. The [decoder kit README](challenge-1-ml-decoder-kit/README.md) documents the supplied baseline experiment, and the [evaluation guide](challenge-1-ml-decoder-kit/evaluation/README.md) gives the detailed commands and artifact formats for our additions.

## Research question and approach

The practical question is whether a small learned decision can save decoding time without giving up the logical accuracy of the supplied Relay-BP decoder on the `[[144,12,12]]` Gross code. The implemented candidate runs Relay-BP until its **first solution**, then uses a logistic model to decide whether a hard shot needs the kit's full five-solution Relay-BP run. A converged solution from the first 80-iteration leg is accepted directly. This is a routing policy: Relay-BP produces every candidate correction. The training target is whether the first candidate gets **any** of the 12 logical observables wrong, rather than whether its physical error vector matches the sampled fault.

The code also implements an algebraically correct predecoder interface and validation procedure for committing high-confidence fault mechanisms before routing. That interface can use an eight-iteration memory-BP control or a separately supplied CNN. **No trained CNN, loss-aware decoder, herald schema, or leakage tracker is integrated on `main`.** Consequently, the currently implemented router evaluates the no-loss Stim memory experiment; it does not yet address the challenge's loss-specific objective.

## Literature and design decisions

| Source | Finding relevant here | What this repository does with it |
| --- | --- | --- |
| [Bravyi *et al.*, *Nature* 627, 778–782 (2024)](https://doi.org/10.1038/s41586-024-07107-7) | Bivariate bicycle codes include the Gross `[[144,12,12]]` memory code. Its sparse checks and repeated syndrome circuit make decoding both a space and a time problem. | The kit's [X and Z memory circuits](challenge-1-ml-decoder-kit/circuits/README.md) use 144 data and 144 check qubits over 12 noisy cycles. We retain those circuits and the required physical-noise sweep. |
| [Roffe *et al.*, *Physical Review Research* 2, 043423 (2020)](https://doi.org/10.1103/PhysRevResearch.2.043423) | Belief propagation can stall on quantum LDPC structure; ordered-statistics postprocessing supplies a stronger, more expensive recovery step. | The supplied BP-OSD combination-sweep order-10 decoder is an accuracy and latency reference. We do not describe it as a learned component. |
| [Müller *et al.*, “Improved belief propagation is sufficient for real-time decoding of quantum memory” (2025)](https://arxiv.org/abs/2506.01779) and its [code/data](https://github.com/trmue/relay) | Relay-BP uses memory in message passing and varied memory strengths across legs to escape difficult BP states and search for multiple valid corrections. | The supplied five-solution settings are kept as the strong reference. Our router exposes the first solution, pays for a full rerun only when its frozen risk rule escalates, and counts that rerun in latency. |
| [Gidney, “Stim: a fast stabilizer circuit simulator” (2021)](https://arxiv.org/abs/2103.02202) | Stabilizer-circuit sampling and detector error models make large, repeatable QEC experiments feasible. | Stim produces the saved circuit shots and the error-mechanism matrices. Our optional independent-mechanism DEM samples are labeled training/diagnostic data and are never substituted for the held-out circuit-shot test. |
| [NVIDIA Ising Decoding](https://github.com/NVIDIA/Ising-Decoding) and its [predecoding documentation](https://nvidia.github.io/cudaq-qec/components/qec/ai_predecoding.html) | A learned local correction can simplify a residual syndrome before a global decoder, provided its logical-frame contribution is restored and total pipeline time is measured. | Our predecoder interface implements that residual and logical-frame composition for Gross-code mechanisms. NVIDIA's trained models and reported gains are **not** results for this code or this repository. |
| [Liu *et al.*, “Achieving Optimal-Distance Atom-Loss Correction via Pauli Envelope” (2026)](https://arxiv.org/abs/2603.04156) and [Clifft's loss simulator](https://github.com/unitaryfoundation/clifft) | An atom-loss flag does not uniquely reveal when loss occurred or which errors propagated. A loss-aware experiment needs a physical loss model, observable heralds, and a decoder map for possible effects. | Loss and herald work is being explored on a separate branch. The current saved-shot loader has no herald fields, and the no-loss router results cannot be interpreted as loss performance. |

These sources motivate different pieces of the pipeline; none supplies a transferable performance claim for the combined candidate. The learned router and its validation rule are our implementation choices, evaluated against the supplied decoders on this kit's shots.

## Exact decoding problem

The kit's Gross code uses the bivariate-bicycle polynomials `A = x^3 + y + y^2` and `B = y^3 + x + x^2` on a `12 × 6` torus. An X-memory shot starts in `|+⟩`, and a Z-memory shot starts in `|0⟩`; each has 12 noisy syndrome cycles and a final data measurement. The circuits have 1,728 detector bits and 12 logical-observable bits per shot. The supplied baselines retain 936 check detectors of the memory basis. The other 792 detectors are available in the full circuit record but are not used by these baseline and router decoders.

For each basis, `circuit.dem_matrices` merges duplicate detector-error-model effects and constructs a binary detector matrix `H` of shape `936 × 8,784`, an observable matrix `L` of shape `12 × 8,784`, and one prior probability per error mechanism. A decoder proposes `e ∈ {0,1}^8784` for syndrome `s ∈ {0,1}^936`; the syndrome equation is `H e = s (mod 2)`, and the predicted logical flips are `L e (mod 2)`. A returned correction is scored against all 12 true observable bits even when an iterative decoder does not converge. The 8,784 columns represent **circuit error mechanisms**, not 8,784 data qubits or a 144-bit physical correction.

Shot `k` of X memory and shot `k` of Z memory are separate samples paired for the kit's score. A basis shot fails if any observable prediction is wrong; a pair fails if either basis fails. For `N` pairs, let `P = failures_pair/N`. The reported logical rate per cycle is `1 − (1 − P)^(1/12)`. This convention matters: a per-basis failure fraction, a per-pair failure fraction, and a per-cycle logical rate are different quantities. A change in fault-vector agreement or syndrome weight is not itself a logical gain.

## Baselines and controls

The supplied [`decoders.py`](challenge-1-ml-decoder-kit/decoders.py) defines BP-OSD with up to 80 minimum-sum BP iterations and `OSD_CS` order 10. Relay-BP uses an 80-iteration first leg with memory strength `0.15`, up to 300 further legs of at most 60 iterations with sampled memory strengths, and stops after five converged solutions. The supplied [`baseline.py`](challenge-1-ml-decoder-kit/baseline.py) runs both decoders on fresh sampled shots. Its [`baselines.log`](challenge-1-ml-decoder-kit/baselines.log) is a useful aggregate reproduction of the official kit procedure at `p = 0.003, 0.004, 0.005`, but contains rounded summaries rather than per-shot predictions and timings.

The saved-shot evaluator adds two controls: `relay-first-solution` stops at one solution, and `relay-first-leg-or-full` accepts a converged first leg but otherwise reruns full Relay-BP. These isolate the value of the first solution and of a simple nonlearned routing rule. The full Relay-BP row remains the accuracy reference for the learned router. A separate [`tcbp_jobs_baseline.py`](challenge-1-ml-decoder-kit/tcbp_jobs_baseline.py) probes NVIDIA TCBP-JOBS; it is a standalone baseline experiment, outside the eleven-row comparison catalog and the current paired router report.

## Learned first-solution router

The pipeline is implemented in [`router_features.py`](challenge-1-ml-decoder-kit/evaluation/router_features.py), [`router_train.py`](challenge-1-ml-decoder-kit/evaluation/router_train.py), and [`router.py`](challenge-1-ml-decoder-kit/evaluation/router.py):

1. On each training or validation shot, decode with Relay-BP configured to stop after its first solution. Save the candidate's 12 observable bits and label it wrong if **any** bit differs from truth. Truth is used for labeling and scoring, never passed to the deployed decoder.
2. Call a shot *easy* only if its first solution converged within the initial 80 iterations. Train the risk model on the remaining *hard* shots. Its 17 inputs are first-solution iterations, convergence, mean absolute posterior log-odds clipped at 100, correction weight, and syndrome weight in each of 13 time slices. The fitted model standardizes features and uses L2-regularized logistic loss to estimate `P(first candidate wrong | features)`.
3. Choose a separate threshold for each basis and physical error rate using validation shots only. The selected threshold is the largest one for which at most `MAX_MISSED_WRONG` incorrect **hard** candidates would be accepted. The current frozen models use a budget of zero. A hard candidate is escalated when its risk is **greater** than the threshold; easy candidates bypass the model. Zero observed validation misses is a count on that finite validation set, not a bound of zero test error.
4. At inference, rerun the kit's full five-solution Relay-BP for escalated shots; otherwise return the first candidate. The rerun repeats work already done to obtain the first solution. Both decoder calls and router work belong in the candidate's latency.

The six committed frozen model files cover X and Z at all three required rates. Their validation reports show how restrictive the zero-miss rule is:

| `p` | Basis | Hard validation shots / 5,000 | Wrong hard candidates | Hard shots escalated | Wrong hard candidates accepted |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.003 | X | 163 | 1 | 131 | 0 |
| 0.003 | Z | 165 | 6 | 139 | 0 |
| 0.004 | X | 691 | 36 | 683 | 0 |
| 0.004 | Z | 680 | 29 | 637 | 0 |
| 0.005 | X | 1,799 | 248 | 1,765 | 0 |
| 0.005 | Z | 1,812 | 240 | 1,807 | 0 |

These are **validation candidate** counts, not held-out logical-error or latency gains. In particular, one wrong hard X candidate at `p = 0.003` gives little information about rare test errors. The frozen JSON records model coefficients, feature schema, threshold, basis, `p`, source dataset hashes, an exact byte fingerprint, and a portable semantic fingerprint of the ordered binary `H` and `L` matrices and priors rounded to 14 decimals. The loader checks the semantic fingerprint when present, allowing the same mathematical problem across Mac and Linux builds; the exact hash remains provenance. A failed identity check still needs investigation rather than a manual hash edit.

## Predecoder interface and alternatives under study

[`predecoder_layout.py`](challenge-1-ml-decoder-kit/evaluation/predecoder_layout.py) checks a Gross-code grid representation: 936 basis detectors become `(13,12,6)`; 8,784 mechanism columns become `(122,12,6)` through their verified translation orbits. It verifies all 72 spatial translations and round trips. Spatial padding may wrap the `12 × 6` torus; time does not wrap. This provides a documented input and output contract for a future CNN but contains no trained CNN weights.

For a first-stage binary mechanism commit `c`, [`predecoder.py`](challenge-1-ml-decoder-kit/evaluation/predecoder.py) passes the residual `s ⊕ Hc` downstream and restores the logical frame as `Lc ⊕ prediction(residual)`. This identity is the reason a predecoder can alter the syndrome without silently changing the requested logical answer. The implemented BP control uses memory BP with **at most** eight iterations and commits mechanisms only when the posterior log-odds of fault *absence* are sufficiently negative. The CNN contract accepts one probability or logit per mechanism, not one per data qubit, and requires a frozen model/threshold manifest. [`predecoder_eval.py`](challenge-1-ml-decoder-kit/evaluation/predecoder_eval.py) selects that threshold on validation shots and defines a separate 5,000-shot X-memory `p = 0.004` go/no-go screen before a full sweep. The CNN model and its training are still external to `main`, so rows requiring it are interfaces, not measured decoder results.

The [eleven-row catalog](challenge-1-ml-decoder-kit/evaluation/comparison_rows.py) separates baseline, router, loss, and predecoder comparisons. Rows 1–4 have evaluator implementations; row 5 requires frozen router models; rows 6–8 await integrated loss/herald/tracker inputs; rows 9–11 require validation-frozen BP or CNN commits and their downstream models. Exploratory [failure analysis](https://github.com/Trade42/HackMidwest/tree/gates/challenge-1-ml-decoder-kit/analysis), [loss simulation](https://github.com/Trade42/HackMidwest/tree/loss-gates/challenge-1-ml-decoder-kit/loss), and an [ML-first logical decoder](https://github.com/Trade42/HackMidwest/tree/MLDecoder/challenge-1-ml-decoder-kit/ml_decoder) live on separate branches. Their branch reports describe exploratory conditions; they are not part of the main-branch paired gain result. The ML-first branch predicts logical bits directly, which is a different model target from the mechanism-commit CNN interface above.

## Evaluation design and interpretation

[`reproduce_eval.sh`](challenge-1-ml-decoder-kit/reproduce_eval.sh) organizes dataset generation, baseline/control evaluation, router-feature extraction, training, and plots as separate stages. For every `p`, the saved Stim datasets have 20,000 training, 5,000 validation, and 10,000 final-test shots **per basis**. X-memory at `p = 0.004` also has a separate 5,000-shot go/no-go set. The script assigns distinct seeds to these splits, while the dataset loader records file hashes and rejects split reuse. Reusing tracked NPZ files across machines preserves the exact test shots; using the same Stim seed with a different build or platform does not by itself guarantee identical files. The detailed commands and schemas are in the [evaluation guide](challenge-1-ml-decoder-kit/evaluation/README.md).

[`evaluation.evaluate`](challenge-1-ml-decoder-kit/evaluation/evaluate.py) processes saved shots in file order with one process, scores all 12 prediction bits, and writes `summary.json`, `summary.csv`, per-shot NPZ arrays, and per-shot diagnostic JSON. The kit-compatible timer covers the baseline decoder call. The separate `seconds_total_X/Z/pair` arrays cover the full per-shot path, including observable projection, router inference, optional predecoder work, and fallback reruns. Model/decoder construction, training, and batch throughput are recorded separately. Mean, median, and p99 pair latency, convergence/fallback statistics, package versions, machine settings, and parent-process peak RSS are recorded; parent RSS is not total worker or accelerator memory.

[`evaluation.compare_gain`](challenge-1-ml-decoder-kit/evaluation/compare_gain.py) computes a paired report only after it verifies matching X/Z test-file hashes, circuit identity, physical rate, environment metadata, predictions, failure masks, and pair totals. For either the per-cycle rate or total pair latency, **percent reduction** is `100 × (reference − candidate) / reference`; a negative value means the candidate is worse. It reports candidate-only and reference-only failures, an exact test on discordant pairs, and paired bootstrap intervals. At low failure counts, the logical-gain interval may be unavailable or wide. Even identical saved shots cannot remove machine-load drift from separately timed runs.

The printed `baselines.log` cannot serve as the reference input to that paired report: it uses other sampled shots and has no per-shot outcomes. It remains a valid record of the kit's aggregate baseline reproduction. A claim of percent gain requires complete saved-shot baseline **and** candidate artifacts for the same `p`, both bases, and the same test files. The committed paired reports currently cover two rates on different machines:

| Site and rate | Relay-BP pair failures | Router pair failures | Observed logical-rate reduction | Mean total pair-latency reduction | p99 total pair-latency reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| [ICRN, `p = 0.003`](challenge-1-ml-decoder-kit/results/icrn/gain_p0.003_portable/comparison.json) | 5 / 10,000 | 6 / 10,000 | −20.01% | 44.92% | −25.61% |
| [Mac, `p = 0.004`](challenge-1-ml-decoder-kit/results/mac/gain_p0.004/comparison.json) | 84 / 10,000 | 84 / 10,000 | 0% | 5.47% | −20.22% |

Both are held-out **paired saved-shot** comparisons within their respective sites. The logical-rate bootstrap intervals are wide: `[-200.03%, 40.01%]` at `p = 0.003` and `[-22.34%, 18.25%]` at `p = 0.004`; neither supports a confident logical improvement claim. The Mac mean-latency reduction has a paired bootstrap 95% interval of `[2.02%, 8.96%]`, while its p99 latency worsens. Do not pool latency measurements from ICRN and Mac. The `p = 0.005` paired report and a loss-aware final comparison remain outstanding.

## Reproduction and completion criteria

Run from `challenge-1-ml-decoder-kit/` in a Python 3.12 environment with its `requirements.txt` installed; Relay-BP also needs Rust when built from source. Use the [kit setup instructions](challenge-1-ml-decoder-kit/README.md#setup) and the [evaluation stage guide](challenge-1-ml-decoder-kit/evaluation/README.md). The saved-shot order is **dataset → router features → validation-frozen router → baseline/control evaluate → candidate evaluate → paired compare → plot**. For a fresh site output, `MIDQ_PY=/path/to/python ./exec.sh 0.005 delta` runs the last three evaluation stages after validating both router models. Existing output directories are protected against overwrite, so give each new run its own output path. The final comparison should include all three required physical rates, both memory bases, logical rate, full per-shot latency including fallback, resource measurements, and a plot. Record the exact code revision, NPZ hashes, model hashes, package versions, and machine for each run. Test a loss-aware pipeline separately once its simulator, herald representation, priors, and paired baseline are integrated.

## Our part: a hierarchical decoder with a learned tie-breaker, and a study of atom loss

Two runs of a fast decoder (IonQ's PriorTilt) settle the shots they agree on; where they disagree, relay-BP joins and a neural
network picks the most likely answer (`challenge-1-ml-decoder-kit/hierarchy/`, `neural/`), though on the test shots that network is no better than the best hand-written tie-break, so the gain over relay-BP is PriorTilt's and the hierarchy's. On the test shots our decoder fails fewer pairs than relay-BP at p = 0.004 and 0.005, and as many within the error at p = 0.003; it is level with TCBP-JOBS at p = 0.003 and 0.004, and behind it at p = 0.005; our ensemble rule, frozen beforehand and combined offline from the saved answers of NVIDIA's, IonQ's and the kit's decoders, with no time measured as one decoder, fails 2 pairs against TCBP-JOBS's 3 at p = 0.003 (only the ensemble 0, only TCBP-JOBS 1, exact p = 1.00; joint decoding on 5 disputed shots, about 5 s each), fails 16 pairs against TCBP-JOBS's 41 at p = 0.004 (only the ensemble 1, only TCBP-JOBS 26, exact p = 4e-07; joint decoding on 93 disputed shots, about 24 s each), and equals TCBP-JOBS at p = 0.005 (joint decoding not run).
Under atom loss (our own simplified model), relay-BP with the loss flags fails 4.4% of shots where a loss-blind one
fails 57.7%; our loss-time tracker, which estimates from the syndrome when each atom was lost, recovers 58% (38% to 78%) of what knowing the true loss time would add (`loss/`, `loss_tracker/`).

Every version we built is in one table, each number with its label ([report/tables/master_comparison.md](challenge-1-ml-decoder-kit/report/tables/master_comparison.md)): the frozen hierarchy above (pre-registered); the same hierarchy with a hand-written tie-break in place of the network (`hierarchy/`; the best hand-written rule pre-registered as timed rows at p = 0.003 and 0.004 and offline at 0.005, plain lowest weight pre-registered offline at every p; their other timed rows after-test ablations); TCBP-JOBS in place of relay-BP in it (an after-test ablation); a learned predecoder that gives relay-BP per-shot priors from the network (`predecoder/`; X memory at p = 0.005 a secondary result, chosen after its own gate failed, and the other rows a registered extension run after it) and its fallback to TCBP-JOBS where relay-BP does not converge (`predecoder/fallback.py`; designed after the first test result); and TCBP-JOBS given the same per-shot priors (designed after the test results were seen; formerly a separate branch (predecoder-tcbp); its pair logical error rate at p = 0.005 is 1.11e-03 ± 9.6e-05 per cycle against TCBP-JOBS's 3.05e-03 ± 1.6e-04, and at p = 0.004 (no validation at these p) 8.34e-05 ± 2.6e-05 against 3.42e-04 ± 5.3e-05; about 38 ms per X shot after the speed work, timed in parts on validation shots); a learned selector among the ensemble's members (`selector/`; frozen on validation, its line not met on the test shots); and, on its own simulated shots, noise shaped like neutral-atom hardware (`neutral_atom_noise/`; exploration, the kit's decoders, and the TCBP version, whose head retrained on that noise makes it worse on its own base: 27 failures against 8 in run 1). None of these rows replaces the frozen decoder.

| Decoder | p = 0.003 | p = 0.004 | p = 0.005 |
|---|---|---|---|
| BP-OSD (kit baseline) | 5.01e-04 ± 6.5e-05, 131 ms | 4.02e-03 ± 1.8e-04, 302 ms | 2.30e-02 ± 4.6e-04, 581 ms |
| relay-BP (kit baseline) | 4.17e-05 ± 1.9e-05, 85.3 ms | 6.94e-04 ± 7.6e-05, 319 ms | 7.33e-03 ± 2.5e-04, 1,471 ms |
| PriorTilt, one run | 3.33e-05 ± 1.7e-05, 22.7 ms | 6.19e-04 ± 7.2e-05, 87.3 ms | 5.15e-03 ± 2.1e-04, 264 ms |
| our hierarchy, neural tie-break | 2.50e-05 ± 1.4e-05, 45.7 ms | 3.67e-04 ± 5.5e-05, 216 ms | 4.08e-03 ± 1.9e-04, 825 ms |
| TCBP-JOBS (NVIDIA, GPU) | 2.50e-05 ± 1.4e-05, 3.9 ms | 3.42e-04 ± 5.3e-05, 7.8 ms | 3.05e-03 ± 1.6e-04, 16.9 ms |

Test shots (10,000 pairs per p): logical error rate per cycle ± the kit's error, mean time per pair (CPU rows: the whole decode of a pair in one process; TCBP-JOBS: a GPU decode call alone, not comparable with the CPU rows). Against TCBP-JOBS, NVIDIA's GPU decoder, on the same test pairs: at p = 0.003 TCBP-JOBS and our decoder fail as many pairs (3 against 3; only TCBP-JOBS fails 1, only ours 1, exact p = 1.00, not a significant difference); at p = 0.004 TCBP-JOBS fails fewer pairs than our decoder (41 against 44; only TCBP-JOBS fails 17, only ours 20, exact p = 0.74, not a significant difference); at p = 0.005 TCBP-JOBS fails fewer pairs than our decoder (360 against 479; only TCBP-JOBS fails 95, only ours 214, exact p = 1e-11). TCBP-JOBS against relay-BP's timed row: 3 against 5 at p = 0.003 (exact p = 0.62), 41 against 83 at p = 0.004 (exact p = 4e-07), 360 against 845 at p = 0.005 (exact p = 1e-101). TCBP-JOBS fits its model when it starts, and three fits on the same validation shots fail 91, 96 and 100, so its counts carry about ±5%.

![Logical error rate per cycle against p](challenge-1-ml-decoder-kit/report/figures/ler_vs_p.png)

Write-up, with every table made by a script from committed files: [report/REPORT.md](challenge-1-ml-decoder-kit/report/REPORT.md). Everything we built and why: [information.md](information.md).
