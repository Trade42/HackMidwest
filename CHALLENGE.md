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
