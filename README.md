# Quantum feature mapping for pronoun resolution

This project tests whether an eight-qubit, entangling IQP feature map helps classify the referent of *her* in short, controlled sentences. Each sentence has 14 words and two candidate names. An explicit parser extracts eight semantic cues from the sentence and candidate order; the same cues go to every model. The quantum circuit returns 8 single-qubit Z and 28 pairwise ZZ expectation values, which feed a classical SVM.

The original dataset, [`data/pronouns.csv`](data/pronouns.csv), combines the controlled corpus and its unused configurations, preserving their IDs and fixed splits. The benchmark plots mean accuracy and one-standard-deviation bands across five training-subset seeds, using nested, balanced training sets of 8, 16, 32, 64, 128, and 256 examples by default. It selects model settings on validation data and evaluates each predetermined size on the same 506 fresh cases whose underlying semantic worlds occur nowhere in the original corpus. The original test partition is retained only to define that strict unseen-world cohort.

## Run

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python benchmark.py
.venv/bin/python contextuality.py
```

These commands save `output/benchmark.png` (test accuracy versus training examples) and `output/contextuality.png` (separate memory protocol), both ignored by Git. Pass `--plot path.png` to choose a different location.

Use `benchmark.py --sizes 32 64 128 --seeds 3` for a shorter run. The benchmark compares raw-cue RBF SVM and MLP against non-entangling and entangling IQP features with the same RBF SVM readout. No model trains quantum circuit parameters; the circuit is a fixed feature extractor.

## Current result

Mean ± one standard deviation across five training-subset seeds, with validation-selected classifiers:

| Model | 128 training examples | 256 training examples |
| --- | ---: | ---: |
| Raw cues + RBF SVM | 74.5 ± 2.4% | 84.6 ± 1.5% |
| Raw cues + MLP | **82.7 ± 3.1%** | **93.7 ± 1.7%** |
| Non-entangling IQP readouts + RBF SVM | 77.2 ± 2.1% | 87.2 ± 0.8% |
| Entangling IQP readouts + RBF SVM | 78.2 ± 1.1% | 87.0 ± 1.0% |

These runs show no quantum advantage. The bands measure variation across training subsets on one fixed test set; they are not test-set confidence intervals. This is a synthetic grammar with hand-extracted cues, not general language understanding. The 8-qubit circuit uses 96 logical gates and depth 26 on PennyLane's exact simulator; these are logical circuit resources, not hardware timings.

## Transfer-reference corpus

[`data/pronouns_nuanced.csv`](data/pronouns_nuanced.csv) adds 1,024 short sentences about lending and borrowing. For example, "Ava borrowed Mia's camera; she asked for it back that evening" refers to Mia, who owns the camera. "Mia's camera was borrowed by Ava; she asked for it back that evening" has the same referent even though Mia now appears first. The label depends on the transfer direction, the later action, sentence voice, and candidate order. The corpus also varies names, objects, times, and action wording; every sentence has at most 15 words.

The 640 training, 128 validation, and 256 test rows are balanced by label. Candidate-order variants stay in the same split, and no encoded input repeats. All eight transfer/action/voice combinations occur in training; the test contains new combinations of the remaining sentence details. The corpus is generated reproducibly by `data/make_pronouns_nuanced.py` and still uses explicit cue extraction, not a language model.

```bash
mkdir -p output/nuanced
.venv/bin/python benchmark.py --data data/pronouns_nuanced.csv --plot output/nuanced/benchmark.png > output/nuanced/results.txt
```

At 64 training examples, MLP reaches 98.8 ± 2.8% test accuracy, versus 64.5 ± 3.5% for entangling IQP (five training-subset seeds). The learning curve and full results are in `output/nuanced/`, separate from the original benchmark.

`contextuality.py` is a separate simulated memory task. The two yes/no facts are "Is Ava shorter?" and "Is Ava younger?" A message is prepared before one of these questions is chosen at random. The rule is that the message must hide whether the two answers match (their parity). A classical message obeying this rule can answer correctly at most 75% of the time: it can retain one answer, then must guess when asked the other. At the default perfect visibility, the one-qubit encoding predicts 85.36%. These are the two bars in `output/contextuality.png`; simulated trials are reported in the terminal, not plotted. Without the parity rule, classical memory can retain both facts and reach 100%. This simulates a known [parity-oblivious multiplexing protocol](https://arxiv.org/abs/0805.1463) whose predicted success exceeds the noncontextual bound. It does not imply that the NLP classifier uses contextuality.

## Sentence-based memory games

The [`owner_holder`](contextuality/owner_holder/run.py) and [`conditional_rules`](contextuality/conditional_rules/run.py) experiments each have one Python script and one generated CSV. Run them with:

```bash
.venv/bin/python contextuality/owner_holder/run.py
.venv/bin/python contextuality/conditional_rules/run.py
```

Each script writes `dataset.csv` and `output/accuracy.png` in its own folder. It splits 64 name/object or name/place scenarios into 40 training, 8 validation, and 16 test scenarios **before** making sentence variants. That gives 320 training, 64 validation, and 128 test sentences with no repeated sentence or scenario across splits. The plots evaluate the same untouched test set for every training size; five different training subsets produce the mean and standard-deviation bands. Every test sentence can receive either of two questions, chosen only after a message has been prepared.

A TF-IDF logistic-regression parser learns the two answers from training sentences; validation selects its regularization and which answer the classical bit stores. The classical and quantum memory encodings are fixed. The one-qubit circuit uses the parser's two predicted answers and is measured according to the later question. At 256 training sentences, both parsers correctly decode the held-out facts; each experiment then gives 75.0% for the classical bit and 85.4% for the qubit, with parity-guess accuracy 50.0%. This reproduces the constrained memory protocol on sentence-shaped inputs. It is not evidence that the original IQP pronoun classifier uses contextuality or that a quantum model parses language better.
