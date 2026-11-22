# neural: a convolutional decoder that follows the Gross code's connections

A neural network that reads the detection events of one memory experiment and predicts the
12 logical bits directly. It is built after the one published recipe that works on this
code: "Scalable Neural Decoders for Practical Fault-Tolerant Quantum Computation"
(arXiv:2604.08358, the "Cascade" decoder), scaled down to train on one laptop GPU.

No code was copied: the paper's authors released none, and nothing here comes from another
repository. The folder uses only torch, numpy, scipy and stim, and the kit's own
`circuit.py` and `make_circuit.py`.

| file | what it does |
|------|--------------|
| `graph.py` | the code's structure as index tables, read from the kit; the lighter logical operators |
| `model.py` | the network |
| `sampler.py` | training shots drawn from the detector error model, and the check against Stim |
| `train.py` | the trainer |
| `evaluate_stim.py` | scores a trained decoder on the kit's Stim shots |
| `measure.py` | prints a model's size and the cost of one training step |
| `test_neural.py` | tests |

Run every command from `challenge-1-ml-decoder-kit/`.

## The design in plain words

**The task.** X memory has 1,728 detectors: 936 on X checks (13 rounds of 72) and 792 on
Z checks (11 rounds of 72). The label is 12 bits: which of the 12 X logical operators the
errors flipped. Z memory is the same with X and Z swapped. One network is trained per
memory basis.

**Where the network keeps its numbers.** On every check, in every round, it keeps a vector
of `width` numbers (128 by default): 13 rounds x 144 checks. Inside a layer it also keeps
vectors on the 144 data qubits, in each of the 12 cycles.

**Three ingredients from the paper.**

1. *Layers that follow the code's connections.* A check touches six data qubits, and a
   data qubit is touched by six checks (three X, three Z). One step of a layer lets every
   data qubit collect from its six checks, and the next lets every check collect from its
   six data qubits. Each step has one weight matrix per neighbour and per time offset. The
   same matrices are used at all 72 positions of the 12 x 6 torus, so the layer is a
   convolution on the code's own graph: shifting the syndrome on the torus shifts the
   network's internal vectors in the same way (tested). A data qubit in cycle c exchanges
   with its checks in rounds c and c + 1, the two rounds where an error of that cycle can
   show up.
2. *Pooling on the logical operators.* After the last layer one more step collects on the
   data qubits. For each of the 12 observables, the vectors of the data qubits it reads are
   averaged (over those qubits and over time), and one small network, shared by all 12,
   turns the average into one number: positive means "this bit flipped".
3. *A curriculum.* Training starts at low noise, where a shot holds a few separate errors
   and the network learns quickly, then climbs to the training noise and stays there.

**A layer in detail.** A residual block with a bottleneck, as in the paper: narrow the
vectors from `width` to `width / 4`, do the two collecting steps, widen again, and add the
result to the block's input. Batch normalisation and a SiLU come before every linear step.
The loss is binary cross-entropy averaged over the 12 bits.

**What comes from the kit, not from numbers typed in.** `graph.py` reads the neighbours of
every check from `make_circuit.code_layout`, follows every DETECTOR of the kit's circuit
back to its measurements to find its check and round, and reads each observable's data
qubits from the circuit's OBSERVABLE_INCLUDE lines. A test proves the layout of all 1,728
detectors: a torus shift maps the kit's full error model onto itself, mechanism by
mechanism, with equal probabilities.

**Which detectors it reads.** `--detectors all` (the default) reads all 1,728.
`--detectors basis` reads the 936 that the baselines use. The network is the same in both
cases: with `basis`, the other checks carry a learned "no detector here" vector in every
round. A comparison of the two therefore measures what the other 792 bits are worth, not a
difference in network size.

**The training data.** `sampler.py` draws every shot on the training device from the
kit's detector error model (`circuit.dem_matrices`): each error mechanism fires on its own
with the kit's prior, the detectors are H times the fired mechanisms mod 2, and the labels
are L times them mod 2. Nothing is read from disk, so no shot is seen twice. The labels go
to the loss only; the network's one input is the detector bits (tested).

## Z memory: lighter logical operators, and a warm start from X

**The problem.** The kit's 12 X observables are logical operators on 12 to 14 data qubits
each. Its 12 Z observables sit on 12, 12, 12, 12, 12, 12, 12, 24, 28, 34, 36 and 38 qubits.
A label flips when an odd number of errors falls on its operator, so a heavy operator's
label is closer to a fair coin, and the readout (an average over the operator's support,
then one head shared by all 12) is built for supports of one size. With the kit's operators
the tiny model has 67.2% of Z shots wrong at step 250 of the smoke run, where X has 41.4%
(the table is under "Numbers"). A scratch diagnostic on that run's step-250 weights, which
this folder has no command for, split the 12 bits (750 error-model shots at p = 0.0005):
the seven light operators' bits were right 89 to 96% of the time, the five heavy ones' 63
to 74%.

**The fix: `--logicals light`.** The Gross code looks the same with X and Z checks
exchanged, the two data blocks exchanged and every torus position negated. That carries the
kit's X operators onto 12 valid Z operators with supports of 12 to 14 qubits. The network
predicts those, and the kit's 12 bits follow exactly:

    kit bits = M x light bits + C x check parities   (mod 2)

M is an invertible 12 x 12 matrix and C picks some Z checks; both are solved from the kit's
own matrices in `graph.py`. The parity of a check is the sum of its detectors over all
rounds. Nothing is approximated: on noisy Stim measurement records the converted bits equal
Stim's observables on every shot (tested, Gross and [[72,12,6]], both detector sets). A shot
is wrong in the kit's bits exactly when it is wrong in the light bits, so the kit's failure
count is unchanged; only "bits right" in a training report refers to the light bits. For X
memory the kit's operators are already the light set, and the option changes nothing.

**A warm start from X: `--init-from <X checkpoint>`.** The same symmetry turns the weights
of an X decoder into weights of a Z decoder, exactly: the mirrored network computes on a Z
syndrome what the X network computes on the mirrored syndrome (tested to 1e-10). The two
memory circuits are not mirror images of each other, though: the checks meet their
neighbours in different orders. So a mirrored X decoder is a starting point for Z, not a Z
decoder. Measured: a Z run started from the tiny X smoke checkpoint reports 37.0% of shots
wrong before its first training step (400 error-model shots at p = 0.0005, always-0 85.3%),
where that checkpoint has 16.2% of X shots wrong:

```sh
python -m neural.train --basis Z --logicals light --init-from <the X smoke folder>/checkpoint.pt --out <folder> \
    --tiny --seed 1 --p-start 0.0005 --p-train 0.0005 --steps 0 --report-shots 400 --report-p 0.0005 --threads 1
```

A scratch comparison of the two error models at p = 0.004, which this folder has no command
for, found that the mirrored X model and the Z model share 11,952 of about 67,800 detector
sets, carrying 61% of the fault probability, with equal priors on 95% of those.

```sh
# Z memory from scratch
python -m neural.train --basis Z --logicals light --out runs/cascade_Z_s1 --seed 1 --steps-start 2000 --steps-anneal 12000
# Z memory from an X run of the same size, with a short climb
python -m neural.train --basis Z --logicals light --init-from runs/cascade_X_s1/checkpoint.pt \
    --out runs/cascade_Z_s1 --seed 1 --steps-start 1000 --steps-anneal 6000
```

With the light operators, Z from scratch learns like X: the tiny model has 18.0% of shots
wrong after 16,000 shots at p = 0.0005, against 16.2% for X (the table is under "Numbers").
How much training the warm start saves at the default size was not measured when this was
written.

## Where the paper is silent, and where this differs from it

| Point | The paper | Here |
|---|---|---|
| The factor 1/sqrt(2 x depth) in a block | its figure draws it on the skip path; the caption is ambiguous | on the block's branch: `x + f(x) / sqrt(2 depth)`. `--residual skip` gives the other reading |
| Between the two collecting steps of a block | not stated | a normalisation and a SiLU |
| Time slots of the data qubits | "2 temporal offsets" per step | cycle c exchanges with rounds c and c + 1 |
| The readout's average | "over the data qubits in that observable's support" | over those qubits and over time. No nonlinearity between the last collecting step and the average, so the average over time is taken first (same result, about six times less work in that step; tested) |
| Weights of the two check types | not stated | X and Z checks have their own weights, and so do the left and right data qubits |
| Which logical operators are predicted | not stated | the kit's for X memory; for Z memory a lighter set from which the kit's follow exactly (`--logicals light`) |
| Which detectors | not stated. Its count of 22 neighbouring checks per convolution equals this graph's (tested), which needs both check types | both types, or the basis alone |
| Optimizer | Muon and Lion, with Maximal Update Parameterization | AdamW, PyTorch's default initialisation |
| Curriculum | "at most 2% of the steps"; no values | 4,000 steps at p = 0.001, 24,000 steps of climb in 12 jumps (options) |
| Mixed precision | bfloat16 | float16 with a gradient scaler (`--amp bf16` is an option) |
| Size | width 512, depth 14, batch 3,328, about 3e8 shots, under 100 hours on an H200 | width 128, depth 8, batch 128 (options) |
| The convolution's code | custom Triton kernels | plain PyTorch: `index_select` and `Linear` |

The team's notes (`reference/papers-ml-decoders.md`, A1) agree with the paper on the
architecture. One detail they leave out: the paper's learning rate follows a cosine over
50,000 of its 80,000 steps and ends at one tenth of the peak. The trainer does the same
over `--lr-decay-steps`.

The paper's logical error rate is per logical qubit per round (its equation 1). The kit's
rate is per cycle for a pair of an X shot and a Z shot, where any wrong bit of 12 fails the
shot. The two are not the same number, so the paper's figures cannot be read against the
kit's table directly.

## Commands

The environment is the kit's (`requirements.txt`) plus torch. Training needs only torch,
numpy, scipy and stim.

```sh
# tests (about a minute on a CPU)
python -m pytest -p no:cacheprovider neural

# the sampler against Stim's circuit sampler, and the sampler's speed (one detector set per call)
python -m neural.sampler --p 0.001 --shots 200000 --detectors all --threads 4
python -m neural.sampler --p 0.001 --shots 200000 --detectors basis --threads 4

# train (the defaults are sized for an 8 GB GPU); stop after 2 hours with a checkpoint
python -m neural.train --out runs/cascade_X_s1 --seed 1 --max-hours 2
# continue the same run
python -m neural.train --out runs/cascade_X_s1 --resume runs/cascade_X_s1/checkpoint.pt --max-hours 10

# score a checkpoint on saved Stim shots; --save keeps the numbers and the per-shot logits
python -m neural.evaluate_stim --checkpoint runs/cascade_X_s1/checkpoint.pt --data data/p0.004/validation_X.npz \
    --save runs/cascade_X_s1/validation_X

# where the time of a training step goes on this machine
python -m neural.measure
```

`python -m neural.train --help` lists every option. `--out` receives `train.log`,
`checkpoint.pt` and the kept copies `checkpoint_step<N>.pt`; none of them belongs in git.

### What the trainer prints

```
step 4000 of 400000 | training noise p=0.001 | mean training loss 0.0161 | learning rate 1.00e-03
  averaged weights on 2000 error-model shots per p (not Stim shots); 95% interval of a share: +-0.022 or less
  p=0.001  shots wrong 0.0170 (always-0 0.9725)  bits right 0.9977 (always-0 0.6066)
  ... and the same for p=0.002, 0.003, 0.004 and 0.005
speed: 645 shots/s in training, 0.198 s per step, report 5.0 s, peak GPU memory 3.92 GB in tensors and ...
```

- **shots wrong** is the share of shots with at least one of the 12 bits wrong: the kit's
  failure, for one basis. **bits right** is the share of single bits predicted right.
- **always-0** is what answering "no bit flipped" scores. At p = 0.003 and above it gets
  half the bits right and nearly every shot wrong: the 12 labels are fair coins there, and
  the network has to learn the whole decoding.
- The report shots come from the error model with fixed seeds: the same shots in every
  report of a run. They are a training diagnostic. Results are quoted from
  `evaluate_stim.py` only.
- **How exact a report is.** With 2,000 shots a share is known to +-0.022 or better, which
  shows learning but cannot resolve a rate under about 0.005. Because every report uses the
  same shots, the change between two reports is steadier than that. `--report-shots` sets
  the number and may be changed on resume.
- Lines that start with `speed` depend on the machine. All other lines are identical when
  the same command runs twice on one machine.

### What the evaluation prints

`evaluate_stim.py` prints the failures of one memory basis on Stim circuit shots, the
logical error per shot and per cycle with a 95% Wilson interval, and the batch-one
inference time with the device named. Then it prints a **confidence sweep**, because the
network is meant as the first stage of a hierarchy: it keeps the shots it is sure of, and a
classical decoder takes the rest.

- The confidence of a shot is the smallest |logit| among its 12 outputs.
- For the most confident 10, 25, 50, 75, 90, 95, 99 and 100% of the shots, the table gives
  the threshold, the failures among the kept shots with their interval, and the failures
  among the rest.
- For scale: relay-BP fails about 0.5% of X shots at p = 0.004 and about 0.04% at p = 0.003
  (kit README). A kept set is useful where its failure share is near or below that. On
  5,000 shots a kept set without a single failure still has an upper limit of 0.08% to
  0.8%, so the p = 0.003 bar needs a file of 50,000 shots or more.
- A threshold read from the table is fitted to those shots. Freeze it and report it on
  another file.
- `--save <prefix>` writes `<prefix>.json` (every number) and `<prefix>.npz` (per shot, in
  file order: `logits`, `predictions`, `observables`, `failed`, `confidence`).
  `predictions` and `observables` are always the kit's 12 bits. The `logits` belong to the
  model's own operators: for a `--logicals light` model, `logits > 0` is not the prediction,
  and the file's `logits_are_for` entry says so.
- Files named `test_*` or `go_*` are refused without `--final`.

### Seeds, checkpoints, stopping and resuming

- `--seed` fixes the initial weights and every shot: the shots of step s depend only on
  the seed and s.
- Every schedule (noise, learning rate, moving average) is a function of the step count,
  never of the clock. `--max-hours` only decides when to stop.
- `checkpoint.pt` holds the weights, the moving average of the weights, the optimizer, the
  gradient scaler, the step, every setting (seed, basis, noise levels, model size), all
  earlier reports and the git commit. It is rewritten at every report and every
  `--checkpoint-minutes` (10) in between. A copy `checkpoint_step<N>.pt` is kept every
  `--snapshot-every` steps (10,000): the decoders along the climb.
- **To stop a run, create an empty file named `STOP` in `--out`.** The run prints a report,
  writes the checkpoint, removes the file and ends. Do not kill the process.
- `--resume` takes all settings from the checkpoint. Only `--steps`, `--lr-decay-steps`,
  `--report-every`, `--report-shots` and `--snapshot-every` may be given again. A run
  resumed on the same machine reaches the same weights as an uninterrupted one (tested on
  the CPU). A checkpoint written by older code of this folder loads: no change since the
  first commit touched a parameter.

## Numbers

Two machines. They never share a line.

### This PC (CPU only: Ryzen 7 7730U, 4 threads of lane A, niced, torch 2.15.0+cpu)

**The sampler against Stim's circuit sampler** (the two `neural.sampler` commands above,
seed 7, X memory, 200,000 shots from each sampler). A z score compares one rate between the
two samplers; 1,740 of them should stay under about 4.

| detectors | mechanisms | largest z, 1,728 or 936 detector rates | mean z^2 (1 expected) | largest z, 12 observable rates | mean observable rate, ours / Stim | syndrome weight, mean and variance, ours / Stim | result |
|---|---|---|---|---|---|---|---|
| all | 67,824 | 3.59 | 0.991 | 1.69 | 0.39352 / 0.39232 | 65.245, 295.46 / 65.157, 294.27 | PASS |
| basis | 8,784 | 3.85 | 1.013 | 2.54 | 0.39292 / 0.39232 | 34.093, 101.29 / 34.063, 101.30 | PASS |

p = 0.001 is used because the observables are informative there: at p = 0.004 every one of
them is a fair coin (rate 0.4997), whatever the sampler does. A 20,000-shot run at
p = 0.004 also passes (largest z 3.55 and 3.27).

**The sampler's speed** (same command): 106 ms per batch of 128 for the full error model
(1,205 shots/s) and 14.8 ms for the basis model (8,656 shots/s), a factor of 7.2. On the
GPU it does not matter: the laptop draws a batch of 32 from the full model in 1.9 ms, under
2% of a training step.

**The model's size** (`python -m neural.measure --device cpu`): 893,697 parameters at width
128 and depth 8; 67 MB of activations per shot in training with 32-bit floats.

**Z memory with the kit's operators and with the light ones** (the smoke command below with
`--basis Z`, and with `--basis Z --logicals light`; share of shots wrong at p = 0.0005 on 500
error-model shots, +-0.044):

| step | X memory | Z memory, the kit's operators | Z memory, `--logicals light` |
|---|---|---|---|
| 250 | 0.414 | 0.672 | 0.420 |
| 500 | 0.162 | not reached (the lane cut the run) | 0.180 |

**The CPU smoke run: a tiny model learns at low noise.** Width 32, depth 3 (38,537
parameters), batch 32, 500 steps (16,000 shots) at p = 0.0005, seed 1, about 7 minutes:

```sh
python -m neural.train --out <folder> --tiny --seed 1 --p-start 0.0005 --p-train 0.0005 --steps 500 \
    --report-every 250 --report-shots 500 --report-p 0.0005 0.001 --warmup-steps 100 --lr 2e-3 --threads 4
```

| step | fresh shots seen | p = 0.0005: shots wrong (always-0 0.856) | bits right (always-0 0.744) | p = 0.001, never trained on: shots wrong (always-0 0.972) |
|---|---|---|---|---|
| 0 | 0 | 1.000 | 0.364 | 1.000 |
| 250 | 8,000 | 0.414 | 0.929 | 0.762 |
| 500 | 16,000 | 0.162 | 0.974 | 0.524 |

These are error-model shots (500 per p, so +-0.044). Run twice into two folders, the two
`train.log` files are identical except for the `speed` lines and the last line, which names
the folder.

**The smoke checkpoint on Stim circuit shots** (`evaluate_stim.py`):

| shots | command ends with | failures | logical error per shot, 95% Wilson interval | always-0 would fail |
|---|---|---|---|---|
| sampled here at p = 0.0005, seed 12345 | `--p 0.0005 --shots 5000 --seed 12345` | 868 of 5,000 | 0.174 [0.163, 0.184] | 4,262 |
| the kit's `data/p0.004/validation_X.npz` (seed 201) | `--data data/p0.004/validation_X.npz` | 5,000 of 5,000 | 1.000 [0.999, 1.000] | 5,000 |

The first row shows that a network trained on error-model shots decodes Stim's circuit
shots equally well (the training report said 0.162 on 500 shots). The second row is no
result: this checkpoint never saw noise above 0.0005, and at p = 0.004 it is at chance. It
only shows that the script runs on the kit's file. Batch-one inference of this tiny model
on the shared CPU (2 threads): 10.3 ms mean, 9.3 ms median.

**The confidence sweep of the first row** (the same 5,000 Stim shots; a shot's confidence is
the smallest |logit| of its 12 outputs):

| kept, most confident first | threshold | failures among the kept | share, 95% Wilson interval | failures in the rest |
|---|---|---|---|---|
| 10% | 6.516 | 0 of 500 | 0 [0, 0.0076] | 868 of 4,500 |
| 25% | 3.897 | 2 of 1,250 | 0.0016 [0.0004, 0.0058] | 866 of 3,750 |
| 50% | 2.827 | 14 of 2,500 | 0.0056 [0.0033, 0.0094] | 854 of 2,500 |
| 75% | 0.901 | 230 of 3,750 | 0.061 [0.054, 0.069] | 638 of 1,250 |
| 100% | 0.001 | 868 of 5,000 | 0.174 [0.163, 0.184] | none |

So even this tiny network knows which shots it has wrong: the confident half fails 30 times
less often than the whole. These thresholds are read from the same shots they are scored on;
a threshold for use must be frozen and reported on another file.

### The laptop (RTX 4060 Laptop GPU, 8 GB, Windows, torch 2.14.1+cu130)

**DEM-sampled training reports from an unreviewed run.** The orchestrator ran it and
forwarded its log; the author of this folder did not re-run it. The default model on
X memory with all detectors, seed 1, float16, started at commit f0a95fa:

    python -m neural.train --out runs\cascade_X_s1 --seed 1 --max-hours 2

Share of shots wrong on 2,000 error-model shots per p (+-0.022 or less):

| step | shots seen | trained at | p = 0.001 | p = 0.002 | p = 0.003 | p = 0.004 | p = 0.005 |
|---|---|---|---|---|---|---|---|
| always-0 | | | 0.9725 | 0.9985 | 1.0000 | 1.0000 | 0.9995 |
| 2,000 | 256,000 | p = 0.001 | 0.0385 | 0.2910 | 0.6620 | 0.9100 | 0.9855 |
| 4,000 | 512,000 | p = 0.001 | 0.0170 | 0.1650 | 0.5385 | 0.8525 | 0.9740 |
| 6,000 | 768,000 | p = 0.001375 | not forwarded | not forwarded | 0.409 | not forwarded | not forwarded |
| 12,000 | 1,536,000 | up to p = 0.0025 | 0.0028 | 0.0386 | 0.2108 | 0.5390 | 0.8288 |
| 28,000 (the climb ends) | 3,584,000 | up to p = 0.0055 | not forwarded | not forwarded | 0.0862 | 0.2724 | 0.5724 |

Up to step 4,000 the network had only seen noise of p = 0.001. At step 10,000 the run was
moved to commit 7ff6a68 by `--resume`, with 5,000 report shots per p (+-0.014 or less) from
then on. Its later rows, and every number on Stim shots, belong to the run's own report,
not to this file.

**Speed and memory** at batch 128, alone on the GPU: 645 shots/s (0.198 s per step) at
commit f0a95fa, 623 shots/s with five CPU decoding processes beside it, and 718 shots/s
(0.178 s per step) at commit 7ff6a68 in the same company. 3.92 GB peak in tensors, about
6.2 GB held by PyTorch.

## Speed: where a training step's time goes

`python -m neural.measure --batch 32` on the laptop, run by the orchestrator beside the
training run on the same GPU. The times are inflated by that sharing and move by tens of
percent between two runs; compare within a column.

| part of a step, ms | commit 2becd60 | commit 7ff6a68 |
|---|---|---|
| draw the shots (67,824 mechanisms) | 1.91 | 1.97 |
| forward pass | 34.00 | 32.47 |
| backward pass | 60.11 | 61.39 |
| optimizer, clipping, moving average | 25.28 | 5.33 |
| whole step | 121.30 | 101.15 |

| operation of a residual block, forward and backward, at 7ff6a68 | ms per call | calls per step | ms per step |
|---|---|---|---|
| linear map on six neighbours, 192 -> 64 | 1.120 | 32 | 35.8 |
| normalisation at the bottleneck | 0.887 | 24 | 21.3 |
| gather of the neighbours | 0.649 | 32 | 20.8 |
| normalisation at full width | 1.920 | 8 | 15.4 |
| SiLU at full width | 1.312 | 8 | 10.5 |
| SiLU at the bottleneck | 0.402 | 24 | 9.7 |
| linear 32 -> 128 | 1.179 | 8 | 9.4 |
| linear 128 -> 32 | 1.093 | 8 | 8.8 |
| residual sum at full width | 1.007 | 8 | 8.1 |
| for comparison: gather with PyTorch's own backward | 1.038 | | |
| for comparison: normalisation at full width on a 3-D view (cuDNN) | 25.004 | | |

What this says:

- The network is not short of arithmetic. A block makes about 100 passes over large
  tensors, and normalisation and SiLU are more than a third of its time. Plain PyTorch
  cannot merge them into the linear maps; the paper's authors wrote custom GPU kernels.
- An estimate from the operation count (1,000 to 2,000 shots/s) was 2 to 3 times too high.
- Sampling 67,824 mechanisms per shot is 2% of a step: the basis model would not save time.

Changes that keep the function (same parameters, so an older checkpoint loads):

| commit | change | measured on the laptop |
|---|---|---|
| 6a99ae6 | the gather's backward pass gathers and sums, instead of PyTorch's scatter with atomic additions | 0.65 ms against 1.04 ms per call: about 12 ms of a step |
| 98196dd | the fused AdamW on cuda | with 82ae10d: the bookkeeping fell from 25.3 ms to 5.3 ms, and the whole step from 121 ms to 101 ms |
| 82ae10d | the moving average of the weights is one vector, updated by one operation instead of 270 | see the row above |
| 607681e | the residual factor multiplies the small weight matrix, not the full-width result | nothing outside the noise |

Tried and dropped: the normalisation on a 3-D view, which lets PyTorch use cuDNN, is 13
times slower here.

Changes that would make it faster by changing the model (each needs a fresh run, and what
it costs in accuracy is not known): `--depth 6` takes 0.75 of the time; `--width 96` about
0.7.

## A long run on an 8 GB GPU

Nobody has published a model this small on this code at p = 0.003 to 0.005, so the long
run is an experiment with a stop rule, not a recipe.

**The size.** Width 128, depth 8, batch 128: the default. PyTorch holds about 6.2 GB of the
card at this size (3.92 GB in tensors, the rest cached), so neither the batch nor the model
can grow on 8 GB.

**The time.** At 0.198 s per step: 18,000 steps an hour. The climb ends at step 28,000,
after 1.5 hours. Twelve hours are about 220,000 steps and 28 million shots (the paper's
large model saw 3e8). The plan's stop rule is 24 GPU-hours.

**The run.** Keep the first run going with `--resume`; do not start over. Give the planned
length on resume, so that the learning rate reaches its floor at the end and not far
beyond it:

```sh
python -m neural.train --out runs/cascade_X_s1 --resume runs/cascade_X_s1/checkpoint.pt \
    --steps 230000 --lr-decay-steps 230000 --report-shots 5000 --max-hours 10
```

**What the report rows must reach.** For X memory, the share of shots wrong of the kit's
baselines (kit README, 10,000 shots): BP-OSD 0.0026 at p = 0.003, 0.025 at 0.004 and 0.13
at 0.005; relay-BP 0.0003, 0.0045 and 0.041. A report cannot resolve the lower values:
score saved Stim shots with `evaluate_stim.py`, the last checkpoint and a few kept copies.

**What to watch, and what to change.**

1. After step 28,000 the run trains at p = 0.0055. If the rows for p = 0.003 and 0.004
   still improve from report to report: change nothing. Most of the paper's training
   happens after the climb.
2. If those rows get worse than they were during the climb: the target is too high for
   this size. Start a fresh run with `--p-train 0.0045` and compare it with the kept copies
   (we are judged at 0.003 to 0.005).
3. If the training loss jumps, or the run stops with a non-finite loss: a fresh run with
   `--lr 5e-4`.
4. A faster climb is safe for a new run, on this evidence: at step 4,000 the network had
   46% of shots right at p = 0.003 without having seen noise above 0.001. Use
   `--steps-start 2000 --steps-anneal 12000`.
5. The kit's score needs both bases. Train Z memory with `--basis Z --logicals light`, and
   start it from the X run's weights with `--init-from` (see "Z memory" above).

## What is tested, and what is not

`test_neural.py` (51 tests, about a minute and a half of CPU time on one thread):

- **Structure.** Every check has its six data qubits, equal to the kit's check matrices;
  every data qubit has three checks of each type; the neighbour tables move with the torus;
  two steps reach 22 checks (the paper's count); the 936 basis detectors sit at
  72 x round + check; the full error model of both circuits maps onto itself under a torus
  shift; the labels are the logical operators of the memory basis.
- **Model.** Shifting the syndrome shifts the internal vectors (to 1e-10, in training and
  in evaluation mode); the gather's hand-written backward pass equals PyTorch's bit for bit;
  the readout equals the paper's order of operations; the `basis` input leaves the other
  checks without a detector; the model takes detector bits only; mixed precision runs.
- **Z memory.** The light operators are logical operators of the memory basis and rebuild
  the kit's; on Stim measurement records the converted bits equal Stim's observables, and a
  shot is wrong in both sets or in neither; the sampler's light labels convert to the kit's;
  mirrored weights decode the mirrored syndrome; a run starts from another basis' checkpoint.
- **Sampler.** Detectors equal H times the faults and labels equal L times the faults; the
  faults fire at the kit's priors; the rates agree with Stim on the [[72,12,6]] code; the
  same seed gives the same shots.
- **Trainer.** The schedules; the scoring; a tiny model overfits 16 fixed shots; a
  checkpoint round-trips, loads when written by older code, and a resumed run reaches the
  same weights; Z memory trains; the command prints the same lines twice, keeps its copies,
  stops on a `STOP` file and resumes.
- **Evaluation.** The Wilson interval; the failures on a saved file equal a direct count;
  the confidence sweep keeps the most confident shots first and its counts add up; the saved
  per-shot arrays; held-out files are refused without `--final`.

Not tested:

- **The GPU path, by any test.** This PC has no GPU. Sampling on the device, float16 mixed
  precision, the gradient scaler, the fused AdamW and `--resume` (also from older code to
  newer) ran on the laptop, at commits f0a95fa and 7ff6a68; nothing here checks them.
- **Whether the default model decodes at p = 0.003 to 0.005.** That is the GPU experiment.
- **A Z decoder at the default size.** Z memory was trained only with the tiny model on the
  CPU. Codes other than gross and bb72 are not tested.
- **`--amp bf16` and `--residual skip`** beyond a forward and backward pass.

## Later: a plugin for `evaluation.evaluate`

Once trained models exist for X and Z, a plugin is a small file. The evaluator calls
`make_decoder(problem, options)` once per basis and `decode_shot(full, basis)` per shot,
where `full` holds the 1,728 detector bits. The plugin would load the checkpoint of
`problem.basis` with `neural.train.load_decoder`, run the model on `full` (or on `basis` for
a `--detectors basis` model) and turn its logits into the kit's 12 bits with
`model.kit_bits(detectors, logits > 0)`, as `evaluate_stim.decode` does. That last step
matters: a `--logicals light` model's own bits are not the kit's. It returns
`ShotResult(predictions, {"confidence": ...})`. The smallest |logit| of the 12 is a natural
confidence for a router: the paper reports that such outputs are well calibrated, and the
confidence sweep above shows it on the smoke model.
