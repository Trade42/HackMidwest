# predecoder: the network tells BP, shot by shot, which faults to suspect

**The idea in one paragraph.** Relay-BP, the kit's strongest decoder, starts every shot with
the same fixed guess of how likely each possible fault is (the kit's "priors"). Our trained
neural network has already read the whole shot. This folder gives the network a second output
(a "head") that says, for each of the 8,784 possible faults BP considers, how likely it is that
this fault happened in this shot. Those numbers replace the fixed priors for that one shot,
and relay-BP then decodes as usual. If the network is right about where the faults are, BP
should find the answer sooner and fail less.

Status: built and tested on this PC; the head's training runs on a GPU (commands below).
The measurement is not in yet; every number that is in is listed under "Numbers".

| file | what it does |
|------|--------------|
| `mapping.py` | links the network's fault list (67,824 mechanisms of the full error model) to the decoder's 8,784 columns, and checks the link |
| `head.py` | the head: the network's internal vectors in, one probability per column out |
| `train.py` | trains the head on shots drawn from the error model, with the truly fired faults as labels |
| `evaluate.py` | runs the network and the head on a shot file, saves every shot's logits, times one shot |
| `priors.py` | turns the head's logits into one shot's priors (two knobs), and fits the knobs on train shots |
| `decode.py` | relay-BP built with one shot's priors; the paired test and the Wilson interval |
| `compare.py` | the measurement: the kit's priors against the head's, on the same Stim shots |
| `plugin.py` | the decoder for the team's evaluator (`evaluation.evaluate`) |
| `oracle.py` | the ceiling: relay-BP told the true faults (a diagnostic, never a result) |
| `test_predecoder.py` | tests |

Run every command from `challenge-1-ml-decoder-kit/`.

## How it works, plainly

**Two fault lists.** The network was trained on shots drawn from the full error model of the
memory circuit: 1,728 detectors and 67,824 kinds of fault. The decoders solve a smaller
problem: X memory keeps only the 936 detectors on X checks, and faults that look the same on
those detectors are merged into one column, 8,784 columns in all. The kit builds that
problem (`circuit.dem_matrices`). `mapping.py` follows each of the 67,824 faults to the column
it becomes (or to none: a fault that touches only Z checks is invisible to this problem). A
column "fired" in a shot when an odd number of its faults fired, because two of them cancel;
that is exactly how the kit merged their probabilities. Checked three ways, on every sampled
batch: the merged probabilities equal the kit's priors (to 1e-16); H times the fired columns
equals the shot's 936 detector bits; L times them equals its 12 logical bits.

**The head** (`head.py`). For each column, it looks at the network's vectors on the 2 to 6
detectors the column would light up, and averages them separately over the detectors that
did fire in this shot and those that did not. It adds what kind of column it is (the 8,784
columns are 122 shapes, each repeated at the 72 positions of the torus) and the kit's prior,
and a small network shared by all columns turns that into a correction to the kit's
log-odds. Before training the correction is zero, so an untrained head gives exactly the
kit's priors, and training can only move away from them where the data says so. The head
has the code's symmetry: shifting the shot on the torus shifts its output the same way
(tested to 1e-9).

**Training** (`train.py`). Fresh shots from the full error model every step (no shot is seen
twice) at p = 0.003, 0.004 and 0.005 in turn; the labels are the fired columns. The loss is the
plain cross-entropy, which rewards honest probabilities: a column the head puts at 2% should
fire about 2% of the time. That matters because BP reads the numbers as probabilities. The
network itself stays fixed (frozen); only the head (about 15,000 numbers) learns.

**Two knobs** (`priors.py`). The head has already read the syndrome, and BP reads it again; fed
the head's numbers as they are, BP counts the same evidence twice and may become overconfident.
`alpha` mixes the head with the kit (1 = the head as it is, 0 = the kit), in probabilities or
in log-odds; `eps` keeps every prior inside [eps, 1 - eps]. They are fitted on train shots
only, never on the shots they are scored on.

**The measurement** (`compare.py`). Three versions of relay-BP at the kit's settings, on the
same shots in the same order:

| arm | priors | decoder |
|---|---|---|
| kit | the kit's | built once, decodes the shots in order (the evaluator's "relay-bp") |
| fresh | the kit's | built anew for every shot |
| head | the head's, for this shot | built anew for every shot |

relay-bp takes priors only when a decoder is built, so the head's priors need a new decoder
per shot (about 8 ms, counted in the time). A decoder built once also carries its random state
from shot to shot, which a new one does not. So **head against fresh** changes the priors and
nothing else: that pair decides the question. Head against kit is what a user would see.

**The kill lines** (fixed in the brief before any number): the head must beat the kit's priors
in log-loss on held-out error-model shots and be calibrated; and on the 5,000 shots of
`data/p0.004/validation_X.npz`, relay-BP with the head's priors must fail fewer shots than with
the kit's priors, paired, with p < 0.05 (head against fresh; reported also against kit).

## Commands

```sh
# tests (about 4 minutes on 4 CPU threads)
python -m pytest -p no:cacheprovider predecoder

# the ceiling: relay-BP told the true faults, on DEM shots
python -m predecoder.oracle --p 0.004 --shots 300 --seed 11 --workers 4

# train the head (a GPU; about 0.06 s per step expected on an RTX 4060, not measured yet)
python -m predecoder.train --trunk <cascade_X_s1 checkpoint> --out <run folder> --seed 1 \
    --steps 30000 --batch 64 --p-train 0.003 0.004 0.005 --report-every 2000 --report-shots 500

# the head's logits on a shot file, and the time of one shot
python -m predecoder.evaluate --head <run folder>/head.pt --trunk <cascade_X_s1 checkpoint> \
    --data data/p0.004/train_X.npz --limit 1000 --out <logits_train>
python -m predecoder.evaluate --head <run folder>/head.pt --trunk <cascade_X_s1 checkpoint> \
    --data data/p0.004/validation_X.npz --out <logits_validation>

# fit the two knobs on train shots, then the measurement on validation shots
python -m predecoder.priors --data data/p0.004/train_X.npz --logits <logits_train> --shots 1000 --out <fit.json>
python -m predecoder.compare --data data/p0.004/validation_X.npz --logits <logits_validation> \
    --settings <fit.json> --arms fresh head kit --workers 4 --out <compare folder>

# the team's evaluator
python -m evaluation.evaluate --test-x <X shots> --test-z <Z shots> --decoder predecoder.plugin:make_decoder \
    --plugin-options <options.json> --out <folder>
```

## Numbers

### The ceiling (oracle, DEM shots: a diagnostic, never a decoder result)

`python -m predecoder.oracle --p 0.004 --shots 300 --seed 11 --workers 4` on this PC (lane A,
4 processes), log `logs/oracle_p0.004_300_s11.log`. X memory, 300 shots drawn from the full
error model, each mapped onto the kit's columns (the map held on every shot). relay-BP at the
kit's settings, built per shot. "true at q": the kit's priors with the shot's truly fired
columns raised to q; "and as many false": as many columns that did not fire are raised too.

| priors | failures of 300 | unconverged | mean iterations |
|---|---|---|---|
| the kit's | 1 | 1 | 499 |
| true at 0.05 | 0 | 0 | 44 |
| true at 0.2 | 0 | 0 | 19 |
| true at 0.5 | 0 | 0 | 5 |
| true and as many false, at 0.05 | 0 | 0 | 45 |
| true and as many false, at 0.2 | 0 | 0 | 20 |
| true and as many false, at 0.5 | 1 | 4 | 310 |

What it says: BP reads per-shot priors and they steer it strongly. Raising the true faults
even to 5% cuts the work tenfold, and half wrong guesses at 5% or 20% cost almost nothing.
Confident wrong guesses (50%) start to confuse it. 300 shots hold too few failures (the kit's
priors fail 1) to say anything about failures.
