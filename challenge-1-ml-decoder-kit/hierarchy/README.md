# hierarchy: two fast decodes, and a learned tie-breaker where they disagree

The team's decoder for the required comparison with BP-OSD and relay-BP. It is a hierarchy:

1. A fast decoder (PriorTilt) decodes the shot twice, under two random seeds.
2. If the two answers agree, that is the answer. This happens on about 99.6% of shots at p = 0.004.
3. If they differ, a third decoder (relay-BP) decodes the shot too, and a rule chooses among
   the three answers. The rule is either the neural network of `neural/` ("network": the answer
   the network finds most likely) or a rule without learning ("weight": the most probable
   set of faults).

On 25,000 exploration shots of X memory at p = 0.004 it fails 66 times where relay-BP fails
122, in 71% of relay-BP's mean time (laptop times; table below). These are not test-set
numbers: the one run on the test shots has not happened.

Run every command from `challenge-1-ml-decoder-kit/`.

| file | what it does |
|------|--------------|
| `plugin.py` | the decoder, as a plugin for the team's evaluator (`evaluation.evaluate`) |
| `candidates.py` | decodes a saved shot file with all three decoders and saves every answer |
| `select.py` | scores the rules offline on such files, alone or with the network's logits |
| `compare.py` | checks that two candidate files agree shot for shot (two machines, two runs) |
| `tcbp_member.py` | part 2: TCBP-JOBS (the teammates' GPU decoder) on saved shot files |
| `joint_member.py` | part 2: joint relay-BP on listed shots |
| `ensemble.py` | part 2: the members' answers combined offline by one rule |
| `stages.py` | the three decoders on the kit's problem, and the order they run in |
| `rules.py` | how a rule chooses; shared by the plugin and by `select.py` |
| `test_hierarchy.py` | tests |

## The design in plain words

**Why two runs of the same decoder.** PriorTilt is belief propagation that, when it gets
stuck, nudges a few of its inputs at random and tries again. A different seed gives
different nudges. On an easy shot both seeds land on the same answer. On a hard shot they
often land on different ones, and that disagreement is the cheapest reliable sign that a shot
is hard: on the laptop's 25,000 X shots the two seeds differ on 99 shots (0.40%).

**What happens on a hard shot.** Relay-BP, the baseline, decodes it as a third opinion. Now
there are up to three different answers, each a prediction of the 12 logical bits. The rule
picks one:

- *weight* looks at the corrections themselves. Every decoder returns a set of faults; the
  weight of the set is the sum of `log((1 - prior) / prior)` over its faults, so a lower
  weight is a more probable explanation. The rule takes the lowest weight of the three.
- *network* asks the neural network of `neural/`, which reads the shot's detector bits and
  gives 12 logits, one per logical bit. A candidate's score is the sum over the 12 bits of
  `log sigmoid(logit)` where the candidate says 1 and `log sigmoid(-logit)` where it says 0:
  the log-probability the network gives to that candidate's 12 bits. The highest score wins.
  With an optional constant C the score is that sum minus weight / C.

**Why the network is only a tie-breaker.** Alone, the network (step 30,000 of a run that is
still training) gets 27% of these shots wrong, against 0.5% for relay-BP. But it does not
need to decode. It only has to say which of two or three given answers looks more likely,
on 0.4% of the shots, and that it can do: it picks a right answer more often than the weight
rule does.

**The cost.** Every shot pays for two PriorTilt decodes (about 52 ms each on the laptop,
against 167 ms for relay-BP). 0.4% of shots also pay for relay-BP and one network call. The
worst shot costs more than relay-BP alone; the mean is lower.

### What is learned and what is not

| part | learned? |
|---|---|
| PriorTilt's schedule | fitted by its authors per noise level, on their circuits (not ours). We use it unchanged. |
| relay-BP | the kit's settings, unchanged |
| "the two seeds disagree" as the test for a hard shot | not learned |
| the "weight" rule | not learned |
| the "network" rule | the network's weights are learned, on shots drawn from the kit's error model at many noise levels. It never saw the shot files used here. The rule on top of its logits has no fitted number. |
| the constant C | fitted, when used. Off by default. |

### State, and the two modes

Both decoders keep state from one decode to the next: relay-BP its random memory strengths,
PriorTilt its random stream (measured here: shot 14 of `validation_X.npz` takes 4,654
iterations on a fresh decoder and 6,444 after shots 0 to 13). So an answer depends on which
shots the decoder saw before. Two ways of feeding the shots are used, and they differ shot
for shot:

- **chunk mode** (`candidates.py`, the default; all numbers measured on the laptop). The file
  is cut into chunks of 500 shots in file order. Each chunk gets fresh decoders: relay-BP,
  PriorTilt with seed 1 + chunk index, PriorTilt with seed 1001 + chunk index. All three decode
  every shot. This runs in parallel and gives the same answers on every machine.
- **session mode** (the plugin, and `candidates.py --mode session`). Decoders are built once
  per basis, PriorTilt with seeds 1 and 1001, and decode the shots in file order. Relay-BP
  sees only the shots where the two seeds differ. This is what the evaluator times.

The two modes are the same decoder with different random draws. The test-set number will come
from session mode, through the evaluator.

## Dependencies and licences

- The kit's `requirements.txt` (stim, ldpc, numpy, scipy, relay-bp).
- torch, for the "network" rule only. The CPU build is enough.
- **PriorTilt** (Min Ye and Nicolas Delfosse, IonQ, arXiv:2611.13373),
  https://github.com/ionq-publications/PriorTiltDecoder at commit
  `8eefd37d694b5942fe60ec6d5c82ce97f3d2a450`. **Its licence is CC BY-NC-SA 4.0
  (non-commercial, share-alike). None of its code is in this repository.** It is installed
  from a clone: its Rust core becomes the module `priortilt_rs`, and its tuned schedule is
  read at run time from the clone's `priortilt/params.py`. The core is given the kit's check
  matrix and priors, not PriorTilt's own circuits.

```sh
git clone https://github.com/ionq-publications/PriorTiltDecoder.git ~/PriorTiltDecoder
git -C ~/PriorTiltDecoder checkout 8eefd37d694b5942fe60ec6d5c82ce97f3d2a450
python3.12 -m venv ~/hierarchy-venv
~/hierarchy-venv/bin/pip install -r requirements.txt pytest ~/PriorTiltDecoder/decoder    # needs a Rust toolchain (cargo), like relay-bp
~/hierarchy-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # for the network rule
export PRIORTILT_REPO=~/PriorTiltDecoder
export PYTHONDONTWRITEBYTECODE=1
```

`hierarchy/requirements.txt` lists the same. `candidates.py`, `select.py` and `compare.py`
run without torch.

## Commands

### On another machine: decode shot files with N processes

```sh
python -m hierarchy.candidates --data data/p0.005 --out runs/cand_p0.005 --processes 12
```

- `--data` takes shot files of `make_dataset.py`, or folders. In a folder, files named
  `test_*` or `go_*` are left alone. Named directly, they are refused without `--final`.
- Use one output folder per noise level. Each finished chunk is saved under
  `<out>/parts/`; the same command continues a stopped run. `--chunks a:b` decodes only some
  chunks now.
- Per shot file it writes `<out>/<name>.chunks.npz` and prints two tables: the three
  decoders, and the rules that need no network. The file's arrays are listed at the top of
  `candidates.py`.
- The answers do not depend on `--processes`. Only the times do.

### On another machine: the plugin through the team's evaluator

```sh
echo '{"rule": "network", "checkpoints": {"X": "runs/checkpoint_step30000.pt"}, "device": "cpu", "torch_threads": 1}' > runs/hierarchy_options.json
python -m evaluation.evaluate --test-x <X shots> --test-z <Z shots> \
    --decoder hierarchy.plugin:make_decoder --decoder relay-bp \
    --plugin-options runs/hierarchy_options.json --out runs/<new folder>
```

Options, all optional: `rule` ("network" or "weight"), `checkpoints` (one per basis),
`seeds` (default `[1, 1001]`), `constant`, `device` ("cpu", "cuda" or "auto"), `torch_threads`,
`priortilt_repo`. A basis without a checkpoint uses the "weight" rule, and the settings saved
in `summary.json` say so (`rule`, `rule_asked`, `rule_note`). The evaluator's timer wraps the
whole `decode_shot`: both PriorTilt decodes, relay-BP, the network and the rule. The metrics
of a shot say which stage answered (`stage`), whether the two seeds agreed (`agreed`), which
candidate won (`winner`), and each decoder's convergence, iterations, weight and time; on a
shot where the seeds differ also the three predictions and the network's logits.

### Score the rules offline

```sh
python -m hierarchy.select --candidates <out>/validation_X.chunks.npz --logits <prefix>.npz
```

`--logits` is the file that `python -m neural.evaluate_stim --checkpoint <ckpt> --data <shots> --save <prefix>`
writes for the same shots. Several `--candidates` files (with as many `--logits` files) are
scored as one set. Without `--logits` the network rows are left out.

A constant C is fitted, so it is an input: `--constant 30` scores a C that was chosen
elsewhere; `--fit-candidates <train candidates> --fit-logits <train logits> --constants 10 30 100`
chooses C on other shots and scores it on these. If the fit shots are the scored shots, the
output says so in capitals.

### Compare two runs, and compare the plugin with the offline rules

```sh
python -m hierarchy.compare <one>.npz <other>.npz
python -m hierarchy.candidates --data <X shots> <Z shots> --out <out> --mode session --processes 2
python -m hierarchy.select --candidates <out>/<name>.session.npz --logits <prefix>.npz \
    --evaluator-shots runs/<folder>/01_hierarchy_plugin_make_decoder_shots.npz
```

The last command prints how many shots the evaluator's saved predictions differ from the
rule scored offline on the same shots decoded the same way. It must print 0.

## Numbers

Every number below is from exploration shots (the committed `train_*` and `validation_*`
files of `data/p0.004/`). No `test_*` or `go_*` file was opened by anything in this folder.
Machines never share a table.

### Measured on the laptop, exploration shots

X memory, p = 0.004, `train_X.npz` and `validation_X.npz` together (25,000 shots), chunk
mode. The decoding ran on the laptop (Ryzen 9 8945HS, Windows 10; 1 to 5 decode processes
beside a GPU training run, so compare the times with each other only). The per-shot files,
the laptop's scripts and its record are on the branch `laptop-artifacts` (the files below are
unchanged from commit `0ef3ba0950eb5497c66769d2688016b0d6fbf591` to `d579dba`), under
`priortilt/` and `cascade_X_s1/`. The network is `cascade_X_s1/checkpoint_step30000.pt`
(sha256 `d1f54ea1...0aecb9`, width 128, depth 8, reads all 1,728 detectors). Its time is the
batch-one mean on the laptop's GPU, 18.8 ms.

The table is printed by `select.py` of this folder from the laptop's files, on this PC:

```sh
A=<a checkout of laptop-artifacts>
python -m hierarchy.select \
    --candidates $A/priortilt/results_8b/selector_validation_X.npz $A/priortilt/results_8b/selector_train_X.npz \
    --logits $A/cascade_X_s1/eval/step30000_validation_X_p0.004.npz $A/cascade_X_s1/eval/step30000_train_X_p0.004.npz \
    --constant 10 --constant 30 --constant 100
```

| Rule | Failures of 25,000 | Rate, 95% Wilson interval | Right pick when one existed | Mean time per shot |
|---|---|---|---|---|
| R: relay-BP, kit settings | 122 | 0.488% [0.409%, 0.582%] | 62 of 143 | 166.7 ms |
| P1: PriorTilt, seed 1 + chunk | 102 | 0.408% [0.336%, 0.495%] | 82 of 143 | 52.0 ms |
| P2: PriorTilt, seed 1001 + chunk | 82 | 0.328% [0.264%, 0.407%] | 102 of 143 | 52.4 ms |
| P1; where P2 differs, the lowest weight of R, P1, P2 (**the "weight" rule**) | 77 | 0.308% [0.247%, 0.385%] | 107 of 143 | 118.0 ms |
| the lowest weight among the converged of R, P1, P2 (all three on every shot) | 68 | 0.272% [0.215%, 0.345%] | 116 of 143 | 271.1 ms |
| P1; where P2 differs, the network's most likely (**the "network" rule**) | **66** | 0.264% [0.208%, 0.336%] | 118 of 143 | **118.1 ms** |
| the same with network minus weight / C, for C = 10, 30, 100 | 63, 64, 65 | | 121, 120, 119 of 143 | 118.1 ms |
| a perfect choice among R, P1, P2 (needs the truth) | 41 | 0.164% [0.121%, 0.222%] | 143 of 143 | |
| P1; where P2 differs, a perfect choice (needs the truth) | 50 | 0.200% [0.152%, 0.264%] | 134 of 143 | |

These equal the laptop's own table (`priortilt/results_network_select_step30000.txt`): 122,
102, 82, 77, 68, 66, and 41.

How to read it:

- **Paired with relay-BP on the same shots.** The network rule and relay-BP both fail on 49
  shots; the rule alone fails on 17, relay-BP alone on 73 (exact two-sided binomial test on
  the 90 discordant shots: p = 2e-9). The weight rule: both 61, rule alone 16, relay-BP alone
  61 (p = 2e-7).
- **Is the network better than the weight rule?** 66 against 77. Paired: both fail 58, the
  network rule alone 8, the weight rule alone 19 (p = 0.052). With the later checkpoint of
  the same run (step 40,000): 65 against 77, alone 9 against 21 (p = 0.043). The gain is
  there, and it is small next to the gain of the hierarchy itself.
- **"Right pick when one existed"** counts the 143 shots where R, P1 and P2 are not all equal
  and at least one is right. P1 and P2 differ on 99 shots; on the other shots of the 143 only
  relay-BP differs, and the hierarchy never runs it there.
- **The floor.** With a perfect tie-breaker this hierarchy would fail 50 times, not 41: the
  41 needs relay-BP on every shot. The network rule is at 66.
- **The C rows are fitted on these same shots** and are not results. See the next table.
- No candidate converged on 11 of the 25,000 shots, and every rule fails on all 11.

**The constant, fitted on `train_X` and reported on `validation_X`** (5,000 shots; the same
command with `--candidates` and `--logits` of validation_X alone, and
`--fit-candidates $A/.../selector_train_X.npz --fit-logits $A/.../step30000_train_X_p0.004.npz --constants 10 30 100`).
On the 20,000 train shots C = 10, 30, 100 give 46, 47, 48 failures, so C = 10 is chosen:

| Rule, on validation_X alone | Failures of 5,000 | Rate, 95% Wilson interval |
|---|---|---|
| R: relay-BP | 31 | 0.620% [0.437%, 0.879%] |
| P1 / P2 | 26 / 18 | |
| the "weight" rule | 18 | 0.360% [0.228%, 0.568%] |
| the "network" rule | 17 | 0.340% [0.212%, 0.544%] |
| the "network" rule with C = 10 fitted on train_X | 17 | 0.340% [0.212%, 0.544%] |

On 5,000 shots the constant changes nothing, so the plugin's default is no constant.

**Both bases, without a network** (the four selector files of `priortilt/results_8b/`,
50,000 shots; `--candidates` with all four, no `--logits`). No Z network existed when this
was written.

| Rule | Failures of 50,000 | Rate, 95% Wilson interval | Mean time per shot |
|---|---|---|---|
| R: relay-BP | 245 | 0.490% [0.432%, 0.555%] | 168.8 ms |
| P1 / P2 | 208 / 189 | | 53.8 / 54.7 ms |
| the "weight" rule | 162 | 0.324% [0.278%, 0.378%] | 124.3 ms |

Paired with relay-BP: both fail 134, the weight rule alone 28, relay-BP alone 111
(p = 7e-13). On `validation_Z.npz` alone (5,000 shots): relay-BP 20, the weight rule 15.

sha256 of the inputs (the data files they were made from are named inside them):

| file on `laptop-artifacts` | sha256 |
|---|---|
| `priortilt/results_8b/selector_validation_X.npz` | `60db8f1088bc80536c0d9b353720eb9511ba84cc1fdc3f657bc046728b229b08` |
| `priortilt/results_8b/selector_train_X.npz` | `d23883caac76135e848dc1280fb7c669602d666d05d58c979d642f783fa350bf` |
| `priortilt/results_8b/selector_validation_Z.npz` | `6d490aec0de6b20562dead9933afd1a817895afedf839617e7ed3ee6b72ca253` |
| `priortilt/results_8b/selector_train_Z.npz` | `f2059d31463260a4a1d2036e41d4aca00b5c23f0461c3477558ffe4caad69024` |
| `cascade_X_s1/eval/step30000_validation_X_p0.004.npz` | `b5d3c9e63794b718f641b97d9d1e3f7da71d1c4d0d3eea782a4cd7a5da0b77f1` |
| `cascade_X_s1/eval/step30000_train_X_p0.004.npz` | `4d6b13e840b388d689904f7f87c756315c2a0e11834441930e2d31f3b4f19346` |
| `cascade_X_s1/eval/step40000_validation_X_p0.004.npz` | `f4287f44c82603be463e778258b45265941304031bcedae03638103a16e9ec83` |
| `cascade_X_s1/eval/step40000_train_X_p0.004.npz` | `8cac44c51934e114a486769e96f4338d97366317af307ea6e761789d42f69e33` |
| data: `data/p0.004/validation_X.npz` (Stim seed 201) | `b00ee09c35ed088a12bcba60b8b704b528c47fb156c7d09cf0f171f90a181f3d` |
| data: `data/p0.004/train_X.npz` (Stim seed 101) | `98c0afeb2a0b6b27f3b689e832160a060a4c0aeb37d229f84daa1995a9a0f7cc` |
| data: `data/p0.004/validation_Z.npz` (Stim seed 202) | `843c4daf24e2b56c5988670550bbcdbac8f50b18dbb9d24beb64969146cb8e20` |
| data: `data/p0.004/train_Z.npz` (Stim seed 102) | `14dce11586fa51256f476224d0fe3965e0562f2d8f68cba6192883a65fb6b24a` |

### This PC (Ryzen 7 7730U, Linux; lane A, niced, shared with other work)

Only smoke sizes ran here; this PC was about four times slower per iteration than the
laptop while they ran, so no time from it is quoted for the decoders.

- **The repository code reproduces the laptop's decoding.** The first 20 shots of
  `validation_X.npz`, decoded here by `candidates.py` (`--chunk-size 20 --chunks 0:1`, so
  seeds 1 and 1001 on fresh decoders, as in the laptop's chunk 0), equal the laptop's
  per-shot file for R, P1 and P2 in every field: predicted bits, convergence, iteration
  counts, fault counts and weights (`python -m hierarchy.compare`: 0 differences). Three of
  those shots run 3,000 to 18,000 iterations through PriorTilt's random stage, so the Linux
  build draws the same random numbers as the laptop's Windows build.
- **1 process and 3 processes** give identical files on a 60-shot slice, and so does a plain
  export of the commit with a freshly built environment.
- **The network on a CPU.** The step-30,000 checkpoint loaded through the plugin gives, on
  120 validation shots, logits within 4.2e-5 of the laptop's GPU logits (the logits are of
  size about 4.5), with no sign change. Batch-one inference: 48 ms mean with 2 torch threads.

### The server (AMD EPYC-Milan, Linux), busy with other jobs

`candidates.py` and the plugin ran here on full files, from `integration` at 4e955b4, in a
conda environment with PriorTilt built from the same commit. Ten other processes ran beside
them, so its times are inflated and are not results.

**`candidates.py` on `validation_X.npz`, chunk mode, 2 processes**
(`python -m hierarchy.candidates --data data/p0.004/validation_X.npz --out runs/hier/cand_p0.004 --processes 2`;
output `files/hier/cand_p0.004/validation_X.chunks.npz` on the branch `server-logs`, sha256
`847871867ce9ee240d67760cefe17715b280e28a9eb29d504caa7d1cae4b22c7`; printout `hier.out`):

| Decoder | Failures of 5,000 | Rate, 95% Wilson interval | Unconverged | Mean iterations | Mean time per shot, busy server |
|---|---|---|---|---|---|
| R: relay-BP | 31 | 0.620% [0.437%, 0.879%] | 17 | 565.0 | 186.9 ms |
| P1: PriorTilt, seed 1 + chunk | 26 | 0.520% [0.355%, 0.761%] | 10 | 391.5 | 49.8 ms |
| P2: PriorTilt, seed 1001 + chunk | 18 | 0.360% [0.228%, 0.568%] | 6 | 399.5 | 50.7 ms |

`python -m hierarchy.compare` of that file against the laptop's `selector_validation_X.npz`:
the same data file, and on all 5,000 shots R, P1 and P2 have the same predicted bits,
convergence, iteration counts, fault counts and weights (0 shots differ; largest weight gap 0).
So three builds, the Windows laptop, this PC and the server, decode alike, and
relay-BP's 31 failures on this file are now measured on three machines.

**The plugin through `evaluation.evaluate` on `validation_X.npz` and `validation_Z.npz`**
(a plumbing check on validation shots, network rule for X, weight rule for Z; the plugin's
predictions must equal `select.py` on the session-mode candidates of the same shots, on every
shot): not measured at the time of writing.

## The ensemble (part 2)

**This part is combined offline from per-shot files. Its time is a sum of stage times taken
from those files, possibly from different machines. It is not one decoder timed in one
process, as part 1's row from the team's evaluator is.**

**Why.** The teammates run NVIDIA's TCBP-JOBS (a GPU decoder, `../tcbp_jobs_baseline.py`) as
a reference. On our exploration shots at p = 0.004 it fails less often than part 1's decoder
(numbers below). Three candidates cannot pass it: part 1's floor with a perfect tie-breaker
is 50 failures in 25,000 X shots. What can is more members whose mistakes differ, and a good
choice among them. The claim under test: **the ensemble fails fewer shots than TCBP-JOBS
alone, on the same shots.**

**The members.**

| name | decoder | runs on | made by |
|---|---|---|---|
| P1, P2 | PriorTilt under two seeds | every shot | `candidates.py` |
| T | TCBP-JOBS, preset `tcbp-jobs` | every shot | `tcbp_member.py` |
| R | relay-BP, the kit's settings | the shots a rule does not accept at once | `candidates.py` |
| J | joint relay-BP: the kit's relay-BP on all 1,728 detectors and all about 68,000 mechanisms, a fresh decoder per shot (gate G2's "joint") | the shots where P1, P2 and T are not unanimous | `joint_member.py` |
| the network | `neural/`, one per basis | the same shots | `neural.evaluate_stim --save` |

**The rules** (`ensemble.py`; none has a fitted number):

| rule | accepts at once | otherwise |
|---|---|---|
| a | P1 = P2 | the network picks among R, P1, P2 (part 1's rule) |
| aw | P1 = P2 | the lowest weight of R, P1, P2 (part 1 without learning) |
| b | P1 = P2 = T | the network picks among P1, P2, T, R |
| bw | P1 = P2 = T | the lowest weight of P1, P2, T, R (no learning) |
| c | P1 = P2 = T | the network picks among P1, P2, T, R and, where it exists, J |
| d | two of P1, P2, T agree (the majority answers) | the network picks among P1, P2, T, R |
| e | two of P1, P2, T agree | the network picks among P1, P2, T, R and, where it exists, J |

"The network picks" is part 1's score, among the candidates that decoded the shot. What is
learned: the network's weights, as in part 1. The rule is chosen on the `train_*` files and
reported on the `validation_*` files; the tool refuses to mark a rule chosen on the file it
scores (`--chosen-on`).

**Commands.** `tcbp_member.py` runs in the teammates' TCBP environment (Linux, an NVIDIA
GPU, `pip install cudaq-qec`; it imports neither torch nor PriorTilt). Everything else runs
in the environment of part 1.

```sh
# 1. every member that decodes every shot
python -m hierarchy.candidates --data data/p0.004 --out runs/cand_p0.004 --processes 12
<TCBP python> -m hierarchy.tcbp_member --data data/p0.004 --out runs/tcbp_p0.004 --batch      # serial (no --batch) for a time per shot
python -m neural.evaluate_stim --checkpoint <X checkpoint> --data data/p0.004/validation_X.npz --save runs/net/validation_X

# 2. per shot file: the shots where P1, P2 and T are not unanimous, then joint relay-BP on them
python -m hierarchy.ensemble --members runs/cand_p0.004/validation_X.chunks.npz runs/tcbp_p0.004/validation_X.tcbp.npz \
    --write-selection runs/sel_p0.004/validation_X.selection.npz
python -m hierarchy.joint_member --data data/p0.004/validation_X.npz \
    --selection runs/sel_p0.004/validation_X.selection.npz --out runs/joint_p0.004 --processes 12

# 3. the tables: the training file first, then the validation file with the rule the training file chooses
python -m hierarchy.ensemble --members runs/cand_p0.004/train_X.chunks.npz runs/tcbp_p0.004/train_X.tcbp.npz \
    runs/joint_p0.004/train_X.joint.npz --logits runs/net/train_X.npz --save runs/ens/train_X.json
python -m hierarchy.ensemble --members runs/cand_p0.004/validation_X.chunks.npz runs/tcbp_p0.004/validation_X.tcbp.npz \
    runs/joint_p0.004/validation_X.joint.npz --logits runs/net/validation_X.npz --chosen-on runs/ens/train_X.json runs/ens/train_Z.json
```

`tcbp_member.py --decoder relay` is a plumbing mode without a GPU: the kit's relay-BP stands
in behind the same two calls, and the output is named `*.relay-plumbing.npz`. A file named
`test_*` or `go_*` needs `--final` in `candidates.py`, `tcbp_member.py`, `joint_member.py`
and `neural.evaluate_stim`.

**Numbers.** Not measured at the time of writing: the members' files are being made on the
server.

## Physics checks

- **Every decoder gets the kit's problem.** `stages.kit_problem` builds H (936 x 8,784),
  L (12 x 8,784) and the priors with the kit's `dem_matrices(load_circuit(GROSS_CIRCUITS[basis], p), basis)`
  from the basis and p stored in the shot file, and refuses a file whose `basis_detectors`
  are not that circuit's. In the plugin the problem is the one the evaluator hands over. Both
  record one sha256 of H, L and the priors, so two machines can show they decoded the same
  problem.
- **Failure.** A shot fails when any of the 12 predicted observable bits (L e mod 2) differs
  from the file's. An unconverged candidate is scored on what it returned.
- **Paired comparison.** `select.py` compares every rule with relay-BP on the same shots and
  prints both-fail, rule-only and relay-only counts with an exact two-sided binomial test.
- **The network's input** is the shot's detector bits and nothing else (tested against a
  direct call of the model). The truth is read by `candidates.py` and `select.py` for scoring
  only, and never reaches a decoder or the plugin.
- **A network on other logical operators.** `neural/` trains Z memory with `--logicals light`:
  12 lighter logical operators instead of the kit's. Its logits then belong to those
  operators. The rule turns each candidate's 12 kit bits into the network's operators before
  scoring (`rules.bits_for_network`), with `neural/`'s own `kit_bits` and matrix. A test
  checks that turning them back with `kit_bits` gives the candidate again.
- **Times** state the machine and what the timer wraps. In `candidates.py` the timer wraps
  one decode call, as in the kit's `decoders.score`; a rule's time in `select.py` is the sum
  of the calls it makes. In the evaluator the timer wraps the whole `decode_shot`.

## Tests

```sh
python -m pytest -p no:cacheprovider hierarchy
```

23 tests, about a minute and a half on one loaded CPU core. They cover: the rules against
hand-made cases; `select.py` on five hand-made shots whose outcome under every rule is known,
on a session-mode file, with a fitted constant (and the warning when it is fitted on the
scored shots) and with logits of other shots (refused); the refusal of `test_*` and `go_*`
files; `candidates.py` with 1 and with 4 processes on a small file (every array equal except
the times), a stopped run that continues, and a folder with other settings (refused); the
plugin with stub decoders and a stub network that prefers each candidate in turn; the
network's input; and the whole chain on 24 shots per basis: the plugin through the real
`evaluation.evaluate`, against `select.py` on session-mode candidates, for X with a network
and Z without, and for Z with a `--logicals light` network.

The decoding tests need PriorTilt and are skipped without it; the network tests need torch.
They use the first 24 shots of the committed `train_*` files and small iteration budgets
(`--priortilt-max-rounds`, `--relay-sets`: for tests only, never for a number).

## Next

Not done here, on purpose: this task builds what was measured, and adds no decoder idea.

1. **The one run on the test shots**, through the evaluator, after every choice is frozen.
2. **The other two noise levels.** `candidates.py` on the p = 0.003 and p = 0.005 files, then
   `select.py` with the network's logits there. PriorTilt ships a schedule for each.
3. **A Z network.** The plugin takes a checkpoint per basis, also one trained with
   `--logicals light`. Until one exists, Z uses the weight rule.
4. **A stage for the shots where no candidate converged** (11 of 25,000 X shots, all of them
   failures under every rule). `plugin.py` marks the place, and the metrics flag those shots
   (`any_converged`). Candidates: a larger budget, joint decoding of both check types.
5. **Run relay-BP more often.** The floor with relay-BP only on disagreement is 50 failures;
   with relay-BP on every shot it is 41. A cheap test for "P1 and P2 agree but are wrong"
   would be needed.
6. **More seeds, or a majority,** as further candidates for the network to choose from.
7. **A constant C** (network minus weight / C) is implemented and off: on validation shots it
   changed nothing. Worth another look with a better network.
8. **Per-shot seeds.** Seeding PriorTilt from the shot would make an answer independent of the
   shot order, so chunk mode and session mode would agree shot for shot.
9. **Tail latency.** A hard shot pays for three decoders one after the other; they could run
   side by side.
