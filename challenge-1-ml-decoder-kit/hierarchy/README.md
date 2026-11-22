# hierarchy: fast decoders first, slower ones only where they disagree

Two things live here, for the required comparison with BP-OSD and relay-BP.

**Part 1, the plugin: a hierarchy with a tie-breaker.** It runs through the team's evaluator,
one shot at a time, in one process.

1. A fast decoder (PriorTilt) decodes the shot twice, under two random seeds.
2. If the two answers agree, that is the answer. This happens on about 99.6% of shots at p = 0.004.
3. If they differ, a third decoder (relay-BP) decodes the shot too, and a rule chooses among
   the three answers: the neural network of `neural/` ("network"), or a rule without
   learning ("converged", "weight").

**Part 2, the ensemble: more members, combined offline.** It adds TCBP-JOBS (the teammates'
GPU decoder) as a third always-on member and joint relay-BP as the arbiter where the three
disagree. It is combined from per-shot files; it is not one timed decoder.

Where things stand at p = 0.004, on the 10,000 validation shots (5,000 X, 5,000 Z; no pair
has both shots failing, so failures and failed pairs are the same number). **These are not
test-set numbers.**

| decoder | failures [X Z] | failed pairs of 5,000 | rate per cycle (the kit's) |
|---|---|---|---|
| relay-BP, kit settings | [31 20] | 51 | 8.5e-4 +- 1.2e-4 |
| TCBP-JOBS alone | [16 9] | 25 | 4.2e-4 +- 0.8e-4 |
| part 1's rule with the network, scored offline | [14 16] | 30 | 5.0e-4 +- 0.9e-4 |
| part 1's plugin through the evaluator (X network, Z weight rule) | [20 17] | 37 | 6.2e-4 +- 1.0e-4 |
| **the ensemble, rule f** | **[8 3]** | **11** | **1.8e-4 +- 0.6e-4** |

What the numbers support, and what they do not:

- The hierarchy fails clearly less often than relay-BP, in less mean time (paired tests below).
- The ensemble fails less often than TCBP-JOBS alone on the same shots (11 against 25; paired
  p = 0.008 on X and 0.03 on Z). The rule that does it uses joint decoding and no network.
- **The network's own gain is not established.** As tie-breaker it beats the plain
  lowest-weight rule (62 against 77 failures on 25,000 X shots, paired p = 0.006), but against
  the best rule without learning ("converged") it is level (62 against 67, p = 0.40), and the
  young Z network is behind it (83 against 73). The test run prints this comparison, paired.

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
| `plugin_options.py` | writes the plugin's options file: the network rule per basis only if it is no worse than the best rule without learning |
| `priortilt_plugin.py` | a reference row for the evaluator: PriorTilt once |
| `preflight.py` | runs every tool on a few validation shots; a run on held-out shots starts only if it passes |
| `parallel_check.py` | checks the plugin's side-by-side mode against its sequential mode, shot for shot |
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
- *converged* is the same among the candidates that converged, that is, whose correction
  really reproduces the syndrome (among all three if none did). An unconverged decoder
  returns a guess, and its weight means little. This is the best rule without learning on
  the exploration shots.
- *network* asks the neural network of `neural/`, which reads the shot's detector bits and
  gives 12 logits, one per logical bit. A candidate's score is the sum over the 12 bits of
  `log sigmoid(logit)` where the candidate says 1 and `log sigmoid(-logit)` where it says 0:
  the log-probability the network gives to that candidate's 12 bits. The highest score wins.
  With an optional constant C the score is that sum minus weight / C.

**Why the network is only a tie-breaker.** Alone, the network (step 60,000 of a run that is
still training) gets 12% of these shots wrong, against 0.5% for relay-BP. But it does not
need to decode. It only has to say which of two or three given answers looks more likely,
on 0.4% of the shots. It does that better than the plain weight rule, and so far about as
well as the "converged" rule (numbers below).

**Which rule the plugin uses is decided by a frozen procedure, not by hand.** Per basis:
"network" if, with the final checkpoint, the network rule has no more failures than
"converged" on the p = 0.004 train and validation shots of that basis; otherwise
"converged". `plugin_options.py` applies it and writes the options file.

**The cost.** Every shot pays for two PriorTilt decodes (about 52 ms each on the laptop,
against 167 ms for relay-BP). 0.4% of shots also pay for relay-BP and one network call. The
worst shot costs more than relay-BP alone; the mean is lower.

### What is learned and what is not

| part | learned? |
|---|---|
| PriorTilt's schedule | fitted by its authors per noise level, on their circuits (not ours). We use it unchanged. |
| relay-BP | the kit's settings, unchanged |
| "the two seeds disagree" as the test for a hard shot | not learned |
| the "weight" and "converged" rules | not learned |
| the "network" rule | the network's weights are learned, on shots drawn from the kit's error model at many noise levels. It never saw the shot files used here. The rule on top of its logits has no fitted number. |
| the constant C | fitted, when used. Off by default. |
| which rule a basis uses | chosen by the procedure above on exploration shots, before any held-out shot is opened |

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

Options, all optional: `rule` ("network", "weight" or "converged", or one per basis as
`{"X": "network", "Z": "converged"}`), `checkpoints` (one per basis), `seeds` (default
`[1, 1001]`), `constant`, `device` ("cpu", "cuda" or "auto"), `torch_threads`, `parallel`,
`priortilt_repo`. A basis asked for "network" without a checkpoint uses the "weight" rule, and
the settings saved in `summary.json` say so (`rule`, `rule_asked`, `rule_note`). The
evaluator's timer wraps the whole `decode_shot`: both PriorTilt decodes, relay-BP, the network
and the rule. The metrics of a shot say which stage answered (`stage`), whether the two seeds
agreed (`agreed`), which candidate won (`winner`), and each decoder's convergence,
iterations, weight and time; on a shot where the seeds differ also the three predictions and
the network's logits.

**The options file is written by a command**, from the checkpoints and their logits on the
p = 0.004 exploration shots (the frozen procedure of "The design"):

```sh
python -m hierarchy.plugin_options --out runs/hierarchy_options.json \
    --x-checkpoint <X checkpoint> --x-candidates <validation_X candidates> <train_X candidates> --x-logits <their logits, same order> \
    --z-checkpoint <Z checkpoint> --z-candidates <validation_Z candidates> <train_Z candidates> --z-logits <their logits, same order>
```

It prints one line per basis, `PLUGIN_RULE X network    (network 62 failures, converged 67, on 25000 shots)`.

**Side by side.** With `"parallel": true` the two PriorTilt decodes of a shot run at the same
time: the second one lives in a helper process that is started with the session and sees
every shot in order. The answers are those of the sequential mode (`parallel_check.py`
compares them shot for shot). The session then uses two cores, and its time per shot is wall
time, not CPU time. It is off by default, and the sequential row is the main one.

**Reference rows through the same evaluator:** `--decoder hierarchy.priortilt_plugin:make_decoder`
is PriorTilt once (the same options file); `{"rule": "converged"}` and `{"rule": "weight"}`
are the hierarchy without learning.

### Before a run on held-out shots

```sh
python -m hierarchy.preflight --options runs/hierarchy_options.json --x data/p0.004/validation_X.npz \
    --z data/p0.004/validation_Z.npz --rule f --tcbp-python <TCBP python> --out runs/preflight_<time>
```

It runs the plugin (made as the evaluator makes it, with that options file), `candidates.py`,
`neural.evaluate_stim`, `tcbp_member.py` (in the TCBP Python), `joint_member.py` and
`ensemble.py` on the first 16 shots of two validation files, and ends with `PREFLIGHT PASSED`
and status 0 only if every step worked. It refuses `test_*` and `go_*` files.

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
- **The side-by-side mode equals the sequential one.** `python -m hierarchy.parallel_check
  --x data/p0.004/validation_X.npz --z data/p0.004/validation_Z.npz --shots 200 --options <file>`
  (rule "network" with the step-60,000 checkpoint for X, "converged" for Z; the paper's and the
  kit's full budgets): on the first 200 shots of each file, 0 shots differ in predicted bits,
  `agreed`, `stage`, `winner`, weights or iteration counts. Only one of those 400 shots
  reaches relay-BP and the rule, so the escalation path is compared mostly by the unit test
  (24 shots per basis with small budgets, many of them escalated; also 0 differences). Mean
  wall time per shot here, both threads of two shared cores busy: X 460 ms sequential against
  285 ms side by side, Z 134 ms against 91 ms. The gain is below a factor of two because a
  shot waits for the slower of its two decodes.
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
(a plumbing check on VALIDATION shots, not a result; options: rule "network", the X
checkpoint of step 30,000 on the CPU with one thread, no Z checkpoint, so Z used the weight
rule; single process; files under `files/hier/` on `server-logs`):

```sh
python -m evaluation.evaluate --test-x data/p0.004/validation_X.npz --test-z data/p0.004/validation_Z.npz \
    --decoder hierarchy.plugin:make_decoder --plugin-options runs/hier/options.json --out runs/hier/plumbing_p0.004
python -m hierarchy.candidates --data data/p0.004/validation_X.npz data/p0.004/validation_Z.npz --out runs/hier/cand_session_p0.004 --mode session --processes 2
python -m neural.evaluate_stim --checkpoint <the same checkpoint> --data data/p0.004/validation_X.npz --device cpu --threads 1 --save runs/hier/net30000_validation_X_cpu
python -m hierarchy.select --candidates runs/hier/cand_session_p0.004/validation_X.session.npz --logits runs/hier/net30000_validation_X_cpu.npz \
    --evaluator-shots runs/hier/plumbing_p0.004/01_hierarchy_plugin_make_decoder_shots.npz
python -m hierarchy.select --candidates runs/hier/cand_session_p0.004/validation_Z.session.npz \
    --evaluator-shots runs/hier/plumbing_p0.004/01_hierarchy_plugin_make_decoder_shots.npz
```

| | X (network rule) | Z (weight rule) |
|---|---|---|
| failures of 5,000, the plugin through the evaluator | 20 | 17 |
| failures of 5,000, `select.py` on the session-mode candidates and the CPU logits | 20 | 17 |
| shots where the two answers differ | 0 | 0 |
| shots where the two seeds differ (relay-BP and the rule run) | 29 | 20 |

The evaluator's summary of that run: failed pairs 37 of 5,000, 6.19e-4 +- 1.02e-4 per cycle;
relay-BP and the rule ran on 0.49% of the shots; mean time per pair 235 ms on the busy server
(inflated, not a result). The session-mode counts (20 and 17) differ from the chunk-mode
counts of the same rules on the same shots (17 and 15): same decoders, other random draws.

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
| ac | P1 = P2 | the lowest weight among the converged of R, P1, P2 (part 1's best rule without learning) |
| b | P1 = P2 = T | the network picks among P1, P2, T, R |
| bw | P1 = P2 = T | the lowest weight of P1, P2, T, R (no learning) |
| c | P1 = P2 = T | the network picks among P1, P2, T, R and, where it exists, J |
| d | two of P1, P2, T agree (the majority answers) | the network picks among P1, P2, T, R |
| e | two of P1, P2, T agree | the network picks among P1, P2, T, R and, where it exists, J |
| **f** | P1 = P2 = T | **J's answer where J converged; T's answer where it did not** (no learning) |
| h | P1 = P2 = T | J's answer where J converged; where it did not, the network picks among T, P1, P2, R |
| g | P1 = P2 = T | the answer most of J, T, P1, P2, R give; a tie goes to the network |
| gw | P1 = P2 = T | the same, a tie going to the first of J, T, P1, P2, R in it (no learning) |

"The network picks" is part 1's score, among the candidates that decoded the shot. What is
learned: the network's weights, as in part 1. The rule is chosen on the `train_*` files and
reported on the `validation_*` files; the tool refuses to mark a rule chosen on the file it
scores (`--chosen-on`).

**The frozen rule is f.** In words: P1, P2 and TCBP-JOBS decode every shot. If all three give
the same 12 bits, that is the answer. If not (0.5% of shots at p = 0.004), joint relay-BP
decodes the shot; its answer is taken if its correction reproduces the syndrome, and
TCBP-JOBS's answer otherwise. Nothing in it is learned and nothing is fitted. Why it works:
on the shots where the three disagree, joint relay-BP is wrong only 13% of the time, and
nearly all of its wrong answers are flagged by non-convergence (27 of 32 on the 256 listed
shots; no unconverged answer was right).

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

`--no-joint` tells `ensemble.py` that joint decoding was not run for a file; the rules with
J are then scored with J on no shot. **At p = 0.005 joint decoding is not planned** (it is
slow there, and by the relay-BP authors' data worse than separate decoding), so rule f
returns T's answer on every shot there: the ensemble's row at p = 0.005 equals TCBP-JOBS
alone, and claims nothing.

### The ensemble's numbers at p = 0.004 (exploration shots)

The members' files, all made from the committed `data/p0.004/` files (sha256 above):

| member | files | made on | time per shot in the file |
|---|---|---|---|
| P1, P2, R | `priortilt/results_8b/selector_*.npz` on `laptop-artifacts` (chunk mode; equal to `candidates.py` shot for shot where compared) | the laptop | yes, beside a GPU training |
| T | `files/tcbp_p0.004/*.tcbp.npz` on `server-logs`; `tcbp_member.py --batch`, preset `tcbp-jobs`, cudaq-qec 0.9.0 | the server, NVIDIA A100 | no (batch mode) |
| J | `files/joint_p0.004/*.joint.npz` on `server-logs`; `joint_member.py` on the lists `files/sel_p0.004/*.selection.npz`, 9 processes | the server, AMD EPYC-Milan | yes: mean 24 to 29 s per listed shot, at most 69 s |
| X network | `cascade_X_s1/eval/step60000_*_X_p0.004.npz` on `laptop-artifacts` (step 60,000, the kit's operators) | the laptop's GPU | 17 to 18 ms per call |
| Z network | `files/netZ/netZ_*_Z_p0.004.npz` on `server-logs` (step 22,000, `--logicals light`) | the server's GPU | 16 to 17 ms per call |

Every correction TCBP-JOBS returned reproduces its syndrome, and decoding the first 256
shots of each file a second time changed none of its answers.

**The rule was chosen on the training files** (`train_X` and `train_Z`, 40,000 shots) by the
tool: fewest failures, ties to the earlier rule. Failures per rule, X + Z:

| | R | P1 | T | a | aw | ac | b | bw | c | d | e | **f** | h | g | gw | ORACLE: P1, P2, T, R | ORACLE with J |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train_X (20,000) | 91 | 76 | 37 | 48 | 59 | 51 | 42 | 44 | 41 | 41 | 41 | **20** | 22 | 32 | 29 | 23 | 15 |
| train_Z (20,000) | 103 | 86 | 51 | 67 | 70 | 61 | 58 | 61 | 54 | 55 | 52 | **27** | 26 | 42 | 39 | 29 | 19 |
| both (40,000) | 194 | 162 | 88 | 115 | 129 | 112 | 100 | 105 | 95 | 96 | 93 | **47** | 48 | 74 | 68 | 52 | 34 |

**Reported on the validation files** (5,000 shots each; `--chosen-on` the two training files
marks rule f):

| | failures X | failures Z | X + Z of 10,000 | rate, 95% Wilson interval (X / Z) | later stage runs on |
|---|---|---|---|---|---|
| R: relay-BP alone | 31 | 20 | 51 | 0.62% [0.44, 0.88] / 0.40% [0.26, 0.62] | |
| P1: PriorTilt once | 26 | 20 | 46 | | |
| T: TCBP-JOBS alone | 16 | 9 | 25 | 0.32% [0.20, 0.52] / 0.18% [0.10, 0.34] | |
| a: part 1 with the network | 14 | 16 | 30 | | 24 + 24 shots |
| ac: part 1, best rule without learning | 16 | 12 | 28 | | 24 + 24 |
| b: the network picks among P1, P2, T, R | 13 | 14 | 27 | | 27 + 27 |
| c: the same with J | 12 | 12 | 24 | | 27 + 27 |
| **f: J where it converged, else T** | **8** | **3** | **11** | 0.16% [0.08, 0.32] / 0.06% [0.02, 0.18] | 27 + 27 (0.54%) |
| h: f, with the network where J did not converge | 8 | 4 | 12 | | 27 + 27 |
| ORACLE among P1, P2, T, R, J (needs the truth) | 6 | 2 | 8 | | |

Paired on the same validation shots (both fail, the rule alone, the other alone; exact
two-sided binomial test on the discordant shots):

| rule f against | X | Z |
|---|---|---|
| TCBP-JOBS alone | 8, 0, 8; p = 0.008 | 3, 0, 6; p = 0.031 |
| relay-BP alone | 7, 1, 24; p = 2e-6 | 2, 1, 18; p = 8e-5 |

On the training shots: against TCBP-JOBS 19, 1, 18 (p = 8e-5) on X and 27, 0, 24 (p = 1e-7)
on Z. As pairs (shot k of X with shot k of Z, `ensemble.py --pair`), validation: relay-BP 51
failed pairs of 5,000 (8.5e-4 per cycle), TCBP-JOBS 25 (4.2e-4), rule f 11 (1.8e-4 +- 0.6e-4).
The kill line of this part (the ORACLE not below TCBP-JOBS) is not hit: 8 against 25.

**Time.** Not measured for the ensemble: TCBP-JOBS ran in batch mode here, without a time
per shot. What the stages cost, each on its own machine: P1 + P2 about 105 ms per shot (the
laptop); TCBP-JOBS 43 ms per pair in the teammates' own serial run
(`results/tcbp-jobs_p0.004_*.log`, other shots); joint relay-BP 24 to 29 s per listed shot on
the busy server, on 0.54% of the shots, so about 0.15 s per shot on average. The run on the
test shots times TCBP-JOBS serially on an idle GPU.

**The network's own increment** (each rule with the network against the same rule without
learning, paired; X network of step 60,000, Z network of step 22,000):

| | X: failures, only-network-fails / only-other-fails, p | Z |
|---|---|---|
| a against aw (plain lowest weight), 25,000 shots per basis | 62 against 77; 6 / 21; p = 0.006 | 83 against 85; 11 / 13; p = 0.84 |
| a against ac (lowest weight among the converged) | 62 against 67; 9 / 14; p = 0.40 | 83 against 73; 19 / 9; p = 0.087 |
| h against f, 25,000 shots per basis | 30 against 28; 4 / 2 | 30 against 30; 3 / 3 |

So the network beats the simplest rule on X and is not better than the best hand-written
rule on either basis yet. The plugin therefore takes the network per basis only by the
frozen procedure of part 1, with the final checkpoints. A larger held-out set for this
comparison (80,000 fresh X shots) is being decoded; see `HANDOFF.md`.

Commands (A = a checkout of `laptop-artifacts`, F = a checkout of `server-logs`'s `files/`):

```sh
for s in train validation; do for b in X Z; do
  python -m hierarchy.ensemble --members $A/priortilt/results_8b/selector_${s}_$b.npz $F/tcbp_p0.004/${s}_$b.tcbp.npz \
      $F/joint_p0.004/${s}_$b.joint.npz --logits <the basis' logits of that file> --save out/${s}_$b.json \
      $( [ $s = validation ] && echo --chosen-on out/train_X.json out/train_Z.json )
done; done
python -m hierarchy.ensemble --pair out/validation_X.npz out/validation_Z.npz
```

sha256 of the inputs that are not listed above:

| file | sha256 |
|---|---|
| `files/tcbp_p0.004/validation_X.tcbp.npz` | `7e8a1c6836d7394fe66d678f5894e7ff2824e71eacb8242c771cb3e1e08fab56` |
| `files/tcbp_p0.004/validation_Z.tcbp.npz` | `4821ea1814bf2e00888ddea17382586d9aa5ab5f5ad5db560b07e710a1368f12` |
| `files/tcbp_p0.004/train_X.tcbp.npz` | `a31a7910266bc65ff4fda840aea00a79e9ab16f8ba201443b8ec511c3a1d747e` |
| `files/tcbp_p0.004/train_Z.tcbp.npz` | `8b9b42235806d9d4d11d8c4580e99eba7ef8562a06241e75c825f44496565d23` |
| `files/joint_p0.004/validation_X.joint.npz` | `24f0e9563fe99ce3e8be8480399f2a85e6dcf86fe08fa1e390e4732e70c38c7f` |
| `files/joint_p0.004/validation_Z.joint.npz` | `a70b5ec4fda59bc348bde60894cfba20932638d84f8e8542cce2f8194dbca7f1` |
| `files/joint_p0.004/train_X.joint.npz` | `f8756681efe00b382fb6bff5d01e636c681b1d8f2a7c349dfb4e2201fbe296ec` |
| `files/joint_p0.004/train_Z.joint.npz` | `a8f92a91aee91e627b91b95b0f33cbdf849c1add22294a8f9467b5fb451ecff1` |
| `files/netZ/netZ_validation_Z_p0.004.npz` | `e681aa4c48c690bb46cb7c6f5005ae88313b535a5239de938eef6868c1fd5771` |
| `files/netZ/netZ_train_Z_p0.004.npz` | `a196570bea96398d088f917a6981bb4a8efd4f05dd2f3604b51acab2dce3203c` |
| `cascade_X_s1/eval/step60000_validation_X_p0.004.npz` | `0b1723c9b9612f134f86bf4467a2b96c94f52dbe6323fc751d7c8f5558adcbab` |
| `cascade_X_s1/eval/step60000_train_X_p0.004.npz` | `5b5901f9688231be4baa8deb8d4504ec7d6ee4f17251eedaf8abecb37f63c474` |

## The run on the held-out shots: frozen commit and exact commands

**Frozen commit: `FROZEN_SHA`** (the code; this README names it from the commit after).
What is frozen: the plugin and its rule procedure; the ensemble rule f; the reference rows
relay-BP, PriorTilt once and TCBP-JOBS alone; the ablation rows aw and ac; the commands below.
Nothing below was run on a `test_*` or `go_*` file by this task.

For each p in 0.003, 0.004, 0.005, with D = `data/p<p>` and O = `runs/test_p<p>`, from
`challenge-1-ml-decoder-kit/` at the frozen commit, `PRIORTILT_REPO` set:

```sh
# 0. once, before any test file: the plugin's rule per basis from the final checkpoints, then the preflight
python -m hierarchy.plugin_options --out runs/hierarchy_options.json --x-checkpoint <X final> --z-checkpoint <Z final> \
    --x-candidates <validation_X and train_X candidates at p = 0.004> --x-logits <their logits from the X final checkpoint> \
    --z-candidates <validation_Z and train_Z candidates at p = 0.004> --z-logits <their logits from the Z final checkpoint>
python -m hierarchy.preflight --options runs/hierarchy_options.json --x data/p0.004/validation_X.npz --z data/p0.004/validation_Z.npz \
    --rule f --tcbp-python <TCBP python> --out runs/preflight_<time>            # must end with PREFLIGHT PASSED

# 1. part 1's row and the baselines, through the team's evaluator (one process; the main row is the sequential one)
python -m evaluation.evaluate --test-x $D/test_X.npz --test-z $D/test_Z.npz --decoder hierarchy.plugin:make_decoder \
    --decoder relay-bp --decoder bposd --plugin-options runs/hierarchy_options.json --out $O/evaluate
#    extra timed rows, each in its own --out folder: the same options with "parallel": true (two cores);
#    {"rule": "converged"}; --decoder hierarchy.priortilt_plugin:make_decoder

# 2. the members on every shot
python -m hierarchy.candidates --final --data $D/test_X.npz $D/test_Z.npz --out $O/cand --processes 12
<TCBP python> -m hierarchy.tcbp_member --final --data $D/test_X.npz $D/test_Z.npz --out $O/tcbp        # serial: a time per shot
python -m neural.evaluate_stim --final --checkpoint <X final> --data $D/test_X.npz --save $O/net_X
python -m neural.evaluate_stim --final --checkpoint <Z final> --data $D/test_Z.npz --save $O/net_Z

# 3. p = 0.003 and 0.004 only: the shots where P1, P2 and T are not unanimous, and joint relay-BP on them
for b in X Z; do python -m hierarchy.ensemble --members $O/cand/test_$b.chunks.npz $O/tcbp/test_$b.tcbp.npz \
    --write-selection $O/sel/test_$b.selection.npz; done
python -m hierarchy.joint_member --final --data $D/test_X.npz $D/test_Z.npz \
    --selection $O/sel/test_X.selection.npz $O/sel/test_Z.selection.npz --out $O/joint --processes 12

# 4. the tables: every member alone, every rule, rule f marked, the paired tests and the network's increment; then the pair score
for b in X Z; do python -m hierarchy.ensemble --members $O/cand/test_$b.chunks.npz $O/tcbp/test_$b.tcbp.npz \
    $O/joint/test_$b.joint.npz --logits $O/net_$b.npz --rule f --save $O/ensemble_$b.json; done
#    at p = 0.005, where step 3 is not run: leave the joint file out and add --no-joint
python -m hierarchy.ensemble --pair $O/ensemble_X.npz $O/ensemble_Z.npz
```

What to expect: about 50 listed shots per basis at p = 0.004 (0.45 to 0.56% on the
exploration files), fewer at p = 0.003; the joint step then costs under an hour of one core.
The members of step 2 do not depend on the rule. `candidates.py` decodes in chunk mode, so
relay-BP's answers there come from a decoder that saw every shot; the plugin of step 1 runs
relay-BP only where the seeds differ. Both are stated with their rows.

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
- **The ensemble's members decode the same saved shots.** `ensemble.py` stops if two member
  files name different data files (sha256) or hold another truth, and `joint_member.py` stops
  on a selection list made from another data file. TCBP-JOBS and joint relay-BP get the kit's
  H, L and priors of the file's basis and p (joint: `dem_matrices(circuit)` with all 1,728
  detectors, the same 12 observables). `tcbp_member.py` checks every returned correction
  against the syndrome itself (H e = s).
- **No rule is chosen on the shots it is reported on.** `ensemble.py --chosen-on` refuses the
  .json of the file it scores; the ORACLE rows and the scoring are the only users of the truth.
- **Times** state the machine and what the timer wraps. In `candidates.py` the timer wraps
  one decode call, as in the kit's `decoders.score`; a rule's time in `select.py` is the sum
  of the calls it makes. In the evaluator the timer wraps the whole `decode_shot`.

## Tests

```sh
python -m pytest -p no:cacheprovider hierarchy
```

31 tests, under two minutes on one CPU core that is not shared. Part 1: the rules against
hand-made cases; `select.py` on five hand-made shots whose outcome under every rule is known,
on a session-mode file, with a fitted constant (and the warning when it is fitted on the
scored shots) and with logits of other shots (refused); the refusal of `test_*` and `go_*`
files; `candidates.py` with 1 and with 4 processes on a small file (every array equal except
the times), a stopped run that continues, and a folder with other settings (refused); the
plugin with stub decoders and a stub network that prefers each candidate in turn, its three
rules and the rule per basis; the network's input; the whole chain on 24 shots per basis:
the plugin through the real `evaluation.evaluate`, against `select.py` on session-mode
candidates, for X with a network and Z without, and for Z with a `--logicals light` network;
the side-by-side mode against the sequential one; the reference plugin against P1.
Part 2: `ensemble.py` on seven hand-made shots whose outcome under every rule is known
(unanimity accepts; a split goes to the scorer; a member that did not decode a shot is left
out), the selection list, the rule choice and its refusal on the scored file, the per-shot
output, the pair score and `--no-joint`; `tcbp_member.py` in its plumbing mode, serial and
batched, and its refusals; `joint_member.py` on two listed shots, resumed, and its refusals;
`plugin_options.py`; what `preflight.py` refuses.

The decoding tests need PriorTilt and are skipped without it; the network tests need torch.
They use the first 24 shots of the committed `train_*` files and small iteration budgets
(`--priortilt-max-rounds`, `--relay-sets`: for tests only, never for a number).

## Next

1. **The one run on the test shots**, by the commands above, after the preflight has passed.
2. **The network's increment on a larger held-out set.** 80,000 fresh X shots at p = 0.004
   are being decoded for it (see `HANDOFF.md`). On today's evidence the learned tie-breaker
   is not better than the best rule without learning.
3. **Joint decoding at p = 0.005** on the shots where the members disagree. Not measured: it
   is slow there, and worse than separate decoding as a decoder of every shot. As an arbiter
   on a few percent of the shots it may still help; rule f needs it to differ from TCBP-JOBS.
4. **The 32 shots in 40,000 where P1, P2 and TCBP-JOBS agree and are wrong** are now the
   larger part of rule f's 47 failures. Only a member that runs on every shot can catch them.
   One idea, not built: let the network overrule a unanimous answer when it is confident
   (a threshold fitted on train).
5. **The ensemble as one timed decoder.** Today it is combined offline. As a pipeline it
   needs TCBP-JOBS and PriorTilt side by side, and joint relay-BP on 0.5% of the shots at
   about half a minute each: fine for the mean, poor for the worst shot.
6. **A stage for the shots where no candidate converged** in part 1 (11 of 25,000 X shots,
   all failures). `plugin.py` marks the place; joint relay-BP is the natural candidate.
7. **A constant C** (network minus weight / C) is implemented and off: on validation shots it
   changed nothing.
8. **Per-shot seeds.** Seeding PriorTilt from the shot would make an answer independent of the
   shot order, so chunk mode and session mode would agree shot for shot.
