# How to reproduce our rows of the comparison

Everything runs from `challenge-1-ml-decoder-kit/`. One script has every stage:
`report/reproduce_ours.sh` (run it without arguments for the list). This page goes through the stages in order
and says, for each command, whether it was **run here** (and on what) or **not run here**. "Here" is the PC the
report was written on (AMD Ryzen 7 7730U, Linux, no GPU). "The server" is the team's ICRN notebook server
(Linux, 12 CPUs by quota, an NVIDIA A100 slice of 20 GB), where the rows of the table were measured.

There are two levels of reproduction:

- **The tables and figures from the committed result files:** two commands, about a minute, no decoder needed
  (step 6). This is what `REPORT.md` is made from.
- **The rows themselves:** decode the saved test shots again (steps 1 to 5). The baselines and our decoder run on
  CPUs; relay-BP at p = 0.005 takes about five hours in one process.

## 1. The environment

| Command | What it does | Status |
|---|---|---|
| `report/reproduce_ours.sh env` | Python 3.12 venv in `.venv-ours`; the kit's `requirements.txt` (stim, ldpc, numpy, scipy, relay-bp), pytest, matplotlib, torch (CPU build); clones PriorTilt at commit `8eefd37d694b5942fe60ec6d5c82ce97f3d2a450` beside this repository and builds its Rust core | **run here** on a clean export of the repository (commit `88d39f2`, without `.venv312`), with `PY312=$(uv python find 3.12)`, 4 cores: 184 s, exit 0 |

- `relay-bp` and PriorTilt build from source: a Rust toolchain is needed (https://rustup.rs).
- The script looks for `python3.12`. If your Python 3.12 has another name, give it: `PY312=/path/to/python3.12`, or
  with uv, `PY312=$(uv python find 3.12)`.
- **PriorTilt's licence is CC BY-NC-SA 4.0** (non-commercial, share-alike). None of its code is in this repository;
  it is installed from its authors' repository, https://github.com/ionq-publications/PriorTiltDecoder, and its
  tuned schedule is read from the clone at run time (`PRIORTILT_REPO`).
- Never use the tracked folder `.venv312` (a macOS environment committed by mistake).

## 2. The tests, and a small end-to-end run

| Command | What it does | Status |
|---|---|---|
| `python -m pytest -p no:cacheprovider report` | the report's arithmetic (rate per cycle, Wilson interval, paired counts against the team's `compare_gain`), and both scripts on a stand-in data folder | **run here**: 10 passed |
| `report/reproduce_ours.sh test` | the same plus the tests of `hierarchy/` (the decoder: rules, plugin, the chain through the evaluator on 24 shots) | **run here** in the clean export's fresh environment: 43 passed, 1 skipped (the check of the top README section, before it exists), 72 s |
| `report/reproduce_ours.sh smoke` | cuts the first 24 shots of the committed **training** files at p = 0.004, and decodes them with our decoder and with relay-BP through the team's evaluator (`evaluation.evaluate`) | **run here** in the clean export's fresh environment, with the tie-break without a network (no checkpoints fetched): 15 s, both decoders 0 failures of 24 pairs. It checks the plumbing only; 24 pairs say nothing about accuracy |

## 3. The shot files

The team's saved shots are committed: `data/p0.003/`, `data/p0.004/`, `data/p0.005/`, each with `train_{X,Z}.npz`
(20,000 shots), `validation_{X,Z}.npz` (5,000) and `test_{X,Z}.npz` (10,000). They were made by the teammates
with the kit's `make_dataset.py` (seeds 101/102, 201/202 and 301/302 for X/Z; `./reproduce_eval.sh dataset <p>`).

| Command | Status |
|---|---|
| `./reproduce_eval.sh dataset <p>` (the teammates' script) | **not run here**, on purpose: a Stim seed does not give the same shots on every machine, so the committed files are the experiment. Their sha256 are pinned in `report/rows.py`, and a test checks the files against them (**run here**) |

sha256 of the test files (every row of the table must name these, or `make_tables` stops):

| File | sha256 |
|---|---|
| `data/p0.003/test_X.npz` | `f7bc181399a0a8fd1324950c98a8419c0082d7c34940c03dfb34c9e2cefc1ff3` |
| `data/p0.003/test_Z.npz` | `f5ac5b336f4155870fd8a6792ad61e21d5d76f04e05a9c09419879b6512e7f49` |
| `data/p0.004/test_X.npz` | `53a7cee301600d36f3dad806a88f59ad6d84b681044b2548268c2ea9b5689e94` |
| `data/p0.004/test_Z.npz` | `b61c1325de8852385d19ee5ce4eb41ea939b668f83fb3e994ce0f590218b3d56` |
| `data/p0.005/test_X.npz` | `1b13521cddb2bf5e5d4c7f1e85a8ecaaa3c72a0b99754a3f76b4a5a4a912fc8b` |
| `data/p0.005/test_Z.npz` | `b86ae320d13fa555d8358fe399156d28d4e52b01fabf8eafa1b30e60bdcd0cef` |

## 4. The two networks

The tie-break uses one trained network per memory basis. The checkpoints are not in the source tree; they are on
data-only branches of this repository.

| Network | Branch | Commit | Path | Bytes | sha256 |
|---|---|---|---|---|---|
| X memory, step 120,000 (the end of its run) | `laptop-artifacts` | `2f752e9367f80e1180711ded6b6ca854a5c0c147` | `cascade_X_s1/checkpoint_final_step120000.pt` | 14,701,836 | `d8d1c70f552233300f1c10997b69292092042eb8517f5b6526b607c1f6dc4933` |
| Z memory | *pending: filled in when its training ends and the run on the test shots names it* | | | | |

| Command | What it does | Status |
|---|---|---|
| `report/reproduce_ours.sh checkpoints` | fetches both files into `runs/checkpoints/` and checks their sha256 | the X file's sha256 was checked here from its commit; the stage itself waits for the Z line |
| `python -m neural.train --out runs/cascade_X_s1 --seed 1 --steps 120000 --lr-decay-steps 120000` | trains the X network from nothing (an 8 GB GPU, several hours) | **not run here** (no GPU): it ran on the laptop; its log is `report/data/networks/cascade_X_s1.train.log` |
| `python -m neural.train --basis Z --logicals light --init-from <X checkpoint> --out runs/cascade_Z_s1 --seed 1 --steps-start 1000 --steps-anneal 6000 --steps 100000 --lr-decay-steps 100000` | trains the Z network, starting from the X network's weights mirrored | **not run here**: it ran on the server; its log is `report/data/networks/cascade_Z_s1.train.log` |

Training is seeded: the shots of step s depend only on the seed and s. A run on another GPU model is not expected
to give the same weights bit for bit, which is why the checkpoints are kept.

## 5. The rows on the test shots

One process per row, one thread, through the team's evaluator. Its timer wraps the whole decode of a shot: both
PriorTilt runs, relay-BP where it runs, the network and the rule.

| Command | Row | Status |
|---|---|---|
| `report/reproduce_ours.sh row <p> bposd` | BP-OSD, the kit's settings | **not run here**; measured on the server |
| `report/reproduce_ours.sh row <p> relay-bp` | relay-BP, the kit's settings (about 5 hours at p = 0.005) | **not run here**; measured on the server |
| `report/reproduce_ours.sh row <p> router` | the teammates' router with their frozen models (`results/router_p<p>/model_{X,Z}.json`) | **not run here**; measured on the server |
| `report/reproduce_ours.sh row <p> hierarchy-weight` | our hierarchy with the tie-break by lowest weight (no network, no torch needed) | pending |
| `report/reproduce_ours.sh row <p> hierarchy` | our hierarchy with the neural tie-break (needs step 4) | pending |

What a `row` command runs, written out for our decoder:

```sh
python -m evaluation.evaluate --test-x data/p0.004/test_X.npz --test-z data/p0.004/test_Z.npz \
    --decoder hierarchy.plugin:make_decoder --plugin-options runs/clean/hierarchy_options.json \
    --out runs/clean/eval_p0.004_hierarchy
```

with `hierarchy_options.json` holding `{"rule": "network", "checkpoints": {"X": ..., "Z": ...}, "seeds": [1, 1001], "device": "cpu", "torch_threads": 1}`.

- The counts of a row depend little on the machine, but not nothing. Relay-BP and PriorTilt gave the same answers
  shot for shot on three x86 machines: our Linux PC, a Windows laptop and the Linux server (`analysis/RESULTS.md`,
  `hierarchy/README.md`). A teammate's Mac (arm64) gave relay-BP a different outcome on <!--f:cross4_differ-->65<!--/f--> of the 10,000
  test pairs at p = 0.004 (<!--f:cross4_earlier_pairs-->84<!--/f--> failed pairs against the server's <!--f:cross4_timed_pairs-->83<!--/f-->;
  `report/tables/checks.md`). Expect counts within the statistical error on another machine, and the exact counts on
  the same kind of machine. The times always depend on the machine.
- A decoder keeps random state from shot to shot, so a row decodes its file in order, in one process.
- **TCBP-JOBS** needs an NVIDIA GPU and `cudaq-qec`: `python -m hierarchy.tcbp_member --data data/p<p>/test_X.npz --out runs/tcbp_p<p> --final` (and the same for Z). **Not run here** (no GPU).

## 6. The tables and the figures

| Command | What it does | Status |
|---|---|---|
| `report/reproduce_ours.sh collect` | only if you made rows yourself: copies `runs/clean/eval_p*/` into `report/data/test/`, replacing the committed copies | **not run here** |
| `python -m report.make_tables` | every table: `report/tables/*.md` and `*.csv`, and the tables and numbers inside `REPORT.md`, `APPENDIX.md`, `SLIDES.md` and this file | **run here** on `report/data/`, and in the clean export (`report/reproduce_ours.sh report`, 6 s with the figures) |
| `python -m report.make_plots` | every figure: `report/figures/*.png` and `*.svg` | **run here** on `report/data/`; a second run gives the same files byte for byte, and the clean export's 45 tables and figures were byte-identical to the committed ones |
| `python -m report.sync_data --check` | verifies every file under `report/data/` against the sha256 recorded in `report/data/SOURCES.md` | **run here** |

`python -m report.sync_data` (without `--check`) is how the team copied the result files from the data branches
`server-logs` and `laptop-artifacts` into `report/data/`. It is not a step of the reproduction.
