# predecoder-tcbp: the head's per-shot priors fed to TCBP-JOBS instead of relay-BP

**A separate branch** (`predecoder-tcbp`, from `predecoder` at `e96cc020`). Sal, 14:39 UTC: "i do want to force to use
tcbp since its just better than relay bp ... we should retrain fully for the tcbp instead of the relay bp and change our
pipeline". It was built as a separate branch. On Sal's word ("once results are in you can bring the tcbp version into
main as well", about 15:47 UTC) it was merged into `main` by PR #9 at 16:28:53 UTC, after the X test results. The Z
rows below came after that merge.

**The idea.** The head (`head.py`, unchanged, sha256 `38ab667e...`) reads the network's view of a shot and gives every
column of H a probability that it fired in that shot. `priors.py` mixes those probabilities with the kit's priors, using
two knobs (log-odds alpha, eps). Until now the mixed priors went to relay-BP. This branch gives them to TCBP-JOBS,
NVIDIA's GPU decoder (cudaq-qec 0.9.0, `nv-qldpc-decoder`, preset `tcbp-jobs`), which on its own was already the
team's best single decoder. The head was not retrained. The knobs were refitted for TCBP-JOBS on train shots only.

**The result in short** (X memory; every number is from Stim circuit shots):

- **Validation** (`data/p0.005/validation_X.npz`, 5,000 shots), the kill line fixed in advance: TCBP-JOBS with the head's
  priors (TH) fails **33**, against **91** for TCBP-JOBS alone. Paired, only TH fails 8 and only TCBP-JOBS 66:
  p = 1.8e-12, **PASSED**. Against the same run's TCBP-JOBS with the kit's priors (100): 8 against 75.
- **Test** (registered; designed after the test results were seen; a separate branch):
  - p = 0.005: TH **72** of 10,000, against **186** for TCBP-JOBS alone (22 against 136, p = 2.9e-21), 151 for
    relay-BP with the head's priors, and 107 for the fallback rule;
  - p = 0.004: TH **4** against 20 (p = 0.0004);
  - p = 0.003: 0 against 1, no claim;
  - Z at p = 0.005 (X's knobs, no Z fit): TH **61** against 177 (p = 2.2e-22);
  - **the kit's pair score at p = 0.005: 132 failed pairs of 10,000 against TCBP-JOBS's 360 (p = 3.0e-42), a rate per
    cycle of 1.11e-3 against 3.05e-3.**
- **Latency on the A100, one shot at a time, one process:** network and head 9.4 ms (float32), building TCBP-JOBS with
  the shot's priors 42.6 ms, decoding 3.8 ms; about 56 ms mean in all. Building is the cost: cudaq-qec 0.9.0 cannot
  change a built decoder's priors. Running the network on the GPU changed no outcome on 1,000 validation shots.
- **The same decoder, built faster** (tcbp-speed, a separate branch; section "tcbp-speed" below): with H handed to
  `get_decoder` as a scipy sparse matrix made once (`--h-format csr`), the build takes 34.0 ms instead of 41.2 ms in
  the same run. The 12 bits, the faults and the whole corrections were the same on all 1,000 validation shots tried.
  The registered numbers above were made with the dense path and are unchanged.

**Erratum.** `logs/tcbp_test_registered_spec.md` says "Written 2026-11-22 15:52 UTC" and cites dq-orch-6's decision as
"15:49 UTC". Both true times are 15:42 UTC. The spec's commit, `dad4fd50`, is dated 15:42:59 UTC, before any test number
of this branch. Only the typed times were wrong; the spec itself is unchanged. More generally, every time typed in this
README and in the logs is an estimate made while writing, and the commit times decide. Two others are also a few
minutes late: the knobs file "frozen at 15:28", committed in `86259291` at 15:26:36 UTC; and declarations §2, "15:06",
committed in `26dcea76` at 15:04:38 UTC. Each was committed before the numbers it governs.

| file | what it does |
|------|--------------|
| `tcbp.py` | `decode`: TCBP-JOBS with one decoder built per shot from that shot's priors (arms `fresh`: the kit's priors; `head`: the head's). `fit`: the knob fit on train subsets. `score`: the counts and paired tests, on this PC, from the per-shot files. `probe`: prints a built decoder's interface |
| `test_tcbp.py` | tests without a GPU: the kit's relay-BP stands in for TCBP-JOBS behind the same calls |
| `tcbp_speed.py` | tcbp-speed: `run` (server) times the build's parts and checks that H as a scipy matrix gives the same answers as the dense path; `report` (any machine) prints its tables and compares it with another run |
| `test_tcbp_speed.py` | its tests without a GPU (plumbing only: the stand-in ignores H's form) |
| `logs/tcbp_declarations.md` | what was fixed before each number: the fit, the arms, the test run |

## How it works

- **One model, one decoder per shot.** `nv_qldpc.init_tcbp_model(H)` does not take the priors. It is fitted once per run
  (54 to 88 s on the A100, measured in each run), and every shot's decoder shares it. A decoder takes its priors only when it is built
  (`error_rate_vec`), so each shot gets its own: `get_decoder(..., error_rate_vec=<shot priors>, **preset)`, then
  `decode(syndrome)`. Building takes 41 to 43 ms on the A100, mean per run. That time is counted in each shot's latency.
- **Two environments, split through a file.** The network and head run in an environment with torch
  (`predecoder.evaluate`, here on luminaty's CPU in float32) and write `logits.npy`. TCBP-JOBS runs in the teammates'
  environment on the server, which has no torch, and reads those exact bytes, checked by sha256. The GPU path of
  `evaluate.py` runs in float16, and that moved 33 of 5,000 relay-BP outcomes. Every decoded shot records a hash of the
  logits row it used, and `score` checks those hashes against this PC's file.
- **The control.** The stored TCBP-JOBS answers (`server-logs`, `files/tcbp_p0.005/*.tcbp.npz`) came from one decoder,
  built once and run with `decode_batch`. The `fresh` arm is the same decoder built per shot and run with `decode()`.
  If the two disagree, the per-shot path itself changes answers, and that is reported first.

## Commands

Run from `challenge-1-ml-decoder-kit/`. The network steps need torch (`neural/`'s environment). The TCBP-JOBS steps
need the teammates' TCBP environment (Linux, an NVIDIA GPU, `pip install cudaq-qec` 0.9.0) and no torch. Without a
GPU, `--decoder relay` runs every TCBP step with relay-BP as a stand-in, for the plumbing only.

```sh
# tests (no GPU: relay-BP stands in for TCBP-JOBS)
python -m pytest -p no:cacheprovider predecoder/test_tcbp.py predecoder/test_tcbp_speed.py

# 0. what a built decoder offers (the search for a cheaper path than one decoder per shot)
python -m predecoder.tcbp probe --data data/p0.005/validation_X.npz

# 1. the train subsets: the shots the stored TCBP-JOBS run fails, and 2,000 it gets right
python -m predecoder.subset --data data/p0.005/train_X.npz --per-shot <server-logs: files/tcbp_p0.005/train_X.tcbp.npz> \
    --arm T --right 2000 --seed 5 --out <subsets>
# 2. the head's logits on them, and on the validation file (torch; CPU float32 is the reference)
python -m predecoder.evaluate --head <head.pt 38ab667e...> --trunk <final X trunk d8d1c70f...> \
    --data <subsets>/train_X_T_failed.npz --out <logits_failed> --device cpu --threads 4 --time-shots 0
python -m predecoder.evaluate ... --data <subsets>/train_X_T_right.npz --out <logits_right> ...
python -m predecoder.evaluate ... --data data/p0.005/validation_X.npz --out <logits_validation> ...
# 3. the knob fit (TCBP env, GPU); this made logs/fit_tcbp_X_p0.005_full.json
python -m predecoder.tcbp fit --data <subsets>/train_X_T_failed.npz <subsets>/train_X_T_right.npz --weights 1 9.805 \
    --logits <logits_failed> <logits_right> --out fit_tcbp_X_p0.005.json
# 4. validation: the head's priors, the kit's priors, and the batch check, on one model
python -m predecoder.tcbp decode --data data/p0.005/validation_X.npz --logits <logits_validation> \
    --settings predecoder/logs/fit_tcbp_X_p0.005.json --arms head fresh --batch-check --out <folder>
# 5. score it (any machine; no decoder runs)
python -m predecoder.tcbp score --data data/p0.005/validation_X.npz --tcbp-head <folder>/per_shot.npz \
    --tcbp <files/tcbp_p0.005/validation_X.tcbp.npz> --relay-head <predecoder.compare head arm per_shot.npz> --logits <logits_validation>
# 6. the network on the GPU in float32, timed one shot at a time, and its logits against the CPU's
python -m predecoder.evaluate ... --data data/p0.005/validation_X.npz --limit 1000 --device cuda --precision fp32 \
    --threads 1 --time-shots 1000 --out <gpu logits>
python -m predecoder.tcbp logits-diff --a <logits_validation> --b <gpu logits> --limit 1000
# 7. the registered test runs (once each, --final), then the kit's pair score
python -m predecoder.tcbp decode --data data/p0.005/test_X.npz --logits <test logits> \
    --settings predecoder/logs/fit_tcbp_X_p0.005.json --arms head fresh --batch-check --out <folder> --final
python -m predecoder.tcbp pair --final --data-x data/p0.005/test_X.npz --data-z data/p0.005/test_Z.npz \
    --ours-x <X per_shot.npz> --ours-z <Z per_shot.npz> --tcbp-x <test_X.tcbp.npz> --tcbp-z <test_Z.tcbp.npz>
# 8. tcbp-speed: the build's parts, and H as a scipy matrix checked against the dense path (TCBP env, GPU) ...
python -m predecoder.tcbp_speed run --data data/p0.005/validation_X.npz --logits <logits_validation> \
    --settings predecoder/logs/fit_tcbp_X_p0.005.json --limit 1000 --out <folder>
# ... its tables, and its arms against step 4's head arm (any machine; no decoder runs)
python -m predecoder.tcbp_speed report --summary <folder>/summary.json --per-shot <folder>/check_per_shot.npz \
    --reference <step 4 folder>/per_shot.npz
# the faster build in any tcbp.py command (decode, fit, probe): add --h-format csr
```

The exact server scripts are the orchestrator's: `server-jobs/gpu/060`, `070`, `075`, `080`, `ctl/130` and
`cpu/070`, `080` on `integration`. Their outputs are on `server-logs` under `files/predecoder-tcbp/`.

## Numbers

All runs are X memory at p = 0.005. Server: the team's notebook server, NVIDIA A100-SXM4-80GB, AMD EPYC-Milan,
1 process with 1 CPU thread, cudaq-qec 0.9.0 (cudaqx `9043634b`), preset `tcbp-jobs` with no overrides. The jobs are
the orchestrator's (`server-jobs/gpu/060`, `070`, ...), and their outputs are on `server-logs` under
`files/predecoder-tcbp/`.

### Job 1: the cheaper path, and TCBP-JOBS with the kit's priors built per shot (validation, no head)

`python -m predecoder.tcbp probe` and `decode --arms fresh` on `data/p0.005/validation_X.npz` (5,000 shots, seed 201),
commit `abb05607`, 14:54 to 14:59 UTC. Logs: `logs/tcbp_job1_probe.log`, `logs/tcbp_job1_validation_fresh.log`.

- **The cheaper path does not exist in cudaq-qec 0.9.0.** A built `Decoder` offers only `decode`, `decode_async`,
  `decode_batch`, `get_block_size`, `get_syndrome_size` and `get_version`. `decode()` takes a syndrome of floats, which
  is soft information about the syndrome bits, not priors. Nothing sets the error rates of a decoder once it is built,
  so the decoder is rebuilt for every shot. Not tested: passing H as a scipy sparse matrix, which the docstring calls
  the preferred input and which may build faster.
- **Time per shot**, one at a time: building the decoder takes 40.7 ms and decoding 8.5 ms, mean; p99 82 ms for the
  two together. The model fit, once per run, takes 54 s.
- **The decoder's `converged` flag** is False on 4,858 of 5,000 shots, including shots it gets right. It does not say
  whether the answer reproduces the syndrome, which it does on all 5,000 (checked here as H e = s).
- **Failures:** 96 of 5,000, 0.0192 [0.0157, 0.0234]. The stored TCBP-JOBS file (`files/tcbp_p0.005/validation_X.tcbp.npz`:
  one decoder built once, `decode_batch`, its own model fit) fails 91. Shot by shot the two agree on the 12 bits except
  on 44 shots: both fail 79, only per-shot 17, only stored 12, exact two-sided p = 0.46. **TCBP-JOBS's answers on hard
  shots vary from run to run**, the way relay-BP's do across seeds. Job 3 separates one of the three candidate causes
  (another model fit, batch against single decoding, random draws inside JOBS).

### Job 2: the knob fit (train shots only)

The orchestrator moved the fit to p = 0.005 at 14:47 UTC, before any number. At p = 0.004, TCBP-JOBS fails only 37
of the 20,000 train shots; at p = 0.005 it fails 390. Declared in `logs/tcbp_declarations.md` §1.
- **Subsets:** `predecoder.subset --data data/p0.005/train_X.npz --per-shot <files/tcbp_p0.005/train_X.tcbp.npz>
  --arm T --right 2000 --seed 5`. That gives the 390 train shots the stored TCBP-JOBS run fails, plus 2,000 of the
  19,610 it gets right. The right file has weight 9.805 (`logs/tcbp_train_X_T_subsets_p0.005.json`).
- **Logits:** the head on luminaty's CPU in float32 (`logs/tcbp_evaluate_train_X_T_p0.005.log`). Sent to the server as
  these exact bytes through `laptop-artifacts:predecoder-tcbp/` (commits `64163747` and `666f10bf`), each checked by
  sha256 (`logs/tcbp_transfer_SHA256SUMS`).
- **The runs:** `python -m predecoder.tcbp fit --data <failed> <right> --weights 1 9.805 --logits <failed logits>
  <right logits> --out fit_tcbp_X_p0.005.json`. Job gpu/070, at `abb05607`, stopped at alpha 0.3: JOBS refuses
  priors above 0.5. Job gpu/075, at `83563dbe`, caps priors at 0.5 (§1b) and resumed from the cached rows. With the
  cap, 1 shot and 1 column were capped at alpha 0.3, the first row where it acts.

| priors | failed shots left wrong, of 390 | right shots broken, of 2,000 | score (failed + 9.805 x broken) |
|---|---|---|---|
| the kit's (control, built per shot) | 341 | 5 | 390.0 |
| log-odds alpha 0.1, eps 1e-4 | 175 | 3 | 204.4 |
| 0.2 | 134 | 3 | 163.4 |
| 0.3 | 119 | 5 | 168.0 |
| **0.4** | **107** | **5** | **156.0** |
| 0.5 | 104 | 6 | 162.8 |
| 0.6 | 111 | 7 | 179.6 |
| 0.7 | 112 | 9 | 200.2 |

**Chosen: log-odds, alpha 0.4, eps 1e-4.** It is the unique minimum, inside the declared grid, so the extension rows
do not join. Frozen in `logs/fit_tcbp_X_p0.005.json` (commit `86259291`, 15:26:36 UTC), from the fit's log
(`logs/tcbp_fit_log_at_freeze.log`), before any validation decode with it. The fit tool's own JSON, which arrived later
(`logs/fit_tcbp_X_p0.005_full.json`), chose the same knobs. The extension rows, decoded for the record, scored
0.8: 210.1, 0.9: 215.9, 1.0: 209.1 and 0.05: 240.8.

The control row is the "fresh" check on train. The 390 shots were chosen as the stored run's failures, yet TCBP-JOBS
with the same kit priors, built per shot, gets 49 of them right and breaks 5 others. As with relay-BP, the train
subsets overstate any gain: part of what a setting "fixes" is the decoder's own run-to-run change. The fit ranks
settings; validation gives the unbiased count. TCBP-JOBS takes alpha 0.4, against relay-BP's 0.2 at p = 0.004.

### Job 3: validation (`data/p0.005/validation_X.npz`, 5,000 Stim shots, seed 201): the kill line is PASSED

The knobs are frozen. The logits are luminaty's CPU float32 bytes behind H = 79 (sha256 `14eea54d...`; every
decoded row's hash equals this PC's file on 5,000 of 5,000). One process with one model: the head arm, then the
kit's-priors arm ("fresh"), then the batch check. Server job ctl/130 at `86259291`, 15:29 to 15:40 UTC, alongside the
fit's last rows. Logs: `logs/tcbp_job3_validation.log`, `logs/tcbp_validation_X_p0.005_score.log`.

    python -m predecoder.tcbp decode --data data/p0.005/validation_X.npz --logits <logits> \
        --settings predecoder/logs/fit_tcbp_X_p0.005.json --arms head fresh --batch-check --out <folder>
    python -m predecoder.tcbp score --data data/p0.005/validation_X.npz --tcbp-head <folder>/per_shot.npz \
        --tcbp <files/tcbp_p0.005/validation_X.tcbp.npz> --relay-head <relay-BP head arm per_shot.npz> --logits <logits>

| decoder, the same 5,000 shots | failures | share, 95% interval | build + decode per shot, A100, mean |
|---|---|---|---|
| **TH: TCBP-JOBS with the head's priors** | **33** | **0.0066 [0.0047, 0.0093]** | 42.3 + 3.8 ms (p99 64 ms) |
| TCBP-JOBS with the kit's priors, per shot, this run's model ("fresh") | 100 | 0.0200 [0.0165, 0.0243] | 42.1 + 12.9 ms (p99 119 ms) |
| the same, built once, `decode_batch` (batch check) | 100 | 0.0200 | |
| TCBP-JOBS alone, the stored file (its own model fit) | 91 | 0.0182 [0.0148, 0.0223] | |
| H: relay-BP with the head's priors (luminaty CPU logits) | 79 | 0.0158 [0.0127, 0.0196] | |
| the frozen fallback rule: H where H converged, else T | 50 | 0.0100 [0.0076, 0.0132] | |
| (b2): H where H converged, else TH (not a candidate, §2b) | 44 | 0.0088 [0.0066, 0.0118] | |

| pair (same shots) | both fail | only TH | only the other | exact two-sided p |
|---|---|---|---|---|
| **TH against TCBP-JOBS alone, stored (the kill line)** | 25 | 8 | 66 | **1.8e-12** |
| TH against this run's fresh arm (like for like: same model, only the priors differ) | 25 | 8 | 75 | 9.1e-15 |
| TH against H | 29 | 4 | 50 | 3.8e-11 |
| TH against the frozen fallback rule | 28 | 5 | 22 | 0.0015 |
| TH against (b2) | 29 | 4 | 15 | 0.019 |

How to read it:
- **TCBP-JOBS fails 2.8 to 3 times less with the head's priors.** That holds against the stored file (91) and
  against the same run's kit-priors arm (100). It beats every decoder the team had on these shots, including the
  fallback rule (50). The fit's train rows had pointed this way: 107 of 390 left wrong against 341.
- **TCBP-JOBS's answers vary from run to run.** The cause is not isolated; the model fit (`init_tcbp_model`, once per
  run) is the leading candidate. The batch check equals the fresh arm on all 5,000 shots, both bits and number of
  faults, so batch against single decoding changes nothing within a run. Yet three runs with the kit's priors gave 91,
  96 and 100 failures (job 1 against job 3: bits differ on 42 shots). Against a fit that always differs: on the test
  shots, job 5's kit-priors arm equals the stored file on every shot. Every comparison with a stored file may
  therefore mix two runs' variation; the like-for-like pair is TH against the same run's fresh arm.
- The decoder's own `converged` flag is False on 3,863 shots with the head's priors and 4,858 with the kit's. It is
  not used.
- The choice of (a) over (b2) was made on Sal's word before (b2) was computed (§2b). By the declared rule of §2
  (fewer failures), (a) would also have been chosen: 33 against 44.

### Job 4: the whole pipeline on the GPU (the network on the A100 in float32)

`python -m predecoder.evaluate ... --device cuda --precision fp32 --threads 1 --time-shots 1000 --limit 1000` on the first
1,000 validation shots, then `tcbp logits-diff` against the CPU bytes and `tcbp decode` with the GPU logits. Server job
cpu/070 at `8c25c16d`. Logs: `logs/tcbp_job4_gpu_*`.

- **Network + head, one shot at a time on the A100, float32:** 9.4 ms mean, 8.5 ms median, 19.8 ms p99. With float16
  autocast: 11.5 / 10.1 / 24.9 ms.
- **The float32 logits against the CPU's:** they differ on 893 of 1,000 rows, but only on 2,251 of 8,784,000 entries, by
  at most 0.0156 (one float16 step). Float16 autocast differs on every row (2.9 million entries, up to 1.79).
- **Decoded with TCBP-JOBS and the frozen knobs, the GPU logits give the same 12 bits, and the same number of faults,
  as the CPU logits on all 1,000 shots** (3 failures in both), although the two decodes used different model fits.
- **The pipeline's time per shot on the A100, one process:** network 9.4 + build 41.8 + decode 4.4 ms, about 56 ms
  mean. These are separate timers. The p99 of the sum was not measured as one timer; the parts are 19.8 and 65 ms.
  Most of it is building the decoder, which cudaq-qec 0.9.0 cannot avoid (job 1).

### The registered test runs (designed after the test results were seen; a separate branch)

The spec, `logs/tcbp_test_registered_spec.md`, was committed at `dad4fd50` (15:42:59 UTC) before any of these numbers
existed; see the erratum at the top about its typed time. Each run is one decode with `--final`, from its checkout,
with every input checked by sha256:
- the test file;
- the knobs `76970106...`;
- the logits, which are dq-predecoder's registered CPU float32 bytes, sent in parts and reassembled; every decoded
  row's hash equals luminaty's file on 10,000 of 10,000 shots, at every p.
The runs were server jobs gpu/080 (p = 0.005) and cpu/080 (p = 0.004, then 0.003), on the A100, side by side, 1
process each. Scored here with `tcbp score --final`; logs `logs/tcbp_test_X_p0.00*_{decode.log,score.log,score.json}`.

**X memory, p = 0.005** (`data/p0.005/test_X.npz`, 10,000 shots, seed 301):

| decoder, the same 10,000 shots | failures | share, 95% interval | build + decode per shot, A100, mean (p99) |
|---|---|---|---|
| **TH: TCBP-JOBS with the head's priors** | **72** | **0.0072 [0.0057, 0.0091]** | 42.6 + 3.8 ms (69 ms) |
| TCBP-JOBS with the kit's priors, per shot, this run's model ("fresh") | 186 | 0.0186 [0.0161, 0.0214] | 43.0 + 11.2 ms (99 ms) |
| the same, built once, `decode_batch` | 186 | 0.0186 | |
| TCBP-JOBS alone, the stored file | 186 | 0.0186 [0.0161, 0.0214] | |
| H: relay-BP with the head's priors (dq-predecoder's registered run) | 151 | 0.0151 [0.0129, 0.0177] | |
| the frozen fallback rule: H where H converged, else T | 107 | 0.0107 [0.0089, 0.0129] | |
| (b2): H where H converged, else TH (not a candidate) | 90 | 0.0090 [0.0073, 0.0110] | |

| pair (same shots) | both fail | only TH | only the other | exact two-sided p |
|---|---|---|---|---|
| **TH against TCBP-JOBS alone, stored (the declared pair)** | 50 | 22 | 136 | **2.9e-21** |
| TH against this run's fresh arm | 50 | 22 | 136 | 2.9e-21 |
| TH against H | 65 | 7 | 86 | 2.1e-18 |
| TH against the frozen fallback rule | 57 | 15 | 50 | 1.6e-5 |
| TH against (b2) | 65 | 7 | 25 | 0.0021 |

- TH fails 2.6 times less than TCBP-JOBS alone, as on validation (33 against 91 and 100), and less than every other
  decoder the team had on these shots.
- On the test shots this run's kit-priors arm, the batch check and the stored file agree on every shot (186, the
  same shots). On validation, three model fits gave 91, 96 and 100. So the like-for-like pair and the declared pair
  coincide here.
- The cap at 0.5 changed 3,779 columns on 3,040 of the 10,000 shots.

**X memory, p = 0.004 and p = 0.003, head arm only** (the p = 0.005 knobs, unchanged; no validation at these p):

| p | shots | TH failures | TCBP-JOBS alone, stored | both, only TH, only T | exact p | build + decode, mean (p99) |
|---|---|---|---|---|---|---|
| 0.004 | `test_X.npz`, 10,000, sha256 `53a7cee3...` | **4**, 0.0004 [0.0002, 0.0010] | 20, 0.0020 [0.0013, 0.0031] | 2, 2, 18 | **0.0004** | 43.1 + 1.9 ms (55 ms) |
| 0.003 | `test_X.npz`, 10,000, sha256 `f7bc1813...` | 0, 0 [0, 0.0004] | 1, 0.0001 [0, 0.0006] | 0, 0, 1 | 1 (no claim) | 42.6 + 2.7 ms (60 ms) |

The gain holds at p = 0.004: 5 times fewer failures. At p = 0.003 both are near zero, so no claim can be made there.

**Z memory at p = 0.005, and the kit's pair score** (run 3 of the spec; server job ctl/140, at `76521b78`, after the merge
into main). The Z head (`083147d2...`, dq-predecoder's) was used with X's knobs unchanged: there was no Z fit. Its
logits are dq-predecoder's CPU float32 bytes (validation `db786d0a...`, test `576849a5...`); every decoded row's hash
equals luminaty's files. The gate inside the job: test_Z was opened only if validation_Z beat the stored TCBP-JOBS
file, paired, p < 0.05.

| file | TH failures | TCBP-JOBS alone, stored | both, only TH, only T | exact p | build + decode, mean (p99) |
|---|---|---|---|---|---|
| `validation_Z.npz`, 5,000, sha256 `24f12efa...` (the gate: **GO**) | 35, 0.0070 [0.0050, 0.0097] | 97, 0.0194 [0.0159, 0.0236] | 24, 11, 73 | 2.3e-12 | 42.2 + 10.8 ms (84 ms) |
| `test_Z.npz`, 10,000, sha256 `b86ae320...` | **61**, 0.0061 [0.0048, 0.0078] | 177, 0.0177 [0.0153, 0.0205] | 41, 20, 136 | **2.2e-22** | 41.3 + 4.7 ms (69 ms) |

**The kit's score: pairs of shot k of X and shot k of Z at p = 0.005** (`python -m predecoder.tcbp pair --final ...`,
`logs/tcbp_pair_test_p0.005.{json,log}`; a pair fails when either shot fails; the rate per cycle is `1 - (1 - P)^(1/12)`
with its error from `decoders.per_cycle_rate`):

| decoder | failed pairs of 10,000 [X Z] | share, 95% interval | rate per cycle |
|---|---|---|---|
| **TH: TCBP-JOBS with the head's priors** | **132** [72 61] | 0.0132 [0.0111, 0.0156] | **1.11e-3 ± 0.10e-3** |
| TCBP-JOBS alone (the stored files) | 360 [186 177] | 0.0360 [0.0325, 0.0398] | 3.05e-3 ± 0.16e-3 |

Paired over the 10,000 pairs: both fail 91, only TH 41, only TCBP-JOBS 269, exact two-sided p = 3.0e-42. **TH fails 2.7
times fewer pairs.** This is the registered run 3 (designed after the test results were seen; a separate branch). The
Z knobs were not fitted for Z.

### tcbp-speed: the same decoder, built 7 ms faster (a separate branch; validation)

Branch `tcbp-speed`, code at `e5354788`. Server job `ctl/160-tcbp-speed-1.sh` on `integration` (the orchestrator's),
run 2026-11-22 18:00:52 to 18:05:24 UTC; outputs on `server-logs` under `files/tcbp-speed/`, copies in
`logs/tcbp_speed_job1*`. Machine: NVIDIA A100-SXM4-80GB, MIG slice 2g.20gb. One process, 1 CPU thread
(`OMP_NUM_THREADS=1` and the like), cudaq-qec 0.9.0 (cudaqx `9043634b`), preset `tcbp-jobs` with no overrides. One
model fit (54.1 s) was shared by every arm.

**What changed: only the form in which H is handed to `get_decoder`.**

- The registered runs give the dense uint8 array (936 x 8,784, 8.2 MB). cudaq-qec's docstring says that a dense array
  is copied into a dense tensor and converted to sparse storage on every call.
- `--h-format csr` (or `csc`) gives instead a scipy sparse matrix, made once per run from the same array
  (`tcbp.h_matrix`, checked to hold exactly its 30,672 ones in canonical form).
- The model, the options, the priors and the decoder are the same.
- The default stays `dense`, so the registered commands, their part files and their numbers are unchanged.

**The same answers.** The check used the first 1,000 shots of `data/p0.005/validation_X.npz` (seed 201). The priors
were the head's, with the frozen knobs of `logs/fit_tcbp_X_p0.005.json` (log-odds, alpha 0.4, eps 1e-4), capped at
0.5. The logits were the CPU float32 bytes of jobs 3 and 4 (laptop-artifacts `64163747`, every row's hash checked).
Every shot was decoded by four arms: dense, CSR, CSC and dense again. They took turns in blocks of 100 shots, in an
order that rotated from block to block.

| arm, the same 1,000 shots | failures | 12 bits differ from dense | faults differ | whole correction (8,784 bits) differs |
|------|---:|---:|---:|---:|
| dense (the registered path) | 3 | - | - | - |
| dense again | 3 | 0 | 0 | 0 |
| **CSR** | 3 | **0** | **0** | **0** |
| CSC | 3 | 0 | 0 | 0 |

- Every correction reproduces its syndrome.
- Dense equals dense again, so the decoder is repeatable within the run, and "the same" means something.
- All four arms also equal job 3's head arm on all 1,000 shots, bits and faults (`tcbp_speed report`,
  `logs/tcbp_speed_job1_report_vs_job3.json`). Job 3 is another run with another model fit.

**Time per shot.** Shots ran one at a time. One timer wraps the shot's whole cost: the priors (the head's logits mixed
with the kit's and capped), the build, the decode, and freeing the decoder. The network and head (9.4 ms, job 4) are
not included. Times are in ms.

| arm | total, mean | median | p99 | priors | build | decode | free |
|------|---:|---:|---:|---:|---:|---:|---:|
| dense | 48.7 | 48.2 | 60.6 | 0.16 | 41.2 | 2.7 | 4.6 |
| dense again | 48.6 | 48.1 | 59.9 | 0.16 | 41.2 | 2.7 | 4.6 |
| **CSR** | **41.7** | 41.0 | 53.7 | 0.16 | **34.0** | 2.7 | 4.8 |
| CSC | 41.8 | 41.0 | 54.3 | 0.16 | 34.4 | 2.7 | 4.6 |

- CSR saves 7.2 ms of the build (17 %) and 7.0 ms of the shot (14 %).
- The registered runs' timers (build 42.6 + decode 3.8 ms) left out freeing the decoder, which takes 4.6 ms. Under those
  timers, the build and decode here come to 43.9 ms with dense and 36.7 ms with CSR.
- Adding the network's 9.4 ms (job 4, a separate timer) gives about 46 ms per X shot with CSR, against about 53 ms with
  dense, without the free. With the free, it is about 51 against 58 ms.

**Where the build's time goes.** These are timing probes only: 50 builds each, after one untimed build, with the
priors of shots 0 to 49 (`logs/tcbp_speed_job1_summary.json`, `profile.variants`). Times are mean ms.

| what is built | H dense | H CSR | free |
|------|---:|---:|---:|
| `tcbp-jobs` (the decoder of every number here) | 41.4 | 33.9 | 4.5 |
| `tcbp-jobs`, the model given as a file path | 43.8 | 36.3 | 4.5 |
| `bp10-minllr-jobs` (JOBS, no TCBP model) | 40.5 | 33.6 | 4.6 |
| `tcbp` (the model, BP only: no OSD, no JOBS) | 10.6 | 3.5 | 0.7 |
| `bp60` (bare BP, no model) | 9.6 | 2.6 | 0.6 |

- **A dense H costs about 7 ms in every decoder** (dense minus CSR: 7.5, 7.5, 6.9, 7.1, 7.0 ms).
  - The form of the sparse matrix makes no difference: CSR in uint8, int32, float64 or bool, and CSC, all take 33.8 to
    34.1 ms.
  - For scale, converting this dense H to sparse in Python (`np.nonzero` or scipy) takes 26 to 27 ms. So the conversion
    inside cudaq-qec is faster than Python's, but it still costs 7 ms on every build.
- **JOBS's set-up is most of what remains: about 30 ms** (33.9 ms against `tcbp`'s 3.5 ms, with CSR). It is about the
  same with or without the TCBP model (33.6 against 33.9 ms).
  - The model costs under 1 ms (`tcbp` 3.5 against `bp60` 2.6, which also differ in iterations), and it is 2,538 bytes.
  - Freeing a decoder with JOBS takes 4.5 ms; one without, 0.6 ms.
- **That set-up happens inside the plugin's C++, which Python cannot reach.**
  - cProfile of 50 shots (`logs/tcbp_speed_job1_profile_dense.txt`) sees only the one `get_decoder` call: 2.08 of the
    2.52 s.
  - No native profiler (nsys, ncu, perf, py-spy) is on the server's path.
  - Giving the model as a file path is 2.4 ms slower, not faster.

**Not shown, and caveats.**

- The equal answers are shown on 1,000 validation X shots at p = 0.005, with the head's priors. Not shown: Z, other p,
  the kit's priors, and the test files.
- A sparse H goes through another constructor of the plugin ("sparse" rather than "dense", in its own error messages).
  1,000 shots with equal whole corrections are evidence, not a proof, that the two agree on every shot.
- The registered numbers (validation, test, the pair score) were made with the dense path and were not rerun.
- The GPU is a MIG slice that other jobs may share. The arms took turns, so any load slowed all of them alike.
- **A built decoder's priors still cannot be changed in place.** The Python `Decoder` of 0.9.0 offers no way to do it
  (job 1).
  - The plugin's binary in the PyPI wheel `cudaq_qec_cu13` 0.9.0 was read on luminaty and never run, and it may not be
    the server's exact build.
  - It holds an internal `set_priors`, and the message "the JOBS column costs are fixed at construction".
  - It exports no such function, so this was not tried.
- Not tried: options that change the decoder (batch sizes, thread counts, JOBS's own settings).

## What was not done, and what is not tested

- **The head was not retrained for TCBP-JOBS.** It is relay-BP's head (trained only on error-model shots with the
  fired columns as labels, so it does not depend on the decoder). Only the two knobs were refitted.
- **One knob fit, X at p = 0.005.** It is applied unchanged at p = 0.004 and 0.003, and to Z, where there is no fit.
- **TCBP-JOBS's answers vary from run to run** (cause not isolated; the model fit is the leading candidate). With the
  kit's priors, three runs gave 91, 96 and 100 failures on the same 5,000 validation shots, while on the test shots
  this branch's run equals the stored file on every shot. Each number here belongs to its run. With the head's priors
  the outcome looked much steadier: two runs agreed on all of the first 1,000 validation shots (job 3 against job 4).
- **Checks against leakage and tuning on test:**
  - the decoder never sees the truth: `decode_one` receives the priors and the syndrome only, and the truth is used
    after decoding, for scoring;
  - the knobs come from train shots only;
  - the head's logits are the registered bytes, every decoded row checked by hash;
  - the test files were decoded once each, after the spec's commit;
  - on test, the same run's kit-priors arm reproduces the stored TCBP-JOBS file shot for shot.
  Not done: a truth-shuffle test of `tcbp.py` itself (`plugin.py` has one for the relay path).
- **`evaluate.py`'s change** (`--precision`) was tested here on the CPU by the existing tests: `test_predecoder.py -k
  "evaluate_then_compare or plugin or held_out"`, 6 pass (lane B, 52 s). Its fp32 GPU branch ran in job 4.
- **Priors capped at 0.5** (JOBS refuses more). With the chosen knobs (alpha 0.4), the cap changed 1,836 columns on
  1,528 of the 5,000 validation shots, about one column per capped shot. Over the whole fit grid it changed 495,062
  columns in 15,052 of 28,680 decodes, most of them at the high alphas, where the head's confident columns pass 0.5
  more often. Whether a JOBS-legal mapping other than a cap would do better was not tried.
- **The latency** is three timers (network, build, decode), each measured one shot at a time in one process on the
  A100, sometimes with another TCBP-JOBS process beside it on the GPU. The p99 of the sum was not measured as one
  timer.
- Not tried: passing H to `get_decoder` as a scipy sparse matrix, which may build faster (tried since: "tcbp-speed"
  above, 7 ms faster with the same answers on 1,000 validation shots); a batch decode with
  per-shot priors (cudaq-qec 0.9.0 has none); the test file at p = 0.005 with GPU logits (the registered run uses
  the CPU bytes).
- The part caches of `tcbp.py` are keyed without the logits' precision (the auditor's note); every run here used a
  fresh output folder.
