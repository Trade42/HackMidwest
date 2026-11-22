# predecoder-tcbp: the head's per-shot priors fed to TCBP-JOBS instead of relay-BP

**A separate branch** (`predecoder-tcbp`, from `predecoder` at `e96cc020`). Sal, 14:39 UTC: "i do want to force to use
tcbp since its just better than relay bp ... we should retrain fully for the tcbp instead of the relay bp and change our
pipeline". It merges nowhere unless Sal says so.

**The idea.** The head (`head.py`, unchanged, sha256 `38ab667e...`) reads the network's view of a shot and gives every
column of H a probability that it fired in that shot. `priors.py` mixes those probabilities with the kit's priors, using
two knobs (log-odds alpha, eps). Until now the mixed priors went to relay-BP. This branch gives them to TCBP-JOBS,
NVIDIA's GPU decoder (cudaq-qec 0.9.0, `nv-qldpc-decoder`, preset `tcbp-jobs`), which on its own was already the
team's best single decoder. The head was not retrained. The knobs were refitted for TCBP-JOBS on train shots only.

RESULTS: (filled in below as they arrive)

| file | what it does |
|------|--------------|
| `tcbp.py` | `decode`: TCBP-JOBS with one decoder built per shot from that shot's priors (arms `fresh`: the kit's priors; `head`: the head's). `fit`: the knob fit on train subsets. `score`: the counts and paired tests, on this PC, from the per-shot files. `probe`: prints a built decoder's interface |
| `test_tcbp.py` | tests without a GPU: the kit's relay-BP stands in for TCBP-JOBS behind the same calls |
| `logs/tcbp_declarations.md` | what was fixed before each number: the fit, the arms, the test run |

## How it works

- **One model, one decoder per shot.** `nv_qldpc.init_tcbp_model(H)` does not take the priors. It is fitted once per run
  (about 2 minutes on the A100), and every shot's decoder shares it. A decoder takes its priors only when it is built
  (`error_rate_vec`), so each shot gets its own: `get_decoder(..., error_rate_vec=<shot priors>, **preset)`, then
  `decode(syndrome)`. Building takes about 51 ms on the A100. That time is counted in each shot's latency.
- **Two environments, split through a file.** The network and head run in an environment with torch
  (`predecoder.evaluate`, here on luminaty's CPU in float32) and write `logits.npy`. TCBP-JOBS runs in the teammates'
  environment on the server, which has no torch, and reads those exact bytes, checked by sha256. The GPU path of
  `evaluate.py` runs in float16, and that moved 33 of 5,000 relay-BP outcomes. Every decoded shot records a hash of the
  logits row it used, and `score` checks those hashes against this PC's file.
- **The control.** The stored TCBP-JOBS answers (`server-logs`, `files/tcbp_p0.005/*.tcbp.npz`) came from one decoder,
  built once and run with `decode_batch`. The `fresh` arm is the same decoder built per shot and run with `decode()`.
  If the two disagree, the per-shot path itself changes answers, and that is reported first.

## Commands

(filled in)

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
  on 44 shots: both fail 79, only per-shot 17, only stored 12, exact two-sided p = 0.46. **TCBP-JOBS does not repeat
  itself across runs on hard shots**, the way relay-BP does not across seeds. Which of three causes does this (another
  model fit, batch against single decoding, random draws inside JOBS) is measured in job 3.

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
do not join. Frozen at 15:28 UTC in `logs/fit_tcbp_X_p0.005.json` (commit `86259291`), from the fit's log
(`logs/tcbp_fit_log_at_freeze.log`), before any validation decode with it.

The control row is the "fresh" check on train. The 390 shots were chosen as the stored run's failures, yet TCBP-JOBS
with the same kit priors, built per shot, gets 49 of them right and breaks 5 others. As with relay-BP, the train
subsets overstate any gain: part of what a setting "fixes" is the decoder's own run-to-run change. The fit ranks
settings; validation gives the unbiased count. TCBP-JOBS takes alpha 0.4, against relay-BP's 0.2 at p = 0.004.

(further rows filled in as they arrive)

## What was not done

(filled in)
