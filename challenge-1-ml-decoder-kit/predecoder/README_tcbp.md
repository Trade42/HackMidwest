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

(further rows filled in as they arrive)

## What was not done

(filled in)
