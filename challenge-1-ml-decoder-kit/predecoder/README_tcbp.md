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

(filled in)

## What was not done

(filled in)
