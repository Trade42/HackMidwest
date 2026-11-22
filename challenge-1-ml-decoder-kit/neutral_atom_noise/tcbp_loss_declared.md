# The TCBP pipeline with atom loss: declared before any TCBP number

**Label: exploration, after the main test results; a separate branch (`infleqtion-noise`).** Written 2026-11-22
about 16:35 UTC by the implementer of task `tcbp-loss`, before the server job was written and before any TCBP-JOBS
decode of a loss shot existed. Every number below is reported whatever it shows.

## The shots

Question 2's X loss shot files (Clifft through `loss/`, flags perfect, seed 6100, made by `make_shots.py --loss` on
luminaty, Clifft 0.13.0, Stim 1.16.0), the files of this branch's relay-BP loss rows:

| file | shots | sha256 | Pauli noise | loss per CX | generic-2q p |
|---|---|---|---|---|---|
| `X_f0.4_loss_seed6100.npz` | 2,000 | `9c225b77e7e5a2913267843ed0e2e11518403fa2e47afcb462eeb2a7600d92fd` | factor 0.4 | 0.0010 | 0.0059563 |
| `X_f0.6_loss_seed6100.npz` | 2,000 | `bcdcfdf177d27910b3096a2c1948034e6731fc876d68c861847864361c1e2282` | factor 0.6 | 0.0015 | 0.0089274 |

**The declared prefix: the first 1,000 shots of each file** (shots 0 to 999). The f0.6 file's relay-BP
matched + flags row covers exactly these 1,000 shots, and 2 files x 2,000 shots x 4 arms would not fit the server
window. The stopping rule below may cut it further.

The predecoder's tools read only the kit's shot format, so `tcbp_loss.py convert` writes the same detector bits in
that format (`p` as below); the heralds stay out of the network's input.

## The arms (TCBP-JOBS, cudaq-qec 0.9.0, preset `tcbp-jobs`, one decoder built per shot, one process, the A100)

All four decode the kit's H and L (936 x 8,784, X memory). Only the prior vector differs.

- **(bm) matched, loss-blind:** `priors.matched_priors("X", noise)` of the file's own Pauli noise (floor 1e-9; the
  loss itself is not in it), capped at 0.5. This is the vector of `loss/`'s condition `blind`, the branch's relay-BP
  `matched__blind` row. The heralds are not used.
- **(fm) matched + flags:** (bm)'s matched vector with `loss/`'s flag-aware raise for the shot, condition `starter`
  (`loss.lossmap.PriorBuilder.priors`, the recipe the branch's question 2 used for relay-BP): every flagged atom's
  columns get their bumps (own-flip 0.5, the five hook columns 0.5/6 each, a flagged data atom's twelve per-cycle
  columns 0.5 (c + 0.5)/12 and its final column 0.5), XOR-combined with the base,
  `P = (1 - (1 - 2 base) * prod(1 - 2 v)) / 2`, then capped at 0.5.
- **(fh) head + flags, the TCBP version with loss information:** the network and head of the TCBP pipeline (trunk
  `d8d1c70f...`, head `38ab667e...`, run on the A100 in float32) at the **frozen TCBP knobs** of
  `predecoder/logs/fit_tcbp_X_p0.005.json` on `predecoder-tcbp` (sha256 `7697010678a3...`): log-odds mixing,
  alpha 0.4, eps 1e-4 (`predecoder.priors.shot_priors`). **No refit.** Then the SAME flag raise as (fm), applied
  on top of the head's per-shot vector exactly as on the matched vector (the same columns, the same bumps, the same
  XOR combination), then capped at 0.5.
  - **The base p the head mixes with, and the network's p input: the file's generic-2q p** (the kit's uniform model
    at the p whose two-qubit error equals the file's total two-qubit Pauli error, `noise.two_qubit_error()`):
    **0.0059563 for f0.4, 0.0089274 for f0.6.** No p was ever fitted on these loss files; generic-2q is the
    branch's own unfitted generic picture of them, and it is 1.2 and 1.8 times the top of the network's training
    range (0.003 to 0.005).
- **(bh) head, loss-blind (if time allows):** (fh) without the flag raise: the head's vector, capped at 0.5.

**fh vs fm is not the head alone.** (fh) starts from the kit's uniform model at generic-2q p and (fm) from the
matched model, so fh vs fm measures the network on top of a generic base against no network on the matched base.
Beside it we put the branch's relay-BP `generic-2q__starter` row (the same uniform base, the same flags, no network).

## The pairs

Exact two-sided sign tests on the shots where exactly one arm fails (binomial, one half):

- **fm vs bm:** what the flags add to TCBP-JOBS;
- **fh vs fm:** the network on top of matched + flags (with the base caveat above);
- **fh vs bh**, if (bh) ran: what the flags add on top of the head.

Beside them, on the same shots (the shots both decoded): this branch's relay-BP rows on these files, recounted from
their part files: `relay__matched__starter` (f0.4: 2,000 shots, f0.6: 1,000), `relay__matched__blind` (f0.4: 300,
f0.6: 100) and `relay__generic-2q__starter` (f0.4: 1,000, f0.6: 150). Each TCBP arm is paired with its relay-BP
counterpart: fm with relay matched + starter, bm with relay matched blind, fh with relay generic-2q + starter.
Every failure count has its 95% Wilson interval. A failure is a shot with at least one of the 12 logical bits wrong
(X memory only; no X+Z pairs here). Each file is reported on its own.

## The stopping rule

Decode time on loss shots is unmeasured (the no-loss f0.8 file took about 80 ms per shot and arm). One process,
one fitted model for both files. **Pass 1:** blocks of 100 shots, in the order (f0.4 shots 0-99, f0.6 shots 0-99,
f0.4 shots 100-199, ...), the arms bm, fm, fh one after the other inside each block. **Pass 2**, after pass 1 has
finished every block: (bh) on the same blocks in the same order. No new block of either pass starts after
17:35 UTC. **The scored set of bm, fm and fh on a file is the longest prefix of whole pass-1 blocks of that file;
bh's is its longest prefix of whole pass-2 blocks.** fh vs bh is scored on bh's set only.

## The expectation, stated before any number

The network never saw a lost atom. A lost atom's readout is a random bit, and the detectors next to it fire as
they never did in training, so **the head may mislead on those columns**: (bh) may be worse than (bm), and the
head's priors could fight the flags. **The flags should help TCBP-JOBS as much as they help relay-BP** (fm better
than bm), because the raise tells the decoder exactly which atoms to suspect. Whether (fh) beats (fm) is open: the
head saw only kit noise at p 0.003 to 0.005, a different shape from this noise, and it starts from a generic base.
