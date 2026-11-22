# tcbp-fastpath, round 3 and the p = 0.004 addendum, written before their numbers

Exploration, after the test results; a separate branch (`tcbp-fastpath`). Validation only; no test file is opened.
Committed before any number about these features was computed, not even their range, and before the p = 0.004 server
files (server job ctl/180, started 18:35:58 UTC) were fetched. The commit time decides.

## 1. Round 3: two more checks on the network's view (both p)

This is the third look at the same 5,000 validation shots at p = 0.005, so a pass here is weaker evidence than a
pass in round 1. Round 2's C3 and C5 were swamped by downward moves, where the head is surer than the kit that a
column did not fire. These two checks are signed or unit-free, so that one threshold can mean the same thing at
another p and in another run:
- **C8, the upward move:** s = the sum over the 8,784 columns of max(0, logit(q_j) - logit(pi_j)). q and pi are as in
  `CRITERION_2.md`: the shot's TH priors exactly (mix, clip, cap) and the kit's priors. This is the evidence the head
  adds for faults the kit does not expect.
- **C9, the head's own uncertainty:** s = the sum over the 8,784 columns of the binary entropy, in bits, of the
  head's raw probability sigmoid(logit_j), before any mixing. A shot the head is sure about, either way, scores low.

Easy when s <= t. Same gate (0 added failures on all 5,000 shots), same t* (the largest t that passes), same time
model as C3-C5 (network on every shot, then F on easy or TH on hard). The same diagnostics: half split, and the other
F runs at p = 0.005.

The bar, from round 2's numbers: a network-first check pays 9.4 ms on every shot. It beats C2 (49.3 ms) only with
about 5% of the shots on the fast path, and C6 (43.4 ms) only with about 20%. Its p99 stays near TH's 63 ms.

The choice at p = 0.005 is redone over every check of rounds 1 to 3 by the same rule: the lowest mean among the
checks at t* that pass, and TH alone.

## 2. Addendum to `CRITERION_2.md` Part B (p = 0.004)

- **Little power.** On test at p = 0.004, TH failed 4 and TCBP-JOBS 20 of 10,000. So validation may hold only about
  ten F-only shots, and every check will find a large "safe" share. A pass at p = 0.004 cannot by itself show that
  accuracy is kept. The standing rule applies: a result that looks too good is a bug until shown otherwise.
- Reported beside the choice at p = 0.004:
  - the 95% upper bound on the added-failure rate among the shots on the fast path (3 / fast shots, for 0 seen);
  - TH's own failure rate on the same shots, beside it.
- **Thresholds carried from one p to the other**, both ways, for every check that has a t* at the p it comes from (C2 to C9):
  - each p = 0.005 t*, unchanged, on the p = 0.004 shots: the added failures and the fast share;
  - each p = 0.004 t* on the p = 0.005 shots, where 75 F-only shots give a real test.
- C6's carry is confounded by load: both server jobs shared the GPU with other work. C2, C7, C8 and C9 are not.
