## How the frozen decoder is chosen

Written at 06:52 UTC (commit `1972fbc` on `integration`; its author time 06:52 UTC survives the rebases), before the variant runs finished. The lines marked 07:02 were added then
(commit `a4f6c95`, author time 07:11 UTC), before the full-validation scores of the 1-sweep and 4-sweep explain versions and
the last 1,000 shots of the variants were read. What had been seen by then: the first 500 shots of every variant;
the explain version's score with 2 sweeps; and a 400-shot development run of an earlier, faulty build in which
2 and 4 sweeps differed by 0.001.

1. **The tracker version** is the one with the best mean log P(true round) on the 4,000 validation shots (step 2).
   - (07:02) Versions within 0.005 nats of each other are a tie, and the one with fewer sweeps is taken.
   - (07:02) The validation decode of the chosen version is a record, not a selection input. Only if it is
     lopsidedly worse than the earlier version on the same 1,500 shots (paired sign test p < 0.01) is it treated
     as a bug, and the work stops there.
2. **The decoder variant** is the one with the fewest failures on the first 1,500 validation shots, among: the
   soft posterior (`tracker`), two tempered posteriors (`tracker-T2` flatter, `tracker-T0.5` sharper), the single
   most probable round where its posterior is at least 0.6 (`tracker-hard0.6`), and the posterior over the 72
   slots used as it is (`tracker-slot`). Ties: fewer unconverged shots, then fewer mean iterations.
   - These five were decoded with the tracker version `order1_coupled2`, the best one when the runs started. If
     rule 1 picks a later version, the winning variant is decoded once more with that version on the same
     validation shots, for the record; the variant is not re-chosen.
3. A second attempt for unconverged shots (the tracker's priors first, then the uniform guess where relay-BP did
   not converge) is reported as an extra row computed from the per-shot results. It is not the frozen decoder.
4. The test split is decoded once, with the frozen pair, together with `starter`, `exact` and the ORACLE.
5. (07:02) **The primary number** is the share of the oracle's gain on the 4,000 test shots of this PC (seed 7300).
   The same job on a second seed on the team's server (seed 7310), alone and pooled with the test, is secondary.
   No decision is taken from a part of the test.
