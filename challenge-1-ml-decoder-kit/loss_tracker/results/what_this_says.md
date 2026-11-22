Only what the numbers support:

1. **The syndrome history tells when a lost atom was lost.** The neighbouring checks start to flicker within a
   round of the loss, and a change-point model over the 6 checks puts 80% of the losses within one round,
   calibrated. The accuracy does not fall for late losses (the herald helps there). A single round says little
   (a lost atom's triple still shows "nothing fires" in 20 to 34% of rounds; noise fires a present atom's triple
   in 35%), so the timing comes from several rounds, and losses in the middle rounds are the hardest.
2. **Most lost atoms are not alone** (78% share a check with another lost atom), and coupling the atoms matters
   more than any other piece of the model (mean log P −1.536 → −1.368). Explaining a shared bit from the other
   atom's own detectors, instead of discarding it, helps the tracking further (−1.368 → −1.325).
3. **The tracker's priors beat the published uniform guess with relay-BP**, on two independent 4,000-shot sets:
   147 failures against 192 for `starter` on the test (1.31 times fewer, paired p = 1.3e-5) and 122 against 146
   on the replication (1.20 times, p = 0.026); against `exact`, 1.33 and 1.32 times (p = 4.5e-6 and 5e-4).
4. **It recovers about half of what the true loss round is worth**: 0.58 of the oracle's gain on the test
   (0.38–0.78), 0.49 on the replication, 0.54 pooled (0.38–0.68). The brief's target (at least half) is met on
   the point estimates; with these shot counts it is not established.
5. **Better tracking did not show up as better decoding** between the two best tracker versions. The version
   that explains shared bits finds the round more often, but its failures are within noise of the version
   before it: 54 against 45 on validation, 147 against 144 on the test (42 / 39 discordant, p = 0.82), 122 against
   113 on the replication (45 / 36, p = 0.37). No variant of how the posterior
   is fed to relay-BP (tempered, hard, per slot) beat the plain soft posterior on validation. With 1,500 to
   4,000 shots these comparisons resolve only differences of about 10 to 20%.
6. **The gain left** sits in shots where several atoms are placed more than one round off: 3.0 such atoms per
   shot where the tracker and `exact` fail and the ORACLE does not, against 2.0 on average.
7. These are exploration numbers at one point (X memory, p = 0.004, s = 1e-3, perfect heralds, independent
   loss), on simulated shots. Step 4 says how far the tracker carries to other points.
