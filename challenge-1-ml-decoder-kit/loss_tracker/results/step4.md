What step 4 says:

- **The tracker carries to other loss rates and to Z memory.** Within one round of the truth, for the losses of
  rounds 1 to 9 (kill line 50%):
  - X memory: 79.7% at s = 5e-4 (5 lost atoms per shot), 77.9% at s = 1e-3 (10), 74.0% at s = 2e-3 (19);
  - Z memory at s = 1e-3: 77.9%.
  Over all atoms the exact round is found for 53%, 51% and 46% at the three loss rates in X, and 51% in Z. At
  s = 2e-3, 82% of the lost atoms share X and Z checks with another lost atom (54% at 1e-3), which costs a few
  points. The shape by round is the same everywhere: best at both ends, hardest in the middle.
- **The tables carry over between loss rates.** Tables fitted at s = 1e-3 track the s = 2e-3 and s = 5e-4 shots as
  well as tables fitted at those rates (mean log P −1.446 against −1.446, and −1.254 against −1.257). So the
  tracker does not need refitting when the loss rate changes by a factor of 2. Whether it carries to another
  Pauli strength p was not measured.
- **These are tracking numbers only.** No decoding was run at the other loss rates or in Z memory.
- **The learned tracker does not beat the model** (above), so it was not decoded.
- **Cost:** the frozen tracker takes 8.1 ms per shot (99th percentile 13.9 ms), one shot at a time, in numpy on
  one CPU thread of this laptop-class CPU, and 4.4 ms per shot in batches of 100 (table above). The timer wraps
  the features and the posteriors of all the shot's flagged atoms (about 10). For comparison, the flag-aware
  relay-BP decode takes a median of 248 ms per shot on one CPU (`loss/RESULTS.md`, L4), so the tracker adds about
  3% of the median decode, and 0.5% of the mean (1,785 ms). The tracker's priors also cut the mean relay-BP iterations, from 2,599 to 2,141 on the test.
