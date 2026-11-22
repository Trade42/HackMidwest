What the test says (4,000 shots, seed 7300; every condition on the same shots):

- **The tracker beats the published uniform guess**, significantly: 147 failures against 192 for `starter`
  (paired 75 / 30, sign test p = 1.3e-5, ratio 1.31, bootstrap 1.17–1.48) and against 196 for `exact` (81 / 32,
  p = 4.5e-6, ratio 1.33, 1.18–1.51).
- **It recovers 58% of the oracle's gain** (bootstrap 38–78%): (196 − 147) / (196 − 112). The brief's target was
  at least half. The point estimate meets it; the interval does not exclude less than half.
- **Fewer relay-BP iterations**: 2,141 on average against 2,599 for `exact` (the ORACLE 1,684), and fewer
  unconverged shots (101 against 145).
- **A second attempt** (the rows with `>`, computed from the per-shot results, not the frozen decoder): the tracker
  first, then the uniform guess where relay-BP did not converge, fails 128 times. Given the same second attempt,
  `exact` fails 158 and the ORACLE 106. Like for like, that is (158 − 128) / (158 − 106) = 0.58 of the gain again.
  The second attempt helps every decoder about equally, and it does not change the tracker's share.
