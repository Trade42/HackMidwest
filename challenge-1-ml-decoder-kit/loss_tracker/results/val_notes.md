What the validation shots say (1,500 shots; `exact` fails 69, the ORACLE 31):

- **The soft posterior is the variant with the fewest failures** (45, with the tracker version `order1_coupled2`).
  Flattening it (T = 2: 48), sharpening it (T = 0.5: 52), deciding hard where it is confident (51) and using
  the 72-slot posterior directly (48) are all within a few shots of it. None of these differences is
  significant: two variants differ on 10 to 20 shots, and the counts move by about 7 by chance.
- **Rule 1 and the decoding disagree on these shots.** The version that finds the round best
  (`order1_explain1`: exact round 50.9% against 49.4%, mean log P −1.325 against −1.368) had MORE decoding
  failures here than the version before it: 54 against 45 (22 shots only it fails, 13 only the earlier one
  fails, sign test p = 0.18). The 2-sweep explain version, decoded before the choice, had 51 (17 against 11,
  p = 0.34). Neither is significant, and the rule said in advance that this decode is a record and not a
  selection input, so `order1_explain1` stayed frozen. The earlier version is decoded on the test shots too,
  as a labelled secondary row, to see which of the two the decoder prefers (below).
- **Why it can happen at all:** relay-BP's outcome on the hard shots is touchy. Two posteriors that agree to
  the third digit on average (the 1-sweep and 2-sweep explain versions) still change the outcome of some shots.
  So 1,500 validation shots can rank decoders only to within about 9 failures; they can say that the tracker
  beats the uniform guess (45 to 54 against 69), not which tracker version is best.
- **A second attempt helps.** Decoding again with the uniform guess where the tracker's priors did not converge
  (the row `tracker>exact`) removes 5 more failures in both versions (45 → 40, 54 → 49), with fewer relay-BP
  iterations on average than `exact` alone (2,344 and 2,371 against 2,560). This is an extra row, not the
  frozen decoder.
