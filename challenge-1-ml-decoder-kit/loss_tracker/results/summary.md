## Summary

| | |
|---|---|
| What | A tracker that estimates WHEN each lost data atom was lost, from the detector history of its 6 neighbouring checks (both bases), and gives that estimate to relay-BP as per-shot priors, in place of the published uniform guess. |
| Setting | X memory, p = 0.004, loss s = 1e-3 per atom per CX (about 10 lost data atoms per shot), Clifft shots, relay-BP at the kit's settings. |
| The tracker alone (test shots, against the probes) | its most probable round is the true loss round for 51% of the lost atoms, and within one round of it for 80%. It puts on average 0.38 of its probability on the true round and 0.71 within one round (the uniform guess: 0.083 and 0.24). Calibrated (expected calibration error 0.002). |
| **Decoding, test (seed 7300, 4,000 shots, primary)** | `starter` (uniform guess) 192 failures, `exact` 196, **tracker 147**, ORACLE (true round) 112. Against `starter`: 1.31 times fewer failures (1.17–1.48), paired 75 / 30, p = 1.3e-5. **Share of the oracle's gain recovered: 0.58 (0.38–0.78).** |
| Replication (server, seed 7310, 4,000 shots, secondary) | `starter` 146, `exact` 161, tracker 122, ORACLE 82: share 0.49 (0.25–0.70); 1.20 times fewer failures than `starter` (p = 0.026), 1.32 than `exact`. Pooled with the test (8,000 shots): share 0.54 (0.38–0.68); 1.26 times fewer than `starter` (1.15–1.38, p = 2.6e-6). |
| Target of the brief | recover at least half of the oracle's gain: met on the point estimates (0.58; 0.54 pooled), not with confidence (the intervals reach 0.38). |
| Kill line (step 2) | passed: 78% of the losses of rounds 1 to 9 are placed within one round (line: 50%). |
| Step 4 | The tracking carries over to s = 5e-4 and 2e-3 (80% and 74% of the losses of rounds 1 to 9 within one round) and to Z memory (78%). Tables fitted at s = 1e-3 work as well at the other two rates. A learned tracker (a small GRU) does not beat the model (mean log P −1.390 against −1.325). Cost: 8 ms per shot on one CPU thread in numpy, about 3% of the median relay-BP decode (0.5% of the mean). |
| New? | No paper we found reports a syndrome-derived loss-time estimate feeding a BP decoder (`reference/papers-loss-aware-decoding.md`, section 9). |
