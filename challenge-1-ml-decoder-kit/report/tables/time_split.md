Where our decoder's time goes on the test shots, per pair (an X shot and a Z shot), averaged over all 10,000 pairs. The two PriorTilt runs decode every shot; relay-BP and the tie-break run only on the shots where the two runs disagree. From the plugin's own per-shot timers; the last column is the evaluator's timer, which wraps everything.

| p | Row | Shots where the two runs disagree | Two PriorTilt runs, ms | relay-BP, ms | the network, ms | Whole decode, ms (the evaluator) |
|---|---|---|---|---|---|---|
| 0.003 | our hierarchy, neural tie-break | 2 of 20,000 | 45.0 | 0.5 | 0.0 | 45.7 |
| 0.004 | our hierarchy, neural tie-break | 90 of 20,000 | 178 | 37.6 | 0.2 | 216 |
| 0.005 | our hierarchy, neural tie-break | 609 of 20,000 | 540 | 282 | 1.7 | 825 |
