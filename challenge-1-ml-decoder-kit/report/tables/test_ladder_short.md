**Whose gain is whose, on the test shots**: failed pairs of 10,000 per noise level, every rung scored on the same per-shot answers (one offline decode: relay-BP and PriorTilt under two seeds on every shot, a fresh decoder per 500 shots; not the timed rows above). `APPENDIX.md` B has each rung's paired test against the rung before it and against relay-BP.

| Rung | Whose work | p = 0.003 | p = 0.004 | p = 0.005 | All three p |
|---|---|---|---|---|---|
| relay-BP (kit baseline), a fresh decoder per 500 shots (the ensemble's member) | the kit | 5 | 90 | 794 | 889 |
| PriorTilt, one run (seed A) | IonQ | 6 | 71 | 595 | 672 |
| our hierarchy, lowest-weight tie-break (nothing learned by us) | ours | 3 | 57 | 527 | 587 |
| our hierarchy, lowest weight among the converged candidates (nothing learned by us) | ours | 3 | 53 | 469 | 525 |
| our hierarchy, neural tie-break | ours | 4 | 50 | 490 | 544 |
| ORACLE: a perfect tie-break on the same candidates (a ceiling, not a decoder) | nobody: needs the truth | 3 | 40 | 386 | 429 |
| TCBP-JOBS (NVIDIA, GPU), for reference | NVIDIA | 3 | 41 | 360 | 404 |
