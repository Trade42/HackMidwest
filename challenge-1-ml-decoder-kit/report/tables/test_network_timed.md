**How much the network does, timed against timed.** Our decoder's timed row (the hierarchy with the neural tie-break) against the same hierarchy with a hand-written tie-break in place of the network, each a timed row of the team's evaluator on the same 10,000 test pairs, in one process ("1 core"; "2 cores": the two fast decodes side by side). Only the shots where the two fast decodes disagree can differ. A row marked "run after the test results were seen (an ablation)" is an ablation the team asked for after the test results were seen, not our decoder; the rows at p = 0.003 and 0.004 of the best hand-written rule on one core ran in the test run itself, among its pre-registered arms. Times are comparable to a few percent (`APPENDIX.md` A).

| p | Cores | Without the network: the tie-break | Failed pairs: with the network against without | Both fail | Only with the network fails / only without | Exact p | Mean time per pair, ms: with against without |
|---|---|---|---|---|---|---|---|
| 0.003 | 1 core | the best hand-written rule (lowest weight among the converged candidates) | 3 against 4 | 3 | 0 / 1 | 1.00 | 45.7 against 46.3 |
| 0.003 | 1 core | plain lowest weight [run after the test results were seen (an ablation)] | 3 against 4 | 3 | 0 / 1 | 1.00 | 45.7 against 46.9 |
| 0.003 | 2 cores | the best hand-written rule (lowest weight among the converged candidates) [run after the test results were seen (an ablation)] (each gives its one-core row's answers on every shot) | 3 against 4 | 3 | 0 / 1 | 1.00 | 28.9 against 29.0 |
| 0.004 | 1 core | the best hand-written rule (lowest weight among the converged candidates) | 44 against 52 | 36 | 8 / 16 | 0.15 | 216 against 218 |
| 0.004 | 1 core | plain lowest weight [run after the test results were seen (an ablation)] | 44 against 60 | 39 | 5 / 21 | 0.002 | 216 against 215 |
| 0.004 | 2 cores | the best hand-written rule (lowest weight among the converged candidates) [run after the test results were seen (an ablation)] (each gives its one-core row's answers on every shot) | 44 against 52 | 36 | 8 / 16 | 0.15 | 150 against 150 |
| 0.005 | 1 core | the best hand-written rule (lowest weight among the converged candidates) [run after the test results were seen (an ablation)] | 479 against 481 | 414 | 65 / 67 | 0.93 | 825 against 815 |
| 0.005 | 1 core | plain lowest weight [run after the test results were seen (an ablation)] | 479 against 553 | 434 | 45 / 119 | 7e-09 | 825 against 817 |
| 0.005 | 2 cores | the best hand-written rule (lowest weight among the converged candidates) [run after the test results were seen (an ablation)] (each gives its one-core row's answers on every shot) | 479 against 481 | 414 | 65 / 67 | 0.93 | 608 against 604 |
| 0.003, 0.004, 0.005 together | 1 core | the best hand-written rule (lowest weight among the converged candidates) | 526 against 537 |  | 73 / 84 | 0.42 |  |
| 0.003, 0.004, 0.005 together | 1 core | plain lowest weight | 526 against 617 |  | 50 / 141 | 3e-11 |  |
