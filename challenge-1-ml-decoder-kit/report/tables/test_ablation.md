**The ablation: what the learned part adds.** The same hierarchy on the same candidates, with the best tie-break that needs no learning (lowest weight among the converged candidates) and with the neural network. Only the shots where the two fast decodes disagree can differ.

| p | Counted over | Shots where the two fast decodes disagree | Failures, classical tie-break | Failures, neural tie-break | Only the network rule fails / only the classical rule fails | Exact p |
|---|---|---|---|---|---|---|
| 0.003 | X memory, shots | 2 | 2 | 2 | 0 / 0 | 1.00 |
| 0.003 | Z memory, shots | 2 | 1 | 2 | 1 / 0 | 1.00 |
| 0.003 | pairs (the kit's score) |  | 3 | 4 | 1 / 0 | 1.00 |
| 0.004 | pairs | not measured | not measured | not measured | not measured | not measured |
| 0.005 | pairs | not measured | not measured | not measured | not measured | not measured |
| 0.003 together | pairs (the kit's score) |  | 3 | 4 | 1 / 0 | 1.00 |
