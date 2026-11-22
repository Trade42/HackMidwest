What each decoder needed for its row on the test shots. Memory is the peak resident memory of the evaluator's process (the decoder, the shot file and Python together), as the evaluator records it; start-up is building the decoder for both bases, outside the per-shot time. Values are for p = 0.003 / 0.004 / 0.005.

| Decoder | Runs on | Peak memory, MB | Start-up, s | What is learned in it |
|---|---|---|---|---|
| BP-OSD (kit baseline) | CPU, 1 process, 1 thread | 284 / 284 / 284 | 0.1 / 0.2 / 0.1 | nothing |
| relay-BP (kit baseline) | CPU, 1 process, 1 thread | 284 / 284 / not measured | 0.0 / 0.0 / not measured | nothing |
| relay-BP with a learned router | CPU, 1 process, 1 thread | 284 / 284 / not measured | 0.0 / 0.0 / not measured | a logistic model decides when relay-BP's first answer is enough |
| PriorTilt, one run | not measured | not measured | not measured | 14 parameters, fitted by its authors |
| our hierarchy, lowest-weight tie-break | not measured | not measured | not measured | nothing of ours (PriorTilt's parameters are its authors') |
| our hierarchy, classical tie-break (lowest weight among the converged candidates) | not measured | not measured | not measured | nothing of ours (PriorTilt's parameters are its authors') |
| our hierarchy, neural tie-break | CPU, 1 process, 1 thread | 781 / not measured / not measured | 5.1 / not measured / not measured | a neural network picks among the candidates where the fast decodes disagree |
| our hierarchy, neural tie-break, the two fast decodes side by side on two cores | CPU, 1 process, 1 thread | 778 / not measured / not measured | 11.3 / not measured / not measured | the same decoder as the row above, with the same answers on every shot (checked); only the wall time differs |
| our ensemble (combined offline) | not measured | not measured | not measured | only the rules that use the network have a learned part: see the rule named in the row |
| TCBP-JOBS (NVIDIA, GPU) | GPU, 1 process, 1 thread | not measured / not measured / not measured | not measured / not measured / not measured | a model it fits at start-up, per memory basis |
