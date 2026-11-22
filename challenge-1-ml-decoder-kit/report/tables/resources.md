What each decoder needed for its row on the test shots. Memory is the peak resident memory of the evaluator's process (the decoder, the shot file and Python together), as the evaluator records it; start-up is building the decoder for both bases, outside the per-shot time. Values are for p = 0.003 / 0.004 / 0.005.

| Decoder | Runs on | Peak memory, MB | Start-up, s | What is learned in it |
|---|---|---|---|---|
| BP-OSD (kit baseline) | not measured | not measured | not measured | nothing |
| relay-BP (kit baseline) | not measured | not measured | not measured | nothing |
| relay-BP with a learned router | not measured | not measured | not measured | a logistic model decides when relay-BP's first answer is enough |
| PriorTilt, one run | not measured | not measured | not measured | 14 parameters, fitted by its authors |
| our hierarchy, lowest-weight tie-break | not measured | not measured | not measured | nothing of ours (PriorTilt's parameters are its authors') |
| our hierarchy, neural tie-break | not measured | not measured | not measured | a neural network picks among the candidates where the fast decodes disagree |
| our ensemble (combined offline) | not measured | not measured | not measured | the same network, choosing among more members |
| TCBP-JOBS (NVIDIA, GPU) | not measured | not measured | not measured | a model it fits at start-up, per memory basis |

The server (the team's ICRN notebook): 12 CPUs by quota, 226 GB of memory, a slice of 19,968 MiB of an NVIDIA A100-SXM4-80GB (`report/data/machines/server_job_000.out`, `server_nvidia-smi.txt`).
