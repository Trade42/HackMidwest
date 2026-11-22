**Exploration shots, not test shots: the predecoder's time per X shot.** The team's server, one process and one thread (server job cpu/050), the first 1,000 X shots of `data/p0.005/validation_X.npz` (so no test file was decoded twice); the head and the final X network pinned by sha256 as in the test run. Per X shot, not per pair.

| Part | Mean, ms per X shot | Median | 99th percentile |
|---|---|---|---|
| the network and the head, on the CPU | 26.8 | 26.7 | 28.0 |
| relay-BP with the head's priors, built and run per shot | 185 | 30.1 | 5,467 |
| the two together (the predecoder's whole decode of an X shot) | 212 |  |  |

Relay-BP converged on 992 of the 1,000 shots (142 ms on those) and not on 8 (5,544 ms on those). For scale only, other shots and so no paired comparison: on the test X shots at p = 0.005, relay-BP's timed row takes 760 ms per X shot (its 1,471 ms per pair is an X shot plus a Z shot) and TCBP-JOBS 8.7 ms per X shot (a GPU decode call alone, not comparable with a CPU time).

Source: `report/data/exploration/predecoder/latency_validation_X_p0.005/` (`summary.json`, `per_shot.npz`, `latency.txt`) and `job_cpu_050-predecoder-latency.out`.
